from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.optimize import linear_sum_assignment

from bayes_mapping.acquisition.peak_finding import find_slice_peak_idxs

type FeatureFinder = Callable[..., npt.NDArray[np.intp]]
type MetricLike = str | Enum



def calc_dispersion(
    samples: torch.Tensor,
    coordinates: torch.Tensor,
    distance_metric: MetricLike,
    dispersion_metric: MetricLike,
    feature_finder: FeatureFinder = find_slice_peak_idxs,
    feature_finder_kwargs: dict[str, Any] | None = None,
    distance_metric_kwargs: dict[str, Any] | None = None,
    dispersion_metric_kwargs: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Calculate dispersion among feature sets extracted from samples.
    """
    feature_finder_kwargs = feature_finder_kwargs or {}
    distance_metric_kwargs = distance_metric_kwargs or {}
    dispersion_metric_kwargs = dispersion_metric_kwargs or {}

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape "
            "(n_mc_samples, n_temp_eval), "
            f"but received shape {tuple(samples.shape)}."
        )

    if coordinates.ndim != 1:
        raise ValueError(
            "coordinates must have shape (n_temp_eval,), "
            f"but received shape {tuple(coordinates.shape)}."
        )

    if samples.shape[1] != coordinates.numel():
        raise ValueError(
            "The final dimension of samples must match the number "
            f"of coordinates: {samples.shape[1]} != "
            f"{coordinates.numel()}."
        )

    if samples.shape[0] < 2:
        raise ValueError(
            "At least two posterior samples are required to "
            "calculate feature dispersion."
        )

    coordinates = coordinates.to(
        device=samples.device,
        dtype=samples.dtype,
    )

    metric_name = _metric_name(dispersion_metric)
    squared = bool(dispersion_metric_kwargs.get("squared", False))

    feature_sets = [
        find_features(
            sample=sample,
            coordinates=coordinates,
            feature_finder=feature_finder,
            feature_finder_kwargs=feature_finder_kwargs,
        )
        for sample in samples
    ]

    if metric_name == "pairwise":
        return _calc_pairwise_dispersion(
            feature_sets=feature_sets,
            distance_metric=distance_metric,
            distance_metric_kwargs=distance_metric_kwargs,
            squared=squared,
        )

    if metric_name == "rand_pair":
        if generator is None:
            raise ValueError("A torch.Generator is required for rand_pair dispersion.")

        return _calc_rand_pair_dispersion(
            feature_sets=feature_sets,
            distance_metric=distance_metric,
            distance_metric_kwargs=distance_metric_kwargs,
            squared=squared,
            generator=generator,
        )

    raise NotImplementedError(
        f"Dispersion metric {dispersion_metric!r} is not implemented."
    )


def _calc_pairwise_dispersion(
    feature_sets: list[torch.Tensor],
    distance_metric: MetricLike,
    distance_metric_kwargs: dict[str, Any],
    squared: bool = False,
) -> torch.Tensor:
    """
    Calculate the mean distance over all unordered pairs of feature sets.
    """
    pairwise_distances = []

    for i in range(len(feature_sets)):
        for j in range(i + 1, len(feature_sets)):
            distance = calc_distance(
                features_a=feature_sets[i],
                features_b=feature_sets[j],
                distance_metric=distance_metric,
                **distance_metric_kwargs,
            )

            distance = _as_scalar_tensor(
                distance,
                reference=feature_sets[i],
            )

            if squared:
                distance = distance.square()

            pairwise_distances.append(distance)

    if not pairwise_distances:
        raise ValueError("Pairwise dispersion requires at least two feature sets.")

    return torch.stack(pairwise_distances).mean()


def _calc_rand_pair_dispersion(
    feature_sets: list[torch.Tensor],
    distance_metric: MetricLike,
    distance_metric_kwargs: dict[str, Any],
    generator: torch.Generator,
    squared: bool = False,
) -> torch.Tensor:
    """
    Estimate dispersion using randomly selected disjoint pairs.
    """
    if len(feature_sets) < 2:
        raise ValueError("Random-pair dispersion requires at least two feature sets.")

    remaining_indices = torch.randperm(
        len(feature_sets),
        generator=generator,
    ).tolist()

    rand_pair_distances = []

    for k in range(0, len(remaining_indices) - 1, 2):
        i = remaining_indices[k]
        j = remaining_indices[k + 1]

        distance = calc_distance(
            features_a=feature_sets[i],
            features_b=feature_sets[j],
            distance_metric=distance_metric,
            **distance_metric_kwargs,
        )

        distance = _as_scalar_tensor(
            distance,
            reference=feature_sets[i],
        )

        if squared:
            distance = distance.square()

        rand_pair_distances.append(distance)

    return torch.stack(rand_pair_distances).mean()


def find_features(
    sample: torch.Tensor,
    coordinates: torch.Tensor,
    feature_finder: FeatureFinder = find_slice_peak_idxs,
    feature_finder_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """
    Find features in one sampled curve and return normalized locations.

    The feature finder is expected to return indices into ``sample``.
    """
    feature_finder_kwargs = feature_finder_kwargs or {}

    feature_indices = feature_finder(
        sample,
        **feature_finder_kwargs,
    )

    feature_indices = torch.as_tensor(
        feature_indices,
        device=coordinates.device,
        dtype=torch.long,
    ).reshape(-1)

    if feature_indices.numel() == 0:
        return coordinates.new_empty((0,))

    if torch.any(feature_indices < 0) or torch.any(
        feature_indices >= coordinates.numel()
    ):
        raise IndexError(
            "feature_finder returned indices outside the coordinate array: "
            f"indices={feature_indices.tolist()}, "
            f"n_coordinates={coordinates.numel()}."
        )

    return coordinates[feature_indices]


def _metric_name(metric: MetricLike) -> str:
    """
    Convert either a string or a string-valued enum to a plain string.
    """
    value = getattr(metric, "value", metric)
    return str(value).lower()


def _as_scalar_tensor(
    value: object,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a distance result to a scalar tensor with consistent placement.
    """
    value = torch.as_tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    )

    if value.numel() != 1:
        raise ValueError(
            "Distance functions must return one scalar, "
            f"but received shape {tuple(value.shape)}."
        )

    return value.reshape(())


def calc_distance(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    distance_metric: MetricLike,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Calculate the distance between two feature sets.
    """
    metric = str(getattr(distance_metric, "value", distance_metric)).lower()

    if metric == "gospa":
        return _gospa_distance(
            features_a,
            features_b,
            **kwargs,
        )

    raise NotImplementedError(f"Distance metric {distance_metric!r} not implemented.")


def _gospa_distance(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    cutoff: float = 0.2,
    p: float = 2,
    alpha: float = 2,
) -> torch.Tensor:
    """
    Calculate the GOSPA distance between two finite feature sets.

    Parameters
    ----------
    features_a
        Tensor of feature locations with shape (n_features_a,).

    features_b
        Tensor of feature locations with shape (n_features_b,).

    cutoff
        Maximum localization penalty.

    p
        Exponent used by GOSPA.

    alpha
        Cardinality penalty divisor. The standard GOSPA choice is 2.

    Returns
    -------
    torch.Tensor
        Scalar GOSPA distance.
    """

    device = features_a.device
    dtype = features_a.dtype

    n_a = len(features_a)
    n_b = len(features_b)

    # Handle empty sets.
    if n_a == 0 and n_b == 0:
        return torch.zeros((), device=device, dtype=dtype)
    if n_a == 0 or n_b == 0:
        mismatch = abs(n_a - n_b)
        cost = (cutoff**p / alpha) * mismatch
        return torch.tensor(
            cost,
            device=device,
            dtype=dtype,
        ).pow(1.0 / p)

    # Pairwise localization costs.
    pairwise = torch.cdist(
        features_a[:, None],
        features_b[:, None],
        p=2,
    )
    pairwise = torch.clamp(pairwise, max=cutoff)

    # Solve optimal assignment.
    row_ind, col_ind = linear_sum_assignment(pairwise.detach().cpu().numpy())
    localization_cost = pairwise[row_ind, col_ind].pow(p).sum()

    # Cardinality penalty.
    mismatch = abs(n_a - n_b)
    cardinality_cost = (cutoff**p / alpha) * mismatch

    # Sum and return cost.
    total_cost = localization_cost + cardinality_cost
    return total_cost.pow(1.0 / p)
