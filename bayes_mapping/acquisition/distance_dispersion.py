import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gpytorch
import torch
from jaxtyping import Float
from scipy.special import roots_hermitenorm
from tqdm import tqdm

from bayes_mapping import acquisition
from bayes_mapping.acquisition import spatial_statistics
from bayes_mapping.utils import eval_context

if TYPE_CHECKING:
    from bayes_mapping.active_learning import ActiveLearner


@dataclass
class DispersionGridSpec:
    """Full evaluation grid and row indices for discrete theta slices.

    Attributes
    ----------
    x_eval
        Normalized evaluation coordinates with shape
        ``(n_theta * n_temp_eval, 2)``.
    idxs_by_theta
        Mapping from each raw integer theta value to the corresponding rows
        of ``x_eval``.
    """

    x_eval: torch.Tensor
    idxs_by_theta: dict[int, slice]


class DistanceDispersion:
    """Score candidates by expected reduction in feature dispersion.

    Feature dispersion is estimated from posterior function samples over a
    local window of discrete theta slices. For each candidate, possible 
    observations are generated either by Monte Carlo sampling or
    Gauss-Hermite quadrature. Gaussian conditioning updates the mean for each
    fantasy value and shares one covariance factor per candidate. Common normal
    draws and sample pairings couple baseline and conditioned estimates.
    The score is baseline dispersion minus expected posterior dispersion.

    Parameters
    ----------
    learner
        Active learner containing the GP model, configuration, and data
        generator.
    generator
        Optional random-number generator used for acquisition sampling.
    """

    def __init__(
        self,
        learner: "ActiveLearner",
        generator: torch.Generator | None = None,
    ):
        self.learner = learner
        self.generator = generator

        self.n_fantasies = int(
            getattr(
                self.learner.config,
                "acquisition_n_fantasies",
                10,
            )
        )
        self.n_mc_samples = int(
            getattr(
                self.learner.config,
                "acquisition_n_mc_samples",
                100,
            )
        )
        self.n_temp_eval = int(
            getattr(
                self.learner.config,
                "acquisition_n_temp_eval",
                100,
            )
        )
        self.sample_jitter = float(
            getattr(
                self.learner.config,
                "acquisition_sample_jitter",
                1e-7,
            )
        )

        self.distance_metric = self.learner.config.distance_metric
        self.dispersion_metric = self.learner.config.dispersion_metric
        self.distance_metric_kwargs = self.learner.config.distance_metric_kwargs
        self.dispersion_metric_kwargs = self.learner.config.dispersion_metric_kwargs

        feature_finders = {
            "find_slice_peak_idxs": acquisition.find_slice_peak_idxs,
        }

        self.feature_finder = feature_finders[
            self.learner.config.feature_finder
        ]

        diagram_kwargs = self.learner.config.diagram_generator_kwargs
        self.feature_finder_kwargs = {
            "peak_threshold": diagram_kwargs.get("peak_threshold"),
            "max_n_peaks": diagram_kwargs.get("max_n_peaks"),
            "peak_height_rule": diagram_kwargs.get("peak_height_rule"),
            "extrema": diagram_kwargs.get(
                "extrema",
            ),
        }

        self.fantasy_quadrature = bool(
            getattr(
                self.learner.config,
                "acquisition_fantasy_quadrature",
                False,
            )
        )
        self.n_fantasy_quadrature_points = int(
            getattr(
                self.learner.config,
                "acquisition_fantasy_quadrature_points",
                10,
            )
        )
        self.slice_window_radius = int(
            getattr(
                self.learner.config,
                "acquisition_slice_window_radius",
                4,
            )
        )

        self._validate_config()
        self.grid_spec = self._build_eval_grid()
        self._base_sample_cache: dict[tuple, torch.Tensor] = {}
        # Reuse the same sample pairing for every baseline and conditioned estimate.
        self._pairing_state = generator.get_state().clone() if generator is not None else None

    def _validate_config(self) -> None:
        """Validate acquisition parameters that determine sample dimensions."""
        if self.n_fantasies < 1:
            raise ValueError(
                "acquisition_n_fantasies must be at least 1, "
                f"got {self.n_fantasies}."
            )

        if self.n_fantasy_quadrature_points < 1:
            raise ValueError(
                "acquisition_fantasy_quadrature_points must be at least 1, "
                f"got {self.n_fantasy_quadrature_points}."
            )

        if self.n_mc_samples < 2:
            raise ValueError(
                "acquisition_n_mc_samples must be at least 2, "
                f"got {self.n_mc_samples}."
            )

        if self.n_temp_eval < 2:
            raise ValueError(
                "acquisition_n_temp_eval must be at least 2, "
                f"got {self.n_temp_eval}."
            )

        if self.sample_jitter < 0.0:
            raise ValueError(
                "acquisition_sample_jitter must be nonnegative, "
                f"got {self.sample_jitter}."
            )

        if self.slice_window_radius < 0:
            raise ValueError(
                "acquisition_slice_window_radius must be nonnegative, "
                f"got {self.slice_window_radius}."
            )

    @torch.no_grad()
    def _calculate_feature_dispersion_from_model(
        self,
        model: gpytorch.models.ExactGP,
        theta_values_raw: list[int],
    ) -> torch.Tensor:
        """Estimate summed feature dispersion over selected theta slices.

        Parameters
        ----------
        model
            GP posterior from which function samples are drawn.
        theta_values_raw
            Raw integer theta values included in the local evaluation window.

        Returns
        -------
        torch.Tensor
            Sum of feature-dispersion estimates over the selected theta
            slices.
        """
        x_eval, idxs_by_theta = self._build_local_eval_grid(
            theta_values_raw
        )

        f_samples = self._sample_functions_from_model(
            model=model,
            n_samples=self.n_mc_samples,
            x_eval=x_eval,
            base_samples=self._base_normal_samples_for_region(theta_values_raw),
        )

        return self._calculate_feature_dispersion_from_samples(
            f_samples, x_eval, idxs_by_theta, theta_values_raw
        )

    def _calculate_feature_dispersion_from_samples(
        self,
        f_samples: torch.Tensor,
        x_eval: torch.Tensor,
        idxs_by_theta: dict[int, slice],
        theta_values_raw: list[int],
    ) -> torch.Tensor:
        """Estimate slice dispersion from already sampled latent functions."""
        device, dtype = x_eval.device, x_eval.dtype

        if f_samples.ndim != 2:
            raise ValueError(
                "Expected f_samples with shape "
                "(n_mc_samples, n_local_eval_points), "
                f"got {f_samples.shape}."
            )

        if f_samples.shape[1] != x_eval.shape[0]:
            raise ValueError(
                "The number of sampled function values must match x_eval: "
                f"{f_samples.shape[1]} != {x_eval.shape[0]}."
            )

        slice_dispersions = []

        for theta_raw in theta_values_raw:
            sl = idxs_by_theta[theta_raw]

            slice_x = x_eval[sl]
            slice_temps = slice_x[:, 0]
            slice_samples = f_samples[:, sl]

            if slice_samples.shape[1] < 2:
                continue

            slice_dispersion = spatial_statistics.calc_dispersion(
                samples=slice_samples,
                coordinates=slice_temps,
                distance_metric=self.distance_metric,
                dispersion_metric=self.dispersion_metric,
                feature_finder=self.feature_finder,
                feature_finder_kwargs=self.feature_finder_kwargs,
                distance_metric_kwargs=self.distance_metric_kwargs,
                dispersion_metric_kwargs=self.dispersion_metric_kwargs,
                generator=self._fixed_pairing_generator(),
            )

            slice_dispersions.append(
                torch.as_tensor(
                    slice_dispersion,
                    device=device,
                    dtype=dtype,
                )
            )

        if not slice_dispersions:
            raise ValueError(
                "No selected theta slice contained enough evaluation "
                "points to calculate feature dispersion."
            )

        return torch.stack(slice_dispersions).sum()

    def __call__(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        """Evaluate acquisition scores for candidate points."""
        return self.forward(X)

    @torch.no_grad()
    def forward(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        """Evaluate expected feature-dispersion reduction.

        Parameters
        ----------
        X
            Candidate points in normalized coordinates with shape ``(m, d)``.

        Returns
        -------
        torch.Tensor
            Acquisition scores with shape ``(m,)``. Larger values indicate
            greater expected reductions in feature dispersion.
        """
        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        X = X.to(
            device=device,
            dtype=dtype,
        )

        # Start a fresh common-draw bank for this acquisition evaluation.
        self._base_sample_cache.clear()

        # Baseline dispersion depends on theta but not candidate temperature.
        baseline_dispersion_by_theta: dict[int, torch.Tensor] = {}

        scores = torch.empty(
            X.shape[0],
            dtype=dtype,
            device=device,
        )

        for i in tqdm(
            range(X.shape[0]),
            desc="Candidates",
            position=1,
        ):
            x_cand = X[i : i + 1]

            candidate_theta_raw = self._candidate_theta_raw(
                x_cand
            )
            local_theta_values = self._local_theta_values(
                candidate_theta_raw
            )

            if candidate_theta_raw not in baseline_dispersion_by_theta:
                baseline_dispersion_by_theta[candidate_theta_raw] = (
                    self._calculate_feature_dispersion_from_model(
                        model=self.learner.model,
                        theta_values_raw=local_theta_values,
                    )
                )

            baseline_dispersion = baseline_dispersion_by_theta[
                candidate_theta_raw
            ]

            x_eval, idxs_by_theta = self._build_local_eval_grid(local_theta_values)
            weights, means, centered = self._conditioned_draws(
                x_cand=x_cand, x_eval=x_eval, theta_values_raw=local_theta_values,
            )
            post_dispersions = [
                self._calculate_feature_dispersion_from_samples(
                    mean.unsqueeze(0) + centered,
                    x_eval, idxs_by_theta, local_theta_values,
                )
                for mean in means
            ]
            expected_post_dispersion = torch.dot(weights, torch.stack(post_dispersions))

            scores[i] = (
                baseline_dispersion
                - expected_post_dispersion
            )

        return scores

    def _build_eval_grid(self) -> DispersionGridSpec:
        """Construct the full normalized temperature-theta evaluation grid.

        Returns
        -------
        DispersionGridSpec
            Evaluation coordinates and row slices for each raw theta value.
        """
        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        dg = self.learner.data_generator

        # Use a uniform normalized temperature grid on every theta slice.
        temp_eval = torch.linspace(
            0.0,
            1.0,
            self.n_temp_eval,
            device=device,
            dtype=dtype,
        )

        theta_values_raw = list(
            range(
                dg.min_theta,
                dg.max_theta + 1,
            )
        )

        # Transform raw integer theta values into normalized GP coordinates.
        theta_values = torch.tensor(
            [
                (theta_raw - dg.min_theta) / dg.theta_range
                for theta_raw in theta_values_raw
            ],
            device=device,
            dtype=dtype,
        )

        x_slices: list[torch.Tensor] = []
        idxs_by_theta: dict[int, slice] = {}
        start = 0

        for theta_raw, theta_value in zip(
            theta_values_raw,
            theta_values,
            strict=True,
        ):
            theta_block = theta_value.expand_as(temp_eval)

            x_slice = torch.stack(
                [temp_eval, theta_block],
                dim=-1,
            )
            x_slices.append(x_slice)

            stop = start + x_slice.shape[0]
            idxs_by_theta[int(theta_raw)] = slice(
                start,
                stop,
            )
            start = stop

        return DispersionGridSpec(
            x_eval=torch.vstack(x_slices),
            idxs_by_theta=idxs_by_theta,
        )

    def _build_local_eval_grid(
        self,
        theta_values_raw: list[int],
    ) -> tuple[
        torch.Tensor,
        dict[int, slice],
    ]:
        """Extract selected theta slices and construct local row indices.

        Parameters
        ----------
        theta_values_raw
            Raw integer theta values to include.

        Returns
        -------
        x_eval
            Concatenated local evaluation grid.
        idxs_by_theta
            Mapping from each selected theta value to its rows in the local
            grid.
        """
        x_slices: list[torch.Tensor] = []
        local_idxs_by_theta: dict[int, slice] = {}
        start = 0

        for theta_raw in theta_values_raw:
            global_slice = self.grid_spec.idxs_by_theta[
                theta_raw
            ]
            x_slice = self.grid_spec.x_eval[
                global_slice
            ]

            x_slices.append(x_slice)

            # Re-index the extracted block relative to the local grid.
            stop = start + x_slice.shape[0]
            local_idxs_by_theta[theta_raw] = slice(
                start,
                stop,
            )
            start = stop

        if not x_slices:
            raise ValueError(
                "At least one theta slice must be selected."
            )

        return (
            torch.vstack(x_slices),
            local_idxs_by_theta,
        )
    def _base_normal_samples(
        self, *, n_samples: int, dimension: int, base_grid_key: str,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Cache standard normals so comparisons share the same random draws."""
        key = (base_grid_key, n_samples, dimension, device, dtype)
        if key not in self._base_sample_cache:
            self._base_sample_cache[key] = torch.randn(
                n_samples, dimension, device=device, dtype=dtype,
                generator=self.generator,
            )
        return self._base_sample_cache[key]

    def _base_normal_samples_for_region(
        self, theta_values_raw: list[int],
    ) -> torch.Tensor:
        """Select local columns from one shared full-grid normal draw bank."""
        full_grid = self.grid_spec.x_eval
        eps = self._base_normal_samples(
            n_samples=self.n_mc_samples, dimension=full_grid.shape[0],
            base_grid_key="slice", device=full_grid.device, dtype=full_grid.dtype,
        )
        indices = torch.cat([
            torch.arange(self.grid_spec.idxs_by_theta[theta].start,
                         self.grid_spec.idxs_by_theta[theta].stop,
                         device=full_grid.device)
            for theta in theta_values_raw
        ])
        return eps.index_select(1, indices)

    def _fixed_pairing_generator(self) -> torch.Generator | None:
        """Replay the same random-pair permutation without consuming draw randomness."""
        if self.generator is None:
            return None
        generator = torch.Generator(device=self.generator.device)
        generator.set_state(self._pairing_state)
        return generator

    def _conditioned_draws(
        self, *, x_cand: torch.Tensor, x_eval: torch.Tensor,
        theta_values_raw: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fantasy weights, means, and shared centered function draws."""
        if self.fantasy_quadrature:
            fantasy_y, weights = self._sample_points_quadrature(
                model=self.learner.model, x_new=x_cand,
                n_points=self.n_fantasy_quadrature_points,
            )
        else:
            fantasy_y = self._sample_fantasized_observations(
                model=self.learner.model, x_new=x_cand, n_fantasies=self.n_fantasies,
            )
            weights = torch.full_like(fantasy_y, 1.0 / fantasy_y.numel())
        means, factor = self._conditioned_sample_components(
            x_cand=x_cand, fantasy_y=fantasy_y, x_eval=x_eval,
        )
        # The covariance does not depend on fantasy_y: factor and multiply once.
        eps = self._base_normal_samples_for_region(theta_values_raw)
        centered = eps @ factor.T
        return weights, means, centered

    @torch.no_grad()
    def _conditioned_sample_components(
        self,
        *,
        x_cand: Float[torch.Tensor, "1 d"],
        fantasy_y: torch.Tensor,
        x_eval: torch.Tensor,
    ) -> tuple[Float[torch.Tensor, "n_fantasies n_eval"], torch.Tensor]:
        """Return conditioned means and their shared latent covariance factor."""
        if x_eval.ndim != 2 or x_eval.shape[0] == 0:
            raise ValueError("x_eval must contain at least one evaluation point.")
        if x_cand.ndim != 2 or x_cand.shape != (1, x_eval.shape[1]):
            raise ValueError("x_cand must contain exactly one candidate point.")
        if fantasy_y.numel() == 0:
            raise ValueError("At least one fantasy observation is required.")

        model = self.learner.model
        x_joint = torch.cat([x_eval, x_cand], dim=0)
        with eval_context(model):
            latent_joint = self._get_posterior(model, x_joint, include_noise=False)
            noisy_candidate = self._get_posterior(model, x_cand, include_noise=True)

        n_eval = x_eval.shape[0]
        joint_mean = latent_joint.mean.reshape(-1)
        joint_cov = latent_joint.covariance_matrix
        if joint_mean.numel() != n_eval + 1:
            raise RuntimeError(
                "Expected one scalar GP output per evaluation point and candidate."
            )

        mean_eval = joint_mean[:n_eval]
        mean_cand = joint_mean[n_eval]
        cov_eval = joint_cov[:n_eval, :n_eval]
        cross_cov = joint_cov[:n_eval, n_eval]
        observation_variance = noisy_candidate.variance.reshape(()).clamp_min(1e-30)

        gain = cross_cov / observation_variance
        residuals = fantasy_y.reshape(-1) - mean_cand
        conditioned_means = mean_eval.unsqueeze(0) + residuals.unsqueeze(1) * gain.unsqueeze(0)

        conditioned_cov = cov_eval - torch.outer(cross_cov, cross_cov) / observation_variance
        conditioned_cov = 0.5 * (conditioned_cov + conditioned_cov.T)
        conditioned_cov = conditioned_cov + self.sample_jitter * torch.eye(
            n_eval,
            device=x_eval.device,
            dtype=x_eval.dtype,
        )
        try:
            L = torch.linalg.cholesky(conditioned_cov)
        except RuntimeError as error:
            min_diag = conditioned_cov.diag().min().item()
            raise RuntimeError(
                "Cholesky failed while sampling fantasy-conditioned posterior "
                f"functions. Covariance shape={tuple(conditioned_cov.shape)}, "
                f"jitter={self.sample_jitter}, min_diag={min_diag}."
            ) from error
        return conditioned_means, L

    @torch.no_grad()
    def _sample_fantasized_observations(
        self,
        model: gpytorch.models.ExactGP,
        x_new: Float[torch.Tensor, "1 d"],
        n_fantasies: int,
    ) -> Float[torch.Tensor, "n_fantasies"]:
        """Sample fantasy values at one candidate.

        Parameters
        ----------
        model
            GP model defining the posterior.
        x_new
            Single candidate point in normalized coordinates.
        n_fantasies
            Number of fantasy values to sample.

        Returns
        -------
        torch.Tensor
            Fantasy values with shape ``(n_fantasies,)``.
        """
        with eval_context(model):
            posterior = self._get_posterior(
                model,
                x_new,
                include_noise=True,
            )

            # .variance clamps small values; inspect the raw diagonal for diagnostics.
            raw_variance = posterior.lazy_covariance_matrix.diagonal(dim1=-2, dim2=-1)
            if torch.any(raw_variance < -1e-6):
                warnings.warn(
                    "Negative posterior variance detected before clamping: "
                    f"min={raw_variance.min().item()}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            mean = posterior.mean.reshape(-1)
            variance = (
                posterior.variance
                .reshape(-1)
                .clamp_min(1e-30)
            )

        eps = self._base_normal_samples(
            n_samples=n_fantasies, dimension=mean.numel(),
            base_grid_key="fantasy", device=mean.device, dtype=mean.dtype,
        )

        fantasy_y = (
            mean
            + variance.sqrt() * eps
        )

        return fantasy_y.reshape(-1)

    @torch.no_grad()
    def _sample_points_quadrature(
        self,
        model: gpytorch.models.ExactGP,
        x_new: Float[torch.Tensor, "1 d"],
        n_points: int,
    ) -> tuple[
        Float[torch.Tensor, "n_quadrature_points"],
        Float[torch.Tensor, "n_quadrature_points"],
    ]:
        """Construct Gauss-Hermite fantasy values and normalized weights.

        The quadrature approximates

        ``E[g(Y)] = sum_i weights[i] * g(fantasy_y[i])``,

        where ``Y`` follows the univariate GP posterior at ``x_new``.

        Parameters
        ----------
        model
            GP model defining the posterior.
        x_new
            Single candidate point in normalized coordinates.
        n_points
            Number of Gauss-Hermite quadrature points.

        Returns
        -------
        fantasy_y
            Fantasy values at transformed quadrature nodes.
        weights
            Normalized quadrature weights.
        """
        if n_points < 1:
            raise ValueError(
                f"n_points must be at least 1, got {n_points}."
            )

        with eval_context(model):
            posterior = self._get_posterior(
                model,
                x_new,
                include_noise=True,
            )

            if posterior.mean.numel() != 1:
                raise ValueError(
                    "_sample_points_quadrature supports exactly one "
                    "candidate output, but received "
                    f"{posterior.mean.numel()}."
                )

            # .variance clamps small values; inspect the raw diagonal for diagnostics.
            raw_variance = posterior.lazy_covariance_matrix.diagonal(dim1=-2, dim2=-1)
            if torch.any(raw_variance < -1e-6):
                warnings.warn(
                    "Negative posterior variance detected before clamping: "
                    f"min={raw_variance.min().item()}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            mean = posterior.mean.reshape(())
            variance = (
                posterior.variance
                .reshape(())
                .clamp_min(1e-30)
            )
            std = variance.sqrt()

        # Probabilists' Gauss-Hermite rule integrates exp(-z^2 / 2) f(z).
        nodes_np, weights_np = roots_hermitenorm(
            n_points
        )

        nodes = torch.as_tensor(
            nodes_np,
            device=mean.device,
            dtype=mean.dtype,
        )
        weights = torch.as_tensor(
            weights_np,
            device=mean.device,
            dtype=mean.dtype,
        )

        fantasy_y = mean + std * nodes

        # Normalize for expectation under a standard normal distribution.
        weights = weights / weights.sum()

        return (
            fantasy_y,
            weights,
        )

    @torch.no_grad()
    def _sample_functions_from_model(
        self,
        model: gpytorch.models.ExactGP,
        n_samples: int,
        x_eval: torch.Tensor,
        base_samples: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "n_samples n_eval"]:
        """Draw latent functions from the GP posterior.

        Parameters
        ----------
        model
            GP model defining the posterior.
        n_samples
            Number of posterior functions to draw.
        x_eval
            Evaluation coordinates with shape ``(n_eval, d)``.
        base_samples
            Optional standard-normal draws shared with conditioned sampling.

        Returns
        -------
        torch.Tensor
            Function samples with shape ``(n_samples, n_eval)``.
        """
        device = x_eval.device
        dtype = x_eval.dtype

        with eval_context(model):
            posterior = self._get_posterior(
                model,
                x_eval,
            )

        mean = posterior.mean.reshape(-1)
        cov = posterior.covariance_matrix

        cov = (
            cov
            + self.sample_jitter
            * torch.eye(
                cov.shape[0],
                device=device,
                dtype=dtype,
            )
        )

        try:
            L = torch.linalg.cholesky(
                cov
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Cholesky failed while sampling posterior functions. "
                f"Covariance shape={cov.shape}, "
                f"jitter={self.sample_jitter}, "
                f"min_diag={cov.diag().min().item()}."
            ) from error

        eps = base_samples if base_samples is not None else torch.randn(
            n_samples, mean.numel(), device=device, dtype=dtype,
            generator=self.generator,
        )

        return mean + eps @ L.T

    def _candidate_theta_raw(
        self,
        x_cand: Float[torch.Tensor, "1 d"],
    ) -> int:
        """Convert one normalized candidate theta value to raw integer theta."""
        dg = self.learner.data_generator

        theta_normalized = x_cand[0, 1]
        theta_raw = torch.round(
            theta_normalized * dg.theta_range
            + dg.min_theta
        )
        theta_raw_int = int(
            theta_raw.item()
        )

        return max(
            dg.min_theta,
            min(
                dg.max_theta,
                theta_raw_int,
            ),
        )

    def _local_theta_values(
        self,
        candidate_theta_raw: int,
    ) -> list[int]:
        """Return the clipped theta window surrounding a candidate slice."""
        dg = self.learner.data_generator

        theta_min = max(
            dg.min_theta,
            candidate_theta_raw
            - self.slice_window_radius,
        )
        theta_max = min(
            dg.max_theta,
            candidate_theta_raw
            + self.slice_window_radius,
        )

        return list(
            range(
                theta_min,
                theta_max + 1,
            )
        )

    @staticmethod
    def _get_posterior(
        model: gpytorch.models.ExactGP,
        x: torch.Tensor,
        *,
        include_noise: bool = False,
    ) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate an ExactGP posterior.

        Parameters
        ----------
        model
            GP model defining the posterior.
        x
            Input coordinates.
        include_noise
            If True, return the posterior predictive distribution of noisy
            observations. If False, return the latent-function posterior.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Latent posterior if ``include_noise=False``; noisy predictive
            posterior if ``include_noise=True``.
        """
        posterior = model(x)

        if include_noise:
            posterior = model.likelihood(posterior)

        if not isinstance(
            posterior,
            gpytorch.distributions.MultivariateNormal,
        ):
            raise TypeError(
                "Expected ExactGP to return a MultivariateNormal, "
                f"got {type(posterior).__name__}."
            )

        return posterior
