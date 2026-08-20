import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import gpytorch
import numpy as np
import torch
from jaxtyping import Bool, Float
from tqdm import tqdm

from bayes_phase.acquisition.peak_finding import find_slice_peak_idxs
from bayes_phase.utils import eval_context

if TYPE_CHECKING: 
    from bayes_phase.active_learning import ActiveLearner


@dataclass
class PeakCellGridSpec:
    """
    Grid used to convert sampled peak locations into binary cell occupancy.

    Each cell represents:
        one temperature bin x one theta slice

    The random variable for each cell is Bernoulli.
    """

    x_eval: Float[torch.Tensor, "n_total 2"] # The full set of evaluation locations.
    temp_eval: Float[torch.Tensor, "n_temp"]
    theta_values: Float[torch.Tensor, "n_theta"]
    theta_values_raw: list[int]
    idxs_by_theta: dict[int, slice]
    temp_edges_raw: np.ndarray
    temp_raw: np.ndarray
    n_cells: int
    n_temp_bins: int
    n_theta: int


class PeakCellEntropy:
    """
    Expected information gain for peak/no-peak cell occupancy.

    This acquisition estimates uncertainty over a binary peak map.

    Steps:
    1. Sample functions from the current GP posterior.
    2. For each sample and theta slice, find peaks.
    3. Convert peaks into binary cell hits.
    4. Estimate Bernoulli probabilities p_j for each cell.
    5. Compute summed Bernoulli entropy across cells.
    6. For each candidate x, fantasize observations y_x.
    7. Condition the GP on each fantasy and recompute expected entropy.
    8. Score candidate by expected entropy reduction.
    """

    def __init__(self, learner: "ActiveLearner"):
        """Initialize the acquisition with learner and config features.
        
        Args:
        n_fantasies: Number of fantasy observations per candidate.
        n_mc_samples: Number of posterior function values used to estimate cell-occupancy probabilities.
        n_temp_eval: Controls the density of the grid used for peak finding.
        sample_jitter: Covariance jitter to add before Cholesky sampling of posterior functions.
        """

        self.learner = learner

        self.n_fantasies = int(
            getattr(self.learner.config, "acquisition_n_fantasies", 10)
        )
        self.n_mc_samples = int(
            getattr(self.learner.config, "acquisition_n_mc_samples", 100)
        ) 
        self.n_temp_eval = int(
            getattr(self.learner.config, "acquisition_n_temp_eval", 100)
        ) 
        self.sample_jitter = float(
            getattr(self.learner.config, "acquisition_sample_jitter", 1e-7)
        )

        self.grid_spec = self._build_peak_cell_grid()

    def __call__(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(X)

    @torch.no_grad()
    def forward(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        """
        Implement acquisition function.
        """

        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        X = X.to(device=device, dtype=dtype)

        # Baseline entropy for acquisition. Conditioned entropy will be subtracted from this.
        # Peak-finding configuration shared with the diagram generator.
        diagram_kwargs = getattr(
            self.learner.config,
            "diagram_generator_kwargs",
            {},
        )
        peak_threshold = diagram_kwargs.get("peak_threshold", None)
        max_n_peaks = diagram_kwargs.get("max_n_peaks", None)
        peak_height_rule = diagram_kwargs.get("peak_height_rule", None)
        extrema = diagram_kwargs.get("extrema", "peaks")

        baseline_entropy = self._estimate_peak_cell_entropy_from_model(
            self.learner.model,
            peak_threshold=peak_threshold,
            max_n_peaks=max_n_peaks,
            peak_height_rule=peak_height_rule,
            extrema=extrema,
        )

        # Initialize an empty tensor for acquisition scores
        scores = torch.empty(X.shape[0], dtype=dtype, device=device)

        # Loop over candidates
        for i in tqdm(range(X.shape[0]), desc="Peak-cell entropy candidates", position=1):
            x_cand = X[i : i + 1]

            # Fantasy observations
            fantasy_y = self._sample_fantasized_observations(
                model=self.learner.model,
                x_new=x_cand,
                n_fantasies=self.n_fantasies,
            )

            # Accumulate entropies
            post_entropies = []
            for y_fant in fantasy_y:
                # Condition on fantasy
                fantasy_model = self.learner.model.get_fantasy_model(
                    x_cand,
                    y_fant.reshape(1),
                )
                fantasy_model.eval()

                # Conditional entropy sum
                post_entropy = self._estimate_peak_cell_entropy_from_model(
                    fantasy_model,
                    peak_threshold=peak_threshold,
                    max_n_peaks=max_n_peaks,
                    peak_height_rule=peak_height_rule,
                    extrema=extrema,
                )
                post_entropies.append(post_entropy)

            # Average conditioned entropies
            expected_post_entropy = torch.stack(post_entropies).mean()

            # EIG score is expected reduction in entropy
            scores[i] = baseline_entropy - expected_post_entropy

        return scores
    
    def _build_peak_cell_grid(self) -> PeakCellGridSpec:
        """
        Create eval grid.
        """

        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        dg = self.learner.data_generator
        cfg = self.learner.config

        # Normalized temperature grid from 0 to 1
        temp_eval = torch.linspace(0.0, 1.0, self.n_temp_eval, device=device, dtype=dtype)

        # Build a list of discrete theta values
        theta_range = dg.theta_range
        theta_values_raw = list(range(dg.min_theta, dg.max_theta + 1))

        # Map each discrete raw theta value into the GP's input space.
        theta_values = torch.tensor(
            [(theta_val_raw - dg.min_theta) / theta_range for theta_val_raw in theta_values_raw],
            device=device,
            dtype=dtype,
        )

        # Initialize containers 
        x_slices: list[torch.Tensor] = []
        idxs_by_theta: dict[int, slice] = {}
        start = 0

        for theta_val_raw, theta_val in zip(theta_values_raw, theta_values):
            theta_block = torch.full_like(temp_eval, fill_value=theta_val)
            x_slice = torch.stack([temp_eval, theta_block], dim=-1) # [n_temp_eval, 2]
            x_slices.append(x_slice)

            stop = start + temp_eval.numel()
            idxs_by_theta[int(theta_val_raw)] = slice(start, stop)
            start = stop

        x_eval = torch.vstack(x_slices) # [n_theta*n_temp_eval, 2]

        # Convert normalized temp grid back to raw temperatures for binning peak locations.
        theta_dummy = torch.zeros_like(temp_eval)
        temp_eval_with_dummy_theta = torch.stack([temp_eval, theta_dummy], dim=-1)
        temp_raw = (
            dg.untransform_inputs(temp_eval_with_dummy_theta)[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # Construct raw-temperature cell edges based on scoring_grid_T_range, which is also used for scoring.
        dt = float(getattr(cfg, "scoring_grid_T_range", 50))
        if dt <= 0:
            raise ValueError(f"scoring_grid_T_range must be positive; got {dt}.")
        temp_edges = [dg.min_temp]
        while temp_edges[-1] < dg.max_temp:
            temp_edges.append(min(temp_edges[-1] + dt, dg.max_temp))

        temp_edges_raw = np.asarray(temp_edges, dtype=float)

        # Count cells.
        n_temp_bins = len(temp_edges_raw) - 1
        n_theta = len(theta_values_raw)
        n_cells = n_temp_bins * n_theta

        # Return the evaluation and cell data all together. 
        return PeakCellGridSpec(
            x_eval=x_eval,
            temp_eval=temp_eval,
            theta_values=theta_values,
            theta_values_raw=theta_values_raw,
            idxs_by_theta=idxs_by_theta,
            temp_edges_raw=temp_edges_raw,
            temp_raw=temp_raw,
            n_cells=n_cells,
            n_temp_bins=n_temp_bins,
            n_theta=n_theta,
        )

    @torch.no_grad()
    def _sample_fantasized_observations(
        self,
        model: gpytorch.models.ExactGP,
        x_new: Float[torch.Tensor, "1 d"],
        n_fantasies: int,
    ) -> Float[torch.Tensor, "n_fantasies"]:
        """
        Generate fantasy target values at x_new from the model posterior.
        """

        with eval_context(model):
            posterior = model(x_new)
            if torch.any(posterior.variance < -1e-6):
                warnings.warn(
                    f"Negative posterior variance detected before clamping: "
                    f"min={posterior.variance.min().item()}",
                    RuntimeWarning,
                )
            mean = posterior.mean.reshape(-1)
            var = posterior.variance.reshape(-1).clamp_min(1e-30)

        dist = torch.distributions.Normal(mean, var.sqrt())
        return dist.sample((n_fantasies,)).reshape(-1)

    @torch.no_grad()
    def _sample_functions_from_model(
        self,
        model: gpytorch.models.ExactGP,
        n_samples: int,
    ) -> Float[torch.Tensor, "n_samples n_total"]:
        """
        Generate full sampled functions. This is used both before and after conditioning.
        """

        x_eval = self.grid_spec.x_eval
        device = x_eval.device
        dtype = x_eval.dtype

        # Model posterior.
        with eval_context(model):
            posterior = model(x_eval)
            mean = posterior.mean.reshape(-1)
            cov = posterior.covariance_matrix
            cov = cov + self.sample_jitter * torch.eye(
                cov.shape[0], device=device, dtype=dtype
            )

        # Cholesky decomposition for sampling
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError as e:
            raise RuntimeError(
                f"Cholesky failed while sampling peak-cell functions. "
                f"Covariance shape={cov.shape}, jitter={self.sample_jitter}, "
                f"min_diag={cov.diag().min().item()}"
            ) from e

        eps = torch.randn(
            n_samples,
            mean.numel(),
            device=device,
            dtype=dtype,
        )
        f_samples = mean + eps @ L.T

        return f_samples

    def _estimate_peak_cell_entropy_from_model(
        self,
        model: gpytorch.models.ExactGP,
        peak_threshold: float | None = None,
        max_n_peaks: int | None = None,
        peak_height_rule: Literal["above_median"] | None = None,
        extrema: Literal["peaks", "valleys"] = "peaks",
    ) -> torch.Tensor:
        """
        Given a model, calculate summed entropy across the space.
        """

        # Sample functions.
        f_samples = self._sample_functions_from_model(
            model=model,
            n_samples=self.n_mc_samples,
        )

        # Which cells contain a peak? Any cell with a peak is classified as a hit.
        hit_counts = torch.zeros(
            self.grid_spec.n_cells,
            dtype=torch.float64,
            device=f_samples.device,
        )

        # Accumulate hit_counts.
        for k in range(f_samples.shape[0]):
            occupancy = self._sample_to_peak_cell_occupancy(
                f_samples[k],
                peak_threshold=peak_threshold,
                max_n_peaks=max_n_peaks,
                peak_height_rule=peak_height_rule,
                extrema=extrema,
            )
            hit_counts += occupancy.to(hit_counts.dtype)

        if hit_counts.sum() == 0:
            warnings.warn(
                "No peaks were found in any sampled function. "
                "Peak-cell entropy will be near zero; check peak_threshold, "
                "peak_height_rule, max_n_peaks, and n_temp_eval.",
                RuntimeWarning,
            )

        # Bernoulli hit probability for each cell.
        p = hit_counts / self.n_mc_samples
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1.0 - eps)

        return self._bernoulli_entropy_sum(p)

    def _bernoulli_entropy_sum(
        self,
        p: Float[torch.Tensor, "n_cells"],
    ) -> torch.Tensor:
        """
        Compute the sum of marginal Bernoulli entropies across cells.
        This ignores dependence between cell occupancies.
        """
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1.0 - eps)

        entropy = -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
        return entropy.sum()

    def _sample_to_peak_cell_occupancy(
        self,
        f_sample: Float[torch.Tensor, "n_total"],
        peak_threshold: float | None = None,
        max_n_peaks: int | None = None,
        peak_height_rule: Literal["above_median"] | None = None,
        extrema: Literal["peaks", "valleys"] = "peaks",
    ) -> Bool[torch.Tensor, "n_cells"]:
        """
        Given a sample, determine whether each cell is occupied (contains a peak)
        """

        device = f_sample.device
        occupancy = torch.zeros(
            self.grid_spec.n_cells,
            dtype=torch.bool,
            device=device,
        )

        # Accumulate occupancy one slice at a time
        for theta_pos, theta_val_raw in enumerate(self.grid_spec.theta_values_raw):
            # Grab the sampled function values along temperature for the current theta.
            sl = self.grid_spec.idxs_by_theta[theta_val_raw]
            y_slice = f_sample[sl]

            # Find peaks.
            peak_idxs = self._find_peak_idxs(
                y_slice,
                peak_threshold=peak_threshold,
                max_n_peaks=max_n_peaks,
                peak_height_rule=peak_height_rule,
                extrema=extrema,
            )

            if len(peak_idxs) == 0:
                continue

            # Turn peak indices into raw temperatures.
            peak_T_raw = self.grid_spec.temp_raw[peak_idxs]

            # Map peak raw temperatures into temperature-bin indices.
            temp_bin_idxs = np.searchsorted(
                self.grid_spec.temp_edges_raw,
                peak_T_raw,
                side="left",
            ) - 1

            # Global minimum edge goes in the first bin.
            temp_bin_idxs[np.isclose(
                peak_T_raw,
                self.grid_spec.temp_edges_raw[0],
            )] = 0

            # Verify peaks fall within a bin.
            valid = (
                (temp_bin_idxs >= 0)
                & (temp_bin_idxs < self.grid_spec.n_temp_bins)
            )

            if not np.all(valid):
                invalid_peaks = peak_T_raw[~valid]
                warnings.warn(
                    f"{len(invalid_peaks)} peak(s) fell outside temperature-bin range "
                    f"[{self.grid_spec.temp_edges_raw[0]}, {self.grid_spec.temp_edges_raw[-1]}]. "
                    f"Invalid peak temperatures: {invalid_peaks}",
                    RuntimeWarning,
                )

            # Flatten 2D cell coordinates into 1D cell indices.
            for temp_bin_idx in temp_bin_idxs[valid]:
                cell_idx = theta_pos * self.grid_spec.n_temp_bins + int(temp_bin_idx)
                occupancy[cell_idx] = True

        return occupancy


    def _find_peak_idxs(
        self,
        y_slice: Float[torch.Tensor, "n"],
        *,
        peak_threshold: float | None = None,
        max_n_peaks: int | None = None,
        peak_height_rule: Literal["above_median"] | None = None,
        extrema: Literal["peaks", "valleys"] = "peaks",
    ) -> np.ndarray:
        """Call peak finder."""

        return find_slice_peak_idxs(
            y_slice,
            peak_threshold=peak_threshold,
            max_n_peaks=max_n_peaks,
            peak_height_rule=peak_height_rule,
            extrema=extrema,
        )