from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import gpytorch
import numpy as np
import torch
import warnings
from jaxtyping import Bool, Float
from scipy.signal import find_peaks
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

    x_eval: Float[torch.Tensor, "n_total 2"]
    temp_eval: Float[torch.Tensor, "n_temp"]
    theta_values: Float[torch.Tensor, "n_theta"]
    theta_indices: list[int]
    idxs_by_theta: dict[int, slice]
    temp_edges_raw: np.ndarray
    temp_raw: np.ndarray
    n_cells: int
    n_temp_bins: int
    n_theta: int


class PeakCellCostEntropy:
    """
    Expected information gain for peak/no-peak cell occupancy.
    Can evaluate batches corresponding to ceratin physical simulations,
    with an optional batch cost denominator for the returned score.

    This acquisition estimates uncertainty over a binary peak map.

    Steps:
    1. Sample functions from the current GP posterior.
    2. For each sample and theta slice, find peaks.
    3. Convert peaks into binary cell hits.
    4. Estimate Bernoulli probabilities p_j for each cell.
    5. Compute summed Bernoulli entropy across cells.
    6. For each candidate (or batch) x, fantasize observations y_x.
    7. Condition the GP on each fantasy and recompute expected entropy.
    8. Score candidate by entropy reduction.
    """

    def __init__(self, learner: "ActiveLearner"):
        self.learner = learner

        self.n_fantasies = int(
            getattr(self.learner.config, "peak_cell_n_fantasies", 10)
        )
        self.n_mc_samples = int(
            getattr(self.learner.config, "peak_cell_n_mc_samples", 100)
        ) 
        self.n_temp_eval = int(
            getattr(self.learner.config, "peak_cell_n_temp_eval", 100)
        ) # density of grid for peak finding
        self.sample_jitter = float(
            getattr(self.learner.config, "peak_cell_sample_jitter", 1e-7)
        )

        self.grid_spec = self._build_peak_cell_grid()

    def __call__(
        self,
        cand_actions: list[dict],
    ) -> dict:
        return self.forward(cand_actions)

    @torch.no_grad()
    def forward(
        self,
        cand_actions: list[dict],
    ) -> dict:
        """
        Implement batch/action-aware acquisition function.
        """

        if len(cand_actions) == 0:
            raise ValueError("No candidate sampling actions were generated.")

        # Baseline entropy for acquisition. Conditioned entropy will be subtracted from this.
        baseline_entropy = self._estimate_peak_cell_entropy_from_model(
            self.learner.model
        )

        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype

        scores = torch.empty(len(cand_actions), dtype=dtype, device=device)
        raw_values = torch.empty(len(cand_actions), dtype=dtype, device=device)

        # Loop over candidate sampling actions.
        for i in tqdm(
            range(len(cand_actions)),
            desc="Peak-cell cost candidates",
            position=1,
        ):
            action = cand_actions[i]
            x_batch = action["x_batch"].to(device=device, dtype=dtype)
            cost = float(action["cost"])

            if cost <= 0.0:
                raise ValueError(f"Sampling action has non-positive cost: {action}")

            # Fantasy observations, sampled jointly over the whole batch/action.
            fantasy_y = self._sample_fantasized_observations(
                model=self.learner.model,
                x_new=x_batch,
                n_fantasies=self.n_fantasies,
            )

            # Accumulate entropies.
            post_entropies = []
            for y_fant in fantasy_y:
                # Condition on fantasy batch.
                fantasy_model = self.learner.model.get_fantasy_model(
                    x_batch,
                    y_fant.reshape(-1),
                )
                fantasy_model.eval()

                # Conditional entropy sum.
                post_entropy = self._estimate_peak_cell_entropy_from_model(
                    fantasy_model
                )
                post_entropies.append(post_entropy)

            # Average conditioned entropies.
            expected_post_entropy = torch.stack(post_entropies).mean()

            # Raw information gain is entropy reduction.
            raw_value = baseline_entropy - expected_post_entropy
            raw_values[i] = raw_value

            if self.learner.config.acquisition_function == "peak_cell_cost":
                scores[i] = raw_value / cost
            elif self.learner.config.acquisition_function == "peak_cell_batch":
                scores[i] = raw_value
            else:
                raise ValueError(
                    f"PeakCellCostEntropy does not support "
                    f"{self.learner.config.acquisition_function}"
                )

        best_idx = torch.argmax(scores)
        best_action = cand_actions[int(best_idx)]

        return {
            "method": best_action["method"],
            "x_batch": best_action["x_batch"].to(device=device, dtype=dtype),
            "cost": float(best_action["cost"]),
            "raw_value": raw_values[best_idx].detach().cpu().item(),
            "score": scores[best_idx].detach().cpu().item(),
            "scores": scores,
            "n_candidates_evaluated": len(cand_actions),
        }
    




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

        # All discrete theta indices
        theta_range = dg.theta_range
        theta_indices = list(range(dg.min_theta, dg.max_theta + 1))

        # Normalize theta
        theta_values = torch.tensor(
            [(theta_idx - dg.min_theta) / theta_range for theta_idx in theta_indices],
            device=device,
            dtype=dtype,
        )

        # Set up and fill blocks so we can process one slice at a time.
        # Each x-block contains normalized temperature, coverage pairs for one fixed theta.
        x_blocks = []
        idxs_by_theta = {}
        start = 0

        for theta_idx, theta_val in zip(theta_indices, theta_values):
            theta_block = torch.full_like(temp_eval, fill_value=theta_val)
            x_block = torch.stack([temp_eval, theta_block], dim=-1)
            x_blocks.append(x_block)

            stop = start + temp_eval.numel()
            idxs_by_theta[int(theta_idx)] = slice(start, stop)
            start = stop

        x_eval = torch.vstack(x_blocks)

        # Convert normalized temp grid back to raw temperatures for binning peak locations.
        theta_dummy = torch.zeros_like(temp_eval)
        x2 = torch.stack([temp_eval, theta_dummy], dim=-1)
        temp_raw = (
            dg.untransform_inputs(x2)[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # Use same temperature bins as scoring.
        dt = float(getattr(cfg, "scoring_grid_T_range", 50))
        temp_edges = [dg.min_temp]
        while temp_edges[-1] < dg.max_temp:
            temp_edges.append(min(temp_edges[-1] + dt, dg.max_temp))

        temp_edges_raw = np.asarray(temp_edges, dtype=float)

        # Count cells.
        n_temp_bins = len(temp_edges_raw) - 1
        n_theta = len(theta_indices)
        n_cells = n_temp_bins * n_theta

        return PeakCellGridSpec(
            x_eval=x_eval,
            temp_eval=temp_eval,
            theta_values=theta_values,
            theta_indices=theta_indices,
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
        x_new: Float[torch.Tensor, "n_batch d"],
        n_fantasies: int,
    ) -> Float[torch.Tensor, "n_fantasies n_batch"]:
        """
        Generate fantasy target values at x_new from the joint model posterior.
        """

        with eval_context(model):
            posterior = model(x_new)
            if torch.any(posterior.variance < -1e-6):
                warnings.warn(
                    f"Negative posterior variance detected before clamping: "
                    f"min={posterior.variance.min().item()}",
                    RuntimeWarning,
                )

            cov = posterior.covariance_matrix
            device = cov.device
            dtype = cov.dtype
            cov = cov + self.sample_jitter * torch.eye(
                cov.shape[0],
                device=device,
                dtype=dtype,
            )

            fantasy_dist = gpytorch.distributions.MultivariateNormal(
                posterior.mean.reshape(-1),
                cov,
            )

        return fantasy_dist.rsample(torch.Size([n_fantasies]))

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
    ) -> torch.Tensor:
        """
        Given a model, calculate summed entropy across the space.
        """

        # Sample functions.
        f_samples = self._sample_functions_from_model(
            model=model,
            n_samples=self.n_mc_samples,
        )

        # Which cells contain a peak? Any peak within a cell is classified as a hit.
        hit_counts = torch.zeros(
            self.grid_spec.n_cells,
            dtype=torch.float64,
            device=f_samples.device,
        )
        # Accumulate hit_counts.
        for k in range(f_samples.shape[0]):
            occupancy = self._sample_to_peak_cell_occupancy(f_samples[k])
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
        for theta_pos, theta_idx in enumerate(self.grid_spec.theta_indices):
            # Grab the sampled function values along temperature for the current theta.
            sl = self.grid_spec.idxs_by_theta[theta_idx]
            y_slice = f_sample[sl]

            # Find peaks.
            peak_idxs = self._find_peak_idxs(y_slice)

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
    ) -> np.ndarray:
        """
        Call peak finder.
        """

        diagram_kwargs = getattr(self.learner.config, "diagram_generator_kwargs", {})

        return find_slice_peak_idxs(
            y_slice,
            peak_threshold=diagram_kwargs.get("peak_threshold", None),
            max_n_peaks=diagram_kwargs.get("max_n_peaks", None),
            peak_height_rule=diagram_kwargs.get("peak_height_rule", None),
        )