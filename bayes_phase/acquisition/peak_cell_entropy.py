from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import gpytorch
import numpy as np
import torch
from jaxtyping import Float
from scipy.signal import find_peaks
from tqdm import tqdm

from bayes_phase.utils import eval_context

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner


@dataclass
class PeakCellGridSpec:
    """
    Grid used to convert sampled peak locations into binary cell occupancy.

    Each cell represents:
        one temperature bin x one theta slice

    The random variable for each cell is Bernoulli:
        Z_j = 1 if a sampled diagram has at least one peak in cell j.
    """

    x_eval: Float[torch.Tensor, "n_total 2"]
    temp_eval: Float[torch.Tensor, "n_temp"]
    theta_values: Float[torch.Tensor, "n_theta"]
    theta_indices: list[int]
    idxs_by_theta: dict[int, slice]
    temp_edges_raw: np.ndarray
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
            getattr(
                self.learner.config,
                "peak_cell_n_temp_eval",
                self.learner.config.diagram_generator_kwargs.get("n_temp_grid", 100),
            )
        )
        self.sample_jitter = float(
            getattr(self.learner.config, "peak_cell_sample_jitter", 1e-7)
        )
        self.alpha = float(
            getattr(self.learner.config, "peak_cell_alpha", 1.0)
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
        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        X = X.to(device=device, dtype=dtype)

        baseline_entropy = self._estimate_peak_cell_entropy_from_model(
            self.learner.model
        )

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

            post_entropies = []

            for y_fant in fantasy_y:
                # Condition on fantasy
                fantasy_model = self.learner.model.get_fantasy_model(
                    x_cand,
                    y_fant.reshape(1),
                )
                fantasy_model.eval()

                # Conditional entropy average
                post_entropy = self._estimate_peak_cell_entropy_from_model(
                    fantasy_model
                )
                post_entropies.append(post_entropy)

            # Average conditioned entropies
            expected_post_entropy = torch.stack(post_entropies).mean()

            # Score is expected reduction in entropy
            scores[i] = baseline_entropy - expected_post_entropy

        return scores

    def _build_peak_cell_grid(self) -> PeakCellGridSpec:
        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        dg = self.learner.data_generator
        cfg = self.learner.config

        temp_eval = torch.linspace(
            0.0,
            1.0,
            self.n_temp_eval,
            device=device,
            dtype=dtype,
        )

        theta_range = dg.theta_range
        theta_indices = list(range(dg.min_theta, dg.max_theta + 1))

        theta_values = torch.tensor(
            [
                (theta_idx - dg.min_theta) / theta_range
                for theta_idx in theta_indices
            ],
            device=device,
            dtype=dtype,
        )

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

        # Use same temperature-bin scale as scoring.
        dt = float(getattr(cfg, "scoring_grid_T_range", 50))
        temp_edges = [dg.min_temp]
        while temp_edges[-1] < dg.max_temp:
            temp_edges.append(min(temp_edges[-1] + dt, dg.max_temp))

        temp_edges_raw = np.asarray(temp_edges, dtype=float)

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
        with eval_context(model):
            posterior = model(x_new)

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

        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return mvn.sample((n_samples,))

    def _estimate_peak_cell_entropy_from_model(
        self,
        model: gpytorch.models.ExactGP,
    ) -> torch.Tensor:
        f_samples = self._sample_functions_from_model(
            model=model,
            n_samples=self.n_mc_samples,
        )

        hit_counts = torch.zeros(
            self.grid_spec.n_cells,
            dtype=torch.float64,
            device=f_samples.device,
        )

        for k in range(f_samples.shape[0]):
            occupancy = self._sample_to_peak_cell_occupancy(f_samples[k])
            hit_counts += occupancy.to(hit_counts.dtype)

        # Bernoulli probability for each cell.
        p = hit_counts / self.n_mc_samples
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1.0 - eps)

        return self._bernoulli_entropy_sum(p)
    

    def _bernoulli_entropy_sum(
        self,
        p: Float[torch.Tensor, "n_cells"],
    ) -> torch.Tensor:
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1.0 - eps)

        entropy = -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
        return entropy.sum()

    def _sample_to_peak_cell_occupancy(
        self,
        f_sample: Float[torch.Tensor, "n_total"],
    ) -> Float[torch.Tensor, "n_cells"]:
        device = f_sample.device
        occupancy = torch.zeros(
            self.grid_spec.n_cells,
            dtype=torch.bool,
            device=device,
        )

        for theta_pos, theta_idx in enumerate(self.grid_spec.theta_indices):
            sl = self.grid_spec.idxs_by_theta[theta_idx]
            x_slice = self.grid_spec.x_eval[sl, 0]
            y_slice = f_sample[sl]

            peak_idxs = self._find_peak_idxs(y_slice)

            if len(peak_idxs) == 0:
                continue

            # Convert normalized temperature positions to raw temperature.
            theta_dummy = torch.zeros_like(x_slice)
            x2 = torch.stack([x_slice, theta_dummy], dim=-1)
            T_raw = (
                self.learner.data_generator
                .untransform_inputs(x2)[:, 0]
                .detach()
                .cpu()
                .numpy()
            )

            peak_T_raw = T_raw[peak_idxs]

            # Peaks into temperature bins.
            temp_bin_idxs = np.searchsorted(
                self.grid_spec.temp_edges_raw,
                peak_T_raw,
                side="right",
            ) - 1

            valid = (
                (temp_bin_idxs >= 0)
                & (temp_bin_idxs < self.grid_spec.n_temp_bins)
            )

            for temp_bin_idx in temp_bin_idxs[valid]:
                cell_idx = theta_pos * self.grid_spec.n_temp_bins + int(temp_bin_idx)
                occupancy[cell_idx] = True

        return occupancy

    def _find_peak_idxs(
        self,
        y_slice: Float[torch.Tensor, "n"],
    ) -> np.ndarray:
        diagram_kwargs = getattr(self.learner.config, "diagram_generator_kwargs", {})

        peak_threshold = diagram_kwargs.get("peak_threshold", None)
        max_n_peaks = diagram_kwargs.get("max_n_peaks", 2)

        row = y_slice.detach().cpu()

        if peak_threshold is not None:
            scale = (row.amax() - row.amin()).item()
            prominence = peak_threshold * scale
        else:
            prominence = None

        y_np = row.numpy()
        y_shifted = y_np - np.min(y_np)

        peak_idxs, peak_info = find_peaks(
            y_shifted,
            prominence=prominence,
            height=(None, None),
        )

        if max_n_peaks is not None:
            ordering = np.argsort(-peak_info["peak_heights"], kind="stable")
            peak_idxs = peak_idxs[ordering][:max_n_peaks]

        return peak_idxs