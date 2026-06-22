import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import gpytorch
import numpy as np
import torch
from jaxtyping import Float
from scipy.signal import find_peaks
from tqdm import tqdm

from bayes_phase.acquisition.peak_finding import find_slice_peak_idxs
from bayes_phase.utils import eval_context

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner


@dataclass
class HullGridSpec:
    """
    Organize structural information for about slices and evaluation.

    Attributes
    ----------
    x_eval:
        Full evaluation grid in unit coordinates, shape (n_total, 2).
        This contains the lower-resolution slice grid points where hull variance
        is evaluated.
    peak_x_eval:
        Full peak-finding grid in unit coordinates, shape (n_peak_total, 2).
        This contains the higher-resolution slice grid points where peaks are
        detected before interpolation onto x_eval.
    temp_eval:
        1D temperature grid in normalized [0, 1] coordinates, shape (n_temp,).
    peak_temp_eval:
        1D peak-finding temperature grid in normalized [0, 1] coordinates,
        shape (n_peak_temp,).
    theta_values:
        1D normalized theta slice values, shape (n_theta,).
    idxs_by_theta:
        Dictionary mapping integer theta index -> row indices in x_eval / hull vector.
    peak_idxs_by_theta:
        Dictionary mapping integer theta index -> row indices in peak_x_eval.
    x_eval_np_by_theta:
        Unchanging x values for eval grid.
    peak_x_eval_np_by_theta:
        Unchanging dense x values for peak finding.
    """
    x_eval: Float[torch.Tensor, "n_total 2"]
    peak_x_eval: Float[torch.Tensor, "n_peak_total 2"]
    temp_eval: Float[torch.Tensor, "n_temp"]
    peak_temp_eval: Float[torch.Tensor, "n_peak_temp"]
    theta_values: Float[torch.Tensor, "n_theta"]
    idxs_by_theta: dict[int, slice]
    peak_idxs_by_theta: dict[int, slice]
    x_eval_np_by_theta: dict[int, np.ndarray]
    peak_x_eval_np_by_theta: dict[int, np.ndarray]


class HullVariance:
    """
    Expected variance reduction for learning slice-wise piecewise linear peak representer functions
    of a heat capacity surface.

    Design:
    - One upper hull per discrete theta slice.
    - Detect peaks on a relatively dense temperature grid.
    - Represent each sampled hull by its values on a lower-resolution
      temperature grid.
    - Concatenate all slices into one global hull vector H.
    - Estimate variance of H using the empirical covariance of Monte Carlo hull samples.
    """

    def __init__(self, learner: "ActiveLearner"):
        self.learner = learner

        self.n_fantasies = int(
            getattr(self.learner.config, "hull_n_fantasies", 10)
        )
        self.n_hull_mc_samples = int(
            getattr(self.learner.config, "hull_n_mc_samples", 100)
        )
        self.n_hull_eval_grid = int(
            getattr(self.learner.config, "hull_n_temp_grid", 75)
        )
        self.n_temp_peak_grid = int(
            getattr(self.learner.config, "hull_n_temp_peak_grid", 100)
        )
        self.sample_jitter = float(
            getattr(self.learner.config, "hull_sample_jitter", 1e-8)
        )

        self.grid_spec = self._build_hull_grid()

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
        Compute expected reduction in integrated hull variance for candidates X.
        """
        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        X = X.to(device=device, dtype=dtype)

        # Baseline integrated hull variance Var(H | D)
        baseline_var = self._estimate_integrated_hull_variance_from_model(
            self.learner.model
        )

        scores = torch.empty(X.shape[0], dtype=dtype, device=device)

        # Loop over candidates
        for i in tqdm(range(X.shape[0]), desc="Hull variance candidates", position=1):
            x_cand = X[i : i + 1]

            # Create fantasy observations
            fantasy_y = self._sample_fantasized_observations(
                model=self.learner.model,
                x_new=x_cand,
                n_fantasies=self.n_fantasies,
            )

            post_vars = []

            # Loop over fantasies
            for y_fant in fantasy_y:
                # Condition on fantasy observations
                fantasy_model = self.learner.model.get_fantasy_model(
                    x_cand,
                    y_fant.reshape(1),
                )

                # Compute variance after conditioning on fantasy
                with eval_context(fantasy_model):
                    post_var = self._estimate_integrated_hull_variance_from_model(
                        fantasy_model
                    )

                post_vars.append(post_var)

            # Compute average sum posterior variance
            expected_post_var = torch.stack(post_vars).mean()

            # Score: expected reduction in hull variance
            scores[i] = baseline_var - expected_post_var

        return scores

    def _build_hull_grid(self) -> HullGridSpec:
        """
        Build the dense evaluation grid used to vectorize the slice-wise hulls.
        """
        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype

        temp_eval = torch.linspace(
            0.0, 1.0, self.n_hull_eval_grid, device=device, dtype=dtype
        )

        peak_temp_eval = torch.linspace(
            0.0, 1.0, self.n_temp_peak_grid, device=device, dtype=dtype
        )

        theta_range = self.learner.data_generator.theta_range

        theta_indices = list(range(theta_range + 1))

        theta_values = torch.tensor(
            [idx / theta_range for idx in theta_indices],
            device=device,
            dtype=dtype,
        )

        x_blocks = []
        peak_x_blocks = []
        idxs_by_theta = {}
        peak_idxs_by_theta = {}
        start = 0
        peak_start = 0

        # Blocks are collections of points along a slice.
        for theta_idx, theta_val in zip(theta_indices, theta_values):
            theta_block = torch.full_like(temp_eval, fill_value=theta_val)
            x_block = torch.stack([temp_eval, theta_block], dim=-1)
            x_blocks.append(x_block)

            stop = start + temp_eval.numel()
            idxs_by_theta[int(theta_idx)] = slice(start, stop)
            start = stop

            peak_theta_block = torch.full_like(peak_temp_eval, fill_value=theta_val)
            peak_x_block = torch.stack([peak_temp_eval, peak_theta_block], dim=-1)
            peak_x_blocks.append(peak_x_block)

            peak_stop = peak_start + peak_temp_eval.numel()
            peak_idxs_by_theta[int(theta_idx)] = slice(peak_start, peak_stop)
            peak_start = peak_stop

        x_eval = torch.vstack(x_blocks)
        peak_x_eval = torch.vstack(peak_x_blocks)

        x_eval_np_by_theta = {}
        peak_x_eval_np_by_theta = {}

        for theta_idx in theta_indices:
            sl = idxs_by_theta[int(theta_idx)]
            peak_sl = peak_idxs_by_theta[int(theta_idx)]

            x_eval_np_by_theta[int(theta_idx)] = (
                x_eval[sl, 0].detach().cpu().numpy()
            )
            peak_x_eval_np_by_theta[int(theta_idx)] = (
                peak_x_eval[peak_sl, 0].detach().cpu().numpy()
            )

        return HullGridSpec(
            x_eval=x_eval,
            peak_x_eval=peak_x_eval,
            temp_eval=temp_eval,
            peak_temp_eval=peak_temp_eval,
            theta_values=theta_values,
            idxs_by_theta=idxs_by_theta,
            peak_idxs_by_theta=peak_idxs_by_theta,
            x_eval_np_by_theta=x_eval_np_by_theta,
            peak_x_eval_np_by_theta=peak_x_eval_np_by_theta,
        )

    @torch.no_grad()
    def _sample_fantasized_observations(
        self,
        model: gpytorch.models.ExactGP,
        x_new: Float[torch.Tensor, "1 d"],
        n_fantasies: int,
    ) -> Float[torch.Tensor, "n_fantasies"]:
        """
        Sample fantasized normalized observations y ~ p(f(x_new) | D).
        """
        with eval_context(model):
            posterior = model(x_new)

        mean = posterior.mean.reshape(-1)
        var = posterior.variance.reshape(-1).clamp_min(1e-30)

        dist = torch.distributions.Normal(mean, var.sqrt())
        return dist.sample((n_fantasies,)).reshape(-1)

    @torch.no_grad()
    def _sample_hull_vectors_from_model(
        self,
        model: gpytorch.models.ExactGP,
        n_samples: int,
    ) -> Float[torch.Tensor, "n_samples d_hull"]:
        """
        Sample hull vectors from the posterior induced by model on the peak grid.
        """
        x_eval = self.grid_spec.peak_x_eval
        device = x_eval.device
        dtype = x_eval.dtype

        with eval_context(model):
            posterior = model(x_eval)

        mean = posterior.mean.reshape(-1)
        cov = posterior.covariance_matrix
        cov = cov + self.sample_jitter * torch.eye(
            cov.shape[0], device=device, dtype=dtype
        )

        assert mean.ndim == 1, f"Expected flattened mean to be 1D, got {mean.shape}"
        assert mean.numel() == x_eval.shape[0], (
            f"Expected mean length {x_eval.shape[0]}, got {mean.numel()}"
        )
        assert cov.shape == (x_eval.shape[0], x_eval.shape[0]), (
            f"Expected covariance {(x_eval.shape[0], x_eval.shape[0])}, got {cov.shape}"
        )

        L = torch.linalg.cholesky(cov)

        eps = torch.randn(
            n_samples,
            mean.numel(),
            device=device,
            dtype=dtype,
        )

        f_samples = mean + eps @ L.T

        hull_vectors = []
        for k in range(n_samples):
            hull_vec = self._sample_to_global_hull_vector(f_samples[k])
            hull_vectors.append(hull_vec)

        return torch.stack(hull_vectors, dim=0)

    def _sample_to_global_hull_vector(
        self,
        f_sample: Float[torch.Tensor, "n_peak_total"],
    ) -> Float[torch.Tensor, "d_hull"]:
        """
        Convert one sampled function on the dense stacked peak grid into a
        concatenated global hull vector across all theta slices.
        """
        assert f_sample.ndim == 1, f"Expected 1D f_sample, got shape {f_sample.shape}"
        assert f_sample.shape[0] == self.grid_spec.peak_x_eval.shape[0], (
            f"Expected f_sample length {self.grid_spec.peak_x_eval.shape[0]}, "
            f"got {f_sample.shape[0]}"
        )

        hull_parts = []

        for theta_idx, sl in self.grid_spec.idxs_by_theta.items():
            peak_sl = self.grid_spec.peak_idxs_by_theta[theta_idx]

            x_eval_np = self.grid_spec.x_eval_np_by_theta[theta_idx]
            x_peak_np = self.grid_spec.peak_x_eval_np_by_theta[theta_idx]
            y_peak_slice = f_sample[peak_sl]

            hull_y = self._peak_hull_values_on_grid(
                x_nodes_np=x_peak_np,
                y_nodes=y_peak_slice,
                x_eval_np=x_eval_np,
            )

            hull_parts.append(hull_y)

        return torch.cat(hull_parts, dim=0)

    def _peak_hull_values_on_grid(
        self,
        x_nodes_np: np.ndarray,
        y_nodes: Float[torch.Tensor, "n"],
        x_eval_np: np.ndarray,
    ) -> Float[torch.Tensor, "m"]:
        """
        Build a 'peak hull' using the same peak-finding logic as
        FindPeaksDiagramGenerator, but always include the min-T and max-T
        endpoints as hull vertices.
        """
        device = y_nodes.device
        dtype = y_nodes.dtype

        y_np = y_nodes.detach().cpu().numpy()

        diagram_kwargs = getattr(
            self.learner.config,
            "diagram_generator_kwargs",
            {},
        )

        peak_idxs = find_slice_peak_idxs(
            y_nodes,
            peak_threshold=diagram_kwargs.get("peak_threshold", None),
            max_n_peaks=diagram_kwargs.get("max_n_peaks", None),
            peak_height_rule=diagram_kwargs.get("peak_height_rule", None),
        )

        # Always include slice endpoints.
        endpoint_idxs = np.array([0, len(y_np) - 1], dtype=int)

        # Combine endpoints + peaks, remove duplicates, restore left-to-right order.
        vertex_idxs = np.unique(np.concatenate([endpoint_idxs, peak_idxs]))
        vertex_idxs = np.sort(vertex_idxs)

        x_vertices = x_nodes_np[vertex_idxs]
        y_vertices = y_np[vertex_idxs]

        y_out = np.interp(
            x_eval_np,
            x_vertices,
            y_vertices,
            left=y_vertices[0],
            right=y_vertices[-1],
        )

        return torch.as_tensor(
            y_out,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _estimate_integrated_hull_variance_from_model(
        self,
        model: gpytorch.models.ExactGP,
    ) -> torch.Tensor:
        """
        Estimate integrated pointwise posterior variance of the random hull.

        Steps:
        1. Sample functions from the GP posterior on the dense peak-finding grid.
        2. Convert each sampled function to a global slice-wise hull vector.
        3. Compute pointwise variance of hull values across MC hull samples.
        4. Sum those variances over the hull grid.

        Returns
        -------
        integrated_var
            Scalar variance proxy for global hull uncertainty.
        """
        hull_samples = self._sample_hull_vectors_from_model(
            model=model,
            n_samples=self.n_hull_mc_samples,
        )  # (n_mc, d_hull)

        return self._integrated_hull_variance(hull_samples)

    def _integrated_hull_variance(
        self,
        hull_samples: Float[torch.Tensor, "n d"],
    ) -> torch.Tensor:
        """
        Integrated pointwise variance of sampled hull values.

        This approximates uncertainty in the random hull:
            sum_j Var[H_j]
        """
        if hull_samples.shape[0] < 2:
            raise ValueError("Need at least two hull samples to estimate variance.")

        pointwise_var = hull_samples.var(dim=0, unbiased=True).clamp_min(1e-12) # unbiased=True is Bessel

        return pointwise_var.mean()