from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import torch
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner

from .base_diagram_generator import DiagramOutput
from bayes_phase.utils import eval_context


class SplineDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        learner: "ActiveLearner",
        n_temp_grid: int = 500,
        n_starts: int = 10,
        peak_threshold: float = 0.7,
    ):
        self.learner = learner
        self.n_theta_grid = learner.data_generator.theta_range + 1
        self.n_temp_grid = n_temp_grid
        self.temp_grid = torch.linspace(0, 1, n_temp_grid, dtype=torch.double)
        self.theta_grid = torch.arange(0, self.n_theta_grid) / (self.n_theta_grid + 1)
        X, Y = torch.meshgrid(self.temp_grid, self.theta_grid, indexing="ij")
        self.points = torch.vstack([X.ravel(), Y.ravel()]).T
        self.starts = torch.linspace(0.0, 1.0, n_starts)
        self.peak_threshold = peak_threshold
        self._update()

    def _update(self):
        """Update the posterior latent distribution over grid points."""
        with eval_context(self.learner.model):
            self.pred = self.learner.model(self.points)
        self.distribution = torch.distributions.MultivariateNormal(
            self.pred.mean, self.pred.covariance_matrix
        )

    def generate_diagram(self) -> DiagramOutput:
        """Generate a single sample of the phase diagram boundary.

        Returns
        -------
        diagram_output: DiagramOutput
            A two-element tuple where the first element is a collection of
            points representing the phase diagram boundary, and the second
            element is the GP sample path corresponding to that boundary.
        """
        sample = self.distribution.sample().reshape(
            (self.n_temp_grid, self.n_theta + 1)
        )
        n = self.n_theta + 1
        opts_pred = torch.zeros((n, len(self.starts), 2))
        vals_pred = torch.zeros((n, len(self.starts), 1))
        peaks = []
        for i in range(n):
            spline = CubicSpline(self.temp_grid, -sample[:, i])
            for j, start in enumerate(self.starts):
                opt = minimize(
                    spline,
                    x0=[start],
                    method="Nelder-Mead",
                    bounds=[(0.0, 1.0)],
                )
                opts_pred[i, j, 0] = opt.x.item()
                opts_pred[i, j, 1] = self.theta_grid[i]
                vals_pred[i, j, 0] = -opt.fun
        return self._normalize_peaks(
            opts_pred, vals_pred, threshold=self.peak_threshold
        ), sample
