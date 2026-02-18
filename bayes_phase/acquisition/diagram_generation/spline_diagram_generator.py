from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import gpytorch
import torch
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from .base_diagram_generator import DiagramOutput


class SplineDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelhood: gpytorch.likelihoods.Likelihood,
        n_theta: int,
        n_temp_grid: int = 500,
        n_starts: int = 10,
        peak_threshold: float = 0.7,
    ):
        self.model = model
        self.likelihood = likelhood
        self.n_theta = n_theta
        self.n_temp_grid = n_temp_grid
        self.temp_grid = torch.linspace(
            0, 1, n_temp_grid, dtype=torch.double
        )
        self.theta_grid = torch.arange(0, n_theta + 1) / n_theta
        X, Y = torch.meshgrid(self.temp_grid, self.theta_grid)
        self.points = torch.vstack([X.ravel(), Y.ravel()]).T
        self.pred = self.model(self.points)
        self.distribution = torch.distributions.Normal(
            self.pred.mean, self.pred.variance.sqrt()
        )
        self.starts = torch.linspace(0.0, 1.0, n_starts)
        self.peak_threshold = peak_threshold

    def generate_diagram(self) -> DiagramOutput:
        sample = self.distribution.sample().reshape((self.n_temp_grid, self.n_theta + 1))
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
