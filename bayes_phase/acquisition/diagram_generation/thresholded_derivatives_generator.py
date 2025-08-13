from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import gpytorch
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


class DerivativesDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelhood: gpytorch.likelihoods.Likelihood,
        min_temp: float,
        max_temp: float,
        n_theta: int,
        n_temp_grid: int = 500,
        ddt_threshold: float = 0.25,
        d2dt2_threshold: float = -1.0,
        n_starts: int = 10,
    ):
        self.model = model
        self.likelihood = likelhood
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_theta = n_theta
        self.n_temp_grid = n_temp_grid
        self.temp_grid = torch.linspace(
            min_temp, max_temp, n_temp_grid, dtype=torch.double
        )
        self.theta_grid = torch.arange(1, n_theta + 1, 1, dtype=torch.double)
        X, Y = torch.meshgrid(self.temp_grid, self.theta_grid)
        self.points = torch.vstack([X.ravel(), Y.ravel()]).T
        print(self.points.shape)
        self.pred = self.model(self.model.transform_inputs(self.points))
        self.distribution = torch.distributions.Normal(
            self.pred.mean, self.pred.variance.sqrt()
        )
        self.starts = torch.linspace(min_temp, max_temp, n_starts)
        self.ddt_threshold = ddt_threshold
        self.d2dt2_threshold = d2dt2_threshold

    def generate_diagram(self):
        raise NotImplementedError  # TODO
        sample = self.distribution.sample().reshape((self.n_temp_grid, self.n_theta))
        n = self.n_theta
        opts_pred = torch.zeros((n, len(self.starts), 2))
        vals_pred = torch.zeros((n, len(self.starts), 1))
        for i in range(n):
            spline = CubicSpline(self.temp_grid, -sample[:, i])
            for j, start in enumerate(self.starts):
                opt = minimize(
                    spline,
                    x0=[start],
                    method="Nelder-Mead",
                    bounds=[(self.min_temp, self.max_temp)],
                )
                opts_pred[i, j, 0] = opt.x.item()
                opts_pred[i, j, 1] = i + 1
                vals_pred[i, j, 0] = -opt.fun
        return self._normalize_peaks(
            opts_pred, vals_pred, threshold=self.peak_threshold
        ), sample
