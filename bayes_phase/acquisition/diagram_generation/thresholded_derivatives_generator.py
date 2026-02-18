from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import gpytorch
import torch


class ThresholdedDerivativesDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelhood: gpytorch.likelihoods.Likelihood,
        min_temp: float,
        max_temp: float,
        n_theta: int,
        n_temp_grid: int = 500,
        ddt_threshold: float = 0.05,
        d2dt2_threshold: float = -1.0,
        n_starts: int = 10,
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
        print(self.points.shape)
        self.pred = self.model(self.model.transform_inputs(self.points))
        self.distribution = torch.distributions.Normal(
            self.pred.mean, self.pred.variance.sqrt()
        )
        self.starts = torch.linspace(0.0, 1.0, n_starts)
        self.ddt_threshold = ddt_threshold
        self.d2dt2_threshold = d2dt2_threshold

    def generate_diagram(self):
        points = self.model.transform_inputs(self.points)
        print(points.shape)
        mean, covar = self.model.predict_with_grad(points, compute_covariance="full")
        print(mean.shape)
        print(covar.shape)
        raise NotImplementedError(
            "Diagram generation not fully implemented for thresholded derivatives."
        )
        sample_with_grad = torch.distributions.MultivariateNormal(mean, covar).sample(
            mean.shape
        )
        sample = sample_with_grad[0, :]
        peaks = (sample_with_grad[:, 1] < self.ddt_threshold) & (
            sample_with_grad[:, 2] < self.d2dt2_threshold
        )
        peak_locations = self.points[peaks]

        return peak_locations, sample
