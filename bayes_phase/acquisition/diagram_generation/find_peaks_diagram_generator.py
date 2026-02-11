from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import gpytorch
import torch
from scipy.signal import find_peaks

from .base_diagram_generator import DiagramOutput


class FindPeaksDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelhood: gpytorch.likelihoods.Likelihood,
        min_temp: float,
        max_temp: float,
        n_theta: int,
        n_temp_grid: int = 500,
        n_starts: int = 10,
        peak_threshold: float = 0.06,
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
        self.peak_threshold = peak_threshold

    def generate_diagram(self) -> DiagramOutput:
        sample = self.distribution.sample().reshape((self.n_temp_grid, self.n_theta))
        m, _ = sample.shape
        scale = (sample.amax() - sample.amin()).item()
        peak_locations = []
        for i, row in enumerate(sample.T):
            peaks, _ = find_peaks(row, prominence=self.peak_threshold * scale)
            temps = self.temp_grid[peaks]
            thetas = self.theta_grid[i].tile(temps.numel())
            peak_locations.append(torch.stack((temps, thetas)).T)

        peaks = torch.vstack(peak_locations)
        return peaks, sample
