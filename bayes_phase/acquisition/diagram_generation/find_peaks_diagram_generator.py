from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import gpytorch
import torch
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from .base_diagram_generator import DiagramOutput
from bayes_phase.utils import eval_context


class FindPeaksDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelhood: gpytorch.likelihoods.Likelihood,
        theta_range: int,
        n_temp_grid: int = 500,
        n_starts: int = 10,
        peak_threshold: float = 0.02,
    ):
        self.model = model
        self.likelihood = likelhood
        self.n_theta_grid = theta_range + 1
        self.n_temp_grid = n_temp_grid
        self.temp_grid = torch.linspace(0.0, 1.0, n_temp_grid, dtype=torch.double)
        self.theta_grid = torch.arange(0, self.n_theta_grid) / theta_range
        X, Y = torch.meshgrid(self.temp_grid, self.theta_grid, indexing="ij")
        self.points = torch.stack([X.ravel(), Y.ravel()]).T
        self.peak_threshold = peak_threshold
        self._update()

    def _update(self):
        with eval_context(self.model):
            self.pred = self.model(self.points)
        self.distribution = torch.distributions.MultivariateNormal(
            self.pred.mean, self.pred.covariance_matrix
        )

    def generate_diagram(self) -> DiagramOutput:
        sample = self.distribution.sample().reshape(
            (self.n_temp_grid, self.n_theta_grid)
        )
        m, _ = sample.shape
        scale = (sample.amax() - sample.amin()).item()
        peak_locations = []
        for i, row in enumerate(sample.T):
            peaks, _ = find_peaks(row, prominence=self.peak_threshold * scale)
            temps = self.temp_grid[peaks]
            vals = row[peaks]
            thetas = self.theta_grid[i].tile(temps.numel())
            if (i == 4) and False:
                fig, ax = plt.subplots(1, 1)
                ax.plot(row.detach())
                ax.set_title(f"Theta={i + 1}")
                ax.plot(peaks, vals.detach(), "bo")
                plt.show(block=True)
                plt.close(fig)
            peak_locations.append(torch.stack((temps, thetas)).T)

        peaks = torch.vstack(peak_locations)
        return peaks, sample
