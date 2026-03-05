from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import gpytorch
import torch
from scipy.signal import find_peaks
from typing import Optional

from .base_diagram_generator import DiagramOutput
from bayes_phase.utils import eval_context
from bayes_phase.data_io.base_data_generator import BaseDataGenerator
import numpy as np


class FindPeaksDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        model: gpytorch.models.ExactGP | BaseDataGenerator,
        likelhood: gpytorch.likelihoods.Likelihood,
        theta_range: int,
        n_temp_grid: int = 100,
        n_starts: int = 10,
        peak_threshold: float = 0.1,
        max_n_peaks: Optional[int] = 2,
        jitter: float = 1e-12,
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
        self.max_n_peaks = max_n_peaks
        self.jitter = jitter
        self._update()

    def _update(self):
        with eval_context(self.model):
            self.pred = self.model(self.points)
        self.distribution = torch.distributions.MultivariateNormal(
            self.pred.mean,
            self.pred.covariance_matrix
            + self.jitter * torch.eye(self.pred.mean.numel()),
        )

    def generate_diagram(
        self,
        data_generator: Optional[BaseDataGenerator] = None,
        use_log_transform: bool = True,
        **kwargs,
    ) -> DiagramOutput:
        if data_generator is not None:  # Use ground-truth data
            sample = (
                data_generator.measure_uniform(self.points)
                .mean(-1)
                .reshape((self.n_temp_grid, self.n_theta_grid))
            )
            if use_log_transform:
                sample = sample.log()
        else:
            sample = self.distribution.sample().reshape(
                (self.n_temp_grid, self.n_theta_grid)
            )
        m, _ = sample.shape
        peak_locations = []
        for i, row in enumerate(sample.T):
            scale = (row.amax() - row.amin()).item()
            peaks, peak_info = find_peaks(
                row - row.amin(), prominence=self.peak_threshold * scale, height=0.0
            )
            if self.max_n_peaks is not None:  # Return only n peaks with greatest height
                ordering = np.argsort(-peak_info["peak_heights"], stable=True)
                peaks = peaks[ordering][: self.max_n_peaks]
            temps = self.temp_grid[peaks]
            vals = row[peaks]
            thetas = self.theta_grid[i].tile(temps.numel())
            peak_locations.append(torch.stack((temps, thetas)).T)

        peaks = torch.vstack(peak_locations)
        return peaks, sample
