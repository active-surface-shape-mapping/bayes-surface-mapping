from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import torch
from scipy.signal import find_peaks
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner

from .base_diagram_generator import DiagramOutput
from bayes_phase.utils import eval_context
from bayes_phase.data_io.base_data_generator import BaseDataGenerator
import numpy as np


class FindPeaksDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        learner: "ActiveLearner",
        n_temp_grid: int = 100,
        peak_threshold: float = 0.1,
        max_n_peaks: Optional[int] = 2,
        jitter: float = 1e-12,
    ):
        """Initialize a generator of phase diagram samples using peak-finding.

        Parameters
        ----------
        learner
            The active learner object, for context.
        n_temp_grid
            The number of temperature grid points to use when generating sample
            paths.
        peak_threshold
            The normalized `peak_threshold`, used to limit the prominence of
            peaks found by peak-finding.
        max_n_peaks
            The maximum number of peaks to generate per temperature slice. If
            this is not `None`, the generator will return only the first
            `max_n_peaks` with greatest height.
        jitter:
            Jitter for numerical stability of posterior latent distribution.
        """
        self.learner = learner
        self.n_theta_grid = learner.data_generator.theta_range + 1
        self.n_temp_grid = n_temp_grid
        self.temp_grid = torch.linspace(0.0, 1.0, n_temp_grid, dtype=torch.double)
        self.theta_grid = torch.arange(0, self.n_theta_grid) / (self.n_theta_grid - 1)
        X, Y = torch.meshgrid(self.temp_grid, self.theta_grid, indexing="ij")
        self.points = torch.stack([X.ravel(), Y.ravel()]).T
        self.peak_threshold = peak_threshold
        self.max_n_peaks = max_n_peaks
        self.jitter = jitter
        self._update()

    def _update(self):
        """Update the posterior latent distribution over grid points."""
        with eval_context(self.learner.model):
            self.pred = self.learner.model(self.points)
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
        """Generate a single sample of the phase diagram boundary.

        Returns
        -------
        diagram_output: DiagramOutput
            A two-element tuple where the first element is a collection of
            points representing the phase diagram boundary, and the second
            element is the GP sample path corresponding to that boundary.
        """
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
            thetas = self.theta_grid[i].tile(temps.numel())
            peak_locations.append(torch.stack((temps, thetas)).T)

        peaks = torch.vstack(peak_locations)
        peaks = self._remove_duplicate_points(peaks)
        return peaks, sample
