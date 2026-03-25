from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import torch
from scipy.signal import find_peaks
from typing import Optional, TYPE_CHECKING
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner

from .base_diagram_generator import DiagramOutput
from bayes_phase.utils import eval_context
from bayes_phase.data_io.base_data_generator import BaseDataGenerator


class FindPeaksAndValleysDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        learner: "ActiveLearner",
        n_temp_grid: int = 30,
        jitter: float = 1e-12,
        max_n_peaks: Optional[int] = 2,
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
        self.jitter = jitter
        self.max_n_peaks = max_n_peaks
        self._update()

    def _update(self):
        """Update the posterior latent distribution over grid points."""
        self._count = 0
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
        debug_plot: bool = True,
        plot_every: Optional[int] = 10,
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
        do_plot = debug_plot and (
            (plot_every is None) or (self.learner._i % plot_every == 0)
        )
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
        valley_locations = []
        increasing_locations = []
        decreasing_locations = []
        n_theta = self.n_theta_grid
        if do_plot:
            fig, axs = plt.subplots(
                n_theta, 1, sharex=True, sharey=False, figsize=(8, 32)
            )
        else:
            axs = None
        for i, row in enumerate(sample.T):
            # For each row, find the corresponding peaks/valleys/increasing
            # points/decreasing points
            peak_locs, valley_locs, increasing_locs, decreasing_locs = (
                self._apply_to_theta_row(row, i, n_theta, do_plot, axs)
            )
            peak_locations.append(peak_locs)
            valley_locations.append(valley_locs)
            increasing_locations.append(increasing_locs)
            decreasing_locations.append(decreasing_locs)

        peaks = torch.vstack(peak_locations)
        valleys = torch.vstack(valley_locations)
        increasing = torch.vstack(increasing_locations)
        decreasing = torch.vstack(decreasing_locations)
        if do_plot:
            assert axs is not None
            axs[-1].set_xlabel("Temperature")
            plt.tight_layout()
            savepath = (
                Path(self.learner.config.base_savepath)
                / "diagram_debug_plots"
                / f"iteration_{self.learner._i}"
                / f"n={self._count}.png"
            )
            savepath.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(savepath, dpi=100)
            plt.close()
        self._count += 1
        return DiagramOutput(
            peaks=peaks,
            valleys=valleys,
            increasing=increasing,
            decreasing=decreasing,
            sample=sample,
        )

    def _apply_to_theta_row(
        self,
        row: Float[torch.Tensor, " n"],
        i: int,
        n_theta: int,
        do_plot: bool,
        axs: Optional[dict[int, Axes]],
    ) -> tuple[
        Float[torch.Tensor, "p d"],
        Float[torch.Tensor, "q d"],
        Float[torch.Tensor, "r d"],
        Float[torch.Tensor, "s d"],
    ]:
        """For a single theta-slice, find the locations of peaks, valleys, and
        increasing/decreasing points."""
        # Find peaks of this row
        peaks, peak_info = find_peaks(
            row - row.amin(), height=0, prominence=(-np.inf, np.inf)
        )
        if self.max_n_peaks is not None:  # Return only n peaks with greatest height
            # Sort peaks by descending prominence, and select first max_n_peaks:
            ordering = np.argsort(-peak_info["prominences"], stable=True)
            peaks = peaks[ordering][: self.max_n_peaks]
        # Track valleys in-between peaks
        valleys = []
        sorted_peaks = sorted(peaks)
        # Track increasing and decreasing indices
        increasing_idxs = torch.zeros_like(row, dtype=torch.bool)
        decreasing_idxs = torch.zeros_like(row, dtype=torch.bool)
        if len(sorted_peaks) > 0:
            increasing_idxs[: sorted_peaks[0]] = True
            decreasing_idxs[sorted_peaks[0] :] = True
        for j in range(len(sorted_peaks) - 1):
            # Look at the section between each pair of peaks
            left_idx = sorted_peaks[j]
            right_idx = sorted_peaks[j + 1]
            section = row[left_idx:right_idx]
            if len(section) == 0:
                # Peaks are adjacent, skip
                continue
            # Find the valley in-between left and right peaks
            valley, valley_info = find_peaks(
                section.amax() - section, height=0, prominence=(-np.inf, np.inf)
            )
            # In case there are more than one, choose the one with greatest prominence:
            valley_ordering = np.argsort(-valley_info["prominences"], stable=True)
            valley = valley[valley_ordering][:1]
            valley += left_idx  # Shift indices to correct for start of section
            if len(valley) > 0:  # Valleys in-between
                valleys.append(valley[0])
                increasing_idxs[valley[0] : right_idx] = True
                decreasing_idxs[valley[0] : right_idx] = False
                increasing_idxs[left_idx : valley[0]] = False
                decreasing_idxs[left_idx : valley[0]] = True
            else:  # No valleys in-between
                if section[0] < section[-1]:  # Increasing
                    increasing_idxs[left_idx:right_idx] = True
                    decreasing_idxs[left_idx:right_idx] = False
                elif section[0] > section[-1]:  # Decreasing
                    decreasing_idxs[left_idx:right_idx] = True
                    increasing_idxs[left_idx:right_idx] = False
                else:  # Flat
                    increasing_idxs[left_idx:right_idx] = False
                    decreasing_idxs[left_idx:right_idx] = False
        valleys = np.array(valleys)
        # Find corresponding temperature and theta coordinates
        peak_temps = self.temp_grid[peaks]
        valley_temps = self.temp_grid[valleys]
        peak_thetas = self.theta_grid[i].tile(peak_temps.numel())
        valley_thetas = self.theta_grid[i].tile(valley_temps.numel())
        peak_locs = torch.stack((peak_temps, peak_thetas)).T
        valley_locs = torch.stack((valley_temps, valley_thetas)).T
        # 1st derivative should be zero at peaks/valleys:
        increasing_idxs[peaks] = False
        increasing_idxs[valleys] = False
        decreasing_idxs[peaks] = False
        decreasing_idxs[valleys] = False
        increasing_temps = self.temp_grid[increasing_idxs]
        increasing_thetas = self.theta_grid[i].tile(increasing_temps.numel())
        increasing_locs = torch.stack((increasing_temps, increasing_thetas)).T
        decreasing_temps = self.temp_grid[decreasing_idxs]
        decreasing_thetas = self.theta_grid[i].tile(decreasing_temps.numel())
        decreasing_locs = torch.stack((decreasing_temps, decreasing_thetas)).T
        # Make sure we aren't double-counting:
        peak_idxs = torch.zeros_like(row, dtype=torch.bool)
        peak_idxs[peaks] = True
        valley_idxs = torch.zeros_like(row, dtype=torch.bool)
        valley_idxs[valleys] = True
        assert not (peak_idxs & valley_idxs).any()
        assert not (peak_idxs & increasing_idxs).any()
        assert not (peak_idxs & decreasing_idxs).any()
        assert not (valley_idxs & increasing_idxs).any()
        assert not (valley_idxs & decreasing_idxs).any()
        assert not (increasing_idxs & decreasing_idxs).any()

        if do_plot:
            assert axs is not None
            with torch.no_grad():
                axs[n_theta - i - 1].scatter(
                    increasing_temps,
                    row[increasing_idxs],
                    marker="^",
                    color="red",
                    alpha=0.333,
                )
                axs[n_theta - i - 1].scatter(
                    decreasing_temps,
                    row[decreasing_idxs],
                    marker="v",
                    color="blue",
                    alpha=0.333,
                )
                axs[n_theta - i - 1].scatter(
                    peak_temps,
                    row[peaks],
                    marker="*",
                    color="red",
                    s=100,
                )
                axs[n_theta - i - 1].scatter(
                    valley_temps,
                    row[valleys],
                    marker="*",
                    color="blue",
                    s=100,
                )
                axs[n_theta - i - 1].plot(self.temp_grid, row.detach(), color="black")
                axs[n_theta - i - 1].set_ylabel(f"θ={i + 1}")
        return peak_locs, valley_locs, increasing_locs, decreasing_locs
