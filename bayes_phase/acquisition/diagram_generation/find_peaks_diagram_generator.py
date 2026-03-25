from bayes_phase.acquisition.diagram_generation import BaseDiagramGenerator
import torch
from scipy.signal import find_peaks
from typing import Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner

from .base_diagram_generator import DiagramOutput
from bayes_phase.utils import eval_context
from bayes_phase.data_io.base_data_generator import BaseDataGenerator


class FindPeaksDiagramGenerator(BaseDiagramGenerator):
    def __init__(
        self,
        learner: "ActiveLearner",
        n_temp_grid: int = 100,
        peak_threshold: Optional[float] = None,
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
            if self.peak_threshold is not None:
                scale = (row.amax() - row.amin()).item()
                prominence = self.peak_threshold * scale
            else:
                prominence = None
            peaks, peak_info = find_peaks(
                row - row.amin(), prominence=prominence, height=0.0
            )
            if self.max_n_peaks is not None:  # Return only n peaks with greatest height
                ordering = np.argsort(-peak_info["peak_heights"], stable=True)
                peaks = peaks[ordering][: self.max_n_peaks]
            temps = self.temp_grid[peaks]
            thetas = self.theta_grid[i].tile(temps.numel())
            peak_locations.append(torch.stack((temps, thetas)).T)

        peaks = torch.vstack(peak_locations)
        peaks = self._remove_duplicate_points(peaks)
        self._count += 1
        return DiagramOutput(
            peaks=peaks, valleys=None, increasing=None, decreasing=None, sample=sample
        )

    def diagrams(
        self,
        n_diagrams: int,
        plot: bool = False,
        plot_full_every: Optional[int] = None,
        **kwargs,
    ) -> list[DiagramOutput]:
        """Get a list of phase diagram boundaries.

        Parameters
        ----------
        n_diagrams
            The number of diagrams to generate.

        Returns
        -------
        diagrams
            A list of `DiagramOutputs`.
        """
        self._update()
        diagrams = []
        for i in range(n_diagrams):
            diagrams.append(self.generate_diagram(**kwargs))
        if plot:
            try:
                self.plot(diagrams)
            except Exception as e:
                print(f"Failed to plot full phase boundary diagrams: {e}")
                plt.close()
            if (plot_full_every is not None) and (
                self.learner._i % plot_full_every == 0
            ):  # Separate plots for each sample, to debug
                try:
                    [
                        self.plot([diagram], suffix=f"_{i}")
                        for i, diagram in enumerate(diagrams)
                    ]
                except Exception as e:
                    print(f"Failed to plot individual phase boundary diagrams: {e}")
                    plt.close()
        return diagrams

    def plot(
        self,
        diagrams: list[DiagramOutput],
        suffix: str = "",
    ):
        with torch.no_grad():
            n_theta = self.n_theta_grid
            mean = self.pred.mean.detach().reshape(
                (self.n_temp_grid, self.n_theta_grid)
            )
            stdv = (
                self.pred.variance.detach()
                .sqrt()
                .reshape((self.n_temp_grid, self.n_theta_grid))
            )
            fig, axs = plt.subplots(
                n_theta, 1, sharex=True, sharey=False, figsize=(8, 32)
            )
            for i in range(n_theta):
                for peaks, _, _, _, sample in diagrams:
                    assert peaks is not None
                    assert sample is not None
                    l = axs[n_theta - i - 1].plot(
                        self.temp_grid, sample[:, i], alpha=0.333
                    )
                    color = l[0].get_color()
                    peak_idxs = peaks[:, 1].isclose(self.theta_grid[i])
                    these_peaks = peaks[peak_idxs]
                    for this_peak in these_peaks[:, 0]:
                        axs[n_theta - i - 1].axvline(
                            this_peak,
                            color=color,
                            alpha=0.333,
                        )
                axs[n_theta - i - 1].plot(self.temp_grid, mean[:, i], color="black")
                # axs[n_theta - i - 1].fill_between(
                #     self.temp_grid,
                #     mean[:, i] - 1.96 * stdv[:, i],
                #     mean[:, i] + 1.96 * stdv[:, i],
                #     color="black",
                #     alpha=0.333,
                # )
                axs[n_theta - i - 1].plot(
                    self.temp_grid,
                    mean[:, i] - 1.96 * stdv[:, i],
                    color="black",
                    linestyle="dashed",
                )
                axs[n_theta - i - 1].plot(
                    self.temp_grid,
                    mean[:, i] + 1.96 * stdv[:, i],
                    color="black",
                    linestyle="dashed",
                )
                # Plot training points
                train_x = self.learner.train_x
                train_y = self.learner.train_y
                idxs = train_x[:, 1].isclose(torch.tensor(float(i) / (n_theta - 1)))
                axs[n_theta - i - 1].scatter(train_x[idxs, 0], train_y[idxs, 0], s=200)
                axs[n_theta - i - 1].set_ylabel(f"θ={i + 1}")
            axs[-1].set_xlabel("Temperature")
            plt.tight_layout()
            savepath = (
                Path(self.learner.config.base_savepath)
                / "diagram_plots"
                / f"iteration_{self.learner._i}"
                / f"n={self._count}{suffix}.png"
            )
            savepath.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(savepath, dpi=100)
            plt.close()
