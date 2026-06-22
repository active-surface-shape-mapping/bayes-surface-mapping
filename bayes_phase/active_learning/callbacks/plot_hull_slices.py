import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from bayes_phase.acquisition.peak_finding import find_slice_peak_idxs


@torch.no_grad()
def plot_hull_slices(
    learner: "ActiveLearner",
    n_temp: int = 200,
    n_posterior_samples: int = 3,
):
    """
    Plot posterior mean slices with peak-hull overlays.

    Includes:
      - posterior mean
      - 95% credible interval
      - mean peak hull
      - posterior sample paths
      - sample-specific peak hulls
    """

    device = learner.train_x.device
    dtype = learner.train_x.dtype

    diagram_kwargs = getattr(learner.config, "diagram_generator_kwargs", {})

    temp_grid = torch.linspace(0.0, 1.0, n_temp, device=device, dtype=dtype)

    theta_vals = torch.arange(
        learner.data_generator.min_theta,
        learner.data_generator.max_theta + 1,
        device=device,
        dtype=dtype,
    )

    theta_grid = (
        theta_vals - learner.data_generator.min_theta
    ) / learner.data_generator.theta_range

    x, y = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    grid = torch.stack([x.ravel(), y.ravel()]).T

    # Use latent posterior for function uncertainty.
    y_pred = learner.model(grid)

    y_mean = (y_pred.mean * learner._y_std + learner._y_mean).reshape(-1)
    y_sd = (y_pred.variance.sqrt() * learner._y_std).reshape(-1)

    sample_paths = y_pred.rsample(torch.Size([n_posterior_samples]))
    sample_paths = sample_paths * learner._y_std + learner._y_mean
    sample_paths = sample_paths.reshape(n_posterior_samples, -1)

    grid_raw = learner.data_generator.untransform_inputs(grid.detach())
    T_raw = grid_raw[:, 0]
    theta_raw = grid_raw[:, 1]

    n_theta = len(theta_vals)
    fig, axs = plt.subplots(n_theta, 1, figsize=(8, 3 * n_theta), sharex=True)

    if n_theta == 1:
        axs = [axs]

    train_x = learner.data_generator.untransform_inputs(learner.train_x.detach())
    train_y = learner.train_y_orig.detach().reshape(-1)

    next_x = learner.data_generator.untransform_inputs(learner.next_x.detach())
    next_y = learner.next_y_orig.detach().reshape(-1)

    def peak_hull_for_slice(T_np: np.ndarray, y_np: np.ndarray):
        peaks = find_slice_peak_idxs(
            y_np,
            peak_threshold=diagram_kwargs.get("peak_threshold", None),
            max_n_peaks=diagram_kwargs.get("max_n_peaks", None),
            peak_height_rule=diagram_kwargs.get("peak_height_rule", None),
        )

        endpoint_idxs = np.array([0, len(y_np) - 1], dtype=int)
        vertex_idxs = np.unique(np.concatenate([endpoint_idxs, peaks]))
        vertex_idxs = np.sort(vertex_idxs)

        T_vertices = T_np[vertex_idxs]
        y_vertices = y_np[vertex_idxs]

        y_peak_hull = np.interp(
            T_np,
            T_vertices,
            y_vertices,
            left=y_vertices[0],
            right=y_vertices[-1],
        )

        return y_peak_hull, vertex_idxs

    for j, theta_val in enumerate(theta_vals):
        ax = axs[n_theta - j - 1]

        mask = theta_raw == theta_val

        T_slice = T_raw[mask]
        y_slice = y_mean[mask]
        sd_slice = y_sd[mask]
        sample_slices = sample_paths[:, mask]

        order = torch.argsort(T_slice)
        T_slice = T_slice[order]
        y_slice = y_slice[order]
        sd_slice = sd_slice[order]
        sample_slices = sample_slices[:, order]

        T_np = T_slice.detach().cpu().numpy()
        y_np = y_slice.detach().cpu().numpy()

        # Credible interval.
        ax.fill_between(
            T_slice.cpu(),
            (y_slice - 1.96 * sd_slice).cpu(),
            (y_slice + 1.96 * sd_slice).cpu(),
            color="royalblue",
            alpha=0.15,
            label="95% credible interval" if j == 0 else None,
        )

        sample_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Posterior samples and their peak hulls.
        for k in range(n_posterior_samples):
            y_sample_np = sample_slices[k].detach().cpu().numpy()
            y_sample_hull, _ = peak_hull_for_slice(T_np, y_sample_np)

            color = sample_colors[k % len(sample_colors)]

            ax.plot(
                T_np,
                y_sample_np,
                color=color,
                alpha=0.30,
                linewidth=1.0,
                label="Posterior samples" if (j == 0 and k == 0) else None,
            )

            ax.plot(
                T_np,
                y_sample_hull,
                color=color,
                linestyle="--",
                alpha=0.65,
                linewidth=1.0,
                label="Sample peak hulls" if (j == 0 and k == 0) else None,
            )

        # Mean peak hull.
        y_peak_hull, vertex_idxs = peak_hull_for_slice(T_np, y_np)

        T_vertices = T_np[vertex_idxs]
        y_vertices = y_np[vertex_idxs]

        ax.plot(
            T_slice.cpu(),
            y_slice.cpu(),
            color="black",
            label="Posterior mean" if j == 0 else None,
        )

        ax.plot(
            T_np,
            y_peak_hull,
            color="red",
            linestyle="--",
            label="Mean peak hull" if j == 0 else None,
        )

        ax.scatter(
            T_vertices,
            y_vertices,
            color="red",
            s=35,
            marker="^",
            label="Mean peak-hull vertices" if j == 0 else None,
            zorder=5,
        )

        mask_train = train_x[:, 1] == theta_val
        ax.scatter(
            train_x[mask_train, 0],
            train_y[mask_train],
            color="blue",
            s=20,
            label="Train" if j == 0 else None,
            zorder=6,
        )

        if next_x[0, 1] == theta_val:
            ax.scatter(
                next_x[:, 0],
                next_y,
                color="orange",
                s=40,
                marker="x",
                label="Next" if j == 0 else None,
                zorder=7,
            )

        ax.set_ylabel(f"θ={int(theta_val.item())}")

    axs[-1].set_xlabel("Temperature")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    out_dir = Path(learner.config.base_savepath) / "peak_hull_slice_plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    savepath = out_dir / f"iteration_{learner._i:03d}.png"
    plt.savefig(savepath, dpi=120)
    plt.close()


    # -------------------------------------------------------------------------
    # Bare-bones plot: posterior mean + mean peak hull only.
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(n_theta, 1, figsize=(8, 3 * n_theta), sharex=True)

    if n_theta == 1:
        axs = [axs]

    for j, theta_val in enumerate(theta_vals):
        ax = axs[n_theta - j - 1]

        mask = theta_raw == theta_val

        T_slice = T_raw[mask]
        y_slice = y_mean[mask]

        order = torch.argsort(T_slice)
        T_slice = T_slice[order]
        y_slice = y_slice[order]

        T_np = T_slice.detach().cpu().numpy()
        y_np = y_slice.detach().cpu().numpy()

        y_peak_hull, vertex_idxs = peak_hull_for_slice(T_np, y_np)

        T_vertices = T_np[vertex_idxs]
        y_vertices = y_np[vertex_idxs]


        ax.plot(
            T_np,
            y_np,
            color="black",
            linewidth=1.5,
            label="Posterior mean" if j == 0 else None,
        )

        ax.plot(
            T_np,
            y_peak_hull,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Mean peak hull" if j == 0 else None,
        )

        ax.scatter(
            T_vertices,
            y_vertices,
            color="red",
            s=35,
            marker="^",
            label="Mean peak-hull vertices" if j == 0 else None,
            zorder=5,
        )

        mask_train = train_x[:, 1] == theta_val
        ax.scatter(
            train_x[mask_train, 0],
            train_y[mask_train],
            color="blue",
            s=20,
            label="Train" if j == 0 else None,
            zorder=6,
        )

        ax.set_ylabel(f"θ={int(theta_val.item())}")

    axs[-1].set_xlabel("Temperature")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    plain_savepath = out_dir / f"iteration_{learner._i:03d}_plain.png"
    plt.savefig(plain_savepath, dpi=120)
    plt.close()