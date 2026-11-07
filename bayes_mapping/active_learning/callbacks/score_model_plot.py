import json
import warnings
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely as shp
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from bayes_mapping.acquisition.diagram_generation.base_diagram_generator import (
    DiagramOutput,
)
from bayes_mapping.utils import eval_context

from .plot_mst import _plot_mst_diagrams


def _plot_mst_diagrams_with_grid(
    learner: "ActiveLearner",
    diagrams: Iterator[DiagramOutput],
    ax: Axes,
    color: str | None = None,
    grid: gpd.GeoDataFrame | None = None,
    true_hits: gpd.GeoDataFrame | None = None,
):
    """Create side-by-side plots of GT phase diagram and sampled phase diagrams."""
    if grid is None:
        grid = _generate_grid(learner)

    grid.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=6, aspect=None)

    if true_hits is not None and len(true_hits) > 0:
        true_hits.boundary.plot(ax=ax, color="red", linewidth=1.2, zorder=7, aspect=None)

    ax.set_aspect("auto")

    # Reuse the MST callback plotting logic so scorer obeys
    # learner.config.tree_graph_type ("MST" or "MSTCurve")
    _plot_mst_diagrams(
        learner=learner,
        diagrams=diagrams,
        ax=ax,
        color=color,
    )


def _generate_grid(
    learner: "ActiveLearner",
) -> gpd.GeoDataFrame:
    """Build rectangular scoring boxes."""
    cfg = learner.config
    dg = learner.ground_truth_data_generator

    t_min = dg.min_temp
    t_max = dg.max_temp

    dt = cfg.scoring_grid_T_range

    # Score all integer theta levels in the ground-truth domain.
    theta_values = list(range(
        dg.min_theta,
        dg.max_theta + 1,
    ))

    records = []

    t0 = t_min
    t_idx = 0
    while t0 < t_max:
        t1 = min(t0 + dt, t_max)

        for theta_idx, theta_val in enumerate(theta_values):
            # Box centered on integer theta value
            th0 = theta_val - 0.5
            th1 = theta_val + 0.5

            records.append(
                {
                    "t_bin": t_idx,
                    "theta_bin": theta_idx,
                    "theta_value": theta_val,
                    "t_min": t0,
                    "t_max": t1,
                    "theta_min": th0,
                    "theta_max": th1,
                    "geometry": shp.box(t0, th0, t1, th1),
                }
            )

        t0 = t1
        t_idx += 1

    return gpd.GeoDataFrame(records, geometry="geometry", crs=None)


def _assign_ridge_points_to_grid(
    learner: "ActiveLearner",
    grid: gpd.GeoDataFrame,
    ridge_x: torch.Tensor,
) -> gpd.GeoDataFrame:
    """Assign each ridge point to exactly one scoring cell.
    When a point falls on the boundary between two grid boxes, it is assigned to the lower box.
    """

    # Check if ridges are present
    if ridge_x.numel() == 0:
        out = grid.iloc[0:0].copy()
        out["contains_ridge_point"] = pd.Series(dtype=bool)
        return out

    # Convert to physical units
    xys = learner.data_generator.untransform_inputs(ridge_x.detach()).cpu().numpy()
    T = xys[:, 0]
    theta = xys[:, 1]

    # Reconstruct bin edges from the scoring grid.
    t_edges = np.r_[grid["t_min"].unique(), grid["t_max"].max()]
    t_edges = np.unique(t_edges)

    theta_edges = np.r_[grid["theta_min"].unique(), grid["theta_max"].max()]
    theta_edges = np.unique(theta_edges)

    # Assign values to bins. Values on edges are assigned to the lower bin, except in the lowest temperature box.
    t_bin = np.searchsorted(t_edges, T, side="left") - 1
    theta_bin = np.searchsorted(theta_edges, theta, side="left") - 1
    t_bin[np.isclose(T, t_edges[0])] = 0
    theta_bin[np.isclose(theta, theta_edges[0])] = 0

    valid = (
        (t_bin >= 0) & (t_bin < len(t_edges) - 1) &
        (theta_bin >= 0) & (theta_bin < len(theta_edges) - 1)
    )

    if not np.all(valid):
        warnings.warn(
            "Ridge points found outside the scoring grid. "
            "Check grid bounds and ridge values.",
            RuntimeWarning,
        )

    # Bin indices.
    key_to_index = {
        (int(row.t_bin), int(row.theta_bin)): idx
        for idx, row in grid.iterrows()
    }

    # Convert bin coordinates into the corresponding grid row index in the geodataframe.
    hit_indices = {
        key_to_index[(int(tb), int(thb))]
        for tb, thb in zip(t_bin[valid], theta_bin[valid])
    }

    # Mark occupied grid cells.
    hits = grid.loc[sorted(hit_indices)].copy()
    hits["contains_ridge_point"] = True
    return hits


def _plot_metrics_by_iteration(
    learner: "ActiveLearner",
    csv_path: Path,
) -> None:
    """Plot score metrics by iteration."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("iteration")

    colors = plt.get_cmap("Set1").colors

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: recall / precision / F1
    ax[0].plot(
        df["iteration"], df["recall"],
        linewidth=1, label="Recall", color=colors[0]
    )
    ax[0].plot(
        df["iteration"], df["precision"],
        linewidth=1, label="Precision", color=colors[1]
    )
    ax[0].plot(
        df["iteration"], df["f1"],
        linewidth=1, label="F1 score", color=colors[2]
    )

    if learner.config.acquisition_function == "random":
        literate_acquisition_function = "Random sampling"
    elif learner.config.acquisition_function == "dispersion":
        literate_acquisition_function = "EFDI"
    elif learner.config.acquisition_function == "eig":
        literate_acquisition_function = "EIG"
    elif learner.config.acquisition_function == "sobol":
        literate_acquisition_function = "Sobol sampling"
    else:
        literate_acquisition_function = learner.config.acquisition_function

    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Score")
    ax[0].set_title("Recall, precision, and F1 score")
    ax[0].set_ylim(0.0, 1.0)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Right panel: log likelihood
    ax[1].plot(
        df["iteration"],
        df["log_likelihood"],
        linewidth=1,
        color="purple",
        label="Log likelihood",
    )
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Log likelihood")
    ax[1].set_title("Log likelihood")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.suptitle(
        f"Performance by iteration: {literate_acquisition_function}",
        fontsize=12,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(csv_path.parent / "score_metrics.pdf")
    plt.close(fig)

def _calc_score(
    learner: "ActiveLearner",
    grid: gpd.GeoDataFrame,
    true_idx: set[int],
    n_true_peaks: int,
    diagrams: list[DiagramOutput],
    neighbor_distance: dict[str, float | int],
    model_hyperparameters: dict[str, float],
    alpha: float = 1.0,
    hit_counts: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate and save scores by iteration."""
    n_scoring_samples = len(diagrams)
    total_boxes = len(grid)
    n_true = len(true_idx)

    if n_true == 0:
        raise RuntimeError(
            "Scoring failed: ground-truth ridge points do not hit any grid boxes."
        )
    if n_scoring_samples == 0:
        raise RuntimeError("Scoring failed: no posterior diagrams were provided.")

    recall_sum = 0.0
    precision_sum = 0.0
    f1_sum = 0.0

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    predicted_peak_count_sum = 0
    abs_peak_count_err_sum = 0

    # Per-cell hit counts across posterior diagrams
    if hit_counts is None:
        hit_counts = _compute_hit_counts(learner, grid, diagrams)

    for out in diagrams:
        ridge_x = out.peaks

        predicted_peak_count_sum += ridge_x.shape[0]
        abs_peak_count_err_sum += abs(ridge_x.shape[0] - n_true_peaks)
        

        hit_gdf = _assign_ridge_points_to_grid(
            learner=learner,
            grid=grid,
            ridge_x=ridge_x,
        )

        pred_idx = set(hit_gdf.index.tolist())

        tp = len(pred_idx & true_idx)
        fp = len(pred_idx - true_idx)
        fn = len(true_idx - pred_idx)

        recall = tp / n_true
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        recall_sum += recall
        precision_sum += precision
        f1_sum += f1

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn


    hit_prob = (hit_counts + alpha) / (n_scoring_samples + 2 * alpha)

    true_mask = np.zeros(len(grid), dtype=np.float64)
    true_mask[list(true_idx)] = 1.0

    # Log likelihood of the truth, assuming independent Bernoulli.
    log_likelihood = np.sum(
        true_mask * np.log(hit_prob) +
        (1.0 - true_mask) * np.log(1.0 - hit_prob)
    )

    # Log likelihood/# grid cells, which can be easier to compare across grid sizes
    mean_log_likelihood = log_likelihood / len(grid)

    new_x_raw = (
        learner.data_generator
        .untransform_inputs(learner.next_x.detach())
        .cpu()
        .numpy()
        .tolist()
    )

    metrics = {
        "iteration": int(learner._i),
        "acquisition": str(learner.config.acquisition_function),
        "sampling_method": learner.last_sampling_method,
        "n_points_sampled_iter": int(learner.next_x.shape[0]),
        "n_points_sampled_total": int(learner.n_points_sampled_total),
        "simulation_cost_iter": float(learner.last_simulation_cost),
        "simulation_cost_total": float(learner.simulation_cost_total),
        "wall_clock_time": float(getattr(learner, "_ti", learner._t0) - learner._t0),
        "total_cost_with_acquisition": float(
            learner.simulation_cost_total + (getattr(learner, "_ti", learner._t0) - learner._t0)
        ),
        "acquisition_raw_value": float(learner.last_acquisition_raw_value),
        "acquisition_score": float(learner.last_acquisition_score),
        "n_candidates_evaluated": int(learner.last_n_candidates_evaluated),
        "n_scoring_samples": n_scoring_samples,
        "n_grid_boxes": total_boxes,
        "n_true_boxes": n_true,
        "n_true_peaks": n_true_peaks,
        "mean_predicted_peak_count": predicted_peak_count_sum / n_scoring_samples,
        "mean_abs_peak_count_error": abs_peak_count_err_sum / n_scoring_samples,
        "mean_true_positives": tp_sum / n_scoring_samples,
        "mean_false_positives": fp_sum / n_scoring_samples,
        "mean_false_negatives": fn_sum / n_scoring_samples,
        "recall": recall_sum / n_scoring_samples,
        "precision": precision_sum / n_scoring_samples,
        "f1": f1_sum / n_scoring_samples,
        "log_likelihood": float(log_likelihood),
        "mean_log_likelihood": float(mean_log_likelihood),
    }

    metrics.update(neighbor_distance)
    metrics.update(model_hyperparameters)
    metrics.update({
        "sampled_x": json.dumps(new_x_raw),
    }) # sampled_x is the point chosen and sampled during this iteration, before scoring.
    metrics_savepath = (
        Path(learner.config.base_savepath)
        / "scoring"
        / "scores.csv"
    )
    metrics_savepath.parent.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame([metrics])

    if metrics_savepath.exists():
        df.to_csv(metrics_savepath, mode="a", header=False, index=False)
    else:
        df.to_csv(metrics_savepath, mode="w", header=True, index=False)

    _plot_metrics_by_iteration(learner, metrics_savepath)
    return metrics


def _compute_grid_probs(
    learner: "ActiveLearner",
    grid: gpd.GeoDataFrame,
    diagrams: list[DiagramOutput],
    alpha: float = 1.0,
    hit_counts: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute Laplace-smoothed probability that each grid cell contains
    at least one ridge point across sampled diagrams.

    Returns a copy of `grid` with added columns:
        - hit_count: number of diagrams that hit the cell
        - hit_prob:  Laplace-smoothed occupancy probability
    """
    n_samples = len(diagrams)
    if n_samples == 0:
        raise RuntimeError("No diagrams provided for probability computation.")

    # Initialize counts for each grid cell
    if hit_counts is None:
        hit_counts = _compute_hit_counts(learner, grid, diagrams)

    # Laplace-smoothed empirical hit probability per cell
    # p = (c + alpha) / (N + 2*alpha)
    hit_prob = (hit_counts + alpha) / (n_samples + 2 * alpha)

    # Return augmented grid
    grid_out = grid.copy()
    grid_out["hit_count"] = hit_counts
    grid_out["hit_prob"] = hit_prob

    return grid_out


def _compute_hit_counts(
    learner: "ActiveLearner",
    grid: gpd.GeoDataFrame,
    diagrams: list[DiagramOutput],
) -> np.ndarray:
    """Count how many sampled diagrams hit each grid cell."""
    hit_counts = np.zeros(len(grid), dtype=np.int64)

    for out in diagrams:
        ridge_x = out.peaks

        hit_gdf = _assign_ridge_points_to_grid(
            learner=learner,
            grid=grid,
            ridge_x=ridge_x,
        )

        pred_idx = set(hit_gdf.index.tolist())
        for idx in pred_idx:
            hit_counts[idx] += 1

    return hit_counts

def _ridge_x_to_physical(
    learner: "ActiveLearner",
    ridge_x: torch.Tensor,
) -> np.ndarray:
    """Convert ridge coordinates from normalized to physical coordinates."""
    if ridge_x.numel() == 0:
        return np.empty((0, 2), dtype=float)

    return (
        learner.data_generator
        .untransform_inputs(ridge_x.detach())
        .cpu()
        .numpy()
    )

def _same_slice_temperature_errors_K(
    true_ridge_x: np.ndarray,
    predicted_ridge_x: np.ndarray,
) -> list[float]:
    """Return predicted-to-true nearest-temperature errors by theta slice."""
    true_peaks = np.atleast_2d(
        np.asarray(true_ridge_x, dtype=float)
    )
    predicted_peaks = np.atleast_2d(
        np.asarray(predicted_ridge_x, dtype=float)
    )

    if true_peaks.size == 0 or predicted_peaks.size == 0:
        return []

    errors_K = []

    for predicted_peak in predicted_peaks:
        predicted_temp_K = predicted_peak[0]
        predicted_theta = predicted_peak[1]

        true_same_slice = true_peaks[
            np.isclose(true_peaks[:, 1], predicted_theta)
        ]

        if len(true_same_slice) == 0:
            continue

        errors_K.append(
            float(
                np.min(
                    np.abs(
                        true_same_slice[:, 0]
                        - predicted_temp_K
                    )
                )
            )
        )

    return errors_K


def _calculate_neighbor_distance(
    learner: "ActiveLearner",
    gt_ridge_x: torch.Tensor,
    posterior_diagrams: list[DiagramOutput],
) -> dict[str, float]:
    """Calculate mean same-slice absolute temperature error."""
    gt_peaks_raw = _ridge_x_to_physical(
        learner=learner,
        ridge_x=gt_ridge_x,
    )

    diagram_mean_errors_K = []

    for diagram in posterior_diagrams:
        predicted_peaks_raw = _ridge_x_to_physical(
            learner=learner,
            ridge_x=diagram.peaks,
        )

        diagram_errors_K = _same_slice_temperature_errors_K(
            true_ridge_x=gt_peaks_raw,
            predicted_ridge_x=predicted_peaks_raw,
        )

        if diagram_errors_K:
            diagram_mean_errors_K.append(
                float(np.mean(diagram_errors_K))
            )

    return {
        "same_slice_temp_error_mean_K": (
            float(np.mean(diagram_mean_errors_K))
            if diagram_mean_errors_K
            else float("nan")
        ),
    }


def _extract_model_hyperparameters(
    learner: "ActiveLearner",
) -> dict[str, float]:
    """Extract fitted GP hyperparameters for the scoring CSV."""
    lengthscale = (
        learner.model.base_kernel.lengthscale
        .detach()
        .cpu()
        .reshape(-1)
    )

    metrics = {
        "outputscale": float(
            learner.model.covar_module.outputscale
            .detach()
            .cpu()
            .item()
        ),
        "noise": float(
            learner.model.likelihood.noise
            .detach()
            .cpu()
            .item()
        ),
    }

    if lengthscale.numel() == 1:
        metrics["lengthscale"] = float(lengthscale.item())
    elif lengthscale.numel() == 2:
        metrics["lengthscale_temperature"] = float(
            lengthscale[0].item()
        )
        metrics["lengthscale_theta"] = float(
            lengthscale[1].item()
        )
    else:
        for dim, value in enumerate(lengthscale):
            metrics[f"lengthscale_dim_{dim}"] = float(
                value.item()
            )

    return metrics

def score_model_plot(learner: "ActiveLearner"):
    """Overall function to calculate and save scores, along with plots illustrating the scoring process."""
    # Build dense evaluation grid for the background heatmaps
    n_temp = 200
    temp_grid = torch.linspace(0.0, 1.0, n_temp)
    n_theta = learner.data_generator.theta_range
    theta_grid = torch.arange(0, n_theta + 1) / n_theta
    x, y = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    eval_grid = torch.stack([x.ravel(), y.ravel()]).T

    # Ground-truth field on dense grid
    y_gt = learner.ground_truth_data_generator.measure_uniform(eval_grid)
    y_gt_mean = y_gt.mean(-1)
    if learner.config.use_log_transform:
        y_gt_mean = y_gt_mean.log()

    # Model prediction on dense grid
    with eval_context(learner.model):
        y_pred = learner.model(eval_grid)
    y_pred_mean = y_pred.mean * learner._y_std + learner._y_mean

    # Build scoring grid in physical units. Reset index
    scoring_grid = _generate_grid(learner).reset_index(drop=True)

    # Get ground-truth diagram, determine true scoring boxes
    gt_diagrams = learner.diagram_generator.diagrams(
        1, data_generator=learner.ground_truth_data_generator
    )
    gt_item = gt_diagrams[0]
    gt_ridge_x = gt_item.peaks
    n_true_peaks = gt_ridge_x.shape[0]

    # print("GT ridge shape:", gt_ridge_x.shape)
    # print("GT ridge normalized:", gt_ridge_x)

    # if gt_ridge_x.numel() > 0:
    #     gt_raw = (
    #         learner.data_generator
    #         .untransform_inputs(gt_ridge_x)
    #         .detach()
    #         .cpu()
    #         .numpy()
    #     )
    #     print("GT ridge physical:\n", gt_raw)

    true_hits = _assign_ridge_points_to_grid(
        learner=learner,
        grid=scoring_grid,
        ridge_x=gt_ridge_x,
    )
    true_idx = set(true_hits.index.tolist())

    # Calculate and save score metrics
    posterior_diagrams = list(
        learner.diagram_generator.diagrams(learner.config.n_scoring_samples)
    )

    neighbor_distance = _calculate_neighbor_distance(
        learner=learner,
        gt_ridge_x=gt_ridge_x,
        posterior_diagrams=posterior_diagrams,
    )

    model_hyperparameters = _extract_model_hyperparameters(
        learner=learner,
    )

    hit_counts = _compute_hit_counts(
        learner=learner,
        grid=scoring_grid,
        diagrams=posterior_diagrams,
    )


    alpha = 1.0

    metrics = _calc_score(
        learner=learner,
        grid=scoring_grid,
        true_idx=true_idx,
        n_true_peaks=n_true_peaks,
        diagrams=posterior_diagrams,
        neighbor_distance=neighbor_distance,
        model_hyperparameters=model_hyperparameters,
        alpha=alpha,
        hit_counts=hit_counts,
    )

    grid_probs = _compute_grid_probs(
        learner=learner,
        grid=scoring_grid,
        diagrams=posterior_diagrams,
        alpha=alpha,
        hit_counts=hit_counts,
    )

    # Plot setup
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(18, 6), constrained_layout=True)

    # Reshape to (n_theta + 1, n_temp) for plotting
    z_gt = y_gt_mean.reshape((n_temp, n_theta + 1)).T.detach().cpu().numpy()
    z_pred = y_pred_mean.reshape((n_temp, n_theta + 1)).T.detach().cpu().numpy()

    # Build physical coordinate centers
    t_centers = torch.linspace(
        learner.data_generator.min_temp,
        learner.data_generator.max_temp,
        n_temp,
    ).cpu().numpy()

    theta_centers = torch.linspace(
        learner.data_generator.min_theta,
        learner.data_generator.max_theta,
        n_theta + 1,
    ).cpu().numpy()

    # Convert centers to edges for pcolormesh
    dt = t_centers[1] - t_centers[0]
    dtheta = theta_centers[1] - theta_centers[0]

    t_edges = np.concatenate([
        [t_centers[0] - 0.5 * dt],
        0.5 * (t_centers[:-1] + t_centers[1:]),
        [t_centers[-1] + 0.5 * dt],
    ])

    theta_edges = np.concatenate([
        [theta_centers[0] - 0.5 * dtheta],
        0.5 * (theta_centers[:-1] + theta_centers[1:]),
        [theta_centers[-1] + 0.5 * dtheta],
    ])

    ax[0, 0].pcolormesh(
        t_edges,
        theta_edges,
        z_gt,
        shading="flat",
        cmap="viridis",
        rasterized=True,
    )
    ax[0, 0].set_title("Ground truth")
    ax[0, 0].set_aspect("auto")

    ax[0, 1].pcolormesh(
        t_edges,
        theta_edges,
        z_pred,
        shading="flat",
        cmap="viridis",
        rasterized=True,
    )
    mean_error_K = metrics["same_slice_temp_error_mean_K"]

    mean_error_text = (
        f"{mean_error_K:.1f} K"
        if np.isfinite(mean_error_K)
        else "N/A"
    )

    ax[0, 1].set_title(
        "Sampled phase diagrams\n"
        f"Recall={metrics['recall']:.3f}, "
        f"Precision={metrics['precision']:.3f}, "
        f"F1={metrics['f1']:.3f}\n"
        f"Average predicted ridge point temperature error={mean_error_text}"
    )
    ax[0, 1].set_aspect("auto")

    grid_probs.plot(
        column="hit_prob",
        ax=ax[0, 2],
        cmap="cividis",
        linewidth=0,
        legend=False,
        vmin=0.0,
        vmax=1.0,
        zorder=1,
    )
    ax[0, 2].set_title(
        f"Ridge point probability"
    )
    ax[0, 2].set_aspect("auto")

    # Training data and newest selected point on middle panel
    if getattr(learner.config, "show_sampled_locations", True):
        xs = learner.data_generator.untransform_inputs(learner.train_x.detach())
        ax[0, 1].scatter(xs[:, 0], xs[:, 1], color="black", marker=".", zorder=2)

        new_xs = learner.data_generator.untransform_inputs(learner.next_x.detach())
        ax[0, 1].scatter(new_xs[:, 0], new_xs[:, 1], color="orange", s=45, zorder=10)

    for i in range(3):
        ax[0, i].set_xlabel("Temperature")
        ax[0, i].set_ylabel("Theta")

    # Plot MSTs + scoring boxes
    _plot_mst_diagrams_with_grid(
        learner=learner,
        diagrams=iter([gt_item]),
        ax=ax[0, 0],
        color="red",
        grid=scoring_grid,
        true_hits=true_hits,
    )

    n_plot_diagrams = min(8, len(posterior_diagrams))

    _plot_mst_diagrams_with_grid(
        learner=learner,
        diagrams=iter(posterior_diagrams[:n_plot_diagrams]),
        ax=ax[0, 1],
        color=None,
        grid=scoring_grid,
        true_hits=true_hits,
    )

    _plot_mst_diagrams_with_grid(
        learner=learner,
        diagrams=iter([gt_item]),
        ax=ax[0, 2],
        color="red",
        grid=scoring_grid,
        true_hits=None,
    )

    # Force GT lines to top (right panel only)
    for line in ax[0, 2].lines:
        line.set_zorder(10)

    for i in range(3):
        ax[0, i].set_xlim(
            learner.data_generator.min_temp,
            learner.data_generator.max_temp,
        )
        ax[0, i].set_ylim(
            learner.data_generator.min_theta-0.5,
            learner.data_generator.max_theta+0.5,
        )


    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="cividis")
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax[0, :], fraction=0.025, pad=0.02)
    cbar.set_label("Ridge point probability")

    # Save figure
    savepath = (
        Path(learner.config.base_savepath)
        / "scoring"
        / "scoring_plots"
        / f"iteration_{learner._i}.pdf"
    )
    savepath.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(savepath)
    plt.close()
