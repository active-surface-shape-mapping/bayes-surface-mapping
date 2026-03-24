import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import pandas as pd
import shapely as shp
import geopandas as gpd

from bayes_phase.acquisition.diagram_generation.base_diagram_generator import (
    DiagramOutput,
)
from bayes_phase.utils import eval_context



def _create_mst(x_norm: torch.Tensor, n_theta: int, learner: "ActiveLearner") -> torch.Tensor:
    """MST in normalized space, forbidding edges within same discrete theta level."""
    x = x_norm.detach().to(dtype=torch.float64, device="cpu")
    n = x.shape[0]
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long)

    # All pairs of indices. offset = 1 means, don't include self-pairs.
    i, j = torch.triu_indices(n, n, offset=1)

    # Assume normalized k/n_theta -> map back to integer k
    theta_idx = (x[:, 1] * n_theta).round().to(torch.int64)

    # No connections within a theta-value
    keep = theta_idx[i] != theta_idx[j] 
    i, j = i[keep], j[keep]

    # Limit connections across an excessive range of temperatures, for visual clarity
    dT = (x[i, 0] - x[j, 0]).abs()
    keep2 = dT <= learner.config.max_T_span
    i, j = i[keep2], j[keep2]

    # Weight by full Euclidean distance in (T_norm, theta_norm):
    w = (x[i] - x[j]).norm(dim=1).tolist()

    # Generate MST
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from((int(a), int(b), float(c)) for a, b, c in zip(i, j, w))
    mst = nx.minimum_spanning_tree(G) 
    return torch.tensor(list(mst.edges()), dtype=torch.long)

def plot_mst_diagrams(
    learner: "ActiveLearner",
    diagrams: Iterator[DiagramOutput],
    ax: Axes,
    color: Optional[str] = None,
    grid: Optional[gpd.GeoDataFrame] = None,
    true_hits: Optional[gpd.GeoDataFrame] = None,
):
    """Create side-by-side plots of GT phase diagram and sampled phase diagrams."""
    if grid is None:
        grid = _generate_grid(learner)

    grid.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=3, aspect=None)

    if true_hits is not None and len(true_hits) > 0:
        true_hits.boundary.plot(ax=ax, color="red", linewidth=1.2, zorder=5, aspect=None)

    ax.set_aspect("auto")

    n_theta = learner.data_generator.theta_range
    cmap = plt.get_cmap("tab20")

    for idx, (ridge_x, _, _, _, _sample) in enumerate(diagrams):
        if ridge_x.numel() == 0:
            continue

        edges = _create_mst(ridge_x, n_theta=n_theta, learner=learner)
        xys = learner.data_generator.untransform_inputs(ridge_x.detach()).cpu()

        if color is None:
            line_color = cmap(idx % cmap.N)
            line_alpha = 0.85
            line_width = 1.0
            point_size = 10
        else:
            line_color = color
            line_alpha = 1.0
            line_width = 1.2
            point_size = 11

        ax.scatter(xys[:, 0], xys[:, 1], s=point_size, color=line_color, zorder=4)

        for a, b in edges.tolist():
            ax.plot(
                xys[[a, b], 0],
                xys[[a, b], 1],
                color=line_color,
                lw=line_width,
                alpha=line_alpha,
                zorder=4,
            )


def _generate_grid(
    learner: "ActiveLearner",
) -> gpd.GeoDataFrame:
    """Build rectangular scoring boxes."""
    cfg = learner.config
    dg = learner.data_generator

    t_min = dg.min_temp
    t_max = dg.max_temp

    dt = cfg.scoring_grid_T_range

    # Integer theta levels to score
    theta_values = list(range(1, 17))

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


def _intersect_ridge_points_grid(
    learner: "ActiveLearner",
    grid: gpd.GeoDataFrame,
    ridge_x: torch.Tensor,
) -> gpd.GeoDataFrame:
    """Return grid cells containing ridge points at integer theta values."""
    if ridge_x.numel() == 0:
        out = grid.iloc[0:0].copy()
        out["contains_ridge_point"] = pd.Series(dtype=bool)
        return out

    # Physical coordinates
    xys = learner.data_generator.untransform_inputs(ridge_x.detach()).cpu().numpy()

    pts = pd.DataFrame({
        "Temperature": xys[:, 0],
        "Theta": xys[:, 1],
    })

    point_gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["Temperature"], pts["Theta"]),
        crs=grid.crs,
    )

    hits = gpd.sjoin(grid, point_gdf, how="inner", predicate="contains")
    hits = hits[~hits.index.duplicated(keep="first")].copy()
    hits["contains_ridge_point"] = True
    return hits


def _plot_metrics_over_iterations(
    learner: "ActiveLearner",
    csv_path: Path) -> None:
    """Plot precision, recall, and F1 scores by iteration."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("iteration")

    colors = plt.get_cmap("Set1").colors

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["iteration"], df["recall"], linewidth = 1, marker="d", markersize = 3, label="Recall", color = colors[0])
    ax.plot(df["iteration"], df["precision"], linewidth = 1, marker="d", markersize = 3, label="Precision", color = colors[1])
    ax.plot(df["iteration"], df["f1"], linewidth = 1, marker="d", markersize = 3, label="F1 score", color = colors[2])


    if learner.config.acquisition_function == "random":
        literate_acquisition_function = "Random sampling"
    elif learner.config.acquisition_function == "mutual_information":
        literate_acquisition_function = "Mutual information"
    elif learner.config.acquisition_function == "eig":
        literate_acquisition_function = "EIG"
    elif learner.config.acquisition_function == "sobol":
        literate_acquisition_function = "Sobol sampling"
    else:
        literate_acquisition_function = learner.config.acquisition_function

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title(f"Performance over iterations: {literate_acquisition_function}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(csv_path.parent / "score_metrics.png", dpi=150)
    plt.close(fig)

def _calc_score(
    learner: "ActiveLearner",
    grid: gpd.GeoDataFrame,
    true_idx: set[int],
    diagrams: list[DiagramOutput],
) -> dict[str, float | int]:
    """Calculate and save scores by iteration."""
    n_scoring_samples = len(diagrams)
    total_boxes = int(len(grid))
    n_true = int(len(true_idx))

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

    for ridge_x, _, _, _, _sample in diagrams:
        hit_gdf = _intersect_ridge_points_grid(
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

    metrics = {
        "iteration": int(learner._i),
        "acquisition": str(learner.config.acquisition_function),
        "n_scoring_samples": n_scoring_samples,
        "n_grid_boxes": total_boxes,
        "n_true_boxes": n_true,
        "mean_true_positives": tp_sum / n_scoring_samples,
        "mean_false_positives": fp_sum / n_scoring_samples,
        "mean_false_negatives": fn_sum / n_scoring_samples,
        "recall": recall_sum / n_scoring_samples,
        "precision": precision_sum / n_scoring_samples,
        "f1": f1_sum / n_scoring_samples,
    }

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

    _plot_metrics_over_iterations(learner, metrics_savepath)
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
    y_gt = learner.data_generator.measure_uniform(eval_grid)
    y_gt_mean = y_gt.mean(-1)
    if learner.config.use_log_transform:
        y_gt_mean = y_gt_mean.log()

    # Model prediction on dense grid
    with eval_context(learner.model):
        y_pred = learner.model(eval_grid)
    y_pred_mean = y_pred.mean * learner._y_std + learner._y_mean

    # Build scoring grid in physical units
    scoring_grid = _generate_grid(learner)

    # Get ground-truth diagram, determine true scoring boxes
    gt_diagrams = learner.diagram_generator.diagrams(
        1, data_generator=learner.data_generator
    )
    gt_item = gt_diagrams[0]
    if gt_item is None:
        raise RuntimeError("Ground-truth diagram generator returned no diagrams.")
    gt_ridge_x, _, _, _, gt_sample = gt_item

    true_hits = _intersect_ridge_points_grid(
        learner=learner,
        grid=scoring_grid,
        ridge_x=gt_ridge_x,
    )
    true_idx = set(true_hits.index.tolist())

    # Calculate and save score metrics
    posterior_diagrams = list(
        learner.diagram_generator.diagrams(learner.config.n_scoring_samples)
    )

    metrics = _calc_score(
        learner=learner,
        grid=scoring_grid,
        true_idx=true_idx,
        diagrams=posterior_diagrams,
    )

    # Plot setup
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))

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
    )
    ax[0, 0].set_title("Ground truth")
    ax[0, 0].set_aspect("auto")

    ax[0, 1].pcolormesh(
        t_edges,
        theta_edges,
        z_pred,
        shading="flat",
        cmap="viridis",
    )
    ax[0, 1].set_title(
        f"Sampled phase diagrams\n"
        f"Recall={metrics['recall']:.3f}, "
        f"Precision={metrics['precision']:.3f}, "
        f"F1={metrics['f1']:.3f}"
    )
    ax[0, 1].set_aspect("auto")

    # Training data and newest selected point on right panel
    xs = learner.data_generator.untransform_inputs(learner.train_x.detach())
    ax[0, 1].scatter(xs[:, 0], xs[:, 1], color="black", marker=".", zorder=6)

    new_xs = learner.data_generator.untransform_inputs(learner.next_x.detach())
    ax[0, 1].scatter(new_xs[:, 0], new_xs[:, 1], color="orange", s=100, zorder=7)

    for i in range(2):
        ax[0, i].set_xlabel("Temperature")
        ax[0, i].set_ylabel("Theta")

    # Plot MSTs + scoring boxes
    plot_mst_diagrams(
        learner=learner,
        diagrams=iter([(gt_ridge_x, _, _, _, gt_sample)]),
        ax=ax[0, 0],
        color="red",
        grid=scoring_grid,
        true_hits=true_hits,
    )

    plot_mst_diagrams(
        learner=learner,
        diagrams=iter(posterior_diagrams),
        ax=ax[0, 1],
        color=None,
        grid=scoring_grid,
        true_hits=true_hits,
    )

    for i in range(2):
        ax[0, i].set_xlim(
            learner.data_generator.min_temp,
            learner.data_generator.max_temp,
        )
        ax[0, i].set_ylim(
            learner.data_generator.min_theta-0.5,
            learner.data_generator.max_theta+0.5,
        )

    # Save figure
    savepath = (
        Path(learner.config.base_savepath)
        / "scoring"
        / "scoring_plots"
        / f"iteration_{learner._i}.png"
    )
    savepath.parent.mkdir(exist_ok=True, parents=True)

    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
    plt.close()