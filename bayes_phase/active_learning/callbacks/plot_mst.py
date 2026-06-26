import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from typing import Iterator, Optional
from scipy.interpolate import make_splprep, splev


from bayes_phase.acquisition.diagram_generation.base_diagram_generator import (
    DiagramOutput,
)
from bayes_phase.utils import eval_context


def _create_mst(x_norm: torch.Tensor, n_theta: int, learner: "ActiveLearner") -> torch.Tensor:
    """
    MST in normalized space, forbidding selected edges.
    These MSTs are strictly a visual convenience for plotting.
    """
    x = x_norm.detach().to(dtype=torch.float64, device="cpu")
    n = x.shape[0]
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long)

    # All pairs of indices. offset = 1 means, don't include self-pairs.
    i, j = torch.triu_indices(n, n, offset=1)

    # Assume normalized k/n_theta -> map back to integer k
    theta_idx = (x[:, 1] * n_theta).round().to(torch.int64)

    # No connections within a theta-value.
    keep = theta_idx[i] != theta_idx[j]
    i, j = i[keep], j[keep]

    # Forbid connections where either endpoint is in a forbidden low-temp region.
    x_raw = learner.data_generator.untransform_inputs(x)
    forbidden = _is_forbidden_region(x_raw, learner)

    # No connections if either points is in a forbidden low-temp region
    keep = ~(forbidden[i] | forbidden[j])
    i, j = i[keep], j[keep]

    # Limit connections across an excessive range of values.
    dist = (x[i] - x[j]).norm(dim=1)
    keep = dist <= learner.config.max_tree_conn_dist
    i, j = i[keep], j[keep]

    # Weight by Euclidean distance in (T_norm, theta_norm):
    w = (x[i] - x[j]).norm(dim=1).tolist()

    # Generate MST
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from((int(a), int(b), float(c)) for a, b, c in zip(i, j, w))

    mst = nx.minimum_spanning_tree(G)
    return torch.tensor(list(mst.edges()), dtype=torch.long)



def _plot_mst_diagrams(
    learner: "ActiveLearner",
    diagrams: Iterator[DiagramOutput],
    ax: Axes,
    color: Optional[str] = None,
):
    """Plots MSTs or MST-consistent smooth curves."""

    # Separate colors per sample
    cmap = plt.get_cmap("tab10")
    n_theta = learner.data_generator.theta_range

    for k, out in enumerate(diagrams):
        diagram = out.peaks

        if diagram.shape[0] < 2:
            continue

        c = color if color is not None else cmap(k % 10)

        xys = learner.data_generator.untransform_inputs(diagram)

        if not learner.config.connect_the_dots:
            ax.scatter(
                xys[:, 0],
                xys[:, 1],
                marker="x",
                s=40,
                color=c,
            )
            continue

        edges = _create_mst(diagram, n_theta=n_theta, learner=learner)

        if learner.config.show_points_tree_graphs:
            ax.scatter(xys[:, 0], xys[:, 1], s=5, color=c)

        if learner.config.tree_graph_type == "MST":
            for a, b in edges.tolist():
                ax.plot(xys[[a, b], 0], xys[[a, b], 1], color=c, lw=1)

        elif learner.config.tree_graph_type == "MSTCurve":
            branches = _identify_branches(ridge_points=xys, edges=edges)
            splines = _fit_mst_curves(branches)

            for spline in splines:
                for seg in spline["segments"]:
                    t = np.linspace(0.0, 1.0, 100) # sample points along a segment

                    # Hermite basis functions.
                    # The cubic curves are a function of these.
                    h00 = 2 * t**3 - 3 * t**2 + 1
                    h10 = t**3 - 2 * t**2 + t
                    h01 = -2 * t**3 + 3 * t**2
                    h11 = t**3 - t**2

                    # xy_smooth is the hermite
                    xy_smooth = ( 
                        h00[:, None] * seg["p0"] # Scaling
                        + h10[:, None] * seg["h"] * seg["m0"]
                        + h01[:, None] * seg["p1"]
                        + h11[:, None] * seg["h"] * seg["m1"]
                    )

                    ax.plot(
                        xy_smooth[:, 0],
                        xy_smooth[:, 1],
                        color=c,
                        linewidth=1.0,
                        alpha=0.8,
                    )

        else:
            raise ValueError("Valid tree plot types are MST and MSTCurve.")


def _identify_branches(
    ridge_points,
    edges,
):
    """Subdivide phase diagram boundaries into branches to make it possible to fit a spline for each branch."""

    n_points = len(ridge_points)

    # Outer loop: Loop through points.

    # Inner loop: Loop through edges and count the connections to each point.
    neighbors = {i: [] for i in range(n_points)}
    for a, b in edges.tolist():
        neighbors[a].append(b)
        neighbors[b].append(a)

    # Classify points in the following categories.
    # A point with one connection is a branch endpoint.
    endpoints = [i for i in range(n_points) if len(neighbors[i]) == 1]
    # A point with more than two connections is a split point (ex. the triple point).
    split_points = [i for i in range(n_points) if len(neighbors[i]) > 2]

    # Classify all connections as unexplored. Initialize branches object.
    unexplored = {frozenset((a, b)) for a, b in edges.tolist()} # Frozensest treats edges as unordered.
    branches = []

    def trace_branch(start_idx, next_idx):
        branch = [start_idx]
        prev_idx = start_idx
        curr_idx = next_idx

        while True:
            edge = frozenset((prev_idx, curr_idx))
            if edge not in unexplored:
                break
            unexplored.remove(edge)
            branch.append(curr_idx)

            if curr_idx in endpoints or curr_idx in split_points:
                break

            candidates = [j for j in neighbors[curr_idx] if j != prev_idx]
            if not candidates:
                break

            prev_idx, curr_idx = curr_idx, candidates[0]

        return branch

    # Outer loop: Over split points. We will walk outward from split points to identify branches.
    for split_idx in split_points:

        # Nested loop: Over the connections of the current split point.
        for next_idx in neighbors[split_idx]:
            edge = frozenset((split_idx, next_idx))
            if edge not in unexplored:
                continue

            # Inner loop: Along branch connections for this branch.
            # Find the next unexplored connection and set the point at its end as the new active point.
            # Repeat until we reach an endpoint.
            branch_indices = trace_branch(split_idx, next_idx)

            # Record the points along the branch.
            branches.append(
                {
                    "indices": branch_indices,
                    "points": ridge_points[branch_indices],
                }
            )

    # If there aren't any split points, begin with endpoints.
    if not split_points and endpoints:
        for start_idx in endpoints:

            # Only start a branch if this endpoint still has an unexplored connection.
            next_idx = None
            for candidate_idx in neighbors[start_idx]:
                edge = frozenset((start_idx, candidate_idx))
                if edge in unexplored:
                    next_idx = candidate_idx
                    break

            if next_idx is None:
                continue

            branch_indices = trace_branch(start_idx, next_idx)
            branches.append(
                {
                    "indices": branch_indices,
                    "points": ridge_points[branch_indices],
                }
            )

    # Return an object with the points along each branch, including the split point and endpoint.
    return branches
        

def _fit_mst_curves(branches):

    splines = []

    # Loop over branches
    for branch in branches:
        pts = branch["points"].detach().cpu().numpy()

        if len(pts) < 2:
            continue # Require at least 2 points to connect

        # Remove duplicate / near-duplicate consecutive points
        diffs = pts[1:] - pts[:-1]
        seg_lengths = np.sqrt((diffs**2).sum(axis=1)) # Euclidean distance between consecutive points
        keep = np.r_[True, seg_lengths > 1e-8]
        pts = pts[keep]

        if len(pts) < 2:
            continue

        # u is like arc length
        diffs = pts[1:] - pts[:-1]
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        u = np.r_[0.0, np.cumsum(seg_lengths)]

        if u[-1] <= 1e-12:
            continue

        # Estimate one tangent vector per point from neighboring points
        tangents = np.zeros_like(pts)

        if len(pts) == 2: # If 2 points -> Straight line
            tangents[0] = pts[1] - pts[0]
            tangents[1] = pts[1] - pts[0]
        else: # Use finite difference
            tangents[0] = (pts[1] - pts[0]) / max(u[1] - u[0], 1e-12)
            tangents[-1] = (pts[-1] - pts[-2]) / max(u[-1] - u[-2], 1e-12)

            for i in range(1, len(pts) - 1):
                du = max(u[i + 1] - u[i - 1], 1e-12)
                tangents[i] = (pts[i + 1] - pts[i - 1]) / du # Scale by arc length

        tangent_max = 1.5
        tangent_norms = np.sqrt((tangents**2).sum(axis=1))
        mask = tangent_norms > tangent_max
        tangents[mask] *= (tangent_max / tangent_norms[mask])[:, None]


        # Record Hermite segment data for each consecutive pair of points
        segments = []
        for i in range(len(pts) - 1):
            h = u[i + 1] - u[i]
            if h <= 1e-12:
                continue

            segments.append(
                {
                    "p0": pts[i],
                    "p1": pts[i + 1],
                    "m0": tangents[i],
                    "m1": tangents[i + 1],
                    "h": h,
                }
            )

        if not segments:
            continue

        splines.append(
            {
                "indices": branch["indices"],
                "points": pts,
                "segments": segments,
            }
        )

    return splines



def plot_mst(
    learner: "ActiveLearner"):

    n_temp = 200
    temp_grid = torch.linspace(0.0, 1.0, n_temp)
    n_theta = learner.data_generator.theta_range
    theta_grid = torch.arange(0, n_theta + 1) / (n_theta)
    x, y = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    grid = torch.stack([x.ravel(), y.ravel()]).T
    y_gt = learner.data_generator.measure_uniform(grid)
    y_gt_mean = y_gt.mean(-1)
    if learner.config.use_log_transform:
        y_gt_mean = y_gt_mean.log()
    with eval_context(learner.model):
        y_pred = learner.model(grid)
    y_pred_mean = y_pred.mean * learner._y_std + learner._y_mean
    y_pred_std = y_pred.stddev * learner._y_std
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    extent = (
        learner.data_generator.min_temp,
        learner.data_generator.max_temp,
        learner.data_generator.min_theta - 0.5,
        learner.data_generator.max_theta + 0.5,
    )
    ax[0, 0].imshow(
        y_gt_mean.reshape((n_temp, n_theta + 1)).T,
        interpolation="nearest",
        extent=extent,
        origin="lower",
    )
    ax[0, 0].set_aspect("auto")
    ax[0, 1].imshow(
        y_pred_mean.reshape((n_temp, n_theta + 1)).detach().T,
        interpolation="nearest",
        extent=extent,
        origin="lower",
    )
    savepath = (
        Path(learner.config.base_savepath) / "mst_plots" / f"iteration_{learner._i}.png"
    )
    ax[0, 1].set_aspect("auto")
    xs = learner.data_generator.untransform_inputs(learner.train_x.detach())
    ax[0, 1].scatter(xs[:, 0], xs[:, 1], color="black", marker=".")
    new_xs = learner.data_generator.untransform_inputs(learner.next_x.detach())
    ax[0, 1].scatter(new_xs[:, 0], new_xs[:, 1], color="orange", s=100)
    for i in range(2):
        ax[0, i].set_xlabel("Temperature")
        ax[0, i].set_ylabel("Theta")

    ax[0,0].set_title("Ground truth phase diagram boundary")
    ax[0,1].set_title("Sampled phase diagrams and training points")

    savepath.parent.mkdir(exist_ok=True, parents=True)

    # Plot phase diagram boundaries
    gt_diagrams = learner.diagram_generator.diagrams(1, data_generator=learner.data_generator)
    _plot_mst_diagrams(learner, gt_diagrams, ax[0, 0], color="red")  

    diagrams = learner.diagram_generator.diagrams(10)
    _plot_mst_diagrams(learner, diagrams, ax[0, 1])

    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
    plt.close()



def _filter_points_above_slice_median(
    peaks: torch.Tensor,
    sample: torch.Tensor,
    learner: "ActiveLearner",
) -> torch.Tensor:
    """
    Helper that allows us to draw connections only between points above the median of a slice. 
    Doesn't affect active learning computations, just clarity of plots.
    """

    if peaks.numel() == 0:
        return peaks

    n_theta = learner.data_generator.theta_range
    n_temp = sample.shape[0]

    keep = []

    for p in peaks:
        temp_u = p[0]
        theta_u = p[1]

        theta_idx = int(round(float(theta_u * n_theta)))
        temp_idx = int(round(float(temp_u * (n_temp - 1))))

        theta_idx = max(0, min(theta_idx, sample.shape[1] - 1))
        temp_idx = max(0, min(temp_idx, sample.shape[0] - 1))

        slice_y = sample[:, theta_idx]
        slice_median = torch.median(slice_y)

        keep.append(sample[temp_idx, theta_idx] > slice_median)

    keep = torch.tensor(keep, dtype=torch.bool, device=peaks.device)
    return peaks[keep]


def _is_forbidden_region(x_raw: torch.Tensor, learner: "ActiveLearner") -> torch.Tensor:
    """
    True for points in regions where MST connections should be forbidden.

    Config format:
        forbidden_tree_regions:
          - theta_min: 2
            theta_max: 6
            temp_max: 200
    """
    regions = getattr(learner.config, "forbidden_tree_regions", [])

    temp = x_raw[:, 0]
    theta = x_raw[:, 1].round().to(torch.int64)

    forbidden = torch.zeros(x_raw.shape[0], dtype=torch.bool, device=x_raw.device)

    for region in regions:
        if region["theta_min"] > region["theta_max"]:
            raise ValueError(
                f"Invalid forbidden_tree_region: {region}"
            )
        forbidden |= (
            (theta >= int(region["theta_min"]))
            & (theta <= int(region["theta_max"]))
            & (temp < float(region["temp_max"]))
        )

    return forbidden
