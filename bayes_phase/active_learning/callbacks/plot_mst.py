import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from typing import Iterator, Optional

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


def _plot_mst_diagrams(
    learner: "ActiveLearner",
    diagrams: Iterator[DiagramOutput],
    ax: Axes,
    color: Optional[str] = None,
):
    # Separate colors per sample
    cmap = plt.get_cmap("tab10")
    n_theta = learner.data_generator.theta_range

    for k, (diagram, _sample) in enumerate(diagrams):
        c = color if color is not None else cmap(k % 10)

        edges = _create_mst(diagram, n_theta=n_theta, learner = learner)

        xys = learner.data_generator.untransform_inputs(diagram)  # plot in physical units
        # Optional: dots at ridge points
        ax.scatter(xys[:, 0], xys[:, 1], s=11, color=c)

        for a, b in edges.tolist():
            ax.plot(xys[[a, b], 0], xys[[a, b], 1], color=c, lw=1)


def plot_mst(learner: "ActiveLearner"):
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

    savepath.parent.mkdir(exist_ok=True, parents=True)

    # Plot phase diagram boundaries
    gt_diagrams = learner.diagram_generator.diagrams(3, data_generator=learner.data_generator)
    _plot_mst_diagrams(learner, gt_diagrams, ax[0, 0], color="red")  

    diagrams = learner.diagram_generator.diagrams(10)
    _plot_mst_diagrams(learner, diagrams, ax[0, 1])

    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
    plt.close()
