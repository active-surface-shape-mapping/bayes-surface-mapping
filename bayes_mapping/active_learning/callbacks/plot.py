import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from typing import Iterator, Optional

from bayes_mapping.acquisition.diagram_generation.base_diagram_generator import (
    DiagramOutput,
)
from bayes_mapping.utils import eval_context


def plot_diagrams(
    learner: "ActiveLearner",
    diagrams: Iterator[DiagramOutput],
    ax: Axes,
    color: Optional[str] = None,
):
    for diagram, _, _, _, sample in diagrams:
        # x = (
        #     diagram[:, 0] * learner.data_generator.temp_range
        #     + learner.data_generator.min_temp
        # )
        # y = (
        #     diagram[:, 1] * learner.data_generator.theta_range
        #     + learner.data_generator.min_theta
        # )
        xys = learner.data_generator.untransform_inputs(diagram)
        ax.scatter(xys[:, 0], xys[:, 1], marker="|", s=200, color=color)


def plot(learner: "ActiveLearner"):
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
        Path(learner.config.base_savepath) / "plots" / f"iteration_{learner._i}.pdf"
    )
    ax[0, 1].set_aspect("auto")
    xs = learner.data_generator.untransform_inputs(learner.train_x.detach())
    ax[0, 1].scatter(xs[:, 0], xs[:, 1], color="black", marker=".")
    new_xs = learner.data_generator.untransform_inputs(learner.next_x.detach())
    ax[0, 1].scatter(new_xs[:, 0], new_xs[:, 1], color="orange", s=100)
    for i in range(2):
        ax[0, i].set_xlabel("Temperature (°C)")
        ax[0, i].set_ylabel("Theta")

    savepath.parent.mkdir(exist_ok=True, parents=True)

    # Plot phase diagram boundaries
    gt_diagrams = learner.diagram_generator.diagrams(
        10, data_generator=learner.data_generator, debug_plot=False
    )
    plot_diagrams(learner, gt_diagrams, ax[0, 0], color="red")
    diagrams = learner.diagram_generator.diagrams(10, debug_plot=False)
    plot_diagrams(learner, diagrams, ax[0, 1])

    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
    plt.close()
