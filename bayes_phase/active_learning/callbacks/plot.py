import torch
import matplotlib.pyplot as plt
from pathlib import Path


def plot(learner: "ActiveLearner"):
    n_temp = 300
    temp_grid = torch.linspace(0.0, 1.0, n_temp)
    n_theta = learner.data_generator.max_theta
    theta_grid = torch.arange(0, n_theta + 1) / (n_theta + 1)
    x, y = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    grid = torch.stack([x.ravel(), y.ravel()]).T
    y_gt = learner.data_generator.measure_uniform(grid)
    y_gt_mean = y_gt.mean(-1)
    if learner.config.use_log_transform:
        y_gt_mean = y_gt_mean.log()
    y_pred = learner.model.likelihood(learner.model(grid))
    y_pred_mean = y_pred.mean * learner._y_std + learner._y_mean
    y_pred_std = y_pred.stddev * learner._y_std
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    extent = (
        learner.data_generator.min_temp,
        learner.data_generator.max_temp,
        learner.data_generator.min_theta,
        learner.data_generator.max_theta,
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
        Path(learner.config.base_savepath) / "plots" / f"iteration_{learner._i}.png"
    )
    ax[0, 1].set_aspect("auto")
    xs = learner.data_generator.untransform_inputs(learner.train_x.detach())
    ax[0, 1].scatter(xs[:, 0], xs[:, 1], color="black", s=100)
    new_xs = learner.data_generator.untransform_inputs(learner.next_x.detach())
    ax[0, 1].scatter(new_xs[:, 0], new_xs[:, 1], color="orange", s=100)
    savepath.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
