import torch
import math
from jaxtyping import Float
from typing import Any, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from pathlib import Path
import gpytorch
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner

from bayes_phase.utils import eval_context
from .conditioning import EPConditioner, ESSConditioner, BaseConditioner


class MutualInformation:
    """
    Class to to compute mutual information between the current model's posterior and the model's posterior over ridges.
    I(X;Y) = H(Y) - H(Y|X)
    Where H(Y) is the model's posterior at a point.
    H(Y|X) the model's posterior conditioned on a set of ridges. To be estimated by Monte Carlo, E[H(Y|X)].

    """

    def __init__(self, learner: "ActiveLearner"):
        self.learner = learner
        self.conditioner = self._init_conditioner()
        if self.learner.config.mi_n_mc_samples is None:
            raise ValueError("`mi_n_mc_samples` must be set in config.")

    def forward(
        self, X: Float[torch.Tensor, "n d"], plot_every: Optional[int] = 50
    ) -> Float[torch.Tensor, "m "]:
        n_mc_samples = int(self.learner.config.mi_n_mc_samples)

        # ---- H(Y): baseline entropy at each candidate (univariate Gaussian) ----
        with eval_context(self.learner.model):
            predictions = self.learner.model(X[:, torch.newaxis, :])
        var_base = predictions.variance.ravel()

        # H(N(., var)) = 0.5 * log(2*pi*e*var)
        H_base = 0.5 * torch.log(var_base) + 0.5 * math.log(
            2.0 * math.pi * math.e
        )  # (m,)

        # ---- Sample functions to induce ridge hypotheses ----
        mc_samples = self.learner.diagram_generator.diagrams(n_mc_samples, debug_plot=True)

        # ---- Monte Carlo estimate of E[ H(Y | ridge) ] ----
        H_cond_sum = torch.zeros_like(H_base)
        n_used = 0

        # Monte Carlo sampling
        for i, diagram_output in tqdm(
            enumerate(mc_samples), desc="MI Monte-Carlo Sampling", position=1
        ):
            try:
                var_cond, info = self.conditioner.condition(
                    x_test=X, diagram_output=diagram_output, learner=self.learner
                )
                if (
                    i == 0
                    and (plot_every is not None)
                    and (
                        (self.learner._i == 0)
                        or (self.learner._i % plot_every == 0)
                    )
                ):
                    _show_conditioning_3d(
                        learner=self.learner,
                        diagram_output=diagram_output,
                        conditioner=self.conditioner,
                        slice_thetas=(2, 14),
                    )
            except StopIteration:
                continue

            # Sigma_cond corresponds to joint over [f(X), f'(X increasing), f'(decreasing), f''(ridge)]; extract f(X)
            H_cond = 0.5 * torch.log(var_cond) + 0.5 * math.log(2.0 * math.pi * math.e)

            H_cond_sum += H_cond
            n_used += 1
            if (plot_every is not None) and (
                (self.learner._i == 0) or (self.learner._i % plot_every == 0)
            ):
                try:
                    self.plot(
                        X,
                        info,
                        diagram_output.peaks,
                        diagram_output.valleys,
                        predictions,
                        suffix=f"_{i}",
                    )
                except Exception as e:
                    raise e
                    # print(f"Failed to plot MI conditioning plots: {e}")
                    # plt.close()

        if n_used == 0:
            raise RuntimeError(
                "MutualInformation: all MC samples failed (no ridges or EP failures)."
            )

        H_cond_mean = H_cond_sum / float(n_used)

        # ---- MI per candidate ----
        scores = H_base - H_cond_mean

        # Plot H(Y), H(Y|X), and MI.
        if (plot_every is not None) and (
            (self.learner._i == 0) or (self.learner._i % plot_every == 0)
        ):
            _plot_mi_terms_2d_3d(
                learner=self.learner,
                X=X,
                H_base=H_base,
                H_cond_mean=H_cond_mean,
                scores=scores,
                make_2d_graphs=True,
                make_3d_graphs=True,
            )

        return scores

    def __call__(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(X)

    def plot(
        self,
        X: Float[torch.Tensor, "n d"],
        info: dict[str, Any],
        ridge_x: Optional[Float[torch.Tensor, "m1 d"]],
        valley_x: Optional[Float[torch.Tensor, "m2 d"]],
        prior_preds: gpytorch.distributions.MultivariateNormal,
        n_samples: int = 30,
        suffix: str = "",
        jitter=1e-6,
    ):
        with torch.no_grad():
            # prior_dist = torch.distributions.MultivariateNormal(
            #     prior_preds.mean,
            #     prior_preds.covariance_matrix + jitter * torch.eye(sigma.shape[0]),
            # )  # Before conditioning
            n_theta = self.learner.data_generator.theta_range + 1
            posterior_distrib = None
            posterior_samples = None
            if ("posterior_mean" in info) and ("posterior_covariance" in info):
                mu = info["posterior_mean"]
                sigma = info["posterior_covariance"]
                mean = mu.detach()
                stdv = sigma.detach().diagonal().sqrt()
                covariance_matrix = sigma + jitter * torch.eye(sigma.shape[0])
                try:
                    posterior_distrib = torch.distributions.MultivariateNormal(
                        loc=mean, covariance_matrix=covariance_matrix
                    )
                except Exception as e:
                    print(f"Error building MVN while plotting: {e}")
            if info.get("posterior_samples") is not None:
                posterior_samples = info["posterior_samples"]
                n_test = X.shape[0]
                mean = (
                    posterior_samples[:, :, :n_test].mean(dim=(0, 1))
                    + prior_preds.mean.ravel()
                )
                stdv = posterior_samples[:, :, :n_test].std(dim=(0, 1))
            fig, axs = plt.subplots(
                n_theta, 1, sharex=True, sharey=False, figsize=(8, 32)
            )
            for i in range(n_theta):
                this_theta = torch.tensor(float(i) / (n_theta - 1))
                idxs = X[:, 1].isclose(this_theta)
                grid = X[idxs, 0]
                sorted_idxs = grid.argsort()
                grid = grid[sorted_idxs]
                m = mean[idxs][sorted_idxs]
                sd = stdv[idxs][sorted_idxs]
                pm = prior_preds.mean[idxs][sorted_idxs]
                psd = prior_preds.variance.sqrt()[idxs][sorted_idxs]
                if posterior_distrib is not None:
                    # Plot samples from posterior distribution
                    for j in range(n_samples):
                        sample = posterior_distrib.sample()[idxs][sorted_idxs]
                        axs[n_theta - i - 1].plot(grid, sample, alpha=0.2)
                if posterior_samples is not None:
                    # Plot samples from ESS MCMC Chain
                    n_chains = posterior_samples.shape[0]
                    n_samples = posterior_samples.shape[1]
                    thin_samples_by = (n_chains * n_samples) // 200
                    for j in range(0, n_samples, thin_samples_by):
                        for k, color in zip(range(n_chains), TABLEAU_COLORS):
                            axs[n_theta - i - 1].plot(
                                grid,
                                posterior_samples[k, j, : X.shape[0]][idxs][sorted_idxs]
                                + prior_preds.mean.ravel()[idxs][sorted_idxs],
                                color=color,
                                alpha=0.05,
                            )
                axs[n_theta - i - 1].plot(grid, pm, color="black")
                axs[n_theta - i - 1].plot(
                    grid,
                    pm - 1.96 * psd,
                    color="black",
                    linestyle="dashed",
                )
                axs[n_theta - i - 1].plot(
                    grid,
                    pm + 1.96 * psd,
                    color="black",
                    linestyle="dashed",
                )
                axs[n_theta - i - 1].plot(grid, m, color="tab:blue")
                axs[n_theta - i - 1].plot(
                    grid,
                    m - 1.96 * sd,
                    color="tab:blue",
                    linestyle="dashed",
                )
                axs[n_theta - i - 1].plot(
                    grid,
                    m + 1.96 * sd,
                    color="tab:blue",
                    linestyle="dashed",
                )
                if ridge_x is not None:
                    # Plot conditioning ridge points
                    ridge_idxs = ridge_x[:, 1].isclose(this_theta)
                    these_ridges = ridge_x[ridge_idxs, 0]
                    sorted_ridge_idxs = these_ridges.argsort()
                    these_ridges = these_ridges[sorted_ridge_idxs]
                    for ridge in these_ridges:
                        axs[n_theta - i - 1].axvline(
                            ridge,
                            color="red",
                        )
                if valley_x is not None:
                    # Plot conditioning valley points
                    valley_idxs = valley_x[:, 1].isclose(this_theta)
                    these_valleys = valley_x[valley_idxs, 0]
                    sorted_valley_idxs = these_valleys.argsort()
                    these_valleys = these_valleys[sorted_valley_idxs]
                    for valley in these_valleys:
                        axs[n_theta - i - 1].axvline(
                            valley,
                            color="blue",
                        )
                # Plot training points
                train_x = self.learner.train_x
                train_y = self.learner.train_y
                idxs = train_x[:, 1].isclose(torch.tensor(float(i) / (n_theta - 1)))
                axs[n_theta - i - 1].scatter(train_x[idxs, 0], train_y[idxs, 0], s=50)
                axs[n_theta - i - 1].set_ylabel(f"θ={i + 1}")
            axs[-1].set_xlabel("Temperature")
            plt.tight_layout()
            savepath = (
                Path(self.learner.config.base_savepath)
                / "mi_conditioning_plots"
                / f"iteration_{self.learner._i}"
                / f"sample{suffix}.png"
            )
            savepath.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(savepath, dpi=100)
            plt.close()

    def _init_conditioner(self) -> BaseConditioner:
        if self.learner.config.mi_conditioning_strategy is None:
            raise ValueError(
                "`config.mi_conditioning` strategy must be set to use MI acquisition function."
            )
        elif self.learner.config.mi_conditioning_strategy == "EP":
            return EPConditioner()
        elif self.learner.config.mi_conditioning_strategy == "ESS":
            return ESSConditioner()
        else:
            raise ValueError(
                f"Unrecognized conditioning strategy {self.learner.config.mi_conditioning_strategy}"
            )




# Create and save plots of H(Y), E[H(Y|X)], and MI on the candidate set.
@torch.no_grad()
def _plot_mi_terms_2d_3d(
    learner: "ActiveLearner",
    X: torch.Tensor,
    H_base: torch.Tensor,
    H_cond_mean: torch.Tensor,
    scores: torch.Tensor,
    make_2d_graphs: bool = True,
    make_3d_graphs: bool = True,
):
    """Plot H(Y), E[H(Y|X)], and MI across the candidate set in 3D HTML and 2D PNG."""
    cand_u = X.detach()
    cand_raw = learner.data_generator.untransform_inputs(cand_u)
    T_raw = cand_raw[:, 0]
    theta_raw = cand_raw[:, 1]

    with torch.no_grad():
        y_pred = learner.model.likelihood(learner.model(cand_u))
    z_mean = (y_pred.mean * learner._y_std + learner._y_mean).reshape(-1)

    H_base = H_base.detach().reshape(-1)
    H_cond_mean = H_cond_mean.detach().reshape(-1)
    scores = scores.detach().reshape(-1)

    x_np = T_raw.detach().cpu().numpy()
    y_np = theta_raw.detach().cpu().numpy()
    z_np = z_mean.detach().cpu().numpy()

    h_base_np = H_base.detach().cpu().numpy()
    h_cond_np = H_cond_mean.detach().cpu().numpy()
    mi_np = scores.detach().cpu().numpy()

    method_label = "Mutual information"

    # Sampled points
    n_init = int(learner.config.init_n_points)
    xs_all = learner.data_generator.untransform_inputs(learner.train_x.detach())
    z_all = learner.train_y_orig.detach().reshape(-1)

    camera = dict(eye=dict(x=-1.6, y=-1.8, z=0.8))
    x_range = [
        float(learner.data_generator.min_temp),
        float(learner.data_generator.max_temp),
    ]
    y_range = [
        float(learner.data_generator.min_theta),
        float(learner.data_generator.max_theta),
    ]

    zmin = float(z_mean.min().detach().cpu().item())
    zmax = float(z_mean.max().detach().cpu().item())
    zpad = 0.05 * (zmax - zmin + 1e-12)
    z_range = [zmin - zpad, zmax + zpad]

    # Shared color limits for H(Y) and E[H(Y|X)], for visual comparability
    hmin = float(min(np.nanmin(h_base_np), np.nanmin(h_cond_np)))
    hmax = float(max(np.nanmax(h_base_np), np.nanmax(h_cond_np)))

    customdata = np.column_stack([h_base_np, h_cond_np, mi_np])

    def _scatter3d(
        term_np: np.ndarray,
        title: str,
        colorbar_x: float,
        colorbar_y: float,
        colorbar_len: float,
        cmin: float | None = None,
        cmax: float | None = None,
    ):
        return go.Scatter3d(
            x=x_np,
            y=y_np,
            z=z_np,
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.75,
                color=term_np,
                colorscale="Plasma",
                showscale=True,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=title,
                    x=colorbar_x,
                    y=colorbar_y,
                    len=colorbar_len,
                ),
            ),
            customdata=customdata,
            showlegend=False,
            hovertemplate=(
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                "Posterior mean: %{z}<br>"
                "H(Y): %{customdata[0]}<br>"
                "E[H(Y|X)]: %{customdata[1]}<br>"
                "MI: %{customdata[2]}<extra></extra>"
            ),
        )

    if make_3d_graphs:
        out_dir = Path(learner.config.base_savepath) / "MI_term_plots_3d"
        out_dir.mkdir(exist_ok=True, parents=True)
        savepath_html = out_dir / f"iteration_{learner._i:03d}.html"
        y_range_3d = [y_range[0], 17.0]

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene"}, {"type": "scene"}],
                [{"type": "scene"}, None],
            ],
            subplot_titles=("H(Y)", "E[H(Y|X)]", "MI", ""),
            horizontal_spacing=0.03,
            vertical_spacing=0.08,
        )

        # H(Y) scatterplot
        fig.add_trace(
            _scatter3d(
                h_base_np,
                "H(Y)",
                colorbar_x=0.46,
                colorbar_y=0.79,
                colorbar_len=0.34,
                cmin=hmin,
                cmax=hmax,
            ),
            row=1,
            col=1,
        )
        # E[H(Y|X)] scatterplot
        fig.add_trace(
            _scatter3d(
                h_cond_np,
                "E[H(Y|X)]",
                colorbar_x=1.02,
                colorbar_y=0.79,
                colorbar_len=0.34,
                cmin=hmin,
                cmax=hmax,
            ),
            row=1,
            col=2,
        )
        # MI scatterplot
        fig.add_trace(
            _scatter3d(
                mi_np,
                "MI",
                colorbar_x=0.46,
                colorbar_y=0.21,
                colorbar_len=0.34,
            ),
            row=2,
            col=1,
        )

        # Previously sampled points: black X
        if xs_all.numel() > 0:
            prev_trace = go.Scatter3d(
                x=xs_all[:, 0].detach().cpu().numpy(),
                y=xs_all[:, 1].detach().cpu().numpy(),
                z=z_all.detach().cpu().numpy(),
                mode="markers",
                marker=dict(symbol="x", size=4, color="black", opacity=1.0),
                showlegend=False,
                hovertemplate=(
                    "Previous Sample<br>"
                    "Temp: %{x}<br>"
                    "Theta: %{y}<br>"
                    "Observed value: %{z}<extra></extra>"
                ),
            )
            fig.add_trace(prev_trace, row=1, col=1)
            fig.add_trace(prev_trace, row=1, col=2)
            fig.add_trace(prev_trace, row=2, col=1)



        scene_layout = dict(
            xaxis=dict(title="Temperature", range=x_range),
            yaxis=dict(title="Coverage", range=y_range_3d),
            zaxis=dict(title="Posterior mean", range=z_range),
            camera=camera,
        )

        fig.update_layout(
            title_text=f"{method_label} terms | Iteration {learner._i}",
            scene=scene_layout,
            scene2=scene_layout,
            scene3=scene_layout,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        html = fig.to_html(include_plotlyjs=True, full_html=True)

        sync_js = """
<script>
(function () {
  const gd = document.querySelector('.plotly-graph-div');
  if (!gd) return;
  let isUnderRelayout = false;
  gd.on('plotly_relayout', function(e) {
    if (isUnderRelayout) return;
    const keys = Object.keys(e || {});
    const cameraKey = keys.find(k => k.endsWith('.camera'));
    if (!cameraKey) return;
    const camera = e[cameraKey];
    isUnderRelayout = true;
    Plotly.relayout(gd, {
      'scene.camera': camera,
      'scene2.camera': camera,
      'scene3.camera': camera
    }).then(() => { isUnderRelayout = false; });
  });
})();
</script>
"""
        html = html.replace("</body>", sync_js + "\n</body>")

        with open(savepath_html, "w") as f:
            f.write(html)

    if make_2d_graphs:
        out_dir = Path(learner.config.base_savepath) / "MI_term_plots_2d"
        out_dir.mkdir(exist_ok=True, parents=True)
        savepath_png = out_dir / f"iteration_{learner._i:03d}.png"

        fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(18, 6), constrained_layout=True)

        sc0 = ax[0, 0].scatter(
            x_np,
            y_np,
            c=h_base_np,
            cmap="cividis",
            marker="s",
            s=140,
            linewidths=0,
            vmin=hmin,
            vmax=hmax,
        )
        ax[0, 0].set_title("H(Y)")

        sc1 = ax[0, 1].scatter(
            x_np,
            y_np,
            c=h_cond_np,
            cmap="cividis",
            marker="s",
            s=140,
            linewidths=0,
            vmin=hmin,
            vmax=hmax,
        )
        ax[0, 1].set_title("E[H(Y|X)]")

        sc2 = ax[0, 2].scatter(
            x_np,
            y_np,
            c=mi_np,
            cmap="cividis",
            marker="s",
            s=140,
            linewidths=0,
        )
        ax[0, 2].set_title("MI")


        # Overlay training points
        for j in range(3):
            if xs_all.numel() > 0:
                ax[0, j].scatter(
                    xs_all[:, 0].detach().cpu().numpy(),
                    xs_all[:, 1].detach().cpu().numpy(),
                    color="black",
                    s=10,
                    zorder=3,
                )


        for i in range(3):
            ax[0, i].set_xlabel("Temperature")
            ax[0, i].set_ylabel("Theta")
            ax[0, i].set_xlim(x_range)
            ax[0, i].set_ylim([0, 17.0])
            ax[0, i].set_aspect("auto")

        cbar0 = fig.colorbar(sc0, ax=ax[0, 0], fraction=0.046, pad=0.04)
        cbar0.set_label("H(Y)")
        cbar1 = fig.colorbar(sc1, ax=ax[0, 1], fraction=0.046, pad=0.04)
        cbar1.set_label("E[H(Y|X)]")
        cbar2 = fig.colorbar(sc2, ax=ax[0, 2], fraction=0.046, pad=0.04)
        cbar2.set_label("MI")

        fig.suptitle(
            f"{method_label} terms | Iteration {learner._i}",
            fontsize=12,
        )

        plt.savefig(savepath_png, dpi=120)
        plt.close(fig)



@torch.no_grad()
def _show_conditioning_3d(
    learner: "ActiveLearner",
    diagram_output,
    conditioner,
    slice_thetas: tuple[int, int] = (2, 14),
    n_temp: int = 200,
):
    """Show one conditioning example on a dense grid."""
    out_dir = Path(learner.config.base_savepath) / "MI_conditioning_3d"
    out_dir.mkdir(exist_ok=True, parents=True)
    savepath_html = out_dir / f"iteration_{learner._i:03d}.html"

    dg = learner.data_generator
    n_theta = dg.theta_range

    # Dense evaluation grid in unit space
    temp_grid = torch.linspace(
        0.0,
        1.0,
        n_temp,
        device=learner.train_x.device,
        dtype=learner.train_x.dtype,
    )
    theta_grid = torch.arange(
        0,
        n_theta + 1,
        device=learner.train_x.device,
        dtype=learner.train_x.dtype,
    ) / n_theta
    xg, yg = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    eval_grid = torch.stack([xg.ravel(), yg.ravel()]).T
    n_eval = eval_grid.shape[0]

    # Condition on the same ridge sample, but now on the dense plotting grid
    try:
        _, info = conditioner.condition(
            x_test=eval_grid,
            diagram_output=diagram_output,
            learner=learner,
        )
    except StopIteration:
        return

    if ("posterior_mean" not in info) or ("posterior_covariance" not in info):
        return

    # Physical coordinates for plotting
    eval_grid_raw = dg.untransform_inputs(eval_grid)
    T_raw = eval_grid_raw[:, 0]
    theta_raw = eval_grid_raw[:, 1]

    # Unconditioned posterior on dense grid
    with eval_context(learner.model):
        pred0 = learner.model(eval_grid[:, None, :])
    mean0 = pred0.mean.reshape(-1) * learner._y_std + learner._y_mean
    std0 = pred0.variance.reshape(-1).clamp_min(0.0).sqrt() * learner._y_std

    # Conditioned posterior on dense grid
    mean1 = info["posterior_mean"][:n_eval].reshape(-1)
    cov1 = info["posterior_covariance"][:n_eval, :n_eval]
    std1 = cov1.diagonal().clamp_min(0.0).sqrt()

    mean1 = mean1 * learner._y_std + learner._y_mean
    std1 = std1 * learner._y_std

    # Training points
    xs_train = dg.untransform_inputs(learner.train_x.detach())
    z_train = learner.train_y_orig.detach().reshape(-1)

    # Ridge points from this sample
    ridge_x = diagram_output.peaks
    ridge_raw = None
    ridge_mean0 = None
    ridge_mean1 = None
    if ridge_x is not None and ridge_x.numel() > 0:
        ridge_raw = dg.untransform_inputs(ridge_x.detach())

        with eval_context(learner.model):
            ridge_pred0 = learner.model(ridge_x[:, None, :])
        ridge_mean0 = ridge_pred0.mean.reshape(-1) * learner._y_std + learner._y_mean

        try:
            _, ridge_info = conditioner.condition(
                x_test=ridge_x,
                diagram_output=diagram_output,
                learner=learner,
            )
            ridge_mean1 = ridge_info["posterior_mean"][: ridge_x.shape[0]].reshape(-1)
            ridge_mean1 = ridge_mean1 * learner._y_std + learner._y_mean
        except StopIteration:
            ridge_mean1 = None
    

    # Shared axis ranges / camera
    x_range = [float(dg.min_temp), float(dg.max_temp)]
    y_range = [0.0, 17.0]

    zmin = float(
        min(
            mean0.min().detach().cpu().item(),
            mean1.min().detach().cpu().item(),
            z_train.min().detach().cpu().item() if z_train.numel() > 0 else float("inf"),
        )
    )
    zmax = float(
        max(
            mean0.max().detach().cpu().item(),
            mean1.max().detach().cpu().item(),
            z_train.max().detach().cpu().item() if z_train.numel() > 0 else float("-inf"),
        )
    )
    zpad = 0.05 * (zmax - zmin + 1e-12)
    z_range = [zmin - zpad, zmax + zpad]

    cmin = float(
        min(
            mean0.min().detach().cpu().item(),
            mean1.min().detach().cpu().item(),
        )
    )
    cmax = float(
        max(
            mean0.max().detach().cpu().item(),
            mean1.max().detach().cpu().item(),
        )
    )

    camera = dict(eye=dict(x=-1.6, y=-1.8, z=0.8))

    theta_a, theta_b = slice_thetas

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Posterior mean before conditioning",
            "Posterior mean after conditioning",
            "Overlay: before and after conditioning",
            f"Theta = {theta_a} and {theta_b} slices",
        ),
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    # Top-left: unconditioned posterior mean
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=theta_raw.detach().cpu().numpy(),
            z=mean0.detach().cpu().numpy(),
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.75,
                color=mean0.detach().cpu().numpy(),
                colorscale="Blues",
                showscale=True,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Mean", x=0.46, y=0.79, len=0.34),
            ),
            showlegend=False,
            hovertemplate=(
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                "Posterior mean (before): %{z}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Top-right: conditioned posterior mean
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=theta_raw.detach().cpu().numpy(),
            z=mean1.detach().cpu().numpy(),
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.75,
                color=mean1.detach().cpu().numpy(),
                colorscale="Greens",
                showscale=True,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Mean", x=1.02, y=0.79, len=0.34),
            ),
            showlegend=False,
            hovertemplate=(
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                "Posterior mean (after): %{z}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # Bottom-left: overlay of full clouds
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=theta_raw.detach().cpu().numpy(),
            z=mean0.detach().cpu().numpy(),
            mode="markers",
            marker=dict(size=2, opacity=0.35, color="blue"),
            showlegend=False,
            hovertemplate=(
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                "Posterior mean (before): %{z}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=theta_raw.detach().cpu().numpy(),
            z=mean1.detach().cpu().numpy(),
            mode="markers",
            marker=dict(size=2, opacity=0.35, color="green"),
            showlegend=False,
            hovertemplate=(
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                "Posterior mean (after): %{z}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # Training points on top row and overlay
    if xs_train.numel() > 0:
        train_trace = go.Scatter3d(
            x=xs_train[:, 0].detach().cpu().numpy(),
            y=xs_train[:, 1].detach().cpu().numpy(),
            z=z_train.detach().cpu().numpy(),
            mode="markers",
            marker=dict(symbol="x", size=4, color="black", opacity=1.0),
            showlegend=False,
            hovertemplate=(
                "Training point<br>"
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                "Observed value: %{z}<extra></extra>"
            ),
        )
        fig.add_trace(train_trace, row=1, col=1)
        fig.add_trace(train_trace, row=1, col=2)
        fig.add_trace(train_trace, row=2, col=1)

    # Ridge points on top row and overlay
    if ridge_raw is not None and ridge_raw.numel() > 0:
        ridge_z0 = (ridge_mean0 + 0.1).detach().cpu().numpy()
        ridge_trace0 = go.Scatter3d(
            x=ridge_raw[:, 0].detach().cpu().numpy(),
            y=ridge_raw[:, 1].detach().cpu().numpy(),
            z=ridge_z0,
            mode="markers",
            marker=dict(symbol="diamond", size=4, color="orange", opacity=1.0),
            showlegend=False,
            hovertemplate=(
                "Ridge point<br>"
                "Temp: %{x}<br>"
                "Theta: %{y}<extra></extra>"
            ),
        )
        fig.add_trace(ridge_trace0, row=1, col=1)

        if ridge_mean1 is not None:
            ridge_z1 = (ridge_mean1 + 0.1).detach().cpu().numpy()
        else:
            ridge_z1 = ridge_z0

        ridge_trace1 = go.Scatter3d(
            x=ridge_raw[:, 0].detach().cpu().numpy(),
            y=ridge_raw[:, 1].detach().cpu().numpy(),
            z=ridge_z1,
            mode="markers",
            marker=dict(symbol="diamond", size=4, color="orange", opacity=1.0),
            showlegend=False,
            hovertemplate=(
                "Ridge point<br>"
                "Temp: %{x}<br>"
                "Theta: %{y}<extra></extra>"
            ),
        )
        fig.add_trace(ridge_trace1, row=1, col=2)

        ridge_overlay = go.Scatter3d(
            x=ridge_raw[:, 0].detach().cpu().numpy(),
            y=ridge_raw[:, 1].detach().cpu().numpy(),
            z=ridge_z1,
            mode="markers",
            marker=dict(symbol="diamond", size=4, color="orange", opacity=1.0),
            showlegend=False,
            hovertemplate=(
                "Ridge point<br>"
                "Temp: %{x}<br>"
                "Theta: %{y}<extra></extra>"
            ),
        )
        fig.add_trace(ridge_overlay, row=2, col=1)

    def _add_slice_to_bottom_right(slice_theta: int, y_before: float, y_after: float):
        slice_mask = torch.isclose(
            theta_raw,
            torch.tensor(float(slice_theta), device=theta_raw.device, dtype=theta_raw.dtype),
        )

        slice_T = T_raw[slice_mask].detach().cpu().numpy()
        order = np.argsort(slice_T)
        slice_T = slice_T[order]

        m0 = mean0[slice_mask].detach().cpu().numpy()[order]
        s0 = std0[slice_mask].detach().cpu().numpy()[order]
        m1 = mean1[slice_mask].detach().cpu().numpy()[order]
        s1 = std1[slice_mask].detach().cpu().numpy()[order]

        y0 = np.full_like(slice_T, y_before, dtype=float)
        y1 = np.full_like(slice_T, y_after, dtype=float)

        # Training points at this theta
        if xs_train.numel() > 0:
            train_slice_mask = np.isclose(
                xs_train[:, 1].detach().cpu().numpy(),
                float(slice_theta),
            )
            if train_slice_mask.any():
                xs_slice = xs_train[train_slice_mask]
                z_slice = z_train[train_slice_mask]

                fig.add_trace(
                    go.Scatter3d(
                        x=xs_slice[:, 0].detach().cpu().numpy(),
                        y=np.full(xs_slice.shape[0], y_before),
                        z=z_slice.detach().cpu().numpy(),
                        mode="markers",
                        marker=dict(symbol="x", size=4, color="black"),
                        showlegend=False,
                        hovertemplate=(
                            "Training point<br>"
                            "Temp: %{x}<br>"
                            f"Theta: {slice_theta}<br>"
                            "Observed value: %{z}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=2,
                )

        # Ridge points at this theta
        if ridge_raw is not None and ridge_raw.numel() > 0:
            ridge_slice_mask = np.isclose(
                ridge_raw[:, 1].detach().cpu().numpy(),
                float(slice_theta),
            )
            if ridge_slice_mask.any():
                ridge_slice = ridge_raw[ridge_slice_mask]

                if ridge_mean1 is not None:
                    ridge_slice_z = (
                        ridge_mean1[ridge_slice_mask] + 0.1
                    ).detach().cpu().numpy()
                else:
                    ridge_slice_z = (
                        ridge_mean0[ridge_slice_mask] + 0.1
                    ).detach().cpu().numpy()

                fig.add_trace(
                    go.Scatter3d(
                        x=ridge_slice[:, 0].detach().cpu().numpy(),
                        y=np.full(ridge_slice.shape[0], slice_theta),
                        z=ridge_slice_z,
                        mode="markers",
                        marker=dict(symbol="diamond", size=4, color="orange"),
                        showlegend=False,
                        hovertemplate=(
                            "Ridge point<br>"
                            "Temp: %{x}<br>"
                            f"Theta: {slice_theta}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=2,
                )

        # Before-conditioning mean and 95% CI
        fig.add_trace(
            go.Scatter3d(
                x=slice_T,
                y=y0,
                z=m0,
                mode="markers",
                marker=dict(size=2, color="blue", opacity=1.0),
                showlegend=False,
                hovertemplate=(
                    "Temp: %{x}<br>"
                    f"Theta: {slice_theta}<br>"
                    "Posterior mean (before): %{z}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=slice_T,
                y=y0,
                z=m0 + 1.96 * s0,
                mode="markers",
                marker=dict(size=1.5, color="blue", opacity=0.1),
                showlegend=False,
                hovertemplate=(
                    "Temp: %{x}<br>"
                    f"Theta: {slice_theta}<br>"
                    "Upper 95% CI (before): %{z}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=slice_T,
                y=y0,
                z=m0 - 1.96 * s0,
                mode="markers",
                marker=dict(size=1.5, color="blue", opacity=0.1),
                showlegend=False,
                hovertemplate=(
                    "Temp: %{x}<br>"
                    f"Theta: {slice_theta}<br>"
                    "Lower 95% CI (before): %{z}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

        # After-conditioning mean and 95% CI
        fig.add_trace(
            go.Scatter3d(
                x=slice_T,
                y=y1,
                z=m1,
                mode="markers",
                marker=dict(size=2, color="green", opacity=1.0),
                showlegend=False,
                hovertemplate=(
                    "Temp: %{x}<br>"
                    f"Theta: {slice_theta}<br>"
                    "Posterior mean (after): %{z}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=slice_T,
                y=y1,
                z=m1 + 1.96 * s1,
                mode="markers",
                marker=dict(size=1.5, color="green", opacity=0.1),
                showlegend=False,
                hovertemplate=(
                    "Temp: %{x}<br>"
                    f"Theta: {slice_theta}<br>"
                    "Upper 95% CI (after): %{z}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=slice_T,
                y=y1,
                z=m1 - 1.96 * s1,
                mode="markers",
                marker=dict(size=1.5, color="green", opacity=0.1),
                showlegend=False,
                hovertemplate=(
                    "Temp: %{x}<br>"
                    f"Theta: {slice_theta}<br>"
                    "Lower 95% CI (after): %{z}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    # Put both slices in the single bottom-right panel
    _add_slice_to_bottom_right(theta_a, y_before=theta_a + 0.1, y_after=theta_a - 0.1)
    _add_slice_to_bottom_right(theta_b, y_before=theta_b + 0.1, y_after=theta_b - 0.1)

    scene_layout = dict(
        xaxis=dict(title="Temperature", range=x_range),
        yaxis=dict(title="Theta", range=y_range),
        zaxis=dict(title="Log scaled heat capacity", range=z_range),
        camera=camera,
    )

    fig.update_layout(
        title_text=f"Conditioning diagnostic | Iteration {learner._i}",
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        scene4=scene_layout,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html = fig.to_html(include_plotlyjs=True, full_html=True)

    sync_js = """
<script>
(function () {
  const gd = document.querySelector('.plotly-graph-div');
  if (!gd) return;
  let isUnderRelayout = false;
  gd.on('plotly_relayout', function(e) {
    if (isUnderRelayout) return;
    const keys = Object.keys(e || {});
    const cameraKey = keys.find(k => k.endsWith('.camera'));
    if (!cameraKey) return;
    const camera = e[cameraKey];
    isUnderRelayout = true;
    Plotly.relayout(gd, {
      'scene.camera': camera,
      'scene2.camera': camera,
      'scene3.camera': camera,
      'scene4.camera': camera
    }).then(() => { isUnderRelayout = false; });
  });
})();
</script>
"""
    html = html.replace("</body>", sync_js + "\n</body>")

    with open(savepath_html, "w") as f:
        f.write(html)