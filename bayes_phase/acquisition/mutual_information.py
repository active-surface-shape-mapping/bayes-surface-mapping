import torch
import math
from jaxtyping import Float
from typing import Any, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from pathlib import Path
import gpytorch
from tqdm import tqdm

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

    def forward(
        self, X: Float[torch.Tensor, "n d"], plot_every: Optional[int] = 10
    ) -> Float[torch.Tensor, "m "]:
        n_mc_samples = int(self.learner.config.n_mc_samples)

        # # ---- H(Y): baseline entropy at each candidate (univariate Gaussian) ----
        # with torch.no_grad():
        #     mean_X, cov_X = posterior_latent(model, X)  # cov over f(X)
        # var_base = cov_X.diagonal().clamp_min(1e-30)  # (m,)
        with eval_context(self.learner.model):
            predictions = self.learner.model(X[:, torch.newaxis, :])
        var_base = predictions.variance.ravel()

        # H(N(., var)) = 0.5 * log(2*pi*e*var)
        H_base = 0.5 * torch.log(var_base) + 0.5 * math.log(
            2.0 * math.pi * math.e
        )  # (m,)

        # ---- Sample functions to induce ridge hypotheses ----
        mc_samples = self.learner.diagram_generator.diagrams(n_mc_samples)

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
            except StopIteration:
                continue

            # Sigma_cond corresponds to joint over [f(X), f''(ridge)]; extract f(X)
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
