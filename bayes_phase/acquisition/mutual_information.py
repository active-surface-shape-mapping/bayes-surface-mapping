import torch
import math
from jaxtyping import Float

from bayes_phase.utils import eval_context

# FIXME: Noise approach between MI and EIG in this draft may be inconsistent.
#        Confirm whether acquisition should condition on noisy or latent values.
#        Ensure no hidden GPyTorch prediction cache interactions.


class MutualInformation:
    """
    Class to to compute mutual information between the current model's posterior and the model's posterior over ridges.
    I(X;Y) = H(Y) - H(Y|X)
    Where H(Y) is the model's posterior at a point.
    H(Y|X) the model's posterior conditioned on a set of ridges. To be estimated by Monte Carlo, E[H(Y|X)].

    """

    def __init__(self):
        pass

    def forward(self, learner: "ActiveLearner") -> Float[torch.Tensor, "m "]:
        model = learner.model
        X = learner.cand_x
        n_mc_samples = int(learner.config.n_mc_samples)

        # # ---- H(Y): baseline entropy at each candidate (univariate Gaussian) ----
        # with torch.no_grad():
        #     mean_X, cov_X = posterior_latent(model, X)  # cov over f(X)
        # var_base = cov_X.diagonal().clamp_min(1e-30)  # (m,)
        with eval_context(learner.model):
            predictions = learner.model(X[:, torch.newaxis, :])
        var_base = predictions.variance.ravel()

        # H(N(., var)) = 0.5 * log(2*pi*e*var)
        H_base = 0.5 * torch.log(var_base) + 0.5 * math.log(
            2.0 * math.pi * math.e
        )  # (m,)

        # ---- Sample functions to induce ridge hypotheses ----
        mc_samples = learner.diagram_generator.diagrams(n_mc_samples)

        # ---- Monte Carlo estimate of E[ H(Y | ridge) ] ----
        H_cond_sum = torch.zeros_like(H_base)
        n_used = 0

        # Monte Carlo sampling
        for ridge_x, _ in mc_samples:
            if ridge_x.numel() == 0:
                # No ridges (or ridge-finder failed) for this sample -> skip sample
                continue

            # Condition on ridges using EP
            with torch.no_grad():
                out = model.pred_condition_1st_and_2nd_deriv(
                    x_test=X,
                    x_ridge=ridge_x,
                    verbose=False,
                )

            if out is None:
                # IF EP returned None -> skip
                continue

            mu_cond, Sigma_cond = out

            # Sigma_cond corresponds to joint over [f(X), f''(ridge)]; extract f(X)
            m = X.shape[0]
            var_cond = Sigma_cond[:m, :m].diagonal().clamp_min(1e-30)  # (m,)
            H_cond = 0.5 * torch.log(var_cond) + 0.5 * math.log(2.0 * math.pi * math.e)

            H_cond_sum += H_cond
            n_used += 1

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
        learner,
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(learner)
