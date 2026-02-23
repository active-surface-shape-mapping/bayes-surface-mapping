import torch
import math
from jaxtyping import Float
from .acquisition_utilities import sample_from_gaussian, posterior_latent, find_ridges

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

    def forward(
        self, 
        learner: "ActiveLearner"
    ) -> Float[torch.Tensor, "m "]:

        model = learner.model
        X = learner.cand_x
        n_mc_samples = int(learner.config.n_mc_samples)

        # ---- H(Y): baseline entropy at each candidate (univariate Gaussian) ----
        with torch.no_grad():
            mean_X, cov_X = posterior_latent(model, X)  # cov over f(X)
        var_base = cov_X.diagonal().clamp_min(1e-30)   # (m,)

        # H(N(., var)) = 0.5 * log(2*pi*e*var)
        H_base = 0.5 * torch.log(var_base) + 0.5 * math.log(2.0 * math.pi * math.e)  # (m,)

        # ---- Sample functions to induce ridge hypotheses ----
        # f_samples: (m, S) each column is one sampled function over all candidates
        with torch.no_grad():
            f_samples = sample_from_gaussian(mean_X, cov_X, n_samples=n_mc_samples)

        # ---- Monte Carlo estimate of E[ H(Y | ridge) ] ----
        H_cond_sum = torch.zeros_like(H_base)
        n_used = 0

        # Monte Carlo sampling
        for s in range(n_mc_samples):
            f_s = f_samples[:, s]  # (m,)

            # Ridge finder expects X and f-values on X
            # It returns indices + ridge_x (points in input space)
            try:
                _, ridge_x = find_ridges(
                    X,
                    f_s,
                )
            except Exception:
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
                # EP returned None (if you ever change it to do so) -> skip
                continue

            mu_cond, Sigma_cond = out

            # Sigma_cond corresponds to joint over [f(X), f''(ridge)] (per your code).
            # The first m entries are f(X). So take the top-left block diagonal.
            m = X.shape[0]
            var_cond = Sigma_cond[:m, :m].diagonal().clamp_min(1e-30)  # (m,)
            H_cond = 0.5 * torch.log(var_cond) + 0.5 * math.log(2.0 * math.pi * math.e)

            H_cond_sum += H_cond
            n_used += 1

        if n_used == 0:
            raise RuntimeError("MutualInformation: all MC samples failed (no ridges or EP failures).")

        H_cond_mean = H_cond_sum / float(n_used)

        # ---- MI per candidate ----
        scores = H_base - H_cond_mean
        return scores
        
        
    def __call__(
        self,
        learner,
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(learner)
