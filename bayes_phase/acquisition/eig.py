import math
import torch
from jaxtyping import Float
from typing import TYPE_CHECKING

from bayes_phase.utils import eval_context

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner


class EIG:
    """
    Local expected information gain for a noisy observation y at candidate x.

        EIG = (1/2)ln(k_D(x,x)+sigma^2) - (1/2)ln(sigma^2)

    Since sigma^2 is constant across candidates, this is monotone in the
    posterior variance Var[f | D].
    """

    def __init__(
        self,
        learner: "ActiveLearner",
    ):
        self.learner = learner

    def forward(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        model = self.learner.model

        with eval_context(model):
            # Posterior at candidate points
            posterior = model(X[:, torch.newaxis, :])

        # Predictive variance k_D(x, x)
        var_f = posterior.variance.ravel().clamp_min(1e-30)

        # Observation noise sigma^2 from GaussianLikelihood
        noise_var = model.likelihood.noise.to(dtype=var_f.dtype, device=var_f.device)
        noise_var = noise_var.clamp_min(1e-30)

        scores = 0.5 * torch.log(var_f + noise_var) - 0.5 * torch.log(noise_var)
        return scores

    def __call__(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(X)