from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

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


    def __init__(self, learner: "ActiveLearner"):
        self.learner = learner

    def forward(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        model = self.learner.model

        device = self.learner.train_x.device
        dtype = self.learner.train_x.dtype
        X = X.to(device=device, dtype=dtype)

        with torch.no_grad(), eval_context(model):
            posterior = model(X)

        var_f = posterior.variance.reshape(-1).clamp_min(1e-30)

        noise_var = model.likelihood.noise.to(
            dtype=var_f.dtype,
            device=var_f.device,
        ).reshape(-1)

        if noise_var.numel() == 1:
            noise_var = noise_var.expand_as(var_f)

        noise_var = noise_var.clamp_min(1e-30)

        scores = 0.5 * torch.log(var_f + noise_var) - 0.5 * torch.log(noise_var)
        return scores

    def __call__(
        self,
        X: Float[torch.Tensor, "m d"],
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(X)