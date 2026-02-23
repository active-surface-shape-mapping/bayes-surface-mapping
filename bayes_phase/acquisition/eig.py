import torch
from typing import Optional
from jaxtyping import Float
from .acquisition_utilities import posterior_latent, entropy_over_representer_set

# FIXME: Noise approach between MI and EIG in this draft may be inconsistent.
#        Confirm whether acquisition should condition on noisy or latent values.
#        Ensure no hidden GPyTorch prediction cache interactions.

class EIG:

    """
    Class to to compute expected information gain (EIG) acquisition across a candidate set.

    Novick, et al. (2024): https://doi.org/10.1039/D4MH00432A
    EIG(x,D) = H[F_D] - E_y[H[F_{D U (x,y)}]], where H is the differential entropy.

    """

    def __init__(
        self,
    ):
        pass

    def forward(
        self, 
        learner: "ActiveLearner"
    ) -> Float[torch.Tensor, "m "]:

        model = learner.model
        X = learner.cand_x
        
        model.eval()
        model.likelihood.eval()

        with torch.no_grad():
            _ = model(X)  # creates prediction_strategy caches

        # Base entropy
        H0 = entropy_over_representer_set(model, X)
        scores = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)

        with torch.no_grad():
            mean_x, _ = posterior_latent(model, X)

        for j in range(X.shape[0]):
            x_new = X[j:j+1]    # (1,d)
            y_new = mean_x[j:j+1].view(1,1) # (1,1); entropy/EIG does not depend on y

            # Fantasy model = posterior conditioned on adding (x_new, y_new)
            fantasy = model.get_fantasy_model(x_new, y_new.view(-1))

            # Entropy after conditioning
            Hj = entropy_over_representer_set(fantasy, X)

            # EIG = entropy before conditioning - entropy after conditioning
            scores[j] = H0 - Hj

        return scores
        
    def __call__(
        self,
        learner,
    ) -> Float[torch.Tensor, "m"]:
        return self.forward(learner)