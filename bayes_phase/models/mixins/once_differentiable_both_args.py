import torch
from abc import abstractmethod, ABC
from jaxtyping import Float
from bayes_phase.models.mixins import OnceDifferentiable
from linear_operator import LinearOperator


class OnceDifferentiableBothArgs(OnceDifferentiable, ABC):
    """Mixin class for a twice-differentiable kernel."""

    @abstractmethod
    def d2dxtzt(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        """Evaluate the second derivative (w.r.t. temperature) of the kernel at `x`."""
        pass
