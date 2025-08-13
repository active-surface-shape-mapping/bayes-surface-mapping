import torch
from abc import abstractmethod, ABC
from jaxtyping import Float
from linear_operator import LinearOperator


class OnceDifferentiable(ABC):
    """Mixin class for a once-differentiable kernel."""

    @abstractmethod
    def ddxt(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        """Evaluate the first derivative (w.r.t. temperature) of the kernel at `x`."""
        pass
