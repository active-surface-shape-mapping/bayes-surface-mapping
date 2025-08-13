import torch
from abc import abstractmethod, ABC
from jaxtyping import Float
from bayes_phase.models.mixins import TwiceDifferentiable, OnceDifferentiableBothArgs
from linear_operator import LinearOperator


class TwiceDifferentiableBothArgs(TwiceDifferentiable, OnceDifferentiableBothArgs, ABC):
    """Mixin class for a twice-differentiable kernel."""

    @abstractmethod
    def d3dxt2zt(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        """Evaluate the mixed derivative (w.r.t. temperature) of the kernel in both arguments."""
        ...

    @abstractmethod
    def d4dxt2zt2(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        """Evaluate the mixed derivative (w.r.t. temperature) of the kernel in both arguments."""
        ...
