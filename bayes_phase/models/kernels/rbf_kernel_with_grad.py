import gpytorch
from jaxtyping import Float
import torch
from linear_operator import LinearOperator
from bayes_phase.models.mixins import TwiceDifferentiableBothArgs


class RBFKernelWithGrad(gpytorch.kernels.RBFKernel, TwiceDifferentiableBothArgs):
    """Lightweight wrapper around GPyTorch RBF kernel."""

    def ddxt(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        """Ref: https://pmc.ncbi.nlm.nih.gov/articles/PMC7595302/#pone.0235885.t001"""
        return (
            super().__call__(x, z)
            * (z[:, 0] - x[:, 0:1])
            / super().lengthscale[0, 0] ** 2.0
        )

    def d2dxt2(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        """Ref: https://pmc.ncbi.nlm.nih.gov/articles/PMC7595302/#pone.0235885.t001"""
        return (
            super().__call__(x, z)
            * ((x[:, 0:1] - z[:, 0]) ** 2 / super().lengthscale[0, 0] ** 2.0 - 1.0)
            / super().lengthscale[0, 0] ** 2.0
        )

    def d2dxtzt(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        # 2g (2g(xt - zt)^2 + 1) k(x,z)
        # where g = 1 / ls^2
        return (
            -1.0
            * super().__call__(x, z)
            * ((x[:, 0:1] - z[:, 0]) ** 2 / super().lengthscale[0, 0] ** 2.0 - 1.0)
            / super().lengthscale[0, 0] ** 2.0
        )

    def d3dxt2zt(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        # 2g (2g(xt - zt)^2 + 1) k(x,z)
        # where g = 1 / ls^2
        g = 1 / (2.0 * super().lengthscale[0, 0] ** 2)
        xmz = x[:, 0:1] - z[:, 0]
        kxz = super().__call__(x, z)
        return 2.0 * g * kxz * xmz * (-4.0 * g + 4.0 * g**2 * xmz**2 - 2.0 * g)

    def d4dxt2zt2(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        # 4g^2 * k(x, z) * (xj - zj) (2g * (xj - zj)^2 - 1)
        # where g = 1 / ls^2
        g = 1 / (2.0 * super().lengthscale[0, 0] ** 2)
        xmz = x[:, 0:1] - z[:, 0]
        kxz = super().__call__(x, z)
        return 4.0 * g**2 * kxz * (3.0 - 12.0 * g * xmz**2 + 4.0 * g**2 * xmz**4)
