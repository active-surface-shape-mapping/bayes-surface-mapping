import gpytorch
from jaxtyping import Float
import torch
import math
from linear_operator import LinearOperator
from bayes_phase.models.mixins import TwiceDifferentiableBothArgs


class Matern52KernelWithGrad(gpytorch.kernels.MaternKernel, TwiceDifferentiableBothArgs):
    """Lightweight wrapper around GPyTorch Matern kernel with nu = 2.5.
    Radial expressions are used.
    Derivatives of the Matern(5/2) kernel were calculated with Wolfram Mathematica 14.2:
    https://www.wolfram.com/mathematica/"""

    def __init__(self, nu: float = 2.5, **kwargs):
        """
        Parameters
        ----------
        nu:
            Smoothness parameter. This kernel supports Matérn(5/2) only (nu=2.5).
        **kwargs:
            Forwarded to gpytorch.kernels.MaternKernel (e.g., ard_num_dims, active_dims, etc.).
        """
        if nu != 2.5:
            raise RuntimeError("Matern52KernelWithGrad supports nu = 2.5 only")
        super().__init__(nu=2.5, **kwargs)

    def _ls_vec(self) -> torch.Tensor:
        # Returns scalar () for 1D isotropic, or (d,) for ARD
        return self.lengthscale.squeeze()

    def _ls_for_dim(self, deriv_dim: int) -> torch.Tensor:
        ls = self._ls_vec()
        return ls if ls.ndim == 0 else ls[deriv_dim]

    def _pairwise_u_v2_R(self, x: torch.Tensor, z: torch.Tensor, deriv_dim: int = 0):
        # In this code, u is the lengthscale-normalized difference along the derivative dimension,
        # and R is the full ARD-normalized distance. The quantity v^2 equals the squared distance in
        # all non-derivative dimensions, i.e. v^2 = R^2 - u^2.
        diff = x[:, None, :] - z[None, :, :]  # (n1, n2, d)
        ls = self._ls_vec()  # () or (d,)

        if ls.ndim == 0:
            diff = diff / ls  # scalar scaling
        else:
            diff = diff / ls[None, None, :]  # ARD scaling

        u = diff[..., deriv_dim]          # (n1, n2)
        r2 = (diff * diff).sum(dim=-1)    # (n1, n2)
        R = torch.sqrt(r2)                # (n1, n2)
        v2 = r2 - u * u                    # (n1, n2)
        return u, v2, R

    # Returns covariance between function value t-derivatives at 'x' and function values at 'z'.
    def ddxt(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        u, v2, R = self._pairwise_u_v2_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()

        sqrt5 = math.sqrt(5.0)
        five_thirds = 5.0 / 3.0
        exp_term = torch.exp(-sqrt5 * R)

        dkdu = (-five_thirds * exp_term - five_thirds * sqrt5 * exp_term * R) * u
        return invls * dkdu

    # Returns covariance between function value t-second derivatives at 'x' and function values at 'z'.
    def d2dxt2(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        u, v2, R = self._pairwise_u_v2_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls2 = invls**2

        sqrt5 = math.sqrt(5.0)
        five_thirds = 5.0 / 3.0
        exp_term = torch.exp(-sqrt5 * R)

        d2kdu2 = (
            -five_thirds * exp_term
            - five_thirds * sqrt5 * exp_term * R
            + 25.0 / 3.0 * exp_term * (u**2)
        )

        return invls2 * d2kdu2

    # Returns covariance between function value t-first derivatives at 'x' and function value t-first derivatives at 'z'
    def d2dxtzt(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        u, v2, R = self._pairwise_u_v2_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls2 = invls**2

        sqrt5 = math.sqrt(5.0)
        five_thirds = 5.0 / 3.0
        exp_term = torch.exp(-sqrt5 * R)

        d2kdu2 = (
            -five_thirds * exp_term
            - five_thirds * sqrt5 * exp_term * R
            + 25.0 / 3.0 * exp_term * (u**2)
        )

        return -invls2 * d2kdu2

    # Returns covariance between function value t-second derivatives at 'x' and function value t-first derivatives at 'z'
    def d3dxt2zt(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        u, v2, R = self._pairwise_u_v2_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls3 = invls**3

        sqrt5 = math.sqrt(5.0)
        exp_term = torch.exp(-sqrt5 * R)

        # Zero handling: For R=0, calculate with R_safe then replace with the limit of d3kdu3 as R-> 0
        eps = 1e-12
        mask = R < eps

        # Safe denominator to avoid division-by-zero / NaNs before masking
        R_safe = torch.where(mask, torch.ones_like(R), R)

        d3kdu3 = (25.0 / 3.0) * exp_term * u * (sqrt5 * (u**2) - 3.0 * R) / R_safe

        # Diagonal limit: diag(d3kdu3) = 0
        d3kdu3 = torch.where(mask, torch.zeros_like(d3kdu3), d3kdu3)

        return invls3 * d3kdu3


    # Returns covariance between function value t-second derivatives at 'x' and function value t-second derivatives at 'z'
    def d4dxt2zt2(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        u, v2, R = self._pairwise_u_v2_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls4 = invls**4

        sqrt5 = math.sqrt(5.0)
        exp_term = torch.exp(-sqrt5 * R)

        # Zero handling: For R=0, calculate with R_safe then replace with the limit as R-> 0
        eps = 1e-12
        mask = R < eps

        R_safe = torch.where(mask, torch.ones_like(R), R)
        R2 = R_safe**2

        numerator = (
            3.0 * (u**2) * (R_safe - 2.0 * sqrt5 * v2)
            + 3.0 * v2 * R_safe
            + 5.0 * (u**4) * (R_safe - sqrt5)
        )

        d4kdu4 = (25.0 / 3.0) * exp_term * numerator / (R_safe**3)

        # Diagonal limit (dimensionless): diag(d4kdu4) = 25  -> full diag = 25 / ℓ_T^4 after invls4
        d4kdu4 = torch.where(mask, torch.full_like(d4kdu4, 25.0), d4kdu4)

        return invls4 * d4kdu4
