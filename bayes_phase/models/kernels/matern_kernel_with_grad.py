import gpytorch
from jaxtyping import Float
import torch
import math
from linear_operator import LinearOperator
from bayes_phase.models.mixins import TwiceDifferentiableBothArgs



class MaternKernelWithGrad(gpytorch.kernels.MaternKernel, TwiceDifferentiableBothArgs):
    """Lightweight wrapper around GPyTorch Matern kernel with nu = 2.5.
    Radial expressions are used.
    Derivatives of the Matern(5/2) kernel were calculated with Wolfram Mathematica 14.2:
    https://www.wolfram.com/mathematica/"""

    def _ls_vec(self) -> torch.Tensor:
        # Returns scalar () for 1D isotropic, or (d,) for ARD
        return self.lengthscale.squeeze()

    def _ls_for_dim(self, deriv_dim: int) -> torch.Tensor:
        ls = self._ls_vec()
        return ls if ls.ndim == 0 else ls[deriv_dim]

    def _pairwise_u_s_R(self, x: torch.Tensor, z: torch.Tensor, deriv_dim: int = 0):
        diff = x[:, None, :] - z[None, :, :]          # (n1, n2, d)
        ls = self._ls_vec()                            # () or (d,)

        if ls.ndim == 0:
            diff = diff / ls                           # scalar scaling
        else:
            diff = diff / ls[None, None, :]            # ARD scaling

        u = diff[..., deriv_dim]             # (n1, n2)
        r2 = (diff * diff).sum(dim=-1)       # (n1, n2)
        R = torch.sqrt(r2)                   # (n1, n2)
        s = r2 - u * u                       # (n1, n2)
        return u, s, R


    # Returns covariance between function value t-derivatives at 'x' and function values at 'z'.
    def ddxt(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        u,s,R = self._pairwise_u_s_R(x, z, deriv_dim=0)
    
        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        
        sqrt5 = math.sqrt(5.0)
        five_thirds = 5.0/3.0
        exp_term =  torch.exp(-sqrt5*R)

        dkdu = (-five_thirds*exp_term-five_thirds*sqrt5*exp_term*R)*u
        return invls*dkdu

    # Returns covariance between function value t-second derivatives at 'x' and function values at 'z'.
    def d2dxt2(
        self, x: Float[torch.Tensor, "n1 d"], z: Float[torch.Tensor, "n2 d"]
    ) -> Float[torch.Tensor, "n1 n2"] | LinearOperator:
        u,s,R = self._pairwise_u_s_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls2 = invls**2

        sqrt5 = math.sqrt(5.0)
        five_thirds = 5.0/3.0
        exp_term =  torch.exp(-sqrt5*R)

        d2kdu2 = (-five_thirds*exp_term - 
        five_thirds*sqrt5*exp_term*R + 
        25.0/3.0*exp_term*(u**2)
        )

        return invls2*d2kdu2

    # Returns covariance between function value t-first derivatives at 'x' and function value t-first derivatives at 'z'
    def d2dxtzt(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        u,s,R = self._pairwise_u_s_R(x, z, deriv_dim=0)

        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls2 = invls**2

        sqrt5 = math.sqrt(5.0)
        five_thirds = 5.0/3.0
        exp_term =  torch.exp(-sqrt5*R)

        d2kdu2 = (-five_thirds*exp_term - 
        five_thirds*sqrt5*exp_term*R + 
        25.0/3.0*exp_term*(u**2)
        )

        return -invls2*d2kdu2

    # Returns covariance between function value t-second derivatives at 'x' and function value t-first derivatives at 'z'
    def d3dxt2zt(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        u,s,R = self._pairwise_u_s_R(x, z, deriv_dim=0)
        
        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls3 = invls**3
        
        sqrt5 = math.sqrt(5.0)
        exp_term =  torch.exp(-sqrt5*R)

        # Zero handling: For R=0, use the limit of d3kdu3 as R-> 0
        d3kdu3 = torch.zeros_like(R)
        eps = 1e-12
        mask = R >= eps
        d3kdu3[mask] = (25.0*exp_term[mask]*u[mask] - 
        (25.0*sqrt5*exp_term[mask]*(u[mask]**3))/(3.0*R[mask])
        )

        return -invls3*d3kdu3


    # Returns covariance between function value t-second derivatives at 'x' and function value t-second derivatives at 'z'
    def d4dxt2zt2(
        self, x: gpytorch.Tensor, z: gpytorch.Tensor
    ) -> gpytorch.Tensor | LinearOperator:
        u,s,R = self._pairwise_u_s_R(x, z, deriv_dim=0)
        
        ls0 = self._ls_for_dim(0)
        invls = ls0.reciprocal()
        invls4 = invls**4

        sqrt5 = math.sqrt(5.0)
        exp_term =  torch.exp(-sqrt5*R)

        # Zero handling: For R=0, use the limit of d4kdu4 as R-> 0
        d4kdu4 = torch.full_like(R,25)
        eps = 1e-12
        mask = R >= eps

        d4kdu4[mask] = (25.0*exp_term[mask] - 
        (50.0*sqrt5*exp_term[mask]*(u[mask]**2))/R[mask] + 
        ((25.0*sqrt5*exp_term[mask])/(3.0*R[mask]**3) + 
        (125.0*exp_term[mask])/(3.0*R[mask]**2))*(u[mask]**4)
        )

        return invls4*d4kdu4

