from bayes_phase.models.kernels import Matern52KernelWithGrad
import torch
import math
import pytest


# Autodiff reference
def _matern52_scalar_autodiff(xi: torch.Tensor, zj: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """
    Matérn(5/2) base kernel (unit output scale), computed as a scalar and made
    to work with autodiff at coincident points by using a Taylor branch for r ~ 0.
    - r is the scaled distance after dividing by length scale(s).
    - This function returns the base kernel value (not multiplied by output scale)
    """
    diff = xi - zj  # (d,)

    ls = lengthscale.squeeze()
    if ls.ndim == 0:
        diff = diff / ls
    else:
        diff = diff / ls  # (d,) broadcast

    r2 = (diff * diff).sum()
    sqrt5 = math.sqrt(5.0)

    # Threshold in scaled-distance units
    r0 = 1e-4
    r0_2 = r0 * r0

    # Second-order Taylor expansion in r^2 near 0: k(r) = 1 - (5/6) r^2 + (25/24) r^4. 
    # Computed using Wolfram Mathematica 14.2 https://www.wolfram.com/mathematica/.
    k_series = 1.0 - (5.0 / 6.0) * r2 + (25.0 / 24.0) * (r2 * r2)

    # Exact branch, avoiding sqrt(0) by clamping r2 only in this branch
    r = torch.sqrt(torch.clamp(r2, min=r0_2))
    t = sqrt5 * r
    k_exact = (1.0 + t + (t * t) / 3.0) * torch.exp(-t)

    return torch.where(r2 < r0_2, k_series, k_exact)


def _pair_derivs_t_autodiff(
    xi: torch.Tensor, zj: torch.Tensor, lengthscale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns derivative covariances (base kernel, unit output scale) wrt T at x and z
    """
    xi = xi.detach().clone().requires_grad_(True)
    zj = zj.detach().clone().requires_grad_(True)

    k = _matern52_scalar_autodiff(xi, zj, lengthscale)

    dk_dxi = torch.autograd.grad(k, xi, create_graph=True)[0]
    ddxt = dk_dxi[0]

    d2k_dxi2 = torch.autograd.grad(dk_dxi[0], xi, create_graph=True)[0]
    d2dxt2 = d2k_dxi2[0]

    d2k_dxidzj = torch.autograd.grad(dk_dxi[0], zj, create_graph=True)[0]
    d2dxtzt = d2k_dxidzj[0]

    d3k_dxi2_dzj = torch.autograd.grad(d2k_dxi2[0], zj, create_graph=True)[0]
    d3dxt2zt = d3k_dxi2_dzj[0]

    d4k_dxi2_dzj2 = torch.autograd.grad(d3k_dxi2_dzj[0], zj, create_graph=True)[0]
    d4dxt2zt2 = d4k_dxi2_dzj2[0]

    return ddxt, d2dxt2, d2dxtzt, d3dxt2zt, d4dxt2zt2


def _all_pairs_autodiff(x: torch.Tensor, z: torch.Tensor, lengthscale: torch.Tensor):
    n1, d = x.shape
    n2, _ = z.shape

    ddxt = torch.empty((n1, n2), dtype=x.dtype, device=x.device)
    d2dxt2 = torch.empty((n1, n2), dtype=x.dtype, device=x.device)
    d2dxtzt = torch.empty((n1, n2), dtype=x.dtype, device=x.device)
    d3dxt2zt = torch.empty((n1, n2), dtype=x.dtype, device=x.device)
    d4dxt2zt2 = torch.empty((n1, n2), dtype=x.dtype, device=x.device)

    for i in range(n1):
        for j in range(n2):
            a, b, c, d3, d4 = _pair_derivs_t_autodiff(x[i], z[j], lengthscale)
            ddxt[i, j] = a
            d2dxt2[i, j] = b
            d2dxtzt[i, j] = c
            d3dxt2zt[i, j] = d3
            d4dxt2zt2[i, j] = d4

    return ddxt, d2dxt2, d2dxtzt, d3dxt2zt, d4dxt2zt2



# Tests-------------------------------------------------------------

def test_matern_kernel_with_grad_requires_nu_2p5():
    _ = Matern52KernelWithGrad(nu=2.5, ard_num_dims=2)

    with pytest.raises(RuntimeError) as err:
        _ = Matern52KernelWithGrad(nu=0.5, ard_num_dims=2)
    assert err.value.args[0] == "Matern52KernelWithGrad supports nu = 2.5 only"


@pytest.mark.parametrize("ard_num_dims", [1, 2])
def test_matern52_derivative_blocks_match_autodiff_with_coincident_points(ard_num_dims: int):
    """
    Test closed-form derivative blocks for Matérn(5/2) against autograd.
    Include coincident and near-coincident pairs to exercise the R~0 handling paths.
    """
    torch.manual_seed(0)

    n1, n2 = 6, 6
    x = torch.rand(n1, ard_num_dims, dtype=torch.double)
    z = torch.rand(n2, ard_num_dims, dtype=torch.double)

    k = Matern52KernelWithGrad(ard_num_dims=ard_num_dims)

    # Nontrivial lengthscales (ARD when d>1)
    if ard_num_dims == 1:
        ls = torch.tensor(0.37, dtype=torch.double)
    else:
        ls = torch.tensor([0.37, 1.23], dtype=torch.double)
    k.lengthscale = ls

    # Force exact coincidences (R = 0)
    z[0] = x[0]
    z[1] = x[1]

    # Force near coincidences that fall below the analytic eps used in closed forms
    ls0 = ls if ls.ndim == 0 else ls[0]
    delta_scaled = 1e-14
    delta_unscaled = delta_scaled * ls0
    z[2] = x[2]
    z[2, 0] = z[2, 0] + delta_unscaled

    # Autodiff reference (base kernel, unit output scale)
    dd_ad, d2_ad, d22_ad, d3_ad, d4_ad = _all_pairs_autodiff(x, z, k.lengthscale)

    # Closed forms under test
    dd = k.ddxt(x, z)
    d2 = k.d2dxt2(x, z)
    d22 = k.d2dxtzt(x, z)
    d3 = k.d3dxt2zt(x, z)
    d4 = k.d4dxt2zt2(x, z)

    # Convert LinearOperators
    dd = dd.to_dense() if hasattr(dd, "to_dense") else dd
    d2 = d2.to_dense() if hasattr(d2, "to_dense") else d2
    d22 = d22.to_dense() if hasattr(d22, "to_dense") else d22
    d3 = d3.to_dense() if hasattr(d3, "to_dense") else d3
    d4 = d4.to_dense() if hasattr(d4, "to_dense") else d4

    # Tolerances
    assert torch.allclose(dd, dd_ad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(d2, d2_ad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(d22, d22_ad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(d3, d3_ad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(d4, d4_ad, rtol=1e-8, atol=1e-8)