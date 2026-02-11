from bayes_phase.models.kernels import MaternKernelWithGrad
import torch
from scipy.differentiate import derivative
import pytest

torch.set_default_dtype(torch.double)


def test_init_matern_kernel_with_grad():
    kernel = MaternKernelWithGrad(ard_num_dims=2)  # NOQA: F841
    kernel = MaternKernelWithGrad(0.5, ard_num_dims=2)  # NOQA: F841
    kernel = MaternKernelWithGrad(nu=1.5, ard_num_dims=2)  # NOQA: F841
    with pytest.raises(RuntimeError) as err:
        kernel = MaternKernelWithGrad(nu=2.0, ard_num_dims=2)  # NOQA: F841
    assert err.value.args == ("nu expected to be 0.5, 1.5, or 2.5",)


def test_first_derivative():
    torch.random.manual_seed(666)
    kernel = MaternKernelWithGrad(ard_num_dims=2)
    n, m = 10, 15
    x = torch.distributions.Uniform(0.0, 1.0).sample((n, 2))
    z = torch.distributions.Uniform(0.0, 1.0).sample((m, 2))
    d = kernel.ddxt(x, z)
    assert d.shape == (n, m)

    # 2nd order finite difference for first x_0 derivative
    dt = 1e-8
    x_prime_1 = x.clone()
    x_prime_1[:, 0] -= dt
    x_prime_2 = x.clone()
    x_prime_2[:, 0] += dt
    d1 = kernel(x_prime_1, z)
    d2 = kernel(x_prime_2, z)
    finite_diff = (d2 - d1) / (2.0 * dt)
    assert torch.allclose(d.to_dense(), finite_diff.to_dense())

    # SciPy derivative
    d2 = torch.full((n, m), torch.nan)
    for i in range(n):
        for j in range(m):

            def f(t):
                numel = t.size
                test_point = torch.full((numel, 2), torch.nan)
                test_point[:, 0] = torch.tensor(t.ravel())
                test_point[:, 1] = x[i, 1].item()
                result = (
                    kernel(test_point, z[j : j + 1, :]).to_dense().detach().numpy().T
                )
                return result

            deriv = derivative(f, x[i : i + 1, 0:1])
            d2[i, j] = deriv.df.item()

    assert torch.allclose(d.to_dense(), d2)


def test_second_derivative():
    torch.random.manual_seed(666)
    kernel = MaternKernelWithGrad(ard_num_dims=2)
    n, m = 10, 15
    x = torch.distributions.Uniform(0.0, 1.0).sample((n, 2))
    z = torch.distributions.Uniform(0.0, 1.0).sample((m, 2))
    dd = kernel.d2dxt2(x, z)
    assert dd.shape == (n, m)

    # 8th-order finite difference for second x_0 derivative.
    dt = 1e-4
    x_prime_1 = x.clone()
    x_prime_1[:, 0] -= 2.0 * dt
    x_prime_2 = x.clone()
    x_prime_2[:, 0] -= dt
    x_prime_1 = x.clone()
    x_prime_1[:, 0] -= 2.0 * dt
    x_prime_2 = x.clone()
    x_prime_2[:, 0] -= dt
    d0 = kernel(x, z)
    d1 = kernel(x_prime_1, z)
    d2 = kernel(x_prime_2, z)
    finite_diff = (d1 - 2.0 * d0 + d2) / (dt**2)
    deltas = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    coeffs = [-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0, -205.0 / 72.0]
    coeffs = coeffs + list(reversed(coeffs))[1:]
    finite_diff = 0.0
    for delta, coeff in zip(deltas, coeffs):
        x_prime = x.clone()
        x_prime[:, 0] += delta * dt
        finite_diff += coeff * kernel(x_prime, z)
    finite_diff /= dt**2
    assert torch.allclose(dd.to_dense(), finite_diff.to_dense(), atol=1e-4)
