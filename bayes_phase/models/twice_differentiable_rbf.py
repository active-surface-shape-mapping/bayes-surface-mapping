import gpytorch
import torch
from typing import Literal, Optional
from .kernels import RBFKernelWithGrad
from jaxtyping import Float
from bayes_phase.models.base_model import BaseModel


class TwiceDifferentiableRBF(gpytorch.models.ExactGP, BaseModel):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self, train_x, train_y, min_temp: float, max_temp: float, n_theta: int
    ):
        # Define likelihood, mean, and kernel:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=gpytorch.priors.LogNormalPrior(0, 1)
        )
        BaseModel.__init__(self, min_temp, max_temp, n_theta)
        super(TwiceDifferentiableRBF, self).__init__(
            self.transform_inputs(
                train_x, min_temp=min_temp, max_temp=max_temp, n_theta=n_theta
            ),
            train_y,
            likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = RBFKernelWithGrad(
            ard_num_dims=2,
            lengthscale_prior=gpytorch.priors.LogNormalPrior(-4, 0.5),
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, outputscale_prior=gpytorch.priors.LogNormalPrior(0, 1)
        )
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze(-1)
        # covar_i = self.task_covar_module(i)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _update_caches(self):
        """Compute and update cached quantitiies."""
        # Compute train covariance matrix + caches:
        x_train = torch.vstack(self.train_inputs)
        y_train = self.train_targets
        prior_mean = self.mean_module(x_train)
        covar_x = self.covar_module(x_train)
        noise_var = self.likelihood.noise_covar.noise
        L = (
            (covar_x + noise_var * torch.eye(y_train.numel()))
            .root_decomposition(method="cholesky")
            .root
        )
        alpha = L.T.solve_triangular(
            L.solve_triangular(y_train - prior_mean, upper=False), upper=True
        )  # (n_train,)
        # Update caches:
        self._L = L
        self._alpha = alpha
        self._n_train = y_train.numel()
        self._x_train = x_train
        self._y_train = y_train

    def eval(self):
        self._update_caches()
        return super().eval()

    def grad_covar(
        self,
        x: Float[torch.Tensor, "n1 d"],
        z: Optional[Float[torch.Tensor, "n2 d"]] = None,
        full: bool = False,
    ) -> Float[torch.Tensor, "3*n1 3*n2"] | Float[torch.Tensor, "3*n1 n2"]:
        """Compute the covariance between observations and temperature derivatives."""
        if z is None:
            z = x

        if full:
            # Compute full covariance, including between derivative terms.
            # Output shape: 3*n1 x 3*n2
            o = self.covar_module.outputscale
            K = self.covar_module(x, z).to_dense()
            K_dxt = o * self.base_kernel.ddxt(x, z).to_dense()
            K_d2dxt2 = o * self.base_kernel.d2dxt2(x, z).to_dense()
            col1 = torch.vstack([K, K_dxt, K_d2dxt2])

            K_d2dxtdzt = o * self.base_kernel.d2dxtzt(x, z).to_dense()
            K_d3dxt2dzt = o * self.base_kernel.d3dxt2zt(x, z).to_dense()
            col2 = torch.vstack([K_dxt.T, K_d2dxtdzt, K_d3dxt2dzt])

            K_d4dxt2zt2 = o * self.base_kernel.d4dxt2zt2(x, z).to_dense()
            col3 = torch.vstack([K_d2dxt2.T, K_d3dxt2dzt.T, K_d4dxt2zt2])

            return torch.hstack([col1, col2, col3])

        # Compute covariance only between observations and derivative terms.
        # Output shape: 3*n1 x n2
        K = self.covar_module(x, z).to_dense()
        K_dt = self.covar_module.outputscale * self.base_kernel.ddxt(x, z).to_dense()
        K_d2dxt2 = (
            self.covar_module.outputscale * self.base_kernel.d2dxt2(x, z).to_dense()
        )
        return torch.vstack([K, K_dt, K_d2dxt2])

    def predict_with_grad(
        self,
        x: Float[torch.Tensor, "n d"],
        compute_covariance: Literal["none", "diagonal", "full"] = "full",
        add_likelihood: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "n 3"],
        Float[torch.Tensor, "3n 3n"] | Float[torch.Tensor, "n 3"] | None,
    ]:
        """Predict function values and temperature gradients at test points `x`"""
        self._update_caches()
        k_xs_x = self.grad_covar(x, self._x_train)  # (3 * n_test, n_train)
        f_pred = k_xs_x @ self._alpha
        f_pred = f_pred.reshape((3, -1)).to_dense().T
        prior_pred_mean = self.mean_module(x)
        f_pred[:, 0] += prior_pred_mean
        # (Co)variance computation:
        if compute_covariance == "full":
            # Return full covariance matrix between observations and temperature derivatives
            # Output shape: 3n x 3n
            v = self._L.solve_triangular(
                k_xs_x.T.to_dense(), upper=False
            )  # (n_train, 3 * n_test)
            k_xs_xs = self.grad_covar(x, full=True)  # (3 * n_test, 3 * n_test)
            covariance = k_xs_xs - v.T @ v
        elif compute_covariance == "diagonal":
            # Return varian matrix of observations and temperature derivatives
            # Output shape: n x 3
            v = self._L.solve_triangular(
                k_xs_x.T.to_dense(), upper=False
            )  # (n_train, 3 * n_test)
            covariance = torch.zeros((x.shape[0], 3))
            for i in range(x.shape[0]):
                k_xs_xs = self.grad_covar(x[i : i + 1, :], full=True).to_dense()
                print(k_xs_xs.shape)
                outer_product = torch.Tensor(
                    [v[:, 3 * i + j].inner(v[:, 3 * i + j]) for j in range(3)]
                )
                print(outer_product.shape)
                covariance[i, :] = k_xs_xs.diag() - outer_product
        else:
            covariance = None
        return f_pred, covariance
