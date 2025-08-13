import gpytorch
import torch
from typing import Literal, Optional
from gpytorch.priors import LogNormalPrior


class DefaultModel2(gpytorch.models.ExactGP):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self, train_x, train_y, min_temp: float, max_temp: float, n_theta: int
    ):
        # Define likelihood, mean, and kernel:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=LogNormalPrior(0, 1)
        )
        super(DefaultModel2, self).__init__(
            self.transform_inputs(
                train_x, min_temp=min_temp, max_temp=max_temp, n_theta=n_theta
            ),
            train_y,
            likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=2,
            lengthscale_prior=LogNormalPrior(-4, 0.5),
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, outputscale_prior=LogNormalPrior(0, 1)
        )
        # Define auxilary mean/kernel for computing gradient
        self.grad_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        )
        self.grad_mean = gpytorch.means.ConstantMeanGrad()
        for p in self.grad_covar.parameters():
            p.requires_grad = False
        for p in self.grad_mean.parameters():
            p.requires_grad = False
        self.likelihood = likelihood
        # Other bookkeeping:
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_theta = n_theta

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze(-1)
        # covar_i = self.task_covar_module(i)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval(self):
        self._update_grad_modules()
        return super().eval()

    def _update_grad_modules(self):
        with torch.no_grad():
            for p1, p2 in zip(
                self.grad_mean.parameters(), self.mean_module.parameters()
            ):
                p1.data = p2.data
            for p1, p2 in zip(
                self.grad_covar.parameters(), self.covar_module.parameters()
            ):
                p1.data = p2.data
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

    def predict_with_grad(
        self,
        x,
        compute_covariance: Literal["none", "partial", "full"] = "full",
        add_likelihood: bool = False,
    ):
        # self._update_grad_modules()
        k_xs_x = self.grad_covar(x, self._x_train)[:, 0:-1:3]  # (3 * n_test, n_train)
        # test = self.grad_covar(x_train)
        # assert torch.allclose(test[0:-1:3, 0:-1:3].to_dense(), covar_x.to_dense()), (
        #     test[0:-1:3, 0:-1:3].to_dense() - covar_x.to_dense()
        # ).mean()
        f_pred = k_xs_x @ self._alpha
        v = self._L.solve_triangular(
            k_xs_x.T.to_dense(), upper=False
        )  # (n_train, 3 * n_test)
        f_pred = f_pred.reshape((-1, 3)).to_dense()
        prior_pred_mean = self.grad_mean(x)
        f_pred += prior_pred_mean
        # (Co)variance computation:
        if compute_covariance == "full":
            k_xs_xs = self.grad_covar(x)  # (3 * n_test, 3 * n_test)
            covariance = k_xs_xs - v.T @ v
        elif compute_covariance == "partial":
            covariance = torch.zeros((x.shape[0], 3))
            for i in range(x.shape[0]):
                k_xs_xs = self.grad_covar(x[i : i + 1, :]).to_dense()
                covariance[i, :] = k_xs_xs.diag() - torch.Tensor(
                    [v[:, 3 * i + j].inner(v[:, 3 * i + j]) for j in range(3)]
                )
        else:
            covariance = None
        return f_pred, covariance

    def predict(self, x):
        k_xs_x = self.covar_module(x, self._x_train)[:, :]  # (n_test, n_train)
        k_xs_xs = self.covar_module(x)  # (n_test, n_test)
        f_pred = k_xs_x @ self._alpha
        v = self._L.solve_triangular(
            k_xs_x.T.to_dense(), upper=False
        )  # (n_train, n_test)
        covariance = k_xs_xs - v.T @ v
        prior_pred_mean = self.grad_mean(x)
        f_pred += prior_pred_mean[:, 0]
        return f_pred, covariance

    def transform_inputs(
        self,
        X: torch.Tensor,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        n_theta: Optional[int] = None,
    ) -> torch.Tensor:
        if min_temp is None:
            min_temp = self.min_temp
        if max_temp is None:
            max_temp = self.max_temp
        if n_theta is None:
            n_theta = self.n_theta
        new_x = torch.zeros_like(X)
        new_x[:, 0] = (X[:, 0] - min_temp) / (max_temp - min_temp)
        new_x[:, 1] = (X[:, 1] - 1) / (n_theta - 1)
        return new_x
