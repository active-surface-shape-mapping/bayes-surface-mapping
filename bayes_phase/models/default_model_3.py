import gpytorch
import torch
from typing import Optional
from gpytorch.priors import LogNormalPrior
import numpy as np


class DefaultModel3(gpytorch.models.ExactGP):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self, train_inputs, train_y, min_temp: float, max_temp: float, n_theta: int
    ):
        # Define likelihood, mean, and kernel:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=LogNormalPrior(0, 1)
        )
        train_x, train_i = train_inputs
        train_x_trainsformed = self.transform_inputs(
            train_x, min_temp=min_temp, max_temp=max_temp, n_theta=n_theta
        )
        super(DefaultModel3, self).__init__(
            (train_x_trainsformed, train_i),
            train_y,
            likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMeanGradGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGradGrad(
            ard_num_dims=2,
            lengthscale_prior=LogNormalPrior(-4, 0.5),
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, outputscale_prior=LogNormalPrior(0, 1)
        )
        self.index_module = gpytorch.kernels.IndexKernel(num_tasks=5)
        # Define auxilary mean/kernel for computing gradient
        for p in self.index_module.parameters():
            p.requires_grad = False
        self.likelihood = likelihood
        # Other bookkeeping:
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_theta = n_theta

    def forward(self, x, i):
        i = i.ravel()
        mean_x = self.mean_module(x).ravel()[i]
        covar = self.covar_module(x).to_dense()
        # mask = i.outer(i)
        n = i.sum()
        covar_x = covar[i, :][:, i].reshape((n, n))
        sigmas = np.exp(np.arange(-11, 3))
        sigmas[0] = 0.0
        distribution = None
        for sigma in sigmas:
            try:
                distribution = gpytorch.distributions.MultivariateNormal(
                    mean_x, covar_x + sigma * torch.eye(n)
                )
                break
            except torch.linalg.LinAlgError:
                continue
        if distribution is None:
            raise ValueError(
                "Covariance is non-positive definite even after adding jitter up to 1e+2"
            )
        return distribution

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
