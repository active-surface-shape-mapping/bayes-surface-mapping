import gpytorch
import torch
from typing import Optional


class StandardExactGP(gpytorch.models.ExactGP):
    """GPyTorch value-only ExactGP for acquisition functions that don't require derivatives."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        base_kernel: gpytorch.kernels.Kernel,
        mean_module: Optional[gpytorch.means.Mean] = None,
    ):
        # Create likelihood object
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=None,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-7),
            )

        # Initialize parent class
        super().__init__(
            train_x,
            train_y.ravel(),
            likelihood,
        )

        # Default to constant mean
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()

        # Store components
        self.mean_module = mean_module
        self.base_kernel = base_kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel,
            outputscale_prior=None,
        )
        self.likelihood = likelihood

    # Forward pass: Return distribution
    def forward(self, x):
        mean_x = self.mean_module(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X):
        return X