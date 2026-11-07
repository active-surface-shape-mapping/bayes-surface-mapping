
import gpytorch
import torch


class StandardExactGP(gpytorch.models.ExactGP):
    """GPyTorch value-only ExactGP for acquisition functions that don't require derivatives."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        base_kernel: gpytorch.kernels.Kernel,
        mean_module: gpytorch.means.Mean | None = None,
        train_yvar: torch.Tensor | None = None,
        learn_additional_noise: bool = True,
        noise_floor: float = 1e-7,
        noise_ceiling: float | None = None,
        noise_prior: gpytorch.priors.Prior | None = None,
        outputscale_prior: gpytorch.priors.Prior | None = None,
    ):

        # Check noise ceiling if supplied.
        if noise_ceiling is not None and noise_ceiling <= noise_floor:
            raise ValueError(
                f"noise_ceiling ({noise_ceiling}) must be greater than "
                f"noise_floor ({noise_floor})."
            )
        
        # Create likelihood object. 
        # If known training data variance (train_yvar) is supplied, use a fixed-noise likelihood and allow learning of additional noise. 
        if train_yvar is None:
            if noise_ceiling is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_prior=noise_prior,
                    noise_constraint=gpytorch.constraints.GreaterThan(noise_floor),
                )
            else:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_prior=noise_prior,
                    noise_constraint=gpytorch.constraints.Interval(noise_floor, noise_ceiling),
                )
        else:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=train_yvar.ravel(),
                learn_additional_noise=learn_additional_noise,
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
            outputscale_prior=outputscale_prior,
        )
        self.likelihood = likelihood

    # Forward pass: Return distribution
    def forward(
        self,
        x: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def transform_inputs(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        return X
