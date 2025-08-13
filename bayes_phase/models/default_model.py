import gpytorch
import torch


class DefaultModel(gpytorch.models.ExactGP):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self, train_x, train_y, min_temp: float, max_temp: float, n_theta: int
    ):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        observation_model = gpytorch.likelihoods.GaussianLikelihood()
        super(DefaultModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        # self.covar_module = self.base_kernel
        # self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=3)
        self.likelihood = likelihood
        self.observation_model = observation_model
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_theta = n_theta

    def forward(self, x):
        if self.training:
            mean_x = self.mean_module(x)[..., 0:1]
            # covar_i = self.task_covar_module(i)
            covar_x = self.covar_module(x)[..., : mean_x.numel(), : mean_x.numel()]
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            mean_x = self.mean_module(x)
            # covar_i = self.task_covar_module(i)
            covar_x = self.covar_module(x)
            # covar = covar_x.mul(covar_i)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X: torch.Tensor) -> torch.Tensor:
        return X
        # return torch.stack(
        #     [
        #         (X[:, 0] - self.min_temp) / (self.max_temp - self.min_temp),
        #         (X[:, 1] - 1) / (self.n_theta)
        #     ]
        # )
