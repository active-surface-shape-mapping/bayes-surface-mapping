"""Small covariance and conditioning checks without hyperparameter fitting."""

import gpytorch
import pytest
import torch

from bayes_phase.models import StandardExactGP


@pytest.mark.parametrize("kernel_name", ["rbf", "matern52"])
def test_gp_covariance_and_posterior(kernel_name):
    # Fixed inputs keep the test deterministic; duplicate query points exercise
    # a valid singular covariance (positive semidefinite, not strictly positive).
    train_x = torch.tensor([[0.0, 0.0], [0.4, 0.7], [1.0, 1.0]], dtype=torch.float64)
    train_y = torch.tensor([[0.2], [1.0], [-0.3]], dtype=torch.float64)
    query_x = torch.tensor([[0.2, 0.3], [0.8, 0.6], [0.2, 0.3]], dtype=torch.float64)
    kernel = (
        gpytorch.kernels.RBFKernel(ard_num_dims=2)
        if kernel_name == "rbf"
        else gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
    )
    model = StandardExactGP(train_x, train_y, kernel).double()
    model.likelihood.noise = 0.05

    def check_distribution(distribution):
        covariance = distribution.covariance_matrix
        assert distribution.mean.shape == (len(query_x),)
        assert covariance.shape == (len(query_x), len(query_x))
        assert torch.isfinite(distribution.mean).all()
        assert torch.isfinite(covariance).all()
        torch.testing.assert_close(covariance, covariance.T, atol=1e-10, rtol=1e-10)
        # Inspect the covariance directly: .variance can hide negatives by clamping.
        assert covariance.diag().min() >= -1e-10
        assert torch.linalg.eigvalsh(covariance).min() >= -1e-10
        return covariance

    with torch.no_grad():
        # forward() returns the prior; model() in eval mode conditions on training data.
        prior = model.forward(query_x)
        prior_covariance = check_distribution(prior)
        model.eval()
        model.likelihood.eval()
        with gpytorch.settings.fast_pred_var(False):
            posterior = model(query_x)
        posterior_covariance = check_distribution(posterior)

        # Independently calculate the standard GP conditioning equations.
        train_covariance = model.covar_module(train_x).to_dense()
        noisy_covariance = train_covariance + model.likelihood.noise * torch.eye(
            len(train_x), dtype=train_x.dtype
        )
        cross_covariance = model.covar_module(query_x, train_x).to_dense()
        residual = train_y.ravel() - model.mean_module(train_x)
        expected_mean = prior.mean + cross_covariance @ torch.linalg.solve(
            noisy_covariance, residual
        )
        expected_covariance = prior_covariance - cross_covariance @ torch.linalg.solve(
            noisy_covariance, cross_covariance.T
        )
        torch.testing.assert_close(posterior.mean, expected_mean)
        torch.testing.assert_close(posterior_covariance, expected_covariance)
        assert (posterior_covariance.diag() <= prior_covariance.diag() + 1e-10).all()

        # Predictive observations add measurement noise to the latent covariance.
        predictive_covariance = check_distribution(model.likelihood(posterior))
        torch.testing.assert_close(
            predictive_covariance,
            posterior_covariance + model.likelihood.noise * torch.eye(
                len(query_x), dtype=query_x.dtype
            ),
        )
