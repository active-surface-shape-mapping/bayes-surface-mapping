"""Validate conditioning against a fresh GPyTorch model with the extra observation."""

from types import SimpleNamespace

import gpytorch
import pytest
import torch

from bayes_mapping.acquisition.distance_dispersion import DistanceDispersion
from bayes_mapping.active_learning.config import ActiveLearnerConfig
from bayes_mapping.models import StandardExactGP


@pytest.fixture(params=["rbf", "matern52"])
def learner(request):
    # Fixed hyperparameters avoid optimization noise and keep these checks quick.
    x = torch.tensor([[0., 0.], [0.2, 0.5], [0.5, 1.], [0.8, 0.], [1., 1.]])
    y = torch.tensor([0.1, 0.8, -0.2, 0.4, 0.3])
    kernel = (gpytorch.kernels.RBFKernel(ard_num_dims=2) if request.param == "rbf"
              else gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    model = StandardExactGP(x, y, kernel).double()
    model.base_kernel.lengthscale = torch.tensor([0.3, 0.6])
    model.covar_module.outputscale = 1.2
    model.likelihood.noise = 0.04
    model.eval()
    config = ActiveLearnerConfig(
        acquisition_n_temp_eval=4, acquisition_n_mc_samples=6,
        acquisition_n_fantasies=3, acquisition_fantasy_quadrature_points=3,
        acquisition_slice_window_radius=1,
        distance_metric_kwargs={"cutoff": 0.2, "p": 1, "alpha": 2},
        diagram_generator_kwargs={"extrema": "peaks", "max_n_peaks": 1},
    )
    return SimpleNamespace(
        model=model, train_x=x, config=config,
        data_generator=SimpleNamespace(min_theta=1, max_theta=3, theta_range=2),
    )


@pytest.mark.parametrize("candidate", [[0.4, 0.5], [0.2, 0.5]])
def test_low_rank_conditioning_matches_gpytorch(learner, candidate):
    """Compare with full GP conditioning, including a repeated training input."""
    acquisition = DistanceDispersion(learner, torch.Generator().manual_seed(42))
    x_eval = acquisition.grid_spec.x_eval
    x_cand = torch.tensor([candidate])
    fantasy_y = torch.tensor([-0.7, 0.2, 1.1])
    before = {name: value.clone() for name, value in learner.model.state_dict().items()}
    means, factor = acquisition._conditioned_sample_components(
        x_cand=x_cand, fantasy_y=fantasy_y, x_eval=x_eval,
    )
    assert means.shape == (3, len(x_eval))
    assert not torch.allclose(means[0], means[-1])
    conditioned_covariance = factor @ factor.T
    assert torch.linalg.eigvalsh(conditioned_covariance).min() > 0

    for index, observation in enumerate(fantasy_y):
        # Build a fresh GP on the augmented dataset. This uses GPyTorch's normal
        # training-data solve, independently of our rank-one update or fantasy API.
        augmented_x = torch.cat([learner.train_x, x_cand])
        augmented_y = torch.cat([learner.model.train_targets, observation.reshape(1)])
        kernel = (gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
                  if isinstance(learner.model.base_kernel, gpytorch.kernels.MaternKernel)
                  else gpytorch.kernels.RBFKernel(ard_num_dims=2))
        reference = StandardExactGP(augmented_x, augmented_y, kernel).double()
        reference.load_state_dict(before)
        reference.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            posterior = reference(x_eval)
        torch.testing.assert_close(means[index], posterior.mean, atol=1e-8, rtol=1e-7)
        # Sampling jitter belongs to the factor, not the statistical covariance.
        expected_covariance = posterior.covariance_matrix + acquisition.sample_jitter * torch.eye(len(x_eval))
        torch.testing.assert_close(conditioned_covariance, expected_covariance, atol=1e-8, rtol=1e-7)
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    torch.testing.assert_close(learner.model.train_inputs[0], learner.train_x)


@pytest.mark.parametrize("quadrature", [False, True], ids=["monte-carlo", "quadrature"])
def test_common_draws_and_single_factor_per_candidate(learner, quadrature, monkeypatch):
    """Shared randomness should remove candidate-order effects in both fantasy paths."""
    learner.config.acquisition_fantasy_quadrature = quadrature
    acquisition = DistanceDispersion(learner, torch.Generator().manual_seed(17))
    # Overlapping windows must select matching columns of one full-grid bank.
    left = acquisition._base_normal_samples_for_region([1, 2])
    right = acquisition._base_normal_samples_for_region([2, 3])
    torch.testing.assert_close(left[:, 4:], right[:, :4], atol=0, rtol=0)
    x_eval, _ = acquisition._build_local_eval_grid([1, 2])
    x_cand = torch.tensor([[0.35, 0.]])
    calls = []
    cholesky = torch.linalg.cholesky

    def count_factorizations(matrix, *args, **kwargs):
        if matrix.shape == (len(x_eval), len(x_eval)):
            calls.append(matrix.shape)
        return cholesky(matrix, *args, **kwargs)

    # Three fantasies must share one Cholesky factor, rather than factor per value.
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "cholesky", count_factorizations)
        weights, means, centered = acquisition._conditioned_draws(
            x_cand=x_cand, x_eval=x_eval, theta_values_raw=[1, 2],
        )
    assert len(calls) == 1
    assert means.shape == (3, len(x_eval))
    assert centered.shape == (6, len(x_eval))
    torch.testing.assert_close(weights.sum(), weights.new_tensor(1.))
    assert torch.isfinite(centered).all()

    candidates = torch.tensor([[0.35, 0.], [0.65, 1.], [0.55, 0.5], [0.35, 0.]])
    first = DistanceDispersion(learner, torch.Generator().manual_seed(99))(candidates)
    reversed_scores = DistanceDispersion(learner, torch.Generator().manual_seed(99))(candidates.flip(0))
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, reversed_scores.flip(0), atol=0, rtol=0)
    torch.testing.assert_close(first[0], first[3], atol=0, rtol=0)
