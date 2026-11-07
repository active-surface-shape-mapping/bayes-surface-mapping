"""Small end-to-end AL runs; no mocked models, acquisitions, or optimizers."""

import json

import polars as pl
import pytest
import torch

from bayes_phase.active_learning.active_learner import ActiveLearner
from bayes_phase.active_learning.config import ActiveLearnerConfig
from bayes_phase.utils import eval_context


ACQUISITIONS = ["random", "sobol", "eig", "dispersion"]


@pytest.fixture
def config(tmp_path):
    # Pytest provides a fresh temporary directory and config for each test case.
    # Three positive, nonconstant heat-capacity curves with a peak per theta.
    paths = []
    temperatures = torch.linspace(100, 500, 17, dtype=torch.float64)
    for replicate in range(3):
        path = tmp_path / f"run{replicate}.arrow"
        columns = {"temp_in_K": temperatures.tolist()}
        for theta in range(1, 4):
            peak = torch.exp(-((temperatures - (220 + 40 * theta)) / 65) ** 2)
            columns[f"θ_{theta}"] = (1 + peak + 0.01 * replicate).tolist()
        pl.DataFrame(columns).write_ipc(path)
        paths.append(str(path))
    # Six starting observations plus two acquisitions keep real GP fits small.
    return ActiveLearnerConfig(
        global_seed=42, n_iters=2, init_n_points=6, iter_n_candidates=3,
        init_dataset_paths=paths, sampling_dataset_paths=paths,
        ground_truth_dataset_paths=paths, base_savepath=str(tmp_path),
        base_kernel_kwargs={"ard_num_dims": 2},
        acquisition_n_mc_samples=4, acquisition_n_temp_eval=12,
        acquisition_n_fantasies=2, acquisition_slice_window_radius=1,
        distance_metric_kwargs={"cutoff": 0.2, "p": 1, "alpha": 2},
        diagram_generator_kwargs={"max_n_peaks": 1, "extrema": "peaks"},
    )


def assert_valid_points(learner, points):
    assert points.ndim == 2 and points.shape[1] == 2, "Invalid point shape"
    assert torch.isfinite(points).all(), "Non-finite coordinates"
    assert ((points >= 0) & (points <= 1)).all(), "Points outside unit square"
    # Theta must be an integer in physical coordinates, even though GP inputs are normalized.
    theta = learner.data_generator.untransform_inputs(points)[:, 1]
    torch.testing.assert_close(theta, theta.round(), atol=1e-5, rtol=0)


# Run all eight combinations: four acquisitions times two candidate strategies.
# Random and Sobol acquisitions draw directly and ignore the candidate strategy.
@pytest.mark.parametrize("acquisition", ACQUISITIONS)
@pytest.mark.parametrize("candidate_strategy", ["sobol", "random"])
def test_al_loop_preserves_invariants(config, acquisition, candidate_strategy):
    """Check candidates, selection, observations, GP updates, and accounting."""
    config.acquisition_function = acquisition
    config.cand_data_strategy = candidate_strategy
    # Seed the run without changing the random state used by other tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.global_seed)
        learner = ActiveLearner(config)
        previous_x = learner.train_x.clone()
        previous_y = learner.train_y_orig.clone()
        completed = []

        def check_iteration(al):
            # This callback runs after a new observation is appended and the GP is refitted.
            nonlocal previous_x, previous_y
            expected_n = config.init_n_points + al._i + 1
            assert al.next_x.shape == (1, 2)
            assert_valid_points(al, al.train_x)
            assert_valid_points(al, al.cand_x)
            assert al.train_x.shape == (expected_n, 2)
            assert al.train_y.shape == al.train_y_orig.shape == (expected_n, 1)
            # Preserve every old row and append exactly the newly acquired observation.
            torch.testing.assert_close(al.train_x[:-1], previous_x)
            torch.testing.assert_close(al.train_y_orig[:-1], previous_y)
            torch.testing.assert_close(al.train_x[-1:], al.next_x)
            torch.testing.assert_close(al.train_y_orig[-1:], al.next_y_orig)
            # With added noise disabled, measuring the same point must reproduce its values.
            expected_y, expected_sigma = al._observe(al.next_x)
            torch.testing.assert_close(al.next_y_orig, expected_y)
            torch.testing.assert_close(al.next_y_sigma, expected_sigma)
            assert torch.isfinite(al.next_y_sigma).all()
            assert (al.next_y_sigma >= 0).all()
            # Normalize all targets again after each append, then pass them to the GP.
            assert torch.isfinite(al.train_y).all()
            torch.testing.assert_close(al.train_y.mean(), al.train_y.new_tensor(0.), atol=1e-5, rtol=0)
            torch.testing.assert_close(al.train_y.std(), al.train_y.new_tensor(1.))
            torch.testing.assert_close(al.model.train_inputs[0], al.train_x)
            torch.testing.assert_close(al.model.train_targets, al.train_y.ravel())
            assert al.diagram_generator.learner.model is al.model
            assert all(torch.isfinite(p).all() for p in al.model.parameters())
            # Check predictions as well as parameters; GPyTorch may clamp negative variances.
            with torch.no_grad(), eval_context(al.model):
                posterior = al.model(al.next_x)
                assert torch.isfinite(posterior.mean).all()
                assert torch.isfinite(posterior.variance).all()
                assert (posterior.variance >= 0).all()
            # Scored acquisitions must select the highest-scoring candidate.
            # Direct draws instead leave the candidate and score arrays empty.
            if acquisition in {"eig", "dispersion"}:
                assert al.cand_x.shape == (config.iter_n_candidates, 2)
                assert al.last_scores.shape == (config.iter_n_candidates,)
                assert torch.isfinite(al.last_scores).all()
                best = al.last_scores.argmax().item()
                torch.testing.assert_close(al.next_x, al.cand_x[best:best + 1])
                assert al.last_acquisition_score == pytest.approx(al.last_scores[best].item())
                assert al.last_n_candidates_evaluated == config.iter_n_candidates
            else:
                assert al.cand_x.shape == (0, 2) and al.last_scores.numel() == 0
                assert al.last_n_candidates_evaluated == 0
            # Sample counts include initialization; accumulated cost counts AL steps only.
            assert al.n_points_sampled_total == expected_n
            cost = config.sampling_methods[0]["cost"]
            assert al.last_simulation_cost == cost
            assert al.simulation_cost_total == pytest.approx((al._i + 1) * cost)
            previous_x, previous_y = al.train_x.clone(), al.train_y_orig.clone()
            completed.append(al._i)

        # Register the checks with the real loop so both iterations are inspected.
        learner.callbacks.append(check_iteration)
        learner.optimize()
    assert completed == list(range(config.n_iters)), "Loop skipped an iteration"
    # Logging happens after callbacks, so inspect the saved records after the run.
    with open(f"{config.base_savepath}/logs.json") as stream:
        records = [json.loads(line) for line in stream]
    assert [row["iteration"] for row in records] == completed
    assert records[-1]["n_points_sampled_total"] == learner.train_x.shape[0]


# Sixteen cases: four bad measurements for each of the four acquisitions.
# Log transformation makes zero and negative measurements non-finite too.
@pytest.mark.parametrize("acquisition", ACQUISITIONS)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 0.0, -1.0])
def test_al_loop_rejects_invalid_observations(config, acquisition, bad_value, monkeypatch):
    """Non-finite observations (including invalid log inputs) cannot be appended."""
    config.acquisition_function = acquisition
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.global_seed)
        learner = ActiveLearner(config)
        initial_x = learner.train_x.clone()
        initial_y = learner.train_y_orig.clone()
        initial_normalized_y = learner.train_y.clone()

        def broken_measurement(points):
            return torch.full((len(points), 3), bad_value, dtype=initial_y.dtype)

        # Fail only new observations, after valid initialization and GP creation.
        # Monkeypatch replaces this instance's measurement method and restores it after the test.
        monkeypatch.setattr(learner.data_generator, "measure_uniform", broken_measurement)
        # The expected error is a pass: the learner must reject the measurement.
        with pytest.raises(ValueError, match="Observed non-finite y values"):
            learner.optimize()
    # Rejection must happen before training data, sample counts, or costs are updated.
    torch.testing.assert_close(learner.train_x, initial_x)
    torch.testing.assert_close(learner.train_y_orig, initial_y)
    torch.testing.assert_close(learner.train_y, initial_normalized_y)
    torch.testing.assert_close(learner.model.train_inputs[0], initial_x)
    torch.testing.assert_close(learner.model.train_targets, initial_normalized_y.ravel())
    assert learner.n_points_sampled_total == config.init_n_points
    assert learner.simulation_cost_total == 0
