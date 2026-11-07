"""Check plotting and scoring against small datasets with known boundaries."""

import json
from functools import wraps
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render files without opening GUI windows during tests.
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pytest
import torch

from bayes_mapping.active_learning.active_learner import ActiveLearner
from bayes_mapping.active_learning.config import ActiveLearnerConfig
from bayes_mapping.active_learning.callbacks.plot import plot_diagrams
from bayes_mapping.active_learning.callbacks.score_model_plot import (
    _assign_ridge_points_to_grid,
    _calc_score,
    _generate_grid,
)
from bayes_mapping.acquisition.diagram_generation.base_diagram_generator import DiagramOutput
from bayes_mapping.acquisition.diagram_generation.find_peaks_diagram_generator import FindPeaksDiagramGenerator
from bayes_mapping.acquisition.peak_finding import find_slice_peak_idxs
from bayes_mapping.data_io import SimulatedDataGenerator


@pytest.fixture
def learner(tmp_path):
    # Peaks at 200, 300, and 400 K lie exactly on the 10 K evaluation grid.
    # The repeat datasets have small offsets but identical peak locations.
    temperatures = torch.linspace(100, 500, 41)
    paths = []
    for replicate in range(3):
        path = tmp_path / f"ridge_{replicate}.arrow"
        columns = {"temp_in_K": temperatures.tolist()}
        for theta in range(1, 4):
            peak = torch.exp(-((temperatures - (100 + 100 * theta)) / 40) ** 2)
            columns[f"θ_{theta}"] = (1 + peak + 0.01 * replicate).tolist()
        pl.DataFrame(columns).write_ipc(path)
        paths.append(str(path))
    config = ActiveLearnerConfig(
        global_seed=42, acquisition_function="sobol", init_n_points=12, n_iters=3,
        init_dataset_paths=paths, sampling_dataset_paths=paths,
        ground_truth_dataset_paths=paths, base_savepath=str(tmp_path),
        base_kernel_kwargs={"ard_num_dims": 2}, n_scoring_samples=3,
        forbidden_tree_regions=[], max_tree_conn_dist=1.0,
        diagram_generator_kwargs={"n_temp_grid": 41, "max_n_peaks": 1},
        callbacks=["plot", "plot_mst", "score_model_plot"],
        callback_every={"plot": 2, "plot_mst": 2, "score_model_plot": 1},
    )
    # Isolate random draws and clean up figures even if an assertion fails.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        try:
            yield ActiveLearner(config)
        finally:
            plt.close("all")


def assert_pdf(path):
    assert path.is_file(), f"Missing plot: {path}"
    assert path.stat().st_size > 100, f"Empty plot: {path}"
    with path.open("rb") as stream:
        assert stream.read(5) == b"%PDF-", f"Invalid PDF: {path}"


@pytest.mark.parametrize("extrema", ["peaks", "valleys"])
@pytest.mark.parametrize("options, expected_indices", [
    ({}, [1, 4, 7]),
    ({"peak_threshold": 0.3}, [1, 7]),
    ({"peak_height_rule": "above_median"}, [1, 7]),
    ({"max_n_peaks": 1}, [7]),
    ({"peak_height_rule": "above_median", "max_n_peaks": 1}, [7]),
    ({"max_n_peaks": 2}, [7, 1]),
])
def test_peak_rules_propagate_to_diagram_boundaries(learner, tmp_path, extrema, options, expected_indices):
    """Peak filters must select the same features in arrays and physical diagrams."""
    # Three local maxima: a tall peak, a small peak below the median, and the
    # tallest peak. Negating the curve gives equivalent valley test cases.
    signal = torch.tensor([0., 5., 0., -4., -3., -4., 0., 6., 0.])
    if extrema == "valleys":
        signal = -signal
    # Positive values allow logarithmic extraction; use linear values here so
    # the specified relative-prominence threshold has an exact interpretation.
    signal = signal + 10
    expected = torch.tensor(expected_indices)
    for values in [signal.numpy(), signal.clone().requires_grad_()]:
        actual = find_slice_peak_idxs(values, extrema=extrema, **options)
        torch.testing.assert_close(torch.as_tensor(actual), expected)
    # Give each theta slice a different scale and offset. Relative prominence
    # and median filtering should preserve the same selected temperatures.
    temperatures = torch.linspace(100, 500, 9)
    path = tmp_path / "multiple_features.arrow"
    pl.DataFrame({"temp_in_K": temperatures.tolist(), **{
        f"θ_{theta}": (signal * theta + theta).tolist() for theta in range(1, 4)
    }}).write_ipc(path)
    data = SimulatedDataGenerator([str(path)])
    diagram_generator = FindPeaksDiagramGenerator(
        learner, n_temp_grid=9, extrema=extrema, **options,
    )
    diagram = diagram_generator.diagrams(
        1, data_generator=data, use_log_transform=False,
    )[0]
    # The current DiagramOutput stores detected features in .peaks even when
    # extrema='valleys'. Check coordinates, not only the number of features.
    physical = data.untransform_inputs(diagram.peaks)
    expected_points = torch.tensor([
        [temperatures[index].item(), float(theta)]
        for theta in range(1, 4) for index in expected_indices
    ])
    torch.testing.assert_close(physical, expected_points)


def test_al_callbacks_save_consistent_outputs(learner):
    """Real callbacks should observe the updated learner without changing it."""
    calls = {callback.__name__: [] for callback in learner.callbacks}
    sampled_points = []

    def checked_callback(callback):
        # Preserve the name: the learner uses it to look up callback frequency.
        @wraps(callback)
        def checked(al):
            before = [tensor.clone() for tensor in (
                al.train_x, al.train_y_orig, al.train_y,
                al.model.train_inputs[0], al.model.train_targets,
            )]
            parameters = [parameter.clone() for parameter in al.model.parameters()]
            figures = set(plt.get_fignums())
            callback(al)
            after = [al.train_x, al.train_y_orig, al.train_y,
                     al.model.train_inputs[0], al.model.train_targets]
            for original, current in zip(before, after, strict=True):
                torch.testing.assert_close(current, original, rtol=0, atol=0)
            for original, current in zip(parameters, al.model.parameters(), strict=True):
                torch.testing.assert_close(current, original, rtol=0, atol=0)
            assert set(plt.get_fignums()) == figures, "Callback leaked a figure"
            calls[callback.__name__].append(al._i)
            if callback.__name__ == "score_model_plot":
                sampled_points.append(al.data_generator.untransform_inputs(al.next_x).tolist())
        return checked

    learner.callbacks = [checked_callback(callback) for callback in learner.callbacks]
    learner.optimize()
    # Frequency 2 means the first iteration and every second completed iteration.
    assert calls == {"plot": [0, 1], "plot_mst": [0, 1], "score_model_plot": [0, 1, 2]}
    root = Path(learner.config.base_savepath)
    for folder in ["plots", "mst_plots", "scoring/scoring_plots"]:
        expected = [0, 1, 2] if folder.startswith("scoring") else [0, 1]
        assert {path.name for path in (root / folder).glob("*.pdf")} == {
            f"iteration_{i}.pdf" for i in expected
        }
        for i in expected:
            assert_pdf(root / folder / f"iteration_{i}.pdf")
    assert_pdf(root / "scoring/score_metrics.pdf")
    scores = pd.read_csv(root / "scoring/scores.csv")
    assert scores["iteration"].tolist() == [0, 1, 2]
    assert scores["n_points_sampled_total"].tolist() == [13, 14, 15]
    assert scores["n_points_sampled_iter"].tolist() == [1, 1, 1]
    torch.testing.assert_close(torch.tensor([json.loads(x) for x in scores["sampled_x"]]),
                               torch.tensor(sampled_points))
    assert scores[["precision", "recall", "f1"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()


@pytest.mark.parametrize("monotonic", [False, True], ids=["known-ridge", "no-peaks"])
def test_diagram_generation_recovers_known_boundary(learner, tmp_path, monotonic):
    """Ground-truth extraction has an exact answer independent of GP fitting."""
    generator = learner.data_generator
    if monotonic:
        # Strictly increasing curves have no interior local maxima.
        temperatures = torch.linspace(100, 500, 41)
        path = tmp_path / "monotonic.arrow"
        pl.DataFrame({"temp_in_K": temperatures.tolist(), **{
            f"θ_{theta}": (1 + temperatures / 500 + theta / 10).tolist()
            for theta in range(1, 4)
        }}).write_ipc(path)
        generator = SimulatedDataGenerator([str(path)])
    diagrams = learner.diagram_generator.diagrams(2, data_generator=generator)
    assert len(diagrams) == 2
    expected = torch.empty((0, 2)) if monotonic else torch.tensor(
        [[200., 1.], [300., 2.], [400., 3.]]
    )
    for diagram in diagrams:
        assert diagram.sample.shape == (41, 3)
        assert torch.isfinite(diagram.sample).all()
        physical = generator.untransform_inputs(diagram.peaks)
        torch.testing.assert_close(physical, expected)
    torch.testing.assert_close(diagrams[0].peaks, diagrams[1].peaks)
    # Basic rendering must also accept an empty boundary and use physical units.
    fig, ax = plt.subplots()
    try:
        plot_diagrams(learner, diagrams[:1], ax)
        torch.testing.assert_close(torch.as_tensor(ax.collections[0].get_offsets().data), expected)
        output = tmp_path / "known_boundary.pdf"
        fig.savefig(output)
        assert_pdf(output)
    finally:
        plt.close(fig)


def test_scoring_distinguishes_exact_shifted_and_empty_boundaries(learner):
    """Score deliberately chosen diagrams so expected metrics are unambiguous."""
    # One AL step provides genuine sampled-point and cost metadata for scoring.
    learner.config.n_iters = 1
    learner.callbacks = []
    learner.optimize()
    grid = _generate_grid(learner).reset_index(drop=True)
    physical_truth = torch.tensor([[200., 1.], [300., 2.], [400., 3.]])
    truth = learner.data_generator.transform_inputs(physical_truth)
    true_hits = _assign_ridge_points_to_grid(learner, grid, truth)
    assert len(true_hits) == 3
    # Move each peak 100 K to a different temperature cell on the same theta slice.
    shifted = learner.data_generator.transform_inputs(
        torch.tensor([[300., 1.], [400., 2.], [300., 3.]])
    )
    returned = []
    for iteration, peaks in enumerate([truth, shifted, torch.empty((0, 2))]):
        learner._i = iteration
        diagram = DiagramOutput(peaks, None, None, None, None)
        metrics = _calc_score(
            learner, grid, set(true_hits.index), 3, [diagram, diagram],
            neighbor_distance={}, model_hyperparameters={},
        )
        expected = 1.0 if iteration == 0 else 0.0
        for name in ["precision", "recall", "f1"]:
            assert metrics[name] == expected
        assert metrics["mean_true_positives"] == (3 if iteration == 0 else 0)
        assert metrics["mean_false_negatives"] == (0 if iteration == 0 else 3)
        assert metrics["mean_false_positives"] == (3 if iteration == 1 else 0)
        assert torch.isfinite(torch.tensor(metrics["log_likelihood"]))
        returned.append(metrics)
    assert returned[0]["log_likelihood"] > returned[1]["log_likelihood"]
    assert returned[0]["log_likelihood"] > returned[2]["log_likelihood"]
    # Verify persistence too: appending must retain all three results in order.
    root = Path(learner.config.base_savepath)
    saved = pd.read_csv(root / "scoring/scores.csv")
    assert saved["iteration"].tolist() == [0, 1, 2]
    for name in ["precision", "recall", "f1", "log_likelihood"]:
        assert saved[name].tolist() == pytest.approx([row[name] for row in returned])
    assert_pdf(root / "scoring/score_metrics.pdf")
