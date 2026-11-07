"""Exercise the CSV-to-statistics-to-report pipeline with known trajectories."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bayes_mapping.active_learning.score_evaluation import run_comparison as comparison


@pytest.fixture
def runs(tmp_path):
    folders = {}
    curves = {
        "EIG": ([0.3, 0.6, 0.8], [100., 40., 20.]),
        "Random": ([0.1, 0.3, 0.6], [120., 80., 40.]),
    }
    for method, (f1, error) in curves.items():
        folders[method] = tmp_path / method
        for seed, offset in enumerate([-0.1, 0., 0.1]):
            path = folders[method] / str(seed) / "scoring/scores.csv"
            path.parent.mkdir(parents=True)
            # Shuffle rows: iteration labels, not CSV row order, determine alignment.
            pd.DataFrame({
                "iteration": [0, 1, 2], "f1": np.array(f1) + offset,
                "same_slice_temp_error_mean_K": np.array(error) + 10 * offset,
            }).iloc[[2, 0, 1]].to_csv(path, index=False)
        # Unrelated folders and incomplete directory layouts must not become seeds.
        (folders[method] / "notes").mkdir()
        (folders[method] / "99").mkdir()
    return folders


def test_comparison_statistics_reports_and_plots(runs, tmp_path):
    metrics = ["f1", "same_slice_temp_error_mean_K"]
    results, summaries = comparison.prepare_run_comparison(
        runs, metrics, n_iters=3, n_boot=1000, random_state=42,
    )
    for metric in metrics:
        assert len(results[metric]) == 18  # Two methods, three seeds, three iterations.
        assert set(results[metric]["seed"]) == {0, 1, 2}
        assert len(summaries[metric]) == 6
        scale = 1 if metric == "f1" else 10
        for method in runs:
            rows = summaries[metric].query("run == @method").sort_values("iteration")
            expected = {
                "f1": {"EIG": [0.3, 0.6, 0.8], "Random": [0.1, 0.3, 0.6]},
                "same_slice_temp_error_mean_K": {
                    "EIG": [100., 40., 20.], "Random": [120., 80., 40.],
                },
            }[metric][method]
            np.testing.assert_allclose(rows["mean"], expected)
            np.testing.assert_allclose(rows["median"], expected)
            np.testing.assert_allclose(rows["sd"], 0.1 * scale)
            np.testing.assert_allclose(rows["q25"], np.array(expected) - 0.05 * scale)
            np.testing.assert_allclose(rows["q75"], np.array(expected) + 0.05 * scale)
            np.testing.assert_allclose(rows["iqr"], 0.1 * scale)
            assert rows["n"].tolist() == [3, 3, 3]
            # All seeds have parallel trajectories: whole-seed resampling must
            # give the same SE at every iteration, close to its analytic value.
            se = rows["bootstrap_se"].to_numpy()
            np.testing.assert_allclose(se, se[0], atol=1e-12)
            assert se[0] == pytest.approx(np.sqrt(2 / 900) * scale, rel=0.12)
        # Inspect plotted data as well as writing a PDF; an empty plot is not a pass.
        for center, band in [("median", "iqr"), ("mean", "std"), ("mean", "bootstrap_se")]:
            fig, ax = comparison.plot_metric_by_method(
                summaries[metric], metric, methods=["EIG", "Random"],
                center=center, band=band, n_iters=2,
            )
            try:
                assert len(ax.collections) == 2
                for line, method in zip(ax.lines[:2], runs, strict=True):
                    rows = summaries[metric].query("run == @method").sort_values("iteration")
                    np.testing.assert_array_equal(line.get_xdata(), [0, 1])
                    np.testing.assert_allclose(line.get_ydata(), rows[center].iloc[:2])
                output = tmp_path / f"{metric}_{band}.pdf"
                fig.savefig(output)
                assert output.read_bytes().startswith(b"%PDF-")
            finally:
                plt.close(fig)
    comparisons = comparison.compare_method_pairs(summaries, [("EIG", "Random")], metrics, n_iters=3)
    for result, difference in zip(comparisons, [0.7 / 3, -80 / 3], strict=True):
        assert result["mean_difference"] == pytest.approx(difference)
        assert result["n_iterations"] == 3
        assert result["EIG_iterations_to_threshold"] == 1
        assert result["Random_iterations_to_threshold"] == 2
        assert result["raw_iteration_speedup"] == 1
        assert result["percent_speedup_vs_run_b"] == 50
    assert comparisons[0]["higher_is_better"] is True
    assert comparisons[1]["higher_is_better"] is False
    report = comparison.format_comparison_report(comparisons)
    assert "EIG - Random — f1" in report
    assert "EIG - Random — same_slice_temp_error_mean_K" in report
    assert "Iterations compared: 3" in report
    assert "EIG iteration to threshold: 1" in report


def test_comparison_detects_incomplete_seed_trajectories(runs):
    # Remove a middle observation from one seed, keeping the other seeds intact.
    path = runs["EIG"] / "1/scoring/scores.csv"
    frame = pd.read_csv(path)
    frame[frame["iteration"] != 1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="incomplete trajectories"):
        comparison.prepare_run_comparison(runs, ["f1"], n_iters=3, n_boot=100)
    # Opting into incomplete data drops whole seeds for bootstrapping, rather
    # than mixing different seed populations across iterations.
    results, summaries = comparison.prepare_run_comparison(
        runs, ["f1"], n_iters=3, n_boot=100, require_complete=False,
    )
    complete = results["f1"].query("run != 'EIG' or seed != 1")
    expected = comparison.bootstrap_mean_se(complete, n_iters=3, n_boot=100)
    actual = summaries["f1"][["run", "iteration", "bootstrap_se"]]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    # Descriptive statistics still use every available observation; only the
    # bootstrap requires complete trajectories. Make that distinction explicit.
    assert summaries["f1"].query("run == 'EIG'")["n"].tolist() == [3, 2, 3]
    # Also reject a requested tail missing from every seed, not just missing rows
    # that can be detected by comparing seeds with one another.
    with pytest.raises(ValueError, match="incomplete trajectories"):
        comparison.prepare_run_comparison(runs, ["f1"], n_iters=4, n_boot=100)
