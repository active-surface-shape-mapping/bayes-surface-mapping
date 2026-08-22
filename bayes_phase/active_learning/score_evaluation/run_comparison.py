"""Utilities for comparing acquisition functions."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


DEFAULT_N_ITERS = 300
DEFAULT_METRIC = "f1"
MARKER_INTERVAL = 50

METRIC_CONFIGS: dict[str, dict[str, Any]] = {
    "f1": {
        "label": "F1 score",
        "higher_is_better": True,
        "threshold": 0.5,
    },
    "mean_abs_peak_count_error": {
        "label": "absolute peak-count error",
        "higher_is_better": False,
        "threshold": 1.0,
    },
    "same_slice_temp_error_mean_K": {
        "label": "predicted ridge point temperature error (K)",
        "higher_is_better": False,
        "threshold": 50.0,
    },
    "log_likelihood": {
        "label": "log likelihood",
        "higher_is_better": True,
        "threshold": None,
    },
}

METRICS = tuple(METRIC_CONFIGS)
PLOT_METRICS = (
    "f1",
    "mean_abs_peak_count_error",
    "same_slice_temp_error_mean_K",
)

METHOD_COLORS = {
    "Random": "orange",
    "EIG": "red",
    "Sobol": "black",
    "Cell": "purple",
    "EFDI": "blue",
}

METHOD_MARKERS = {
    "Random": "h",
    "EIG": "s",
    "Sobol": "D",
    "Cell": "o",
    "EFDI": "^",
}


def method_color(method_name: str) -> str | None:
    """Return the configured color for a method label."""
    for prefix, color in METHOD_COLORS.items():
        if method_name.startswith(prefix):
            return color
    return None


def method_marker(method_name: str) -> str | None:
    """Return the configured marker for a method label."""
    for prefix, marker in METHOD_MARKERS.items():
        if method_name.startswith(prefix):
            return marker
    return None


def method_linestyle(method_name: str) -> str:
    """Return a line style, distinguishing kernel variants when present."""
    if "(RBF)" in method_name:
        return "dotted"
    return "-"


def collect_scores_for_run(
    run_name: str,
    run_dir: str | Path,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Collect one score metric from all numbered seed directories."""
    run_dir = Path(run_dir)
    output_columns = ["iteration", metric, "run", "seed"]

    if not run_dir.exists():
        print(f"Missing run directory: {run_dir}")
        return pd.DataFrame(columns=output_columns)

    seed_dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )

    dfs: list[pd.DataFrame] = []

    for seed_dir in seed_dirs:
        scores_path = seed_dir / "scoring" / "scores.csv"

        if not scores_path.exists():
            print(f"Missing scores file: {scores_path}")
            continue

        try:
            df = pd.read_csv(scores_path, usecols=["iteration", metric])
        except ValueError:
            available_columns = pd.read_csv(scores_path, nrows=0).columns.tolist()
            print(
                f"Missing metric {metric!r} in {scores_path}. "
                f"Available columns: {available_columns}"
            )
            continue

        df["run"] = run_name
        df["seed"] = int(seed_dir.name)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=output_columns)

    return pd.concat(dfs, ignore_index=True)


def collect_all_scores(
    run_folders: Mapping[str, str | Path],
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Collect one score metric for all acquisition methods."""
    all_dfs: list[pd.DataFrame] = []

    for run_name, run_dir in run_folders.items():
        run_df = collect_scores_for_run(
            run_name=run_name,
            run_dir=run_dir,
            metric=metric,
        )
        if not run_df.empty:
            all_dfs.append(run_df)

    if not all_dfs:
        return pd.DataFrame(columns=["iteration", metric, "run", "seed"])

    return pd.concat(all_dfs, ignore_index=True)


def summarize_scores_for_plotting(
    results_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Summarize a metric by acquisition method and iteration."""
    required_columns = {"run", "iteration", metric}
    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise ValueError(
            f"results_df is missing required columns: {sorted(missing_columns)}"
        )

    summary = (
        results_df.groupby(["run", "iteration"])[metric]
        .agg(
            mean="mean",
            median="median",
            sd="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n="count",
        )
        .reset_index()
    )

    summary["iqr"] = summary["q75"] - summary["q25"]
    summary["metric"] = metric
    return summary


def bootstrap_mean_se(
    results_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    n_boot: int = 10_000,
    random_state: int = 0,
    n_iters: int | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Estimate bootstrap SEs by resampling complete seed trajectories."""
    required_columns = {"run", "seed", "iteration", metric}
    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise ValueError(
            f"results_df is missing required columns: {sorted(missing_columns)}"
        )

    if n_boot < 2:
        raise ValueError("n_boot must be at least 2.")
    if n_iters is not None and n_iters < 1:
        raise ValueError("n_iters must be at least 1 when provided.")

    df = results_df.copy()
    if n_iters is not None:
        df = df[df["iteration"] < n_iters]

    rng = np.random.default_rng(random_state)
    bootstrap_dfs: list[pd.DataFrame] = []

    for run, run_df in df.groupby("run"):
        trajectories = (
            run_df.pivot(index="seed", columns="iteration", values=metric)
            .sort_index(axis=1)
        )

        # Explicit reindexing also detects a tail missing from every seed.
        if n_iters is not None:
            trajectories = trajectories.reindex(columns=range(n_iters))

        incomplete_mask = trajectories.isna().any(axis=1)

        if incomplete_mask.any():
            incomplete_seeds = trajectories.index[incomplete_mask].tolist()

            if require_complete:
                raise ValueError(
                    f"Run {run!r} has incomplete trajectories within the "
                    f"requested iteration range. Incomplete seeds: "
                    f"{incomplete_seeds}"
                )

            print(
                f"Run {run!r}: dropping incomplete seeds "
                f"{incomplete_seeds}"
            )
            trajectories = trajectories.loc[~incomplete_mask]

        if trajectories.empty:
            print(
                f"Run {run!r}: no complete seed trajectories remain; "
                "skipping bootstrap."
            )
            continue

        values = trajectories.to_numpy(dtype=float)
        n_seeds, n_iterations = values.shape
        bootstrap_means = np.empty((n_boot, n_iterations), dtype=float)

        for b in range(n_boot):
            sampled_idxs = rng.integers(0, n_seeds, size=n_seeds)
            bootstrap_means[b] = values[sampled_idxs].mean(axis=0)

        bootstrap_se = bootstrap_means.std(axis=0, ddof=1)

        bootstrap_dfs.append(
            pd.DataFrame(
                {
                    "run": run,
                    "iteration": trajectories.columns,
                    "bootstrap_se": bootstrap_se,
                }
            )
        )

    if not bootstrap_dfs:
        return pd.DataFrame(columns=["run", "iteration", "bootstrap_se"])

    return pd.concat(bootstrap_dfs, ignore_index=True)


def prepare_run_comparison(
    run_folders: Mapping[str, str | Path],
    metrics: Iterable[str] = METRICS,
    *,
    n_boot: int = 10_000,
    random_state: int = 0,
    n_iters: int | None = DEFAULT_N_ITERS,
    require_complete: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load, summarize, and bootstrap all requested metrics."""
    results_dfs: dict[str, pd.DataFrame] = {}
    summary_dfs: dict[str, pd.DataFrame] = {}

    for metric in metrics:
        results_df = collect_all_scores(run_folders, metric=metric)
        summary_df = summarize_scores_for_plotting(results_df, metric=metric)
        bootstrap_df = bootstrap_mean_se(
            results_df,
            metric=metric,
            n_boot=n_boot,
            random_state=random_state,
            n_iters=n_iters,
            require_complete=require_complete,
        )

        results_dfs[metric] = results_df
        summary_dfs[metric] = summary_df.merge(
            bootstrap_df,
            on=["run", "iteration"],
            how="left",
        )

    return results_dfs, summary_dfs


def plot_metric_by_method(
    summary_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    methods: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    n_iters: int | None = DEFAULT_N_ITERS,
    threshold: float | None = None,
    center: str = "median",
    band: str | None = None,
    n_std: float = 1.0,
    marker_interval: int = MARKER_INTERVAL,
) -> tuple[Figure, Axes]:
    """Plot a summarized score trajectory for multiple methods."""
    if center not in {"mean", "median"}:
        raise ValueError("center must be 'mean' or 'median'.")

    if band not in {None, "iqr", "std", "bootstrap_se"}:
        raise ValueError(
            "band must be None, 'iqr', 'std', or 'bootstrap_se'."
        )

    if band == "iqr" and center != "median":
        raise ValueError("band='iqr' should be used with center='median'.")

    if band in {"std", "bootstrap_se"} and center != "mean":
        raise ValueError(f"band={band!r} should be used with center='mean'.")

    if marker_interval < 1:
        raise ValueError("marker_interval must be at least 1.")

    metric_config = METRIC_CONFIGS.get(
        metric,
        {
            "label": metric.replace("_", " ").title(),
            "higher_is_better": True,
            "threshold": None,
        },
    )
    metric_label = metric_config["label"]

    df = summary_df.copy()
    if n_iters is not None:
        df = df[df["iteration"] < n_iters]

    if methods is None:
        methods = sorted(df["run"].unique())

    if threshold is None:
        threshold = metric_config.get("threshold")

    if title is None:
        if band == "iqr":
            title = f"Median {metric_label} with IQR by iteration"
        elif band == "std":
            title = f"Mean {metric_label} with ±{n_std:g} SD by iteration"
        elif band == "bootstrap_se":
            title = (
                f"Mean {metric_label} with ±{n_std:g} bootstrap SE "
                "by iteration"
            )
        else:
            title = f"{center.capitalize()} {metric_label} by iteration"

    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        df_method = (
            df[df["run"] == method]
            .sort_values("iteration")
        )

        if df_method.empty:
            print(f"No data found for method: {method}")
            continue

        line = ax.plot(
            df_method["iteration"],
            df_method[center],
            label=method,
            linewidth=1,
            color=method_color(method),
            linestyle=method_linestyle(method),
            marker=method_marker(method),
            markevery=marker_interval,
            markersize=6,
        )[0]

        if band == "iqr":
            lower = df_method["q25"]
            upper = df_method["q75"]
        elif band == "std":
            lower = df_method["mean"] - n_std * df_method["sd"]
            upper = df_method["mean"] + n_std * df_method["sd"]
        elif band == "bootstrap_se":
            lower = (
                df_method["mean"]
                - n_std * df_method["bootstrap_se"]
            )
            upper = (
                df_method["mean"]
                + n_std * df_method["bootstrap_se"]
            )
        else:
            lower = upper = None

        if lower is not None:
            ax.fill_between(
                df_method["iteration"],
                lower,
                upper,
                alpha=0.2,
                color=line.get_color(),
            )

    if threshold is not None:
        ax.axhline(
            y=threshold,
            linestyle="--",
            linewidth=1.5,
            color="grey",
            label=f"{metric_label} = {threshold:g}",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"{center.capitalize()} {metric_label}")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, ax


def compute_average_metric_difference_between(
    summary_df: pd.DataFrame,
    run_a: str,
    run_b: str,
    metric: str = DEFAULT_METRIC,
    n_iters: int | None = 500,
    value_col: str = "mean",
    threshold: float | None = None,
    higher_is_better: bool | None = None,
) -> dict[str, Any]:
    """Compare two methods across the included iterations."""
    metric_config = METRIC_CONFIGS.get(
        metric,
        {
            "label": metric,
            "higher_is_better": True,
            "threshold": None,
        },
    )

    if higher_is_better is None:
        higher_is_better = metric_config["higher_is_better"]

    if threshold is None:
        threshold = metric_config.get("threshold")

    df = summary_df.copy()
    if n_iters is not None:
        df = df[df["iteration"] < n_iters]

    if value_col not in df.columns:
        raise ValueError(f"Missing value column {value_col!r}.")

    wide = (
        df.pivot(index="iteration", columns="run", values=value_col)
        .sort_index()
    )

    missing_runs = {run_a, run_b} - set(wide.columns)
    if missing_runs:
        raise ValueError(
            f"Missing runs in summary dataframe: {sorted(missing_runs)}"
        )

    paired = wide[[run_a, run_b]].dropna()
    diff = paired[run_a] - paired[run_b]

    def first_iteration_at_threshold(series: pd.Series) -> int | None:
        if threshold is None:
            return None

        hits = (
            series[series >= threshold]
            if higher_is_better
            else series[series <= threshold]
        )

        if hits.empty:
            return None
        return int(hits.index.min())

    iter_a = first_iteration_at_threshold(wide[run_a])
    iter_b = first_iteration_at_threshold(wide[run_b])

    if iter_a is None or iter_b is None:
        raw_speedup = None
        percent_speedup = None
    else:
        raw_speedup = iter_b - iter_a
        percent_speedup = (
            100 * raw_speedup / iter_b
            if iter_b != 0
            else None
        )

    if higher_is_better:
        interpretation = (
            "Positive differences favor run_a; "
            "negative differences favor run_b."
        )
    else:
        interpretation = (
            "Negative differences favor run_a; "
            "positive differences favor run_b."
        )

    return {
        "metric": metric,
        "comparison": f"{run_a} - {run_b}",
        "value_column": value_col,
        "mean_difference": diff.mean(),
        "median_difference": diff.median(),
        "sd_difference": diff.std(),
        "n_iterations": diff.notna().sum(),
        "higher_is_better": higher_is_better,
        "difference_interpretation": interpretation,
        "threshold": threshold,
        f"{run_a}_iterations_to_threshold": iter_a,
        f"{run_b}_iterations_to_threshold": iter_b,
        "raw_iteration_speedup": raw_speedup,
        "percent_speedup_vs_run_b": percent_speedup,
    }


def compare_method_pairs(
    summary_dfs: Mapping[str, pd.DataFrame],
    comparisons: Iterable[tuple[str, str]],
    metrics: Iterable[str] = PLOT_METRICS,
    *,
    n_iters: int | None = DEFAULT_N_ITERS,
    value_col: str = "mean",
) -> list[dict[str, Any]]:
    """Compute the notebook-style summary for method pairs and metrics."""
    results: list[dict[str, Any]] = []

    for run_a, run_b in comparisons:
        for metric in metrics:
            if metric not in summary_dfs:
                raise KeyError(f"No summary dataframe found for metric {metric!r}.")

            results.append(
                compute_average_metric_difference_between(
                    summary_df=summary_dfs[metric],
                    run_a=run_a,
                    run_b=run_b,
                    metric=metric,
                    n_iters=n_iters,
                    value_col=value_col,
                )
            )

    return results


def format_comparison_report(
    comparisons: Iterable[Mapping[str, Any]],
) -> str:
    """Format comparison dictionaries as a concise plain-text report."""
    sections: list[str] = []

    for result in comparisons:
        lines = [
            f"{result['comparison']} — {result['metric']}",
            f"Mean difference: {result['mean_difference']:.6g}",
            f"Median difference: {result['median_difference']:.6g}",
            f"SD of difference: {result['sd_difference']:.6g}",
            f"Iterations compared: {result['n_iterations']}",
        ]

        threshold = result.get("threshold")
        if threshold is not None:
            run_a, run_b = result["comparison"].split(" - ", maxsplit=1)
            iter_a = result.get(f"{run_a}_iterations_to_threshold")
            iter_b = result.get(f"{run_b}_iterations_to_threshold")

            lines.extend(
                [
                    f"Threshold: {threshold:g}",
                    f"{run_a} iteration to threshold: {iter_a}",
                    f"{run_b} iteration to threshold: {iter_b}",
                    (
                        "Iteration speedup vs run_b: "
                        f"{result['raw_iteration_speedup']}"
                        if result["raw_iteration_speedup"] is not None
                        else (
                            "Iteration speedup vs run_b: None - one or more comparators "
                            "may not have reached the specified threshold."
                        )
                    ),
                    (
                        "Percent speedup vs run_b: "
                        f"{result['percent_speedup_vs_run_b']}"
                        if result["percent_speedup_vs_run_b"] is not None
                        else (
                            "Percent speedup vs run_b: None - one or more comparators "
                            "may not have reached the specified threshold."
                        )
                    ),
                ]
            )

        lines.append(result["difference_interpretation"])
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
