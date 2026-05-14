"""Episode-regret breakdowns by bucket, oracle stop step, stop-step delta, and tree size.

Hosts the canonical "where is the controller losing?" plots: per-budget-
bucket regret bars, exact/over/undersearch fractions, accuracy by oracle
halt-step bin, mean-regret-vs-delta panels, the regret decomposition
plot, regret-by-tree-size, and the 2D (N_t, T_t) partial-dependence
heatmaps.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from cts.analysis._budgeted._shared import (
    ANALYSIS_FIGURE_DPI,
    FIGSIZE_SINGLE_TALL,
    FIGSIZE_SINGLE_WIDE,
    FIGSIZE_TWO_PANEL_WIDE,
    PARTIAL_DEPENDENCE_TIME_BIN_EDGES,
    _episode_error_type,
)
from cts.analysis._common import (
    _episode_regret_decomposition,
    _oracle_stop_bin_label,
)
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig


def _plot_episode_regret_by_bucket(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Bar plot of mean episode regret per starting-budget bucket. Returns the per-bucket summary."""
    buckets = sorted({episode["budget_bucket_name"] for episode in diagnostics})
    means = []
    stds = []
    counts = []
    for bucket in buckets:
        regrets = [float(episode["regret"]) for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        means.append(statistics.mean(regrets))
        stds.append(statistics.pstdev(regrets))
        counts.append(len(regrets))

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL, constrained_layout=True)
    ax.bar(buckets, means, yerr=stds, color="tab:red", alpha=0.8)
    ax.set_ylabel("Episode regret")
    ax.set_title("Regret by Starting-Budget Bucket")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return {bucket: {"mean_regret": mean, "std_regret": std, "episodes": count} for bucket, mean, std, count in zip(buckets, means, stds, counts)}


def _plot_stop_error_by_bucket(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Grouped bars showing the exact/oversearch/undersearch fractions per budget bucket."""
    buckets = sorted({episode["budget_bucket_name"] for episode in diagnostics})
    error_types = ["exact", "undersearch", "oversearch"]
    fractions: dict[str, list[float]] = {error_type: [] for error_type in error_types}
    summary: dict[str, Any] = {}
    for bucket in buckets:
        bucket_eps = [episode for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        counts = Counter(_episode_error_type(episode) for episode in bucket_eps)
        summary[bucket] = {error_type: counts.get(error_type, 0) / len(bucket_eps) for error_type in error_types}
        for error_type in error_types:
            fractions[error_type].append(summary[bucket][error_type])

    x = np.arange(len(buckets))
    width = 0.25
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_WIDE, constrained_layout=True)
    colors = {"exact": "tab:green", "undersearch": "tab:orange", "oversearch": "tab:red"}
    for idx, error_type in enumerate(error_types):
        ax.bar(x + (idx - 1) * width, fractions[error_type], width=width, label=error_type, color=colors[error_type])
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of episodes")
    ax.set_title("Predicted Stop vs Oracle Stop by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_regret_by_oracle_stop(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Mean regret as a function of the oracle's stop step; flags where the controller bleeds value."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for episode in diagnostics:
        grouped[int(episode["oracle_stop_step"])].append(float(episode["regret"]))
    xs = sorted(grouped)
    means = [statistics.mean(grouped[x]) for x in xs]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL, constrained_layout=True)
    ax.plot(xs, means, marker="o")
    ax.set_xlabel("Oracle stop step")
    ax.set_ylabel("Mean regret")
    ax.set_title("Regret by Oracle Stop Step")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return {str(x): {"mean_regret": means[i], "episodes": len(grouped[x])} for i, x in enumerate(xs)}


def _stop_step_delta_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate regret by stop-step delta overall, per bucket, and per direction.

    Stop-step delta = predicted - oracle. Used downstream by the regret-by-delta
    plot; this function only computes the dictionary, not the figure.
    """
    overall: dict[int, list[float]] = defaultdict(list)
    by_bucket: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    direction_by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for episode in diagnostics:
        delta = int(episode["predicted_stop_step"]) - int(episode["oracle_stop_step"])
        regret = float(episode["regret"])
        bucket = str(episode["budget_bucket_name"])
        overall[delta].append(regret)
        by_bucket[bucket][delta].append(regret)
        if delta == 0:
            direction = "exact"
        elif delta > 0:
            direction = "oversearch"
        else:
            direction = "undersearch"
        direction_by_bucket[bucket][direction].append(regret)

    overall_summary = {
        str(delta): {
            "episodes": len(regrets),
            "mean_regret": statistics.mean(regrets),
        }
        for delta, regrets in sorted(overall.items())
    }
    bucket_summary = {
        bucket: {
            str(delta): {
                "episodes": len(regrets),
                "mean_regret": statistics.mean(regrets),
            }
            for delta, regrets in sorted(delta_map.items())
        }
        for bucket, delta_map in by_bucket.items()
    }
    direction_summary = {
        bucket: {
            direction: {
                "episodes": len(regrets),
                "mean_regret": statistics.mean(regrets) if regrets else 0.0,
            }
            for direction, regrets in sorted(direction_map.items())
        }
        for bucket, direction_map in direction_by_bucket.items()
    }
    return {
        "overall": overall_summary,
        "by_bucket": bucket_summary,
        "direction_by_bucket": direction_summary,
    }


def _plot_regret_by_stop_step_delta(diagnostics: list[dict[str, Any]], out_path: Path) -> None:
    """Six-panel mean-regret-vs-delta plot, one per budget bucket plus overall."""
    buckets = ["overall", "scramble", "medium-small", "medium-large", "large", "very-large"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes_list = list(axes.flat)

    for ax, bucket in zip(axes_list, buckets):
        selected = diagnostics if bucket == "overall" else [episode for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        grouped: dict[int, list[float]] = defaultdict(list)
        for episode in selected:
            delta = int(episode["predicted_stop_step"]) - int(episode["oracle_stop_step"])
            grouped[delta].append(float(episode["regret"]))
        deltas = sorted(grouped)
        means = [statistics.mean(grouped[delta]) for delta in deltas]
        counts = [len(grouped[delta]) for delta in deltas]
        ax.bar(deltas, means, color="tab:red", alpha=0.75)
        ax.set_title(bucket)
        ax.set_xlabel("Predicted stop - Oracle stop")
        ax.set_ylabel("Mean regret")
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate bars with the episode count, but only when there's enough
        # data to trust the bar; sub-50 bins are too noisy to be worth labeling.
        for x, y, count in zip(deltas, means, counts):
            if count >= 50:
                ax.text(x, y, str(count), ha="center", va="bottom", fontsize=7, rotation=90)

    fig.suptitle("Regret by Stop-Step Error")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)


def _regret_decomposition_summary(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, Any]:
    """Aggregate regret decomposition by stop-step delta, overall and per bucket.

    Also exposes the max/mean absolute decomposition residual so the report
    can flag any subtle bug in the per-step cost computation.
    """
    decomposed = [_episode_regret_decomposition(episode, oracle_config) for episode in diagnostics]

    overall: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    by_bucket: dict[str, dict[int, list[dict[str, float | int | str]]]] = defaultdict(lambda: defaultdict(list))
    for row in decomposed:
        delta = int(row["delta"])
        bucket = str(row["budget_bucket_name"])
        overall[delta].append(row)
        by_bucket[bucket][delta].append(row)

    def summarize(groups: dict[int, list[dict[str, float | int | str]]]) -> dict[str, Any]:
        """Mean-aggregate each delta group into a per-delta stats dict."""
        summary: dict[str, Any] = {}
        for delta, rows in sorted(groups.items()):
            summary[str(delta)] = {
                "episodes": len(rows),
                "mean_regret": statistics.mean(float(row["saved_regret"]) for row in rows),
                "mean_halt_reward_term": statistics.mean(float(row["halt_reward_term"]) for row in rows),
                "mean_maintenance_term": statistics.mean(float(row["maintenance_term"]) for row in rows),
                "mean_time_term": statistics.mean(float(row["time_term"]) for row in rows),
                "mean_decomposed_regret": statistics.mean(float(row["decomposed_regret"]) for row in rows),
                "max_abs_residual": max(abs(float(row["residual"])) for row in rows),
            }
        return summary

    residuals = [abs(float(row["residual"])) for row in decomposed]
    return {
        "overall": summarize(overall),
        "by_bucket": {bucket: summarize(groups) for bucket, groups in sorted(by_bucket.items())},
        "max_abs_residual": max(residuals) if residuals else 0.0,
        "mean_abs_residual": statistics.mean(residuals) if residuals else 0.0,
    }


def _plot_regret_decomposition_by_delta(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
    out_path: Path,
) -> None:
    """Four-panel plot showing how regret decomposes by stop-step delta, per selected bucket."""
    selected_buckets = ["overall", "medium-large", "large", "very-large"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes_list = list(axes.flat)

    for ax, bucket in zip(axes_list, selected_buckets):
        selected = diagnostics if bucket == "overall" else [episode for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
        for episode in selected:
            row = _episode_regret_decomposition(episode, oracle_config)
            grouped[int(row["delta"])].append(row)

        deltas = sorted(grouped)
        halt_terms = [statistics.mean(float(row["halt_reward_term"]) for row in grouped[delta]) for delta in deltas]
        maintenance_terms = [statistics.mean(float(row["maintenance_term"]) for row in grouped[delta]) for delta in deltas]
        time_terms = [statistics.mean(float(row["time_term"]) for row in grouped[delta]) for delta in deltas]
        regrets = [statistics.mean(float(row["saved_regret"]) for row in grouped[delta]) for delta in deltas]
        counts = [len(grouped[delta]) for delta in deltas]

        ax.plot(deltas, regrets, marker="o", linewidth=2, color="black", label="total regret")
        ax.plot(deltas, halt_terms, marker="o", linewidth=1.5, label="halt-reward term")
        ax.plot(deltas, maintenance_terms, marker="o", linewidth=1.5, label="maintenance term")
        ax.plot(deltas, time_terms, marker="o", linewidth=1.5, label="time term")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(bucket)
        ax.set_xlabel("Predicted stop - Oracle stop")
        ax.set_ylabel("Mean contribution")
        ax.grid(True, alpha=0.3)
        for x, y, count in zip(deltas, regrets, counts):
            if count >= 50:
                ax.text(x, y, str(count), ha="center", va="bottom", fontsize=7, rotation=90)

    axes_list[0].legend(loc="upper left")
    fig.suptitle("Regret Decomposition by Stop-Step Error")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)


def _plot_regret_by_tree_size(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Episode regret bucketed by initial tree size (equal-mass quintiles of the tree-size distribution)."""
    episode_rows = []
    for episode in diagnostics:
        first_tree_size = int(episode["tree_sizes"][0])
        episode_rows.append((first_tree_size, float(episode["regret"])))
    sizes = np.array([row[0] for row in episode_rows], dtype=np.int64)
    regrets = np.array([row[1] for row in episode_rows], dtype=np.float32)
    # Use quintile cuts so each bin has roughly the same number of episodes.
    quantiles = np.unique(np.percentile(sizes, [0, 20, 40, 60, 80, 100]).astype(int))
    if len(quantiles) < 2:
        quantiles = np.array([sizes.min(), sizes.max()])
    labels = []
    means = []
    summary: dict[str, Any] = {}
    for lower, upper in zip(quantiles[:-1], quantiles[1:]):
        if upper <= lower:
            continue
        # Right-edge inclusion only on the final bin so the boundary tree size
        # isn't double-counted into adjacent bins.
        mask = (sizes >= lower) & (sizes <= upper if upper == quantiles[-1] else sizes < upper)
        bucket_regrets = regrets[mask]
        if bucket_regrets.size == 0:
            continue
        label = f"[{int(lower)}, {int(upper)}{'+' if upper == quantiles[-1] else ')'}"
        labels.append(label)
        means.append(float(bucket_regrets.mean()))
        summary[label] = {"mean_regret": float(bucket_regrets.mean()), "episodes": int(bucket_regrets.size)}
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_TALL, constrained_layout=True)
    ax.bar(labels, means, color="tab:purple")
    ax.set_ylabel("Mean regret")
    ax.set_title("Episode Regret by Initial Tree Size")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_partial_dependence_heatmaps(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Mean predicted advantage and sign accuracy as a 2D heat-map over (N_t, T_t) bins."""
    time_bins = PARTIAL_DEPENDENCE_TIME_BIN_EDGES
    tree_sizes = np.array([row["tree_size"] for row in state_rows], dtype=np.float32)
    # Quartile cuts on the tree-size axis so the bins aren't dominated by
    # the long tail of large trees.
    tree_edges = np.unique(np.percentile(tree_sizes, [0, 25, 50, 75, 100]).astype(int))
    if len(tree_edges) < 2:
        tree_edges = np.array([int(tree_sizes.min()), int(tree_sizes.max())])
    pred_grid = np.full((len(tree_edges) - 1, len(time_bins) - 1), np.nan)
    acc_grid = np.full((len(tree_edges) - 1, len(time_bins) - 1), np.nan)
    count_grid = np.zeros((len(tree_edges) - 1, len(time_bins) - 1), dtype=int)

    for i, (n_lo, n_hi) in enumerate(zip(tree_edges[:-1], tree_edges[1:])):
        for j, (t_lo, t_hi) in enumerate(zip(time_bins[:-1], time_bins[1:])):
            selected = [
                row
                for row in state_rows
                if n_lo <= row["tree_size"] <= n_hi
                and t_lo <= row["time_budget"] < t_hi
            ]
            if not selected:
                continue
            pred_grid[i, j] = statistics.mean(row["predicted_advantage"] for row in selected)
            acc_grid[i, j] = statistics.mean(1.0 if row["sign_correct"] else 0.0 for row in selected)
            count_grid[i, j] = len(selected)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL_WIDE, constrained_layout=True)
    im0 = axes[0].imshow(pred_grid, aspect="auto", cmap="coolwarm")
    axes[0].set_title("Mean Predicted Advantage")
    im1 = axes[1].imshow(acc_grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Sign Accuracy")
    xlabels = [f"[{time_bins[j]}, {time_bins[j + 1]})" if math.isfinite(time_bins[j + 1]) else f"[{time_bins[j]}, inf)" for j in range(len(time_bins) - 1)]
    ylabels = [f"[{int(tree_edges[i])}, {int(tree_edges[i + 1])}]" for i in range(len(tree_edges) - 1)]
    for ax in axes:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("T_t bin")
        ax.set_ylabel("N_t bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)

    return {
        "time_bins": xlabels,
        "tree_bins": ylabels,
        "mean_predicted_advantage": pred_grid.tolist(),
        "sign_accuracy": acc_grid.tolist(),
        "counts": count_grid.tolist(),
    }


def _oracle_stop_distribution(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    """Count of episodes by oracle stop step. Sanity-checks the dataset's halt-step balance."""
    counts = Counter(int(episode["oracle_stop_step"]) for episode in diagnostics)
    return {str(step): counts[step] for step in sorted(counts)}
