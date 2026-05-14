"""Calibration plots: accuracy by halt bin, predicted-vs-target, false rates by time.

Covers state-level diagnostics that ask "how well-calibrated are the
controller's predicted advantages?" — sign accuracy / false-action rates
sliced by time budget, the per-bucket predicted-vs-target scatter, the
|predicted|-binned calibration curve, and the five-panel accuracy /
share / regret / decomposition / error view by oracle halt-step bin.
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
    CALIBRATION_BIN_EDGES,
    FIGSIZE_SINGLE_WIDE,
    FIGSIZE_TWO_PANEL_WIDE,
)
from cts.analysis._common import (
    _episode_regret_decomposition,
    _oracle_stop_bin_label,
)
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig


def _plot_accuracy_by_oracle_stop_bin(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
    out_path: Path,
) -> dict[str, Any]:
    """Five-panel breakdown of accuracy, episode share, regret, decomposition, and stop-step error by oracle-stop bin.

    The canonical "where is the controller losing?" view: stacks first-action
    accuracy on top of exact-stop accuracy, then shows what fraction of
    episodes fall into each bin, the mean regret per bin, the halt/time
    contributions to that regret, and the average over/undersearch magnitude.
    """
    bin_order = ["0", "1", "2", "3", "4-7", "8-15", "16+"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in diagnostics:
        grouped[_oracle_stop_bin_label(int(episode["oracle_stop_step"]))].append(episode)

    exact_rates = []
    first_action_rates = []
    episode_shares = []
    mean_regrets = []
    mean_halt_terms = []
    mean_time_terms = []
    mean_over_errors = []
    mean_under_errors = []
    summary: dict[str, Any] = {}
    total_episodes = len(diagnostics)
    for bin_label in bin_order:
        episodes = grouped.get(bin_label, [])
        count = len(episodes)
        if count == 0:
            # Empty bin: pin all aggregates to zero so the bar still renders.
            exact_rate = 0.0
            first_action_rate = 0.0
            mean_regret = 0.0
            mean_halt_term = 0.0
            mean_time_term = 0.0
            mean_over_error = 0.0
            mean_under_error = 0.0
        else:
            exact_rate = statistics.mean(
                int(int(episode["predicted_stop_step"]) == int(episode["oracle_stop_step"]))
                for episode in episodes
            )
            # First-action accuracy: did the controller agree with the oracle
            # on whether to halt at step 0? Coarser than exact-stop and the
            # metric the trainer optimizes most directly.
            first_action_rate = statistics.mean(
                int((int(episode["predicted_stop_step"]) == 0) == (int(episode["oracle_stop_step"]) == 0))
                for episode in episodes
            )
            decomposed = [_episode_regret_decomposition(episode, oracle_config) for episode in episodes]
            mean_regret = statistics.mean(float(episode["regret"]) for episode in episodes)
            mean_halt_term = statistics.mean(float(row["halt_reward_term"]) for row in decomposed)
            mean_time_term = statistics.mean(float(row["time_term"]) for row in decomposed)
            deltas = [
                int(episode["predicted_stop_step"]) - int(episode["oracle_stop_step"])
                for episode in episodes
            ]
            over_errors = [delta for delta in deltas if delta > 0]
            under_errors = [-delta for delta in deltas if delta < 0]
            mean_over_error = statistics.mean(over_errors) if over_errors else 0.0
            mean_under_error = statistics.mean(under_errors) if under_errors else 0.0
        exact_rates.append(exact_rate)
        first_action_rates.append(first_action_rate)
        episode_shares.append(count / total_episodes if total_episodes else 0.0)
        mean_regrets.append(mean_regret)
        mean_halt_terms.append(mean_halt_term)
        mean_time_terms.append(mean_time_term)
        mean_over_errors.append(mean_over_error)
        mean_under_errors.append(mean_under_error)
        summary[bin_label] = {
            "episodes": count,
            "episode_fraction": count / total_episodes if total_episodes else 0.0,
            "exact_stop_step_accuracy": exact_rate,
            "first_action_accuracy": first_action_rate,
            "mean_regret": mean_regret,
            "mean_halt_reward_term": mean_halt_term,
            "mean_time_term": mean_time_term,
            "mean_oversearch_error": mean_over_error,
            "mean_undersearch_error": mean_under_error,
        }

    # Stacked five-panel layout, shared x-axis so the bin labels line up
    # across accuracy / share / regret / decomposition / error panels.
    x = np.arange(len(bin_order))
    fig, (ax_acc, ax_share, ax_regret, ax_decomp, ax_error) = plt.subplots(
        5,
        1,
        figsize=(9, 15),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1.5, 2, 2, 2]},
        sharex=True,
    )
    width = 0.36
    ax_acc.bar(x - width / 2, exact_rates, width=width, color="tab:blue", label="exact stop")
    ax_acc.bar(x + width / 2, first_action_rates, width=width, color="tab:orange", label="first action")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy by Oracle Halt-Step Bin")
    ax_acc.legend()
    ax_acc.grid(True, axis="y", alpha=0.3)

    ax_share.bar(x, episode_shares, color="tab:gray", alpha=0.8)
    ax_share.set_ylabel("Episode share")
    ax_share.set_title("Episode Share by Oracle Halt-Step Bin")
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_share.grid(True, axis="y", alpha=0.3)

    ax_regret.bar(x, mean_regrets, color="tab:red", alpha=0.8)
    ax_regret.set_ylabel("Mean regret")
    ax_regret.set_title("Mean Regret by Oracle Halt-Step Bin")
    ax_regret.grid(True, axis="y", alpha=0.3)

    ax_decomp.plot(x, mean_regrets, marker="o", linewidth=2, color="black", label="total regret")
    ax_decomp.plot(x, mean_halt_terms, marker="o", linewidth=1.5, color="tab:purple", label="halt-reward term")
    ax_decomp.plot(x, mean_time_terms, marker="o", linewidth=1.5, color="tab:green", label="time term")
    ax_decomp.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_decomp.set_xticks(x)
    ax_decomp.set_xticklabels(bin_order)
    ax_decomp.set_xlabel("Oracle halt-step bin")
    ax_decomp.set_ylabel("Mean contribution")
    ax_decomp.set_title("Regret Decomposition by Oracle Halt-Step Bin")
    ax_decomp.legend()
    ax_decomp.grid(True, axis="y", alpha=0.3)

    ax_error.plot(x, mean_over_errors, marker="o", linewidth=1.8, color="tab:red", label="oversearch error")
    ax_error.plot(x, mean_under_errors, marker="o", linewidth=1.8, color="tab:blue", label="undersearch error")
    ax_error.set_xticks(x)
    ax_error.set_xticklabels(bin_order)
    ax_error.set_xlabel("Oracle halt-step bin")
    ax_error.set_ylabel("Mean stop-step error")
    ax_error.set_title("Mean Halt-Step Error by Oracle Halt-Step Bin")
    ax_error.legend()
    ax_error.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_predicted_vs_target(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Six-panel scatter of predicted vs target advantage per bucket; reports per-bucket correlation."""
    buckets = ["overall"] + sorted({row["budget_bucket_name"] for row in state_rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes_list = list(axes.flat)
    summary: dict[str, Any] = {}
    for ax, bucket in zip(axes_list, buckets):
        selected = state_rows if bucket == "overall" else [row for row in state_rows if row["budget_bucket_name"] == bucket]
        xs = np.array([row["target_advantage"] for row in selected], dtype=np.float32)
        ys = np.array([row["predicted_advantage"] for row in selected], dtype=np.float32)
        if xs.size == 0:
            ax.set_visible(False)
            continue
        # Use the 99th-percentile absolute value as the axis limit so a few
        # outliers don't squish the bulk of the data into the origin.
        limit = float(np.percentile(np.abs(np.concatenate([xs, ys])), 99))
        limit = max(limit, 0.5)
        # Downsample for plot clarity (and PDF file size); rendering 100k+
        # points per panel is both slow and visually meaningless.
        sample_size = min(4000, xs.size)
        if xs.size > sample_size:
            idx = np.linspace(0, xs.size - 1, sample_size, dtype=int)
            xs = xs[idx]
            ys = ys[idx]
        ax.scatter(xs, ys, s=5, alpha=0.2)
        ax.plot([-limit, limit], [-limit, limit], color="tab:red", linewidth=1)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(bucket)
        corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 1 else float("nan")
        summary[bucket] = {"sampled_points": int(xs.size), "correlation": corr}
    for ax in axes_list[len(buckets):]:
        ax.set_visible(False)
    fig.suptitle("Predicted vs Target Advantage")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_false_action_rates_by_time(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Per-T_t false-continue and false-halt rates; isolates the time bucket where each error type dominates."""
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for row in state_rows:
        grouped[int(row["time_budget"])][str(row["state_error_type"])] += 1
    xs = sorted(grouped)
    false_continue = []
    false_halt = []
    summary: dict[str, Any] = {}
    for x in xs:
        total = sum(grouped[x].values())
        fc = grouped[x]["false_continue"] / total
        fh = grouped[x]["false_halt"] / total
        false_continue.append(fc)
        false_halt.append(fh)
        summary[str(x)] = {
            "false_continue_rate": fc,
            "false_halt_rate": fh,
            "states": total,
        }
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_WIDE, constrained_layout=True)
    ax.plot(xs, false_continue, marker="o", label="false continue")
    ax.plot(xs, false_halt, marker="o", label="false halt")
    ax.set_xlabel("Current T_t")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("False Continue / False Halt Rates by T_t")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_sign_accuracy_by_time(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Sign accuracy of the predicted advantage vs current time budget, with sample-count bars."""
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in state_rows:
        grouped[int(row["time_budget"])].append(bool(row["sign_correct"]))
    xs = sorted(grouped)
    accs = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    counts = [len(grouped[x]) for x in xs]
    fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE_WIDE, constrained_layout=True)
    ax1.plot(xs, accs, marker="o", color="tab:blue")
    ax1.set_xlabel("Current T_t")
    ax1.set_ylabel("Sign accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    # Twin axis with a faded bar chart of sample counts so the reader can
    # tell whether a sharp accuracy drop is real or driven by tiny support.
    ax2 = ax1.twinx()
    ax2.bar(xs, counts, color="tab:gray", alpha=0.2)
    ax2.set_ylabel("State count", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")
    ax1.set_title("Sign Accuracy by Current T_t")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return {str(x): {"sign_accuracy": accs[i], "states": counts[i]} for i, x in enumerate(xs)}


def _plot_target_distribution_by_time(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Box plot of the per-step target advantage distribution as a function of time budget."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in state_rows:
        grouped[int(row["time_budget"])].append(float(row["target_advantage"]))
    xs = sorted(grouped)
    box_values = [grouped[x] for x in xs]
    means = [statistics.mean(values) for values in box_values]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.boxplot(box_values, positions=xs, widths=0.6, showfliers=False)
    ax.plot(xs, means, marker="o", color="tab:red", label="mean")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Current T_t")
    ax.set_ylabel("Target advantage")
    ax.set_title("Target Advantage Distribution by Current T_t")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return {
        str(x): {
            "mean_target_advantage": means[i],
            "median_target_advantage": statistics.median(box_values[i]),
            "states": len(box_values[i]),
        }
        for i, x in enumerate(xs)
    }


def _calibration_by_margin(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Sign accuracy as a function of |predicted advantage|; reads as a calibration curve."""
    edges = CALIBRATION_BIN_EDGES
    labels = []
    accs = []
    counts = []
    mean_targets = []
    summary: dict[str, Any] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [row for row in state_rows if lo <= abs(row["predicted_advantage"]) < hi]
        label = f"[{lo:.2f}, {hi:.2f})" if math.isfinite(hi) else f"[{lo:.2f}, inf)"
        labels.append(label)
        if not selected:
            accs.append(float("nan"))
            counts.append(0)
            mean_targets.append(float("nan"))
            continue
        acc = statistics.mean(1.0 if row["sign_correct"] else 0.0 for row in selected)
        mean_target = statistics.mean(abs(row["target_advantage"]) for row in selected)
        accs.append(acc)
        counts.append(len(selected))
        mean_targets.append(mean_target)
        summary[label] = {"sign_accuracy": acc, "mean_abs_target_advantage": mean_target, "states": len(selected)}
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL_WIDE, constrained_layout=True)
    axes[0].bar(labels, accs, color="tab:blue")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Sign accuracy")
    axes[0].set_title("Calibration by |Predicted Advantage|")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(labels, counts, color="tab:gray")
    axes[1].set_ylabel("State count")
    axes[1].set_title("States per Calibration Bin")
    axes[1].tick_params(axis="x", rotation=35)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _same_tree_budget_consistency(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Sanity check: for source trees sampled at multiple budgets, are oracle/predicted stop steps monotone in the budget?

    The oracle's stop step should weakly increase with the starting budget by
    construction. The controller doesn't have to, but if it does it suggests
    well-calibrated time sensitivity. Returns the monotone fractions plus
    a handful of example trajectories for the report.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in diagnostics:
        groups[episode["source_path"]].append(episode)
    total_groups = 0
    oracle_monotone = 0
    predicted_monotone = 0
    same_tree_examples: list[dict[str, Any]] = []
    for source_path, episodes in groups.items():
        if len(episodes) < 2:
            continue
        total_groups += 1
        ordered = sorted(episodes, key=lambda episode: (int(episode["starting_budget"]), episode["budget_bucket_name"]))
        oracle_stops = [int(episode["oracle_stop_step"]) for episode in ordered]
        predicted_stops = [int(episode["predicted_stop_step"]) for episode in ordered]
        if all(b >= a for a, b in zip(oracle_stops, oracle_stops[1:])):
            oracle_monotone += 1
        if all(b >= a for a, b in zip(predicted_stops, predicted_stops[1:])):
            predicted_monotone += 1
        if len(same_tree_examples) < 10:
            same_tree_examples.append(
                {
                    "source_path": source_path,
                    "starting_budgets": [int(episode["starting_budget"]) for episode in ordered],
                    "oracle_stops": oracle_stops,
                    "predicted_stops": predicted_stops,
                }
            )
    return {
        "num_multibudget_source_paths": total_groups,
        "oracle_stop_monotone_fraction": oracle_monotone / max(total_groups, 1),
        "predicted_stop_monotone_fraction": predicted_monotone / max(total_groups, 1),
        "examples": same_tree_examples,
    }
