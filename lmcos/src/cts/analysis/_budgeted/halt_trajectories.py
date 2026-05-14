"""Per-bucket halt-reward trajectory and per-step delta envelopes.

Renders the six-panel halt-reward median/10-90 percentile band (one per
budget bucket plus overall) and the matching plot for per-step deltas
``h[t+1] - h[t]``, used to read off how halt rewards evolve over search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cts.analysis._budgeted._shared import ANALYSIS_FIGURE_DPI


def _plot_halt_reward_trajectory_distribution(
    diagnostics: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """Six-panel envelope plot of halt-reward trajectories per bucket (median + 10-90 pct band)."""
    buckets = ["overall"] + sorted({str(episode["budget_bucket_name"]) for episode in diagnostics})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes_list = list(axes.flat)
    summary: dict[str, Any] = {}

    for ax, bucket in zip(axes_list, buckets):
        selected = diagnostics if bucket == "overall" else [episode for episode in diagnostics if str(episode["budget_bucket_name"]) == bucket]
        max_len = max(len(episode["halt_rewards"]) for episode in selected)
        means = []
        medians = []
        p10s = []
        p90s = []
        counts = []
        for step_index in range(max_len):
            values = [
                float(episode["halt_rewards"][step_index])
                for episode in selected
                if step_index < len(episode["halt_rewards"])
            ]
            if not values:
                continue
            arr = np.array(values, dtype=np.float32)
            means.append(float(arr.mean()))
            medians.append(float(np.median(arr)))
            p10s.append(float(np.percentile(arr, 10)))
            p90s.append(float(np.percentile(arr, 90)))
            counts.append(int(arr.size))
        xs = np.arange(len(means))
        ax.fill_between(xs, p10s, p90s, alpha=0.2, color="tab:blue", label="10-90 pct")
        ax.plot(xs, medians, color="tab:blue", linewidth=2, label="median")
        ax.plot(xs, means, color="tab:red", linewidth=1.5, linestyle="--", label="mean")
        ax.set_title(bucket)
        ax.set_xlabel("Search step")
        ax.set_ylabel("Halt reward")
        ax.grid(True, alpha=0.3)
        if bucket == "overall":
            ax.legend(loc="best")
        summary[bucket] = {
            "mean": means,
            "median": medians,
            "p10": p10s,
            "p90": p90s,
            "counts": counts,
        }

    for ax in axes_list[len(buckets):]:
        ax.set_visible(False)
    fig.suptitle("Halt-Reward Trajectory Distribution")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_halt_reward_delta_trajectory_distribution(
    diagnostics: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """Same shape as the halt-reward trajectory plot but for per-step deltas (h[t+1] - h[t])."""
    buckets = ["overall"] + sorted({str(episode["budget_bucket_name"]) for episode in diagnostics})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes_list = list(axes.flat)
    summary: dict[str, Any] = {}

    for ax, bucket in zip(axes_list, buckets):
        selected = diagnostics if bucket == "overall" else [episode for episode in diagnostics if str(episode["budget_bucket_name"]) == bucket]
        max_len = max(max(len(episode["halt_rewards"]) - 1, 0) for episode in selected)
        means = []
        medians = []
        p10s = []
        p90s = []
        counts = []
        for step_index in range(max_len):
            values = [
                float(episode["halt_rewards"][step_index + 1]) - float(episode["halt_rewards"][step_index])
                for episode in selected
                if step_index + 1 < len(episode["halt_rewards"])
            ]
            if not values:
                continue
            arr = np.array(values, dtype=np.float32)
            means.append(float(arr.mean()))
            medians.append(float(np.median(arr)))
            p10s.append(float(np.percentile(arr, 10)))
            p90s.append(float(np.percentile(arr, 90)))
            counts.append(int(arr.size))
        xs = np.arange(len(means))
        ax.fill_between(xs, p10s, p90s, alpha=0.2, color="tab:green", label="10-90 pct")
        ax.plot(xs, medians, color="tab:green", linewidth=2, label="median")
        ax.plot(xs, means, color="tab:orange", linewidth=1.5, linestyle="--", label="mean")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title(bucket)
        ax.set_xlabel("Search step t")
        ax.set_ylabel("halt_reward[t+1] - halt_reward[t]")
        ax.grid(True, alpha=0.3)
        if bucket == "overall":
            ax.legend(loc="best")
        summary[bucket] = {
            "mean": means,
            "median": medians,
            "p10": p10s,
            "p90": p90s,
            "counts": counts,
        }

    for ax in axes_list[len(buckets):]:
        ax.set_visible(False)
    fig.suptitle("Halt-Reward Delta Trajectory Distribution")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary
