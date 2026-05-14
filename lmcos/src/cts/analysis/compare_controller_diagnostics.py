"""Side-by-side comparison of two controller diagnostics JSONLs.

Reads ``left`` and ``right`` diagnostics files (typically baseline vs
intervention), computes calibration-by-|advantage| and accuracy-by-oracle-
halt-step summaries for each, and writes paired PNG plots so the two runs
can be visually compared. Designed to be called from ad hoc experiment
scripts; the heavy lifting (parsing + per-state row construction) is reused
from ``analyze_budgeted_controller_run``.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from pydantic import BaseModel, ConfigDict

from analyze_budgeted_controller_run import build_state_rows, load_diagnostics
from cts.analysis._common import _oracle_stop_bin_label


class CompareControllerDiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_diagnostics: str
    right_diagnostics: str
    left_label: str
    right_label: str
    output_dir: str
    max_state_rows: int = 500000
    state_sample_seed: int = 0


def _calibration_summary(state_rows: list[dict[str, object]]) -> dict[str, list[float] | list[str] | dict[str, dict[str, float | int]]]:
    """Compute sign-accuracy vs |predicted advantage| bins for the supplied rows.

    Bin edges are fixed (0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, inf) so the two
    diagnostics files plot against the same x-axis. Empty bins return NaN
    accuracy and zero count rather than being dropped, again so the bar
    positions line up across the left/right panels.
    """
    edges = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, math.inf]
    labels: list[str] = []
    accuracies: list[float] = []
    counts: list[int] = []
    summary: dict[str, dict[str, float | int]] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Half-open [lo, hi); the final bin runs to +inf so any
        # arbitrarily large magnitude lands in "16+"-style tail.
        selected = [row for row in state_rows if lo <= abs(float(row["predicted_advantage"])) < hi]
        label = f"[{lo:.2f}, {hi:.2f})" if math.isfinite(hi) else f"[{lo:.2f}, inf)"
        labels.append(label)
        if not selected:
            accuracies.append(float("nan"))
            counts.append(0)
            summary[label] = {"sign_accuracy": float("nan"), "states": 0}
            continue
        accuracy = statistics.mean(1.0 if bool(row["sign_correct"]) else 0.0 for row in selected)
        accuracies.append(accuracy)
        counts.append(len(selected))
        summary[label] = {"sign_accuracy": accuracy, "states": len(selected)}
    return {
        "labels": labels,
        "accuracies": accuracies,
        "counts": counts,
        "summary": summary,
    }


def _oracle_stop_summary(diagnostics: list[dict[str, object]]) -> dict[str, list[float] | list[str] | dict[str, dict[str, float | int]]]:
    """Per-bin aggregates of stop accuracy / regret / over- and under-search error.

    Groups episodes by the oracle halt-step bin (see ``_oracle_stop_bin_label``)
    and emits, for each bin: exact stop-step accuracy, first-action accuracy
    (matches stop==0 vs stop>0), share of all episodes, mean regret, and the
    mean over/undersearch errors. The returned dict is shaped for direct
    consumption by ``_plot_oracle_stop``.
    """
    bin_order = ["0", "1", "2", "3", "4-7", "8-15", "16+"]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for episode in diagnostics:
        grouped[_oracle_stop_bin_label(int(episode["oracle_stop_step"]))].append(episode)

    exact_rates: list[float] = []
    first_action_rates: list[float] = []
    episode_shares: list[float] = []
    mean_regrets: list[float] = []
    mean_over_errors: list[float] = []
    mean_under_errors: list[float] = []
    total_episodes = len(diagnostics)
    summary: dict[str, dict[str, float | int]] = {}

    for bin_label in bin_order:
        episodes = grouped.get(bin_label, [])
        count = len(episodes)
        if count == 0:
            # Empty bins still occupy a slot in the plot; emit zeros rather
            # than NaN so the bars render at the right x position.
            exact_rate = 0.0
            first_action_rate = 0.0
            mean_regret = 0.0
            mean_over_error = 0.0
            mean_under_error = 0.0
        else:
            exact_rate = statistics.mean(
                int(int(episode["predicted_stop_step"]) == int(episode["oracle_stop_step"]))
                for episode in episodes
            )
            # "First action" collapses the prediction to stop-immediately vs
            # search-at-least-once, which is the most coarse usable decision.
            first_action_rate = statistics.mean(
                int((int(episode["predicted_stop_step"]) == 0) == (int(episode["oracle_stop_step"]) == 0))
                for episode in episodes
            )
            mean_regret = statistics.mean(float(episode["regret"]) for episode in episodes)
            # Split (predicted - oracle) into oversearch (positive) and
            # undersearch (negative, reported as a positive magnitude) so
            # the two error modes show up on separate lines.
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
        mean_over_errors.append(mean_over_error)
        mean_under_errors.append(mean_under_error)
        summary[bin_label] = {
            "episodes": count,
            "episode_fraction": count / total_episodes if total_episodes else 0.0,
            "exact_stop_step_accuracy": exact_rate,
            "first_action_accuracy": first_action_rate,
            "mean_regret": mean_regret,
            "mean_oversearch_error": mean_over_error,
            "mean_undersearch_error": mean_under_error,
        }

    return {
        "labels": bin_order,
        "exact_rates": exact_rates,
        "first_action_rates": first_action_rates,
        "episode_shares": episode_shares,
        "mean_regrets": mean_regrets,
        "mean_over_errors": mean_over_errors,
        "mean_under_errors": mean_under_errors,
        "summary": summary,
    }


def _plot_calibration(left: dict[str, object], right: dict[str, object], left_label: str, right_label: str, out_path: Path) -> None:
    """Render the 2x2 calibration figure (accuracy + count, left vs right)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True, sharex="col")
    for col, (label, payload) in enumerate(((left_label, left), (right_label, right))):
        labels = payload["labels"]
        # Top row: sign accuracy per |advantage| bin; bottom row: state count.
        axes[0, col].bar(labels, payload["accuracies"], color="tab:blue")
        axes[0, col].set_ylim(0.0, 1.0)
        axes[0, col].set_ylabel("Sign accuracy")
        axes[0, col].set_title(label)
        axes[0, col].tick_params(axis="x", rotation=35)
        axes[1, col].bar(labels, payload["counts"], color="tab:gray")
        axes[1, col].set_ylabel("State count")
        axes[1, col].tick_params(axis="x", rotation=35)
    fig.suptitle("Calibration by |Predicted Advantage|")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_oracle_stop(left: dict[str, object], right: dict[str, object], left_label: str, right_label: str, out_path: Path) -> None:
    """Render the 4x2 oracle-stop figure: accuracies, share, regret, errors."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True, sharex="col")
    width = 0.36  # paired-bar width for the exact / first-action accuracy panel
    for col, (label, payload) in enumerate(((left_label, left), (right_label, right))):
        x = np.arange(len(payload["labels"]))
        labels = payload["labels"]

        # Row 0: exact stop-step accuracy and first-action accuracy side-by-side.
        axes[0, col].bar(x - width / 2, payload["exact_rates"], width=width, color="tab:blue", label="exact stop")
        axes[0, col].bar(x + width / 2, payload["first_action_rates"], width=width, color="tab:orange", label="first action")
        axes[0, col].set_ylim(0.0, 1.0)
        axes[0, col].set_ylabel("Accuracy")
        axes[0, col].set_title(label)
        axes[0, col].legend()
        axes[0, col].grid(True, axis="y", alpha=0.3)

        # Row 1: fraction of episodes that landed in each bin (formatted as %).
        axes[1, col].bar(x, payload["episode_shares"], color="tab:gray")
        axes[1, col].set_ylabel("Episode share")
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[1, col].grid(True, axis="y", alpha=0.3)

        # Row 2: mean regret per bin.
        axes[2, col].bar(x, payload["mean_regrets"], color="tab:red")
        axes[2, col].set_ylabel("Mean regret")
        axes[2, col].grid(True, axis="y", alpha=0.3)

        # Row 3: over- vs undersearch error magnitudes as paired lines, so
        # the two failure modes can be read off independently per bin.
        axes[3, col].plot(x, payload["mean_over_errors"], marker="o", linewidth=1.8, color="tab:red", label="oversearch error")
        axes[3, col].plot(x, payload["mean_under_errors"], marker="o", linewidth=1.8, color="tab:blue", label="undersearch error")
        axes[3, col].set_xticks(x)
        axes[3, col].set_xticklabels(labels)
        axes[3, col].set_ylabel("Mean stop-step error")
        axes[3, col].set_xlabel("Oracle halt-step bin")
        axes[3, col].legend()
        axes[3, col].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Accuracy vs Oracle Halt-Step Bin")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(config: CompareControllerDiagnosticsConfig) -> None:
    """CLI entry point: parse args, load both diagnostics, emit paired PNGs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_diagnostics = load_diagnostics(Path(config.left_diagnostics))
    right_diagnostics = load_diagnostics(Path(config.right_diagnostics))

    # Use the same seed for both runs so any subsampling is at least
    # deterministic; the sampled subsets aren't aligned across runs but
    # bin-level aggregates are stable enough at 500k rows.
    left_state_rows, _ = build_state_rows(left_diagnostics, max_rows=config.max_state_rows, seed=config.state_sample_seed)
    right_state_rows, _ = build_state_rows(right_diagnostics, max_rows=config.max_state_rows, seed=config.state_sample_seed)

    left_calibration = _calibration_summary(left_state_rows)
    right_calibration = _calibration_summary(right_state_rows)
    left_oracle = _oracle_stop_summary(left_diagnostics)
    right_oracle = _oracle_stop_summary(right_diagnostics)

    _plot_calibration(
        left_calibration,
        right_calibration,
        config.left_label,
        config.right_label,
        output_dir / "calibration_by_margin_compare.png",
    )
    _plot_oracle_stop(
        left_oracle,
        right_oracle,
        config.left_label,
        config.right_label,
        output_dir / "accuracy_by_oracle_stop_bin_compare.png",
    )


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(CompareControllerDiagnosticsConfig, main)
