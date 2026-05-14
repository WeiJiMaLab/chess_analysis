"""Parser and plotter for compute-advantage controller training logs.

Companion to ``analyze_budgeted_controller_run.py`` with a narrower scope:
ingests a single stdout log emitted by the compute-advantage training loop,
extracts per-epoch train/validation/greedy-eval rows by regex, and writes
CSVs, summary JSON, and matplotlib PNGs to an output directory. The log
format is the line schema produced by the training script, so the regexes
below double as the canonical reference for that schema.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict

from cts.analysis._common import GREEDY_RE, TRAIN_RE, VALIDATION_RE


class AnalyzeComputeAdvantageTrainingLogConfig(BaseModel):
    """Config for ``analyze_compute_advantage_training_log``."""

    model_config = ConfigDict(extra="forbid")

    log_path: str
    output_dir: str

# Catch-all for single-key metadata lines (e.g. ``seed=42``); only matched
# when no space appears in the line so it never swallows a metrics row.
METADATA_RE = re.compile(r"^(?P<key>[a-z_]+)=(?P<value>.+)$")
# Multi-field header emitted once at the top of the log describing the
# representation kind and dataset sizes.
HEADER_STATS_RE = re.compile(
    r"^representation=(?P<representation>\w+) "
    r"train_examples=(?P<train_examples>\d+) "
    r"validation_examples=(?P<validation_examples>\d+)$"
)


def _parse_value(key: str, value: str) -> Any:
    """Cast a regex group string to int or float based on its key name."""
    # Integer-valued fields are enumerated explicitly so anything else
    # (loss, MSE, accuracy, etc.) is treated as a float.
    if key in {
        "epoch",
        "total_epochs",
        "train_snapshots",
        "validation_snapshots",
        "evaluated_episodes",
    }:
        return int(value)
    return float(value)


def _row_from_match(match: re.Match[str]) -> dict[str, Any]:
    """Turn a regex match into a typed row dict; absent optional fields become NaN."""
    row: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        row[key] = float("nan") if value is None else _parse_value(key, value)
    return row


def parse_log(log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read a training log and split it into metadata + per-phase row lists.

    Returns a tuple ``(metadata, train_rows, validation_rows, greedy_rows)``.
    Lines that don't match any known pattern are silently dropped — this is
    intentional so noisy progress prints don't fail the parse.

    Args:
        log_path: path to the training stdout log to ingest.
    """
    metadata: dict[str, Any] = {}
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    greedy_rows: list[dict[str, Any]] = []

    for line in log_path.read_text(encoding="utf-8").splitlines():
        # Skip the training script's bracketed status prints and a couple
        # of well-known noisy lines that aren't worth modelling.
        if line.startswith("[compute_advantage]") or line.startswith("project_dir=") or line.startswith("device="):
            continue

        header_match = HEADER_STATS_RE.match(line)
        if header_match is not None:
            metadata["representation"] = header_match.group("representation")
            metadata["train_examples"] = int(header_match.group("train_examples"))
            metadata["validation_examples"] = int(header_match.group("validation_examples"))
            continue

        train_match = TRAIN_RE.match(line)
        if train_match is not None:
            train_rows.append(_row_from_match(train_match))
            continue

        validation_match = VALIDATION_RE.match(line)
        if validation_match is not None:
            validation_rows.append(_row_from_match(validation_match))
            continue

        greedy_match = GREEDY_RE.match(line)
        if greedy_match is not None:
            greedy_rows.append(_row_from_match(greedy_match))
            continue

        # Fallback: single ``key=value`` metadata lines. Require no spaces
        # so it doesn't grab partial matches from the metrics regexes above.
        meta_match = METADATA_RE.match(line)
        if meta_match is not None and " " not in line:
            metadata[meta_match.group("key")] = meta_match.group("value")

    return metadata, train_rows, validation_rows, greedy_rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Dump rows to CSV using the first row's keys as the column order. No-op if empty."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best_by(rows: list[dict[str, Any]], key: str, *, maximize: bool) -> dict[str, Any] | None:
    """Return the row with the extremal value at ``key``, or None on empty input.

    Args:
        rows: rows to scan.
        key: column to compare on.
        maximize: True picks the argmax; False picks the argmin.
    """
    if not rows:
        return None
    return max(rows, key=lambda row: row[key]) if maximize else min(rows, key=lambda row: row[key])


def _plot_training(train_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Save a 2x2 grid of train-side curves (loss, advantage MSE, sign BCE, sign accuracy)."""
    xs = [row["epoch"] for row in train_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    flat_axes = axes.reshape(-1)

    flat_axes[0].plot(xs, [row["train_total_loss"] for row in train_rows], marker="o", color="tab:red")
    flat_axes[0].set_title("Train Total Loss")
    flat_axes[0].set_xlabel("Epoch")
    flat_axes[0].set_ylabel("Loss")

    flat_axes[1].plot(xs, [row["train_advantage_mse"] for row in train_rows], marker="o", color="tab:blue")
    flat_axes[1].set_title("Train Advantage MSE")
    flat_axes[1].set_xlabel("Epoch")
    flat_axes[1].set_ylabel("MSE")

    flat_axes[2].plot(xs, [row["train_sign_bce"] for row in train_rows], marker="o", color="tab:purple")
    flat_axes[2].set_title("Train Sign BCE")
    flat_axes[2].set_xlabel("Epoch")
    flat_axes[2].set_ylabel("Loss")

    flat_axes[3].plot(xs, [row["train_sign_accuracy"] for row in train_rows], marker="o", color="tab:green")
    flat_axes[3].set_title("Train Sign Accuracy")
    flat_axes[3].set_xlabel("Epoch")
    flat_axes[3].set_ylabel("Accuracy")
    # Sign accuracy is a fraction in [0, 1]; pin the axis so runs with weak
    # signal don't auto-zoom to a misleadingly tight range.
    flat_axes[3].set_ylim(0.0, 1.0)

    fig.suptitle("Compute Advantage Training Metrics", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_validation_and_greedy(
    validation_rows: list[dict[str, Any]],
    greedy_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Save a 2x3 grid mixing validation losses/accuracy with greedy rollout stats."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

    # Validation and greedy curves can be sampled on different epoch
    # cadences, so each subplot uses its own x axis.
    vxs = [row["epoch"] for row in validation_rows]
    gxs = [row["epoch"] for row in greedy_rows]

    axes[0, 0].plot(vxs, [row["validation_total_loss"] for row in validation_rows], marker="o", color="tab:red")
    axes[0, 0].set_title("Validation Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(vxs, [row["validation_advantage_mse"] for row in validation_rows], marker="o", color="tab:blue")
    axes[0, 1].set_title("Validation Advantage MSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE")

    axes[0, 2].plot(vxs, [row["validation_sign_bce"] for row in validation_rows], marker="o", color="tab:purple")
    axes[0, 2].set_title("Validation Sign BCE")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")

    # Overlay return vs. oracle on the same axis so the gap is visually
    # obvious — the gap is the headline number for the run.
    axes[1, 0].plot(gxs, [row["average_return"] for row in greedy_rows], marker="o", label="return", color="tab:blue")
    axes[1, 0].plot(
        gxs,
        [row["average_oracle_value"] for row in greedy_rows],
        marker="o",
        label="oracle",
        color="tab:red",
    )
    axes[1, 0].set_title("Greedy Return")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend()

    axes[1, 1].plot(vxs, [row["validation_sign_accuracy"] for row in validation_rows], marker="o", color="tab:green")
    axes[1, 1].set_title("Validation Sign Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0.0, 1.0)

    # Three "boundary" indicators share the bottom-right panel; expansions
    # share the axis even though it lives on a different scale because it
    # tends to co-move with the accuracies in practice.
    axes[1, 2].plot(gxs, [row["exact_stop_step_accuracy"] for row in greedy_rows], marker="o", label="exact", color="tab:purple")
    axes[1, 2].plot(gxs, [row["first_action_accuracy"] for row in greedy_rows], marker="o", label="first action", color="tab:orange")
    axes[1, 2].plot(gxs, [row["average_expansions"] for row in greedy_rows], marker="o", label="expansions", color="tab:brown")
    axes[1, 2].set_title("Greedy Boundary Metrics")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()

    fig.suptitle("Compute Advantage Validation and Greedy Metrics", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(
    metadata: dict[str, str],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    greedy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collapse the parsed rows into a single JSON-serializable summary dict.

    Picks the last-epoch row for each phase plus the best-by-metric rows
    (validation MSE / total loss / sign accuracy, greedy return). The
    summary is what gets written to ``summary.json`` and is the easiest
    place to glance at "did the run work?" without opening the CSVs.
    """
    summary: dict[str, Any] = {
        "metadata": metadata,
        "num_logged_epochs": len(train_rows),
        "num_validation_checkpoints": len(validation_rows),
        "num_greedy_checkpoints": len(greedy_rows),
        "last_train": train_rows[-1] if train_rows else None,
        "last_validation": validation_rows[-1] if validation_rows else None,
        "last_greedy": greedy_rows[-1] if greedy_rows else None,
        "best_validation_by_mse": _best_by(validation_rows, "validation_advantage_mse", maximize=False),
        # Filter out NaN total losses from older log formats before
        # argmin'ing; otherwise the comparison would propagate NaN.
        "best_validation_by_total_loss": _best_by(
            [row for row in validation_rows if not math.isnan(float(row["validation_total_loss"]))],
            "validation_total_loss",
            maximize=False,
        )
        if validation_rows
        else None,
        "best_validation_by_sign_accuracy": _best_by(validation_rows, "validation_sign_accuracy", maximize=True),
        "best_greedy_by_return": _best_by(greedy_rows, "average_return", maximize=True),
    }

    if greedy_rows:
        # Headline numbers: final epoch's greedy stats plus the
        # return-vs-oracle gap (negative = controller underperforms oracle).
        summary["final_eval"] = greedy_rows[-1]
        summary["oracle_return_gap"] = greedy_rows[-1]["average_return"] - greedy_rows[-1]["average_oracle_value"]
    return summary


def main(config: AnalyzeComputeAdvantageTrainingLogConfig) -> None:
    """CLI entry point: parse one log, write CSVs/plots/summary into ``output_dir``."""
    log_path = Path(config.log_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, train_rows, validation_rows, greedy_rows = parse_log(log_path)
    # No training rows means the log is empty or malformed; bail loudly
    # rather than emit an empty CSV/plot the user will misread.
    if not train_rows:
        raise ValueError(f"No training epochs parsed from {log_path}.")

    _write_csv(train_rows, output_dir / "updates.csv")
    if validation_rows:
        _write_csv(validation_rows, output_dir / "validations.csv")
    if greedy_rows:
        _write_csv(greedy_rows, output_dir / "greedy.csv")

    _plot_training(train_rows, output_dir / "training_metrics.png")
    # Validation/greedy plot needs both phases to make sense; skip
    # entirely if either is missing rather than emit half a figure.
    if validation_rows and greedy_rows:
        _plot_validation_and_greedy(validation_rows, greedy_rows, output_dir / "validation_metrics.png")

    summary = build_summary(metadata, train_rows, validation_rows, greedy_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(AnalyzeComputeAdvantageTrainingLogConfig, main)
