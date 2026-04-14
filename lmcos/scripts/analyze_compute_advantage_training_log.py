from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

TRAIN_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"train_advantage_mse=(?P<train_advantage_mse>-?\d+(?:\.\d+)?) "
    r"train_mean_abs_advantage_error=(?P<train_mean_abs_advantage_error>-?\d+(?:\.\d+)?) "
    r"train_sign_accuracy=(?P<train_sign_accuracy>-?\d+(?:\.\d+)?) "
    r"train_snapshots=(?P<train_snapshots>\d+)$"
)

VALIDATION_RE = re.compile(
    r"^validation_epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"validation_advantage_mse=(?P<validation_advantage_mse>-?\d+(?:\.\d+)?) "
    r"validation_mean_abs_advantage_error=(?P<validation_mean_abs_advantage_error>-?\d+(?:\.\d+)?) "
    r"validation_sign_accuracy=(?P<validation_sign_accuracy>-?\d+(?:\.\d+)?) "
    r"validation_snapshots=(?P<validation_snapshots>\d+)$"
)

GREEDY_RE = re.compile(
    r"^greedy_epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"exact_stop_step_accuracy=(?P<exact_stop_step_accuracy>-?\d+(?:\.\d+)?) "
    r"first_action_accuracy=(?P<first_action_accuracy>-?\d+(?:\.\d+)?) "
    r"average_return=(?P<average_return>-?\d+(?:\.\d+)?) "
    r"average_oracle_value=(?P<average_oracle_value>-?\d+(?:\.\d+)?) "
    r"average_regret=(?P<average_regret>-?\d+(?:\.\d+)?) "
    r"average_expansions=(?P<average_expansions>-?\d+(?:\.\d+)?) "
    r"evaluated_episodes=(?P<evaluated_episodes>\d+) "
    r"skipped_episodes=(?P<skipped_episodes>\d+)$"
)

METADATA_RE = re.compile(r"^(?P<key>[a-z_]+)=(?P<value>.+)$")
HEADER_STATS_RE = re.compile(
    r"^representation=(?P<representation>\w+) "
    r"train_examples=(?P<train_examples>\d+) "
    r"validation_examples=(?P<validation_examples>\d+)$"
)


def _parse_value(key: str, value: str) -> Any:
    if key in {
        "epoch",
        "total_epochs",
        "train_snapshots",
        "validation_snapshots",
        "evaluated_episodes",
        "skipped_episodes",
    }:
        return int(value)
    return float(value)


def _row_from_match(match: re.Match[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        row[key] = _parse_value(key, value)
    return row


def parse_log(log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    greedy_rows: list[dict[str, Any]] = []

    for line in log_path.read_text(encoding="utf-8").splitlines():
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

        meta_match = METADATA_RE.match(line)
        if meta_match is not None and " " not in line:
            metadata[meta_match.group("key")] = meta_match.group("value")

    return metadata, train_rows, validation_rows, greedy_rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best_by(rows: list[dict[str, Any]], key: str, *, maximize: bool) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row[key]) if maximize else min(rows, key=lambda row: row[key])


def _plot_training(train_rows: list[dict[str, Any]], out_path: Path) -> None:
    xs = [row["epoch"] for row in train_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    axes[0].plot(xs, [row["train_advantage_mse"] for row in train_rows], marker="o", color="tab:blue")
    axes[0].set_title("Train Advantage MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")

    axes[1].plot(xs, [row["train_mean_abs_advantage_error"] for row in train_rows], marker="o", color="tab:orange")
    axes[1].set_title("Train Mean Abs Advantage Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")

    axes[2].plot(xs, [row["train_sign_accuracy"] for row in train_rows], marker="o", color="tab:green")
    axes[2].set_title("Train Sign Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.0)

    fig.suptitle("Compute Advantage Training Metrics", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_validation_and_greedy(
    validation_rows: list[dict[str, Any]],
    greedy_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    vxs = [row["epoch"] for row in validation_rows]
    gxs = [row["epoch"] for row in greedy_rows]

    axes[0, 0].plot(vxs, [row["validation_advantage_mse"] for row in validation_rows], marker="o", color="tab:blue")
    axes[0, 0].set_title("Validation Advantage MSE")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")

    axes[0, 1].plot(vxs, [row["validation_sign_accuracy"] for row in validation_rows], marker="o", color="tab:green")
    axes[0, 1].set_title("Validation Sign Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.0)

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

    axes[1, 1].plot(gxs, [row["exact_stop_step_accuracy"] for row in greedy_rows], marker="o", label="exact", color="tab:purple")
    axes[1, 1].plot(gxs, [row["first_action_accuracy"] for row in greedy_rows], marker="o", label="first action", color="tab:orange")
    axes[1, 1].plot(gxs, [row["average_expansions"] for row in greedy_rows], marker="o", label="expansions", color="tab:brown")
    axes[1, 1].set_title("Greedy Boundary Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    fig.suptitle("Compute Advantage Validation and Greedy Metrics", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(
    metadata: dict[str, str],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    greedy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metadata": metadata,
        "num_logged_epochs": len(train_rows),
        "num_validation_checkpoints": len(validation_rows),
        "num_greedy_checkpoints": len(greedy_rows),
        "last_train": train_rows[-1] if train_rows else None,
        "last_validation": validation_rows[-1] if validation_rows else None,
        "last_greedy": greedy_rows[-1] if greedy_rows else None,
        "best_validation_by_mse": _best_by(validation_rows, "validation_advantage_mse", maximize=False),
        "best_validation_by_sign_accuracy": _best_by(validation_rows, "validation_sign_accuracy", maximize=True),
        "best_greedy_by_return": _best_by(greedy_rows, "average_return", maximize=True),
    }

    if greedy_rows:
        summary["final_eval"] = greedy_rows[-1]
        summary["oracle_return_gap"] = greedy_rows[-1]["average_return"] - greedy_rows[-1]["average_oracle_value"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze compute-advantage training logs and emit plots.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, train_rows, validation_rows, greedy_rows = parse_log(log_path)
    if not train_rows:
        raise ValueError(f"No training epochs parsed from {log_path}.")

    _write_csv(train_rows, output_dir / "updates.csv")
    if validation_rows:
        _write_csv(validation_rows, output_dir / "validations.csv")
    if greedy_rows:
        _write_csv(greedy_rows, output_dir / "greedy.csv")

    _plot_training(train_rows, output_dir / "training_metrics.png")
    if validation_rows and greedy_rows:
        _plot_validation_and_greedy(validation_rows, greedy_rows, output_dir / "validation_metrics.png")

    summary = build_summary(metadata, train_rows, validation_rows, greedy_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
