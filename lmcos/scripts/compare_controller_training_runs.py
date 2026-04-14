from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                elif key in {"update", "total_updates", "episodes"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
        return rows


def _plot_validation(
    frozen_validations: list[dict[str, Any]],
    unfrozen_validations: list[dict[str, Any]],
    output_path: Path,
    oracle_average_return: float | None,
    oracle_average_expansions: float | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    for validations, label, color in (
        (frozen_validations, "Frozen", "tab:blue"),
        (unfrozen_validations, "Unfrozen", "tab:orange"),
    ):
        xs = [row["update"] for row in validations]
        axes[0].plot(xs, [row["average_return"] for row in validations], marker="o", label=label, color=color)
        axes[1].plot(xs, [row["average_expansions"] for row in validations], marker="o", label=label, color=color)

    if oracle_average_return is not None:
        axes[0].axhline(oracle_average_return, color="tab:red", linestyle="--", label="Oracle")
    if oracle_average_expansions is not None:
        axes[1].axhline(oracle_average_expansions, color="tab:red", linestyle="--", label="Oracle")

    axes[0].set_title("Validation Average Return")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    axes[1].set_title("Validation Average Expansions")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Expansions")
    axes[1].legend()

    fig.suptitle("Frozen vs Unfrozen Controller Validation", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_training(
    frozen_updates: list[dict[str, Any]],
    unfrozen_updates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    series = [
        ("mean_episode_return", "Train Mean Episode Return", "Return"),
        ("mean_episode_length", "Train Mean Episode Length", "Steps"),
        ("halt_rate", "Train Halt Rate", "Rate"),
        ("entropy", "Train Entropy", "Entropy"),
    ]

    for axis, (key, title, ylabel) in zip(axes.flatten(), series):
        for updates, label, color in (
            (frozen_updates, "Frozen", "tab:blue"),
            (unfrozen_updates, "Unfrozen", "tab:orange"),
        ):
            xs = [row["update"] for row in updates]
            axis.plot(xs, [row[key] for row in updates], marker="o", label=label, color=color)
        axis.set_title(title)
        axis.set_xlabel("Update")
        axis.set_ylabel(ylabel)
        axis.legend()

    fig.suptitle("Frozen vs Unfrozen Controller Training", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two controller training run reports.")
    parser.add_argument("--frozen-report-dir", required=True)
    parser.add_argument("--unfrozen-report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-average-return", type=float, default=None)
    parser.add_argument("--oracle-average-expansions", type=float, default=None)
    args = parser.parse_args()

    frozen_dir = Path(args.frozen_report_dir)
    unfrozen_dir = Path(args.unfrozen_report_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_updates = _read_csv(frozen_dir / "updates.csv")
    frozen_validations = _read_csv(frozen_dir / "validations.csv")
    unfrozen_updates = _read_csv(unfrozen_dir / "updates.csv")
    unfrozen_validations = _read_csv(unfrozen_dir / "validations.csv")

    _plot_validation(
        frozen_validations=frozen_validations,
        unfrozen_validations=unfrozen_validations,
        output_path=output_dir / "validation_comparison.png",
        oracle_average_return=args.oracle_average_return,
        oracle_average_expansions=args.oracle_average_expansions,
    )
    _plot_training(
        frozen_updates=frozen_updates,
        unfrozen_updates=unfrozen_updates,
        output_path=output_dir / "training_comparison.png",
    )

    summary = {
        "frozen_best_validation": json.loads((frozen_dir / "summary.json").read_text(encoding="utf-8"))[
            "best_validation_by_return"
        ],
        "unfrozen_best_validation": json.loads((unfrozen_dir / "summary.json").read_text(encoding="utf-8"))[
            "best_validation_by_return"
        ],
        "frozen_last_validation": json.loads((frozen_dir / "summary.json").read_text(encoding="utf-8"))[
            "last_validation"
        ],
        "unfrozen_last_validation": json.loads((unfrozen_dir / "summary.json").read_text(encoding="utf-8"))[
            "last_validation"
        ],
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
