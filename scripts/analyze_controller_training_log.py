from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

UPDATE_RE = re.compile(
    r"^update=(?P<update>\d+)/(?P<total_updates>\d+) "
    r"policy_loss=(?P<policy_loss>-?\d+(?:\.\d+)?) "
    r"value_loss=(?P<value_loss>-?\d+(?:\.\d+)?) "
    r"entropy=(?P<entropy>-?\d+(?:\.\d+)?) "
    r"approx_kl=(?P<approx_kl>-?\d+(?:\.\d+)?) "
    r"clipfrac=(?P<clipfrac>-?\d+(?:\.\d+)?) "
    r"explained_variance=(?P<explained_variance>-?\d+(?:\.\d+)?) "
    r"mean_episode_return=(?P<mean_episode_return>-?\d+(?:\.\d+)?) "
    r"mean_episode_length=(?P<mean_episode_length>-?\d+(?:\.\d+)?) "
    r"halt_rate=(?P<halt_rate>-?\d+(?:\.\d+)?) "
    r"episodes=(?P<episodes>\d+)$"
)

VALIDATION_RE = re.compile(
    r"^validation_update=(?P<update>\d+)/(?P<total_updates>\d+) "
    r"average_return=(?P<average_return>-?\d+(?:\.\d+)?) "
    r"average_expansions=(?P<average_expansions>-?\d+(?:\.\d+)?) "
    r"average_terminal_quality=(?P<average_terminal_quality>-?\d+(?:\.\d+)?)$"
)

FINAL_EVAL_RE = re.compile(
    r"^average_return=(?P<average_return>-?\d+(?:\.\d+)?) "
    r"average_expansions=(?P<average_expansions>-?\d+(?:\.\d+)?) "
    r"average_terminal_quality=(?P<average_terminal_quality>-?\d+(?:\.\d+)?)$"
)


def _to_number_map(match: re.Match[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        if key in {"update", "total_updates", "episodes"}:
            parsed[key] = int(value)
        else:
            parsed[key] = float(value)
    return parsed


def parse_log(log_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    updates: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    final_eval: dict[str, Any] | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        update_match = UPDATE_RE.match(line)
        if update_match is not None:
            updates.append(_to_number_map(update_match))
            continue

        validation_match = VALIDATION_RE.match(line)
        if validation_match is not None:
            validations.append(_to_number_map(validation_match))
            continue

        final_eval_match = FINAL_EVAL_RE.match(line)
        if final_eval_match is not None:
            final_eval = _to_number_map(final_eval_match)

    return updates, validations, final_eval


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_training(updates: list[dict[str, Any]], out_path: Path) -> None:
    xs = [row["update"] for row in updates]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    flat = axes.flatten()

    flat[0].plot(xs, [row["mean_episode_return"] for row in updates], marker="o", color="tab:blue")
    flat[0].set_title("Train Mean Episode Return")
    flat[0].set_xlabel("Update")
    flat[0].set_ylabel("Return")

    flat[1].plot(xs, [row["mean_episode_length"] for row in updates], marker="o", color="tab:orange")
    flat[1].set_title("Train Mean Episode Length")
    flat[1].set_xlabel("Update")
    flat[1].set_ylabel("Steps")

    flat[2].plot(xs, [row["halt_rate"] for row in updates], marker="o", color="tab:green")
    flat[2].set_title("Train Halt Rate")
    flat[2].set_xlabel("Update")
    flat[2].set_ylabel("Rate")

    flat[3].plot(xs, [row["entropy"] for row in updates], marker="o", label="entropy", color="tab:red")
    flat[3].plot(xs, [row["approx_kl"] for row in updates], marker="o", label="approx_kl", color="tab:purple")
    flat[3].plot(xs, [row["clipfrac"] for row in updates], marker="o", label="clipfrac", color="tab:brown")
    flat[3].set_title("Policy Health")
    flat[3].set_xlabel("Update")
    flat[3].set_ylabel("Value")
    flat[3].legend()

    flat[4].plot(xs, [row["value_loss"] for row in updates], marker="o", color="tab:cyan")
    flat[4].set_title("Value Loss")
    flat[4].set_xlabel("Update")
    flat[4].set_ylabel("Loss")

    flat[5].plot(xs, [row["explained_variance"] for row in updates], marker="o", color="tab:pink")
    flat[5].axhline(0.0, color="black", linewidth=1, linestyle="--")
    flat[5].set_title("Explained Variance")
    flat[5].set_xlabel("Update")
    flat[5].set_ylabel("Explained Variance")

    fig.suptitle("Controller Training Metrics", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_validation(
    validations: list[dict[str, Any]],
    out_path: Path,
    oracle_average_return: float | None,
    oracle_average_expansions: float | None,
) -> None:
    xs = [row["update"] for row in validations]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    axes[0].plot(xs, [row["average_return"] for row in validations], marker="o", color="tab:blue")
    if oracle_average_return is not None:
        axes[0].axhline(oracle_average_return, color="tab:red", linestyle="--", label="oracle")
        axes[0].legend()
    axes[0].set_title("Validation Average Return")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Return")

    axes[1].plot(xs, [row["average_expansions"] for row in validations], marker="o", color="tab:orange")
    if oracle_average_expansions is not None:
        axes[1].axhline(oracle_average_expansions, color="tab:red", linestyle="--", label="oracle")
        axes[1].legend()
    axes[1].set_title("Validation Average Expansions")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Expansions")

    fig.suptitle("Controller Validation Metrics", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _best_validation(validations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not validations:
        return None
    return max(validations, key=lambda row: row["average_return"])


def build_summary(
    updates: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    final_eval: dict[str, Any] | None,
    oracle_average_return: float | None,
    oracle_average_expansions: float | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_logged_updates": len(updates),
        "num_validation_checkpoints": len(validations),
        "last_update": updates[-1] if updates else None,
        "last_validation": validations[-1] if validations else None,
        "best_validation_by_return": _best_validation(validations),
        "final_eval": final_eval,
    }

    last_validation = validations[-1] if validations else None
    if last_validation is not None and oracle_average_return is not None:
        summary["oracle_return_gap"] = last_validation["average_return"] - oracle_average_return
        summary["oracle_return_ratio"] = (
            last_validation["average_return"] / oracle_average_return if oracle_average_return != 0.0 else math.nan
        )
    if last_validation is not None and oracle_average_expansions is not None:
        summary["oracle_expansion_gap"] = last_validation["average_expansions"] - oracle_average_expansions
        summary["oracle_expansion_ratio"] = (
            last_validation["average_expansions"] / oracle_average_expansions
            if oracle_average_expansions != 0.0
            else math.nan
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze controller training logs and emit plots.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-average-return", type=float, default=None)
    parser.add_argument("--oracle-average-expansions", type=float, default=None)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updates, validations, final_eval = parse_log(log_path)
    if not updates:
        raise ValueError(f"No training updates parsed from {log_path}.")

    _write_csv(updates, output_dir / "updates.csv")
    if validations:
        _write_csv(validations, output_dir / "validations.csv")

    _plot_training(updates, output_dir / "training_metrics.png")
    if validations:
        _plot_validation(
            validations,
            output_dir / "validation_metrics.png",
            args.oracle_average_return,
            args.oracle_average_expansions,
        )

    summary = build_summary(
        updates,
        validations,
        final_eval,
        args.oracle_average_return,
        args.oracle_average_expansions,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
