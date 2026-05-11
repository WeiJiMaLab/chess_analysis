from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_compute_advantage_training_log import parse_log


def plot_advantage_loss(log_path: Path, output_path: Path, *, title: str | None = None) -> None:
    _, train_rows, validation_rows, _ = parse_log(log_path)
    if not train_rows:
        raise ValueError(f"No training epochs parsed from {log_path}.")
    if not validation_rows:
        raise ValueError(f"No validation epochs parsed from {log_path}.")

    train_epochs = [row["epoch"] for row in train_rows]
    train_mse = [row["train_advantage_mse"] for row in train_rows]
    validation_epochs = [row["epoch"] for row in validation_rows]
    validation_mse = [row["validation_advantage_mse"] for row in validation_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(train_epochs, train_mse, marker="o", linewidth=2, label="Train advantage MSE", color="tab:blue")
    ax.plot(
        validation_epochs,
        validation_mse,
        marker="o",
        linewidth=2,
        label="Validation advantage MSE",
        color="tab:orange",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Advantage MSE")
    ax.set_title(title or "Advantage Loss Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/validation advantage loss from a compute-advantage .out log.")
    parser.add_argument("--log-path", required=True, help="Path to the training .out log file.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the PNG plot. Defaults to <log-dir>/advantage_loss.png.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    output_path = Path(args.output_path) if args.output_path else log_path.with_name("advantage_loss.png")
    plot_advantage_loss(log_path, output_path, title=args.title)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
