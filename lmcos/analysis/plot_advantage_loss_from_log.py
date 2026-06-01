"""Plot train/validation advantage MSE over epochs from a controller log.

Quick visualization helper for ``compute_advantage`` training runs: reads a
``.out`` slurm log, parses the per-epoch advantage MSE via
``analyze_compute_advantage_training_log.parse_log``, and writes a single
two-line PNG showing train and validation curves on the same axes. Intended
for eyeballing convergence, not for paper figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict

from cts.analysis.analyze_compute_advantage_training_log import parse_log


class PlotAdvantageLossFromLogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_path: str
    output_path: Optional[str] = None
    title: Optional[str] = None


def plot_advantage_loss(log_path: Path, output_path: Path, *, title: str | None = None) -> None:
    """Parse ``log_path`` and write the advantage-MSE-over-epoch PNG to ``output_path``.

    Args:
        log_path: path to a ``.out`` training log produced by the controller pipeline.
        output_path: destination PNG; parent dirs are created on demand.
        title: optional plot title; defaults to "Advantage Loss Over Time".
    """
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


def main(config: PlotAdvantageLossFromLogConfig) -> None:
    """CLI entry point: dispatch to ``plot_advantage_loss`` from validated config."""
    log_path = Path(config.log_path)
    # Default the output next to the log file so re-running on the same log
    # overwrites the previous plot rather than scattering PNGs across cwd.
    output_path = Path(config.output_path) if config.output_path else log_path.with_name("advantage_loss.png")
    plot_advantage_loss(log_path, output_path, title=config.title)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(PlotAdvantageLossFromLogConfig, main)
