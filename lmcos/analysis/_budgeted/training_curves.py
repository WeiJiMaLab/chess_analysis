"""Training-loss and greedy-metric curves over training epochs.

Renders the three-panel total / advantage-MSE / sign-BCE loss curves
plus the two-panel greedy-policy return/regret/boundary-accuracy plot,
and surfaces the best-epoch picks (validation MSE, greedy return,
greedy regret) used by the report header.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from cts.analysis._budgeted._shared import (
    ANALYSIS_FIGURE_DPI,
    FIGSIZE_LOSS_THREE_PANEL,
    FIGSIZE_TWO_PANEL_WIDE,
)


def _plot_loss_curves(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Three-panel total/advantage-MSE/sign-BCE loss curves over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_LOSS_THREE_PANEL, constrained_layout=True)
    epochs_train = [row["epoch"] for row in train_rows]
    epochs_val = [row["epoch"] for row in validation_rows]

    axes[0].plot(
        epochs_train,
        [row["train_total_loss"] for row in train_rows],
        marker="o",
        linewidth=2,
        label="train",
    )
    axes[0].plot(
        epochs_val,
        [row["validation_total_loss"] for row in validation_rows],
        marker="o",
        linewidth=2,
        label="validation",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs_train,
        [row["train_advantage_mse"] for row in train_rows],
        marker="o",
        linewidth=2,
        label="train",
    )
    axes[1].plot(
        epochs_val,
        [row["validation_advantage_mse"] for row in validation_rows],
        marker="o",
        linewidth=2,
        label="validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Advantage MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs_train,
        [row["train_sign_bce"] for row in train_rows],
        marker="o",
        linewidth=2,
        label="train",
    )
    axes[2].plot(
        epochs_val,
        [row["validation_sign_bce"] for row in validation_rows],
        marker="o",
        linewidth=2,
        label="validation",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Sign BCE")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)


def _plot_greedy_metrics(greedy_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Two-panel plot of greedy-policy return/regret and boundary accuracies over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL_WIDE, constrained_layout=True)
    epochs = [row["epoch"] for row in greedy_rows]
    axes[0].plot(epochs, [row["average_return"] for row in greedy_rows], marker="o", label="return")
    axes[0].plot(epochs, [row["average_oracle_value"] for row in greedy_rows], marker="o", label="oracle")
    axes[0].plot(epochs, [row["average_regret"] for row in greedy_rows], marker="o", label="regret")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Greedy Returns and Regret")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["exact_stop_step_accuracy"] for row in greedy_rows], marker="o", label="exact stop")
    axes[1].plot(epochs, [row["first_action_accuracy"] for row in greedy_rows], marker="o", label="first action")
    axes[1].plot(epochs, [row["average_expansions"] for row in greedy_rows], marker="o", label="avg expansions")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Greedy Boundary Metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)


def _best_epoch_summary(validation_rows: list[dict[str, Any]], greedy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick out the best-validation-MSE, best-greedy-return, and best-greedy-regret epochs for the report header."""
    best_val = min(validation_rows, key=lambda row: row["validation_advantage_mse"])
    best_return = max(greedy_rows, key=lambda row: row["average_return"])
    best_regret = min(greedy_rows, key=lambda row: row["average_regret"])
    return {
        "best_validation_mse_epoch": best_val["epoch"],
        "best_validation_mse": best_val["validation_advantage_mse"],
        "best_greedy_return_epoch": best_return["epoch"],
        "best_greedy_return": best_return["average_return"],
        "best_greedy_regret_epoch": best_regret["epoch"],
        "best_greedy_regret": best_regret["average_regret"],
    }
