"""Plot best-so-far greedy regret across training epochs for selected controller runs.

Parses `greedy_epoch=N/M ... average_regret=X` lines from slurm .out files and
draws one curve per run on a single axis. Curves are running minima of per-epoch
regret (best-so-far).

Usage:
    python3 scripts/plot_controller_regret_curves.py \\
        --output audit_outputs/controller_regret_curves.pdf
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GREEDY_EPOCH_RE = re.compile(
    r"^greedy_epoch=(?P<epoch>\d+)/\d+ .* average_regret=(?P<regret>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class Run:
    label: str
    path: Path


def parse_regret_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs, regret) extracted from a slurm .out file."""
    epochs: list[int] = []
    regrets: list[float] = []
    with path.open() as fh:
        for line in fh:
            match = GREEDY_EPOCH_RE.match(line)
            if match is None:
                continue
            epochs.append(int(match["epoch"]))
            regrets.append(float(match["regret"]))
    if not epochs:
        raise ValueError(f"No greedy_epoch lines found in {path}")
    return np.asarray(epochs), np.asarray(regrets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "audit_outputs" / "controller_regret_curves.pdf",
    )
    parser.add_argument(
        "--audit-outputs",
        type=Path,
        default=repo_root / "audit_outputs",
        help="Directory containing the .out files.",
    )
    parser.add_argument(
        "--slw01-rerun-out",
        type=Path,
        default=None,
        help=(
            "Path to the historical slw01_rerun [z, N, T] .out file. "
            "If omitted, that curve is skipped."
        ),
    )
    args = parser.parse_args()

    runs: list[Run] = [
        Run("Subtree-weighted [z_t, T_t]", args.audit_outputs / "cts-fittedq_8527146.out"),
        Run("Subtree-weighted [z_t]",      args.audit_outputs / "cts-fittedq_8527147.out"),
        Run("T_t only (new pipeline)",     args.audit_outputs / "cts-fittedq_8550917.out"),
        Run("Rerun encoder + [z_t, T_t]",  args.audit_outputs / "cts-fittedq_8533204.out"),
    ]
    if args.slw01_rerun_out is not None:
        runs.append(Run("Historical main slw01 [z_t, N_t, T_t]", args.slw01_rerun_out))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for run in runs:
        if not run.path.exists():
            print(f"[skip] missing: {run.path}")
            continue
        epochs, regret = parse_regret_curve(run.path)
        best_so_far = np.minimum.accumulate(regret)
        ax.plot(epochs, best_so_far, marker="o", markersize=3, linewidth=1.6, label=run.label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best-so-far greedy regret")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
