"""Plot best-so-far greedy regret for Yotam's May 2026 controller runs.

Usage::

    ./analysis/sync_logs.sh
    python3 analysis/plot_controller_regret_curves.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, MaxNLocator, MultipleLocator

_ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from config import ANALYSIS_ROOT, CONTROLLER_RUNS

DEFAULT_LOGS = ANALYSIS_ROOT / "logs"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "outputs" / "controller_regret_curves.png"

GREEDY_EPOCH_RE = re.compile(
    r"^greedy_epoch=(?P<epoch>\d+)/(?P<max_epoch>\d+) .* "
    r"average_regret=(?P<regret>[-+0-9.eE]+)"
    r"(?: .* evaluated_episodes=(?P<n_episodes>\d+))?"
)


@dataclass(frozen=True)
class GreedyEvalMeta:
    max_epoch: int
    n_validation_episodes: int | None


@dataclass(frozen=True)
class Run:
    label: str
    path: Path


def parse_regret_curve(path: Path) -> tuple[np.ndarray, np.ndarray, GreedyEvalMeta]:
    epochs: list[int] = []
    regrets: list[float] = []
    max_epoch = 0
    n_episodes: int | None = None
    with path.open() as fh:
        for line in fh:
            match = GREEDY_EPOCH_RE.match(line)
            if match is None:
                continue
            epochs.append(int(match["epoch"]))
            regrets.append(float(match["regret"]))
            max_epoch = max(max_epoch, int(match["max_epoch"]))
            if match.group("n_episodes") is not None:
                n_episodes = int(match.group("n_episodes"))
    if not epochs:
        raise ValueError(f"No greedy_epoch lines found in {path}")
    return (
        np.asarray(epochs),
        np.asarray(regrets),
        GreedyEvalMeta(max_epoch=max_epoch, n_validation_episodes=n_episodes),
    )


def _nice_linear_step(span: float) -> float:
    if span <= 0.08:
        return 0.01
    if span <= 0.2:
        return 0.02
    if span <= 0.5:
        return 0.05
    return 0.1


def _configure_y_axis(ax: plt.Axes, values: np.ndarray, *, yscale_log: bool) -> None:
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if yscale_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.set_ylim(ymin * 0.85, ymax * 1.12)
    else:
        step = _nice_linear_step(ymax - ymin)
        lo = np.floor(ymin / step) * step
        hi = np.ceil(ymax / step) * step
        margin = step * 0.5
        ax.set_ylim(max(0.0, lo - margin), hi + margin)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_minor_locator(MultipleLocator(step / 2))


def _subtitle_from_meta(meta: GreedyEvalMeta) -> str:
    parts = [
        f"{meta.max_epoch} controller training epochs",
        "(one full pass over the train set per epoch; not tree count)",
    ]
    if meta.n_validation_episodes is not None:
        parts.append(
            f"greedy eval each epoch on {meta.n_validation_episodes:,} validation episodes"
        )
    return " · ".join(parts)


def plot_runs(
    runs: list[Run],
    output: Path,
    *,
    yscale_log: bool = False,
) -> int:
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plotted = 0
    all_best: list[np.ndarray] = []
    subtitle_meta: GreedyEvalMeta | None = None

    for run in runs:
        if not run.path.exists():
            print(f"[skip] missing: {run.path}")
            continue
        epochs, regret, meta = parse_regret_curve(run.path)
        if subtitle_meta is None:
            subtitle_meta = meta
        best_so_far = np.minimum.accumulate(regret)
        all_best.append(best_so_far)
        ax.plot(
            epochs,
            best_so_far,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=run.label,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit(
            f"No log files under {runs[0].path.parent}. Run ./analysis/sync_logs.sh first."
        )

    stacked = np.concatenate(all_best)
    _configure_y_axis(ax, stacked, yscale_log=yscale_log)

    max_epoch = subtitle_meta.max_epoch if subtitle_meta else 20
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Best-so-far greedy validation regret")
    ax.set_xticks(range(1, max_epoch + 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=min(max_epoch, 20)))
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(loc="upper right", framealpha=0.95)
    if subtitle_meta is not None:
        fig.suptitle(_subtitle_from_meta(subtitle_meta), fontsize=9, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output} ({plotted} curve(s))")
    return plotted


def runs_from_config(logs_dir: Path) -> list[Run]:
    out: list[Run] = []
    for spec in CONTROLLER_RUNS:
        if spec.slurm_log_name is None:
            continue
        out.append(Run(spec.label, logs_dir / spec.slurm_log_name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-y", action="store_true")
    args = parser.parse_args()
    plot_runs(runs_from_config(args.logs), args.output, yscale_log=args.log_y)


if __name__ == "__main__":
    main()
