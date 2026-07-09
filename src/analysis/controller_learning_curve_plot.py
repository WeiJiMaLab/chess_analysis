"""plan.md §G item 2 (2026-07-08) -- restyled PG controller learning curve.

Reads the production PG controller's per-epoch training history (written by
``cts.train.pg_controller_train._save_training_curves`` to
``<packed_dir>/mchalt_controller_training_curve.csv`` during the real 20-epoch run, see
``slurm/pipeline/train_readout_pg.slurm``) and adds two reference lines that do NOT depend on
the trained model at all: "Always Stop" (regret of stopping every episode at step 0) and
"Always Continue" (regret of running every episode to its last step). Both are fixed-k points
on the SAME regret-effort frontier ``analysis.evaluate._compute_frontier_data`` plots (see its
``point(np.zeros(...), ...)`` / ``point(np.full(..., kmax-1), ...)`` calls) -- reimplemented here
directly against the validation episode set (no packed-data cache / GPU needed, since the
Always-Stop/Always-Continue regret only needs the oracle return curves, not z_t features).

Restyled to match the board figures' house style (``analysis.utils.helpers.apply_poster_style``,
``MAIN_COLOR``/``PHASE_COLORS``/``LEGEND_FONTSIZE``) instead of the ad hoc colors the original
(now-lost, see g_cleanup.md) driver script used.

Usage:
    python -m analysis.controller_learning_curve_plot \\
        --config config_minply15_maxply75.yaml \\
        --history-csv /scratch/.../packed/mchalt_controller_training_curve.csv \\
        --out-dir outputs/figures/minply15_maxply75/normative
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.utils.helpers import LEGEND_FONTSIZE, MAIN_COLOR, PHASE_COLORS, apply_poster_style
from analysis.utils.plots import save_pdf_png
from cts._config import load_config
from cts.train.controller_train import (
    ControllerEpisodeDataset,
    ControllerTrainConfig,
    _collect_episode_metadata_and_step_count,
    _oracle_config,
)
from cts.data.preprocess_mc.oracle import return_for_stop_step


def always_stop_continue_regret(episode_metadata, oracle_config) -> tuple[float, float]:
    """Mean regret of the two model-independent fixed-k policies, over ``episode_metadata``.

    "Always Stop" = every episode halts at step 0 (k=0); "Always Continue" = every episode runs
    to its own last step (k=len-1) -- exactly the ``point(np.zeros(...))`` /
    ``point(np.full(..., kmax-1))`` fixed-stop points ``analysis.evaluate._compute_frontier_data``
    plots, just computed directly against ``EpisodeMetadata`` instead of the packed z_t cache
    (these two policies need only the oracle return curve, never z_t).

    Pure aside from ``return_for_stop_step`` calls (no disk I/O), so it's testable on a handful
    of synthetic ``EpisodeMetadata``-like objects.
    """
    stop_regrets, continue_regrets = [], []
    for meta in episode_metadata:
        g = [return_for_stop_step(meta.halt_rewards, meta.tree_sizes, meta.time_budgets, s, oracle_config)
             for s in range(meta.num_steps)]
        oracle_value = max(g)
        stop_regrets.append(oracle_value - g[0])
        continue_regrets.append(oracle_value - g[-1])
    n = len(episode_metadata)
    return sum(stop_regrets) / n, sum(continue_regrets) / n


def _load_history(csv_path: str | Path) -> list[dict]:
    with open(csv_path, newline="") as handle:
        return [{k: float(v) if k != "epoch" else int(float(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def render_controller_learning_curve(history: list[dict], always_stop: float, always_continue: float,
                                     out_dir: str | Path, base: str = "controller_learning_curve") -> str:
    """Board-house-style plot: train E[regret] (soft PG loss) + val greedy regret over epochs,
    plus two horizontal Always-Stop/Always-Continue reference lines. Saves pdf+png via
    ``save_pdf_png``. Returns the pdf path."""
    apply_poster_style()
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(20, 14), constrained_layout=True)
    ax.plot(epochs, [h["train_E_regret"] for h in history], marker="o", lw=3, markersize=9,
           color=PHASE_COLORS[1], label="Train E[regret] (soft PG loss)")
    ax.plot(epochs, [h["val_greedy_regret"] for h in history], marker="o", lw=3, markersize=9,
           color=MAIN_COLOR, label="Val regret (hard greedy stop)")
    ax.axhline(always_continue, color=PHASE_COLORS[3], lw=3, ls="--",
              label=f"Always Continue (k=max) = {always_continue:.3f}")
    ax.axhline(always_stop, color=PHASE_COLORS[2], lw=3, ls=":",
              label=f"Always Stop (k=0) = {always_stop:.3f}")
    ax.set(xlabel="Epoch", ylabel="Regret")
    ax.set_xticks(epochs)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, -0.16),
             ncol=2, frameon=False)
    return save_pdf_png(fig, str(out_dir), base, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_minply15_maxply75.yaml")
    parser.add_argument("--history-csv",
                        default="/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/"
                                "mchalt_controller_training_curve.csv")
    parser.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/normative")
    args = parser.parse_args()

    # Same stage + overrides as the real production run (slurm/pipeline/train_readout_pg.slurm)
    # so the validation episode set + oracle config used for Always-Stop/Continue matches
    # exactly what produced --history-csv's numbers.
    config = load_config(ControllerTrainConfig, args.config, stage="train",
                         sets=["device=cpu", "epochs=20", "pg_episode_batch=1024"])
    oracle_config = _oracle_config(config)
    val_meta, _ = _collect_episode_metadata_and_step_count(
        ControllerEpisodeDataset(str(config.packed_validation_data)), max_episodes=config.pg_max_episodes)
    always_stop, always_continue = always_stop_continue_regret(val_meta, oracle_config)
    print(f"[controller_learning_curve_plot] Always Stop (k=0)={always_stop:.4f} "
         f"Always Continue (k=max)={always_continue:.4f}", flush=True)

    history = _load_history(args.history_csv)
    pdf_path = render_controller_learning_curve(history, always_stop, always_continue, args.out_dir)
    print(f"[controller_learning_curve_plot] -> {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
