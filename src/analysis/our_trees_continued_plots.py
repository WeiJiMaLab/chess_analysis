"""plan.md Phase 2, Agent 2 (our_trees_continued) -- the two REQUIRED plots for both sub-lines
(per this repo's "every step needs a plot as proof of success" convention):

1. ``plot_regret_vs_epoch`` -- loss/regret-vs-epoch curve, EXTENDING the existing training curve
   (``controller_learning_curve_plot.py``'s house style: train E[regret] + val greedy regret,
   with AlwaysStop/AlwaysContinue reference lines) out past the original checkpoint's epoch.

2. ``plot_paired_diff_vs_epoch`` -- point+errorbar chart, x=epoch, y=paired regret DIFFERENCE
   (baseline - checkpoint), two series (AlwaysStop = low bar, SingleHalt* = real bar), horizontal
   line at 0 -- answers "when does it cross into significance, if ever" visually, not just in text.

Both take already-loaded data (CSV rows / JSON results) rather than file paths, so they're
directly unit-testable on tiny synthetic inputs and reusable for either sub-line (e2e or pg) --
this module has NO opinion on which lineage produced its input, only on how to draw it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.evaluate import _C, _rcparams
from analysis.utils.plots import save_pdf_png

_ALWAYS_STOP_COLOR = _C["always"]      # deep red, matches evaluate.py's AlwaysStop convention
_SINGLEHALT_COLOR = _C["singlehalt"]   # pastel red, matches evaluate.py's SingleHalt* convention


def plot_regret_vs_epoch(rows: list[dict], *, always_stop: float, always_continue: float,
                         out_dir: str | Path, base: str, title: str,
                         boundary_epoch: int | None = None) -> str:
    """``rows``: list of ``{"epoch", "train_E_regret", "val_greedy_regret"}`` dicts, ALREADY
    combined across the original + continued runs (epoch numbers globally consistent, e.g. the
    continued run's local epoch N re-labeled as ``original_final_epoch + N``).

    ``boundary_epoch``: if given, draws a vertical dashed line marking where the ORIGINAL
    (already-reported) run ends and the continuation (this agent's new training) begins --
    so the plot self-documents which portion is new.
    """
    _rcparams()
    rows = sorted(rows, key=lambda r: r["epoch"])
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, [r["train_E_regret"] for r in rows], "-o", ms=3.5, lw=1.6, color="#b3382c",
           label="train E[regret] (soft PG loss)")
    ax.plot(epochs, [r["val_greedy_regret"] for r in rows], "-s", ms=3.5, lw=1.6, color="#2c6fb3",
           label="val regret (hard greedy stop)")
    ax.axhline(always_continue, color=_C["never"], lw=1.6, ls="--",
              label=f"Always Continue = {always_continue:.3f}")
    ax.axhline(always_stop, color=_ALWAYS_STOP_COLOR, lw=1.6, ls=":",
              label=f"Always Stop = {always_stop:.3f}")
    if boundary_epoch is not None:
        ax.axvline(boundary_epoch, color="#7A8894", lw=1.2, ls="-.", alpha=0.7)
        ax.text(boundary_epoch, ax.get_ylim()[1], " continuation starts here", fontsize=8,
               color="#7A8894", va="top", ha="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("regret")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.grid(axis="both", color=_C["steps"], lw=0.6, alpha=0.5)
    fig.tight_layout()
    return save_pdf_png(fig, str(out_dir), base, dpi=200)


def plot_paired_diff_vs_epoch(results: list[dict], *, out_dir: str | Path, base: str, title: str,
                              boundary_epoch: int | None = None) -> str:
    """``results``: list of per-epoch result dicts (ONE regime already filtered out) each with
    ``{"epoch", "paired_diff": {"always_stop": {"mean","lo","hi"}, "singlehalt": {...}}}`` --
    the exact shape ``analysis.evaluate_our_trees_continued``'s per-(checkpoint,regime) JSON rows
    have. Positive y = the checkpoint BEATS that baseline (lower regret); CI entirely above 0 =
    statistically confirmed at that epoch.
    """
    _rcparams()
    results = sorted(results, key=lambda r: r["epoch"])
    epochs = np.array([r["epoch"] for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="#2C3E50", lw=1.2, ls="-")
    for name, color, label, dx in (
        ("always_stop", _ALWAYS_STOP_COLOR, "vs AlwaysStop (low bar)", -0.15),
        ("singlehalt", _SINGLEHALT_COLOR, "vs SingleHalt* (real bar)", 0.15),
    ):
        means = np.array([r["paired_diff"][name]["mean"] for r in results])
        los = np.array([r["paired_diff"][name]["lo"] for r in results])
        his = np.array([r["paired_diff"][name]["hi"] for r in results])
        confirmed = np.array([r["paired_diff"][name]["confirmed"] for r in results])
        ax.errorbar(epochs + dx, means, yerr=[means - los, his - means], fmt="o", ms=5.5,
                   color=color, ecolor=color, elinewidth=1.6, capsize=3.5, label=label, zorder=5)
        # Ring the CONFIRMED points so "crosses into significance" reads visually, not just in text.
        if confirmed.any():
            ax.scatter(epochs[confirmed] + dx, means[confirmed], s=140, facecolors="none",
                      edgecolors=color, linewidths=1.8, zorder=6)
    if boundary_epoch is not None:
        ax.axvline(boundary_epoch, color="#7A8894", lw=1.2, ls="-.", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("paired $\\Delta$ regret (baseline $-$ checkpoint)\npositive = checkpoint wins; ringed = 95% CI excludes 0")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="best", frameon=False)
    ax.grid(axis="both", color=_C["steps"], lw=0.6, alpha=0.5)
    fig.tight_layout()
    return save_pdf_png(fig, str(out_dir), base, dpi=200)
