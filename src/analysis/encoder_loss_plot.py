"""plan.md §G item 1 (2026-07-08) -- simplified E1 encoder proof-of-principle plot.

The original ``e1_encoder_loss_vs_epoch`` overlaid the production 6-epoch child-WDL encoder
pretrain run against a later 30-epoch diagnostic rerun (4 lines: train/val loss x 2 runs) to
show the loss plateau starting around epoch 2 was a genuine ceiling, not undertraining stopped
early (see plan.md's "Decision (2026-07-08): 6 epochs is enough"). That comparison is no longer
needed for the proof-of-principle slot -- this module replots ONLY the original 6-epoch
production run's train+val loss (2 lines), sourced from its own driver's stdout log
(``cts.data.build_tree pretrain-child-wdl-encoder``, invoked by ``slurm/pipeline/train_encoder.slurm``,
which prints one ``epoch=E/N train_total_loss=... val_total_loss=...`` summary line per epoch --
see ``_log_epoch`` in ``cts/data/build_tree.py``).

Per-epoch, not per-step: per-batch train_loss IS logged (``phase=train batch=...`` lines,
throttled by ``--log-interval``), but validation loss is only computed once per epoch (the
training loop runs the val pass after each full epoch), so per-epoch is the finest resolution
at which train and val are directly comparable on the same x-axis -- matches plan.md §G's "or
per-epoch if that's all the log supports" allowance for the val side; sub-step sampling would
only be available for train alone.

The identity of the log file backing the CURRENT ``packed/tiny_encoder.pt`` (the checkpoint set
{encoder_epoch001..006, tiny_encoder.pt} in ``$MCP/../packed/``) was recovered by matching file
mtimes (~2026-07-08, epoch checkpoints written 34s apart) against ``slurm/logs/train-encoder_*.out``
mtimes -- ``slurm/logs/train-encoder_10805127.out`` (job 10805127) is the unique match. See
g_cleanup.md for the full trail.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.utils.helpers import LEGEND_FONTSIZE, MAIN_COLOR, PHASE_COLORS, apply_poster_style
from analysis.utils.plots import save_pdf_png

_EPOCH_LINE_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)/(?P<total>\d+)\s+train_total_loss=(?P<train_loss>[\d.]+)\s+"
    r"train_target_entropy=[\d.]+\s+train_loss_gap=[\d.]+\s+train_supervised_edges=\d+\s+"
    r"val_total_loss=(?P<val_loss>[\d.]+)"
)

DEFAULT_LOG_PATH = "slurm/logs/train-encoder_10805127.out"


def parse_epoch_summary_lines(log_text: str) -> list[dict]:
    """Extract ``{epoch, train_loss, val_loss}`` per ``_log_epoch`` summary line, in log order.

    Pure string-parsing function (no disk I/O), so it's directly testable on a small synthetic
    log excerpt without needing a real slurm log on disk.
    """
    rows = []
    for line in log_text.splitlines():
        m = _EPOCH_LINE_RE.match(line.strip())
        if m:
            rows.append({"epoch": int(m.group("epoch")), "train_loss": float(m.group("train_loss")),
                        "val_loss": float(m.group("val_loss"))})
    return rows


def render_encoder_loss_plot(history: list[dict], out_dir: str | Path,
                             base: str = "e1_encoder_loss_vs_epoch") -> str:
    """2-line board-house-style plot: train loss + val loss over epochs, original run only."""
    apply_poster_style()
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(20, 14), constrained_layout=True)
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", lw=3, markersize=9,
           color=MAIN_COLOR, label="Train loss (child-WDL)")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", lw=3, markersize=9,
           color=PHASE_COLORS[3], label="Val loss (child-WDL)")
    ax.set(xlabel="Epoch", ylabel="Child-WDL total loss")
    ax.set_xticks(epochs)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, -0.16),
             ncol=2, frameon=False)
    return save_pdf_png(fig, str(out_dir), base, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH,
                        help="stdout log of the original 6-epoch child-WDL pretrain run")
    parser.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/normative")
    args = parser.parse_args()

    history = parse_epoch_summary_lines(Path(args.log_path).read_text())
    if not history:
        raise ValueError(f"No epoch summary lines found in {args.log_path}")
    pdf_path = render_encoder_loss_plot(history, args.out_dir)
    print(f"[encoder_loss_plot] parsed {len(history)} epochs from {args.log_path} -> {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
