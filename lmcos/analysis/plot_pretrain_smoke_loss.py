"""Parse a child-WDL pretraining Slurm log and plot the loss-vs-epoch curve.

Reads the per-epoch ``epoch=N/M train_total_loss=.. train_loss_gap=.. val_total_loss=..
val_loss_gap=..`` lines emitted by ``cts.data.build_tree pretrain-child-wdl-encoder``,
plus the ``=== TRAIN START <ts> ===`` / ``=== TRAIN END <ts> ===`` markers, and produces:

  * a two-panel figure: cross-entropy (with the data's target-entropy floor) and the
    drivable ``loss_gap`` (CE - target_entropy), train vs validation;
  * a per-epoch wall-clock estimate and a naive full-dataset convergence ETA.

Usage: python -m cts.analysis.plot_pretrain_smoke_loss <slurm_out_log> [out_png]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(
    r"^epoch=(\d+)/(\d+)\s+train_total_loss=([\d.]+)\s+train_target_entropy=([\d.]+)\s+"
    r"train_loss_gap=([\d.]+)\s+train_supervised_edges=(\d+)\s+val_total_loss=([\d.]+)\s+"
    r"val_target_entropy=([\d.]+)\s+val_loss_gap=([\d.]+)\s+val_supervised_edges=(\d+)"
)
START_RE = re.compile(r"=== TRAIN START (\d+) ===")
END_RE = re.compile(r"=== TRAIN END (\d+) ===")


def parse_log(path: str):
    epochs, tr_ce, tr_ent, tr_gap, va_ce, va_ent, va_gap, tr_edges = ([] for _ in range(8))
    start = end = None
    for line in Path(path).read_text().splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epochs.append(int(m.group(1)))
            tr_ce.append(float(m.group(3))); tr_ent.append(float(m.group(4)))
            tr_gap.append(float(m.group(5))); tr_edges.append(int(m.group(6)))
            va_ce.append(float(m.group(7))); va_ent.append(float(m.group(8)))
            va_gap.append(float(m.group(9)))
            continue
        if (s := START_RE.search(line)):
            start = int(s.group(1))
        if (e := END_RE.search(line)):
            end = int(e.group(1))
    return dict(epochs=epochs, tr_ce=tr_ce, tr_ent=tr_ent, tr_gap=tr_gap,
                va_ce=va_ce, va_ent=va_ent, va_gap=va_gap, tr_edges=tr_edges,
                start=start, end=end)


def main(log_path: str, out_png: str) -> None:
    d = parse_log(log_path)
    ep = d["epochs"]
    if not ep:
        raise SystemExit(f"No epoch lines found in {log_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ep, d["tr_ce"], "o-", label="train CE", color="#1f5fa8")
    ax1.plot(ep, d["va_ce"], "s-", label="val CE", color="#d1622b")
    ax1.plot(ep, d["tr_ent"], "--", color="grey", label="target entropy (CE floor)")
    ax1.set_title("Child-WDL cross-entropy"); ax1.set_xlabel("epoch"); ax1.set_ylabel("CE (nats)")
    ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(ep, d["tr_gap"], "o-", label="train loss_gap", color="#1f5fa8")
    ax2.plot(ep, d["va_gap"], "s-", label="val loss_gap", color="#d1622b")
    ax2.axhline(0, color="grey", lw=.8)
    ax2.set_title("loss_gap = CE - target_entropy (drivable KL)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("loss_gap (nats)")
    ax2.legend(); ax2.grid(alpha=.3)

    secs = (d["end"] - d["start"]) if d["start"] and d["end"] else None
    n_ep = len(ep)
    subtitle = f"{n_ep} epochs"
    if secs:
        per_ep = secs / n_ep
        subtitle += f" | {secs}s total, {per_ep:.1f}s/epoch (256 train / 64 val trees, GPU)"
    fig.suptitle("GNN encoder child-WDL pretraining — smoke (lc0_trees subset)\n" + subtitle)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")

    # Convergence / ETA report.
    print("\n=== timing & ETA ===")
    print(f"smoke: {n_ep} epochs over 256 train trees", end="")
    if secs:
        per_ep = secs / n_ep
        edges_per_ep = d["tr_edges"][0]
        print(f", {secs}s total -> {per_ep:.1f}s/epoch, ~{edges_per_ep} supervised train edges/epoch")
        # Linear scale to full 150k trees (compute-bound by supervised edges).
        full_per_ep = per_ep * (150003 / 256)
        print(f"naive full-150k epoch time (linear in trees): {full_per_ep/60:.1f} min/epoch "
              f"= {full_per_ep/3600:.2f} h/epoch (CPU->GPU data loading not modeled)")
    print(f"final val CE={d['va_ce'][-1]:.4f}  val loss_gap={d['va_gap'][-1]:.4f}  "
          f"(target-entropy floor ~{d['va_ent'][-1]:.3f})")
    print(f"train loss_gap {d['tr_gap'][0]:.4f} -> {d['tr_gap'][-1]:.4f}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "analysis/figures/pretrain_smoke_loss.png"
    if not log:
        raise SystemExit("usage: plot_pretrain_smoke_loss.py <slurm_out_log> [out_png]")
    main(log, out)
