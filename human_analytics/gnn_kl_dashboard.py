#!/usr/bin/env python3
"""Glanceable in-flight dashboard for the child-WDL GNN encoder pretrain.

Tails the bucketed-KL JSONL emitted by ``cts.data.build_tree``
(``pretrain-child-wdl-encoder``, ``log_bucketed_kl_path``) and plots the
validation ``loss_gap`` (== overall_mean_kl, the KL we drive to zero) vs
epoch plus the rolling per-epoch ΔKL, so an in-flight run is readable at a
glance. The convergence criterion (gnn_train.md [Q2]) is per-epoch
ΔKL(loss_gap) < 1e-5; that threshold is drawn on the ΔKL panel.

Each JSONL row has at least: ``epoch``, ``overall_mean_kl``,
``overall_edge_count``. ``overall_mean_kl`` is the validation loss_gap.

Usage
-----
One-shot (render once and exit)::

    python human_analytics/gnn_kl_dashboard.py \
        --jsonl /scratch/gpfs/GRIFFITHS/hl4291/GNN_profile/smoke_bucketed_kl.jsonl \
        --out figures/gnn_kl_dashboard.png

Live (re-render every --interval seconds until the file stops growing)::

    python human_analytics/gnn_kl_dashboard.py --jsonl <path> --watch --interval 15
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONVERGENCE_THRESHOLD = 1e-5


def read_rows(jsonl_path: str) -> List[dict]:
    """Read all well-formed JSON rows from the (possibly still-growing) JSONL."""
    rows: List[dict] = []
    if not os.path.isfile(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Partial trailing line while the trainer is mid-write; skip it.
                continue
    rows.sort(key=lambda r: r.get("epoch", 0))
    return rows


def extract_series(rows: List[dict]) -> Tuple[List[int], List[float], List[float]]:
    """Return (epochs, loss_gap/KL per epoch, edge_count per epoch)."""
    epochs = [int(r["epoch"]) for r in rows]
    kl = [float(r["overall_mean_kl"]) for r in rows]
    edges = [float(r.get("overall_edge_count", 0)) for r in rows]
    return epochs, kl, edges


def render(jsonl_path: str, out_path: str) -> int:
    """Render the dashboard once. Returns the number of epochs plotted."""
    rows = read_rows(jsonl_path)
    if not rows:
        print(f"[dashboard] no rows yet at {jsonl_path}", flush=True)
        return 0
    epochs, kl, _ = extract_series(rows)
    # Per-epoch ΔKL = |KL[t] - KL[t-1]| (magnitude of change; convergence test).
    dkl_epochs = epochs[1:]
    dkl = [abs(kl[i] - kl[i - 1]) for i in range(1, len(kl))]

    converged = bool(dkl) and dkl[-1] < CONVERGENCE_THRESHOLD

    fig, (ax_kl, ax_d) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_kl.plot(epochs, kl, marker="o", color="C0")
    ax_kl.set_ylabel("validation loss_gap (KL)")
    ax_kl.set_title(
        f"GNN child-WDL pretrain  |  epoch={epochs[-1]}  "
        f"KL={kl[-1]:.6f}" + ("  [CONVERGED]" if converged else "")
    )
    ax_kl.grid(True, alpha=0.3)

    if dkl:
        ax_d.plot(dkl_epochs, dkl, marker="o", color="C1")
    ax_d.axhline(
        CONVERGENCE_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"ΔKL < {CONVERGENCE_THRESHOLD:g} (converge)",
    )
    ax_d.set_yscale("log")
    ax_d.set_ylabel("per-epoch |ΔKL|")
    ax_d.set_xlabel("epoch")
    ax_d.legend(loc="best")
    ax_d.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(
        f"[dashboard] epochs={len(epochs)} last_KL={kl[-1]:.6f} "
        f"last_dKL={(dkl[-1] if dkl else float('nan')):.2e} "
        f"converged={converged} -> {out_path}",
        flush=True,
    )
    return len(epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="path to bucketed-KL JSONL")
    parser.add_argument(
        "--out",
        default="figures/gnn_kl_dashboard.png",
        help="output PNG (default: figures/gnn_kl_dashboard.png)",
    )
    parser.add_argument(
        "--watch", action="store_true", help="re-render on an interval until the file stops growing"
    )
    parser.add_argument("--interval", type=float, default=15.0, help="watch refresh seconds")
    parser.add_argument(
        "--idle-exits",
        type=int,
        default=4,
        help="in --watch, stop after this many consecutive intervals with no new epoch",
    )
    args = parser.parse_args()

    if not args.watch:
        render(args.jsonl, args.out)
        return

    last_n = -1
    idle = 0
    while True:
        n = render(args.jsonl, args.out)
        if n == last_n:
            idle += 1
            if idle >= args.idle_exits and n > 0:
                print("[dashboard] no new epochs; exiting watch.", flush=True)
                break
        else:
            idle = 0
            last_n = n
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
