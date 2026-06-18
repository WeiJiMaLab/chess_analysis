"""P1' — build a durable per-tree table of policy-entropy-at-temperature, then read it
back for the RT correlation sweep (no stdout parsing; recompute only with --refresh).

lc0's policy prior is over-confident; re-temper it π_T(a) ∝ π(a)^(1/T) (T>1 flattens
toward a wider, more human-like option set; T→∞ → uniform over legal moves, so
H(π_T) → log(branching)). We cache {fen, branching, H_T...} per tree to parquet, then
join to human RT and report Spearman ρ(H(π_T), log RT) vs the raw-branching bar.

  # build the cache (slow; serial — mp is broken in this sandbox), then analyze:
  python human_analytics/p1prime_temp_entropy.py
  # re-analyze instantly from the cache:
  python human_analytics/p1prime_temp_entropy.py --analyze-only
"""
import argparse
import os
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from utils.helpers import apply_poster_style  # noqa: E402

TREES = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
CACHE = Path("/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache")
FIGURES = Path(__file__).resolve().parent.parent / "figures"
TEMPS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
N_TREES, SEED = 80000, 7
CACHE_PARQUET = CACHE / f"p1prime_temps_human_trees_{N_TREES}_{SEED}.parquet"


def _row(path: str):
    torch.set_num_threads(1)
    t = torch.load(path, map_location="cpu", weights_only=False)
    fn = list(t["feature_names"]); nf = t["node_features"].numpy(); par = t["parent_index"].numpy()
    roots = np.where(par < 0)[0]
    if not roots.size:
        return None
    kids = np.where(par == int(roots[0]))[0]
    if kids.size < 2:
        return None
    p = nf[kids, fn.index("prior")].astype(float); s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    p = p / s
    row = {"fen": " ".join(t["root_position_spec"].split()[:4]), "branching": int(len(p))}
    for T in TEMPS:
        q = np.power(p, 1.0 / T); q = q / q.sum(); q = q[q > 0]
        row[f"H_T{T}"] = float(-(q * np.log(q)).sum())
    return row


def build_cache() -> pd.DataFrame:
    names = [e.name for e in os.scandir(TREES) if e.name.endswith(".pt")]
    names = random.Random(SEED).sample(names, min(N_TREES, len(names)))
    rows = []
    for i, nm in enumerate(names):
        try:
            r = _row(os.path.join(TREES, nm))
        except Exception:
            r = None
        if r:
            rows.append(r)
        if i % 10000 == 0:
            print(f"  ...{i}/{len(names)}", flush=True)
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PARQUET)
    print(f"  cached {len(df):,} trees → {CACHE_PARQUET}", flush=True)
    return df


def analyze() -> None:
    df = pd.read_parquet(CACHE_PARQUET)
    conn = duckdb.connect(DB, read_only=True)
    conn.register("_v", df)
    j = conn.execute("""
        SELECT v.*, ln(m.move_time) AS log_T
        FROM _v v JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0
    """).df()
    R = j.rank()

    def partial(x: str) -> float:
        sub = R[[x, "log_T", "branching"]]
        A = np.c_[np.ones(len(sub)), sub["branching"].to_numpy()]
        def resid(col):
            b, *_ = np.linalg.lstsq(A, sub[col].to_numpy(), rcond=None)
            return sub[col].to_numpy() - A @ b
        return float(np.corrcoef(resid(x), resid("log_T"))[0, 1])

    branch_bar = float(R["branching"].corr(R["log_T"]))
    marginal = {T: float(R[f"H_T{T}"].corr(R["log_T"])) for T in TEMPS}
    partials = {T: partial(f"H_T{T}") for T in TEMPS}
    # Does tempering just push H(π) onto the raw legal-move count? Track the convergence.
    to_branch = {T: float(R[f"H_T{T}"].corr(R["branching"].rank())) for T in TEMPS}

    print(f"\nn = {len(j):,} joined moves  ({len(df):,} trees)\n", flush=True)
    print(f"  raw branching          ρ(., logRT) = {branch_bar:+.4f}   (bar to beat)")
    for T in TEMPS:
        print(f"  H(π) @ temp ×{T:<5}       ρ(., logRT) = {marginal[T]:+.4f}   "
              f"[ρ(H_T, branching) = {to_branch[T]:+.4f}]")
    print()
    for T in TEMPS:
        print(f"  partial ρ(H@×{T}, logRT | branching) = {partials[T]:+.4f}")

    plot_temperature_sweep(branch_bar, marginal, partials, n=len(j))


def plot_temperature_sweep(branch_bar: float, marginal: dict, partials: dict, n: int) -> None:
    """Re-tempering the over-confident lc0 prior (T>1) raises the *marginal* H(π)↔logRT
    correlation toward the raw-branching bar — but only because H(π_T) → log(branching) as
    T→∞. The *partial* (controlling for branching) tells the real story: the prior's
    independent contribution is largest at T=1 and erodes (goes negative) as tempering
    collapses H(π) onto the legal-move count. So tempering buys marginal fit by discarding
    the only signal the policy shape adds over branching."""
    ts = TEMPS
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.axhline(branch_bar, ls="--", color="0.4", lw=2,
               label=f"raw branching ρ = {branch_bar:+.2f} (bar to beat)")
    ax.axhline(0, color="0.7", lw=1)
    ax.plot(ts, [marginal[T] for T in ts], "o-", color="#1f77b4", lw=2.5, ms=9,
            label="marginal ρ(H(π_T), logRT)")
    ax.plot(ts, [partials[T] for T in ts], "s-", color="#d62728", lw=2.5, ms=9,
            label="partial ρ(H(π_T), logRT | branching)")
    ax.set_xscale("log")
    ax.set_xticks(ts); ax.set_xticklabels([f"×{T:g}" for T in ts], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("policy temperature T   (π_T ∝ π^(1/T);  T→∞ ⇒ uniform)", fontsize=14)
    ax.set_ylabel("Spearman ρ with log(RT)", fontsize=14)
    ax.set_title(f"Re-tempering lc0's policy prior H(π) (n = {n:,})", fontsize=15, pad=10)
    ax.legend(fontsize=12, loc="center right", frameon=True)
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "hpi_temperature_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute the per-tree cache")
    ap.add_argument("--analyze-only", action="store_true", help="skip build; read cache + correlate")
    args = ap.parse_args()
    if not args.analyze_only and (args.refresh or not CACHE_PARQUET.exists()):
        build_cache()
    analyze()


if __name__ == "__main__":
    main()
