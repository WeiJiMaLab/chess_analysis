"""Light follow-up: join the per-FEN oracle-side signals (voc_signals.parquet) to human
RT and report Spearman(signal, log RT) with percentile-bootstrap 95% CIs, per cost config,
plus the softmax-VOC vs legal-moves bridge. Produces plots + prints findings.

Runs at the MOVE level (one row per human move; the per-FEN tree signal is repeated for
each move at that FEN) — same population as engine.py's RT correlations.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

SIG = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIGDIR = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
MAIN, ACC = "#2E86C1", "#C0392B"

CONFIGS = [
    "power_law_p1.5_x0.25", "power_law_p1.5_x1.0", "power_law_p1.5_x4.0",
    "power_law_p2.8_x0.25", "power_law_p2.8_x1.0", "power_law_p2.8_x4.0",
    "linear_x0.25", "linear_x1.0", "linear_x4.0",
    "quadratic_x0.25", "quadratic_x1.0", "quadratic_x4.0",
]
THETA_GRID = [f"{t:.2f}" for t in np.linspace(0.0, 1.0, 21)]


def _spearman_boot(x, y, n_boot=500, seed=0):
    """Point Spearman + percentile-bootstrap 95% CI (rank-pearson; fast, no normality)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 50 or np.std(x) == 0:
        return (float("nan"), float("nan"), float("nan"), n)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a*a).sum()*(b*b).sum())
        return float((a*b).sum()/d) if d > 0 else float("nan")
    pt = pear(rx, ry)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = pear(rx[idx], ry[idx])
    return (pt, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)), n)


def main():
    sig = pd.read_parquet(SIG)
    print(f"signal trees: {len(sig)}", flush=True)
    # regret_fraction(theta*) per cost: theta* = argmin mean over trees of (oracle_value - retfrac(theta))
    for cfg in CONFIGS:
        ovc = f"oracle_value__{cfg}"
        best_th, best_mean = None, np.inf
        for th in THETA_GRID:
            col = f"retfrac__{cfg}__{th}"
            if col not in sig or ovc not in sig:
                continue
            mr = (sig[ovc] - sig[col]).mean()
            if mr < best_mean:
                best_mean, best_th = mr, th
        if best_th is not None:
            sig[f"regret_fraction__{cfg}"] = sig[ovc] - sig[f"retfrac__{cfg}__{best_th}"]

    conn = duckdb.connect(DB, read_only=True)
    conn.register("_sig", sig[["fen"]])
    moves = conn.execute("""
        SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m
        JOIN (SELECT DISTINCT fen FROM _sig) f ON m.fen = f.fen
        WHERE m.move_time > 0
    """).df()
    conn.close()
    sig = sig.drop(columns=[c for c in ["legal_moves"] if c in sig.columns])  # use the DB's legal_moves (n_possible_moves)
    df = moves.merge(sig, on="fen", how="inner")
    print(f"matched human moves: {len(df):,} across {df.fen.nunique():,} FENs", flush=True)

    rt = df["log_rt"].to_numpy()
    # reference correlations
    print("\n=== reference vs log RT ===", flush=True)
    for name in ["legal_moves", "gain_costfree", "n_steps"]:
        pt, lo, hi, n = _spearman_boot(df[name].to_numpy(float), rt)
        print(f"  {name:16s} rho={pt:+.4f} [{lo:+.4f},{hi:+.4f}] n={n:,}", flush=True)

    # per-signal x cost
    sigs = {"regret_alwaysstop": "regret-alwaysstop (cost-aware Gain)",
            "regret_fraction": "regret-fraction(θ*)",
            "softmax_voc": "softmax-VOC (τ=1)"}
    results = {s: {} for s in sigs}
    print("\n=== Spearman(signal, log RT) by cost config ===", flush=True)
    for s in sigs:
        for cfg in CONFIGS:
            col = f"{s}__{cfg}"
            if col not in df:
                continue
            results[s][cfg] = _spearman_boot(df[col].to_numpy(float), rt)
        print(f"\n  {s}:", flush=True)
        for cfg in CONFIGS:
            if cfg in results[s]:
                pt, lo, hi, n = results[s][cfg]
                print(f"    {cfg:22s} rho={pt:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

    # softmax_voc vs legal_moves (the bridge) at the current-pipeline cost
    cur = "power_law_p2.8_x1.0"
    vcol = f"softmax_voc__{cur}"
    if vcol in df:
        pt, lo, hi, n = _spearman_boot(df[vcol].to_numpy(float), df["legal_moves"].to_numpy(float))
        print(f"\n=== bridge: softmax_voc[{cur}] vs legal_moves: rho={pt:+.4f} [{lo:+.4f},{hi:+.4f}] ===", flush=True)

    # ---- plot 1: corr-vs-cost (bars + CI), one row per signal ----
    fig, axes = plt.subplots(len(sigs), 1, figsize=(9, 9), sharex=True)
    for ax, (s, title) in zip(axes, sigs.items()):
        xs = [c for c in CONFIGS if c in results[s] and np.isfinite(results[s][c][0])]
        pts = [results[s][c][0] for c in xs]
        los = [results[s][c][0]-results[s][c][1] for c in xs]
        his = [results[s][c][2]-results[s][c][0] for c in xs]
        ax.errorbar(range(len(xs)), pts, yerr=[los, his], fmt="o", color=MAIN, capsize=3)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Spearman ρ\nvs log RT", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("Normative signal vs human RT (filtered elo2000) — by cost config", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGDIR}/voc_rt_costsweep.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- plot 2: softmax_voc vs legal_moves (binned) ----
    if vcol in df:
        fig, ax = plt.subplots(figsize=(7, 5))
        b = pd.qcut(df["legal_moves"], 12, duplicates="drop")
        g = df.groupby(b, observed=True).agg(lm=("legal_moves","mean"), v=(vcol,"mean")).dropna()
        ax.plot(g["lm"], g["v"], "o-", color=ACC)
        ax.set_xlabel("legal moves"); ax.set_ylabel(f"softmax-VOC ({cur})")
        ax.set_title("Does the normative VOC track decision width?")
        ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(f"{FIGDIR}/voc_vs_legalmoves.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- plot 3: OSS distribution across cost modes (mult=1) ----
    fig, ax = plt.subplots(figsize=(7,5))
    for cfg, c in [("power_law_p2.8_x1.0",MAIN),("linear_x1.0","#27AE60"),("quadratic_x1.0",ACC)]:
        col = f"oss__{cfg}"
        if col in sig:
            ax.hist(sig[col], bins=range(0,98,4), histtype="step", lw=2, color=c, label=cfg)
    ax.set_xlabel("oracle stop step (OSS)"); ax.set_ylabel("trees"); ax.legend(fontsize=8)
    ax.set_title("OSS distribution by cost shape (mult=1, filtered elo2000)")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(f"{FIGDIR}/oss_dist_elo2000.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved: voc_rt_costsweep, voc_vs_legalmoves, oss_dist_elo2000 (png+pdf)", flush=True)


if __name__ == "__main__":
    main()
