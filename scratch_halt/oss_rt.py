"""OSS-vs-RT: the optimal STOP STEP (argmax_s[h(s)-C(s)]) is genuinely cost-dependent —
the cost moves where the max lands. Correlate OSS under each cost regime with human log-RT
(and with legal moves), bootstrap CIs. Also check OSS is distinct from cost-free Gain.
"""
from __future__ import annotations
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIG = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIGDIR = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
MAIN, ACC = "#2E86C1", "#C0392B"
CONFIGS = ["power_law_p1.5_x0.25","power_law_p1.5_x1.0","power_law_p1.5_x4.0",
           "power_law_p2.8_x0.25","power_law_p2.8_x1.0","power_law_p2.8_x4.0",
           "linear_x0.25","linear_x1.0","linear_x4.0",
           "quadratic_x0.25","quadratic_x1.0","quadratic_x4.0"]


def _sb(x, y, n_boot=500, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    def pear(a, b):
        a = a-a.mean(); b = b-b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
        return float((a*b).sum()/d) if d > 0 else np.nan
    pt = pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def main():
    sig = pd.read_parquet(SIG)
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", sig[["fen"]])
    moves = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); conn.close()
    sig = sig.drop(columns=[c for c in ["legal_moves"] if c in sig.columns])
    df = moves.merge(sig, on="fen", how="inner")
    rt = df["log_rt"].to_numpy(); lm = df["legal_moves"].to_numpy(float)
    print(f"matched moves: {len(df):,}", flush=True)
    print(f"ref: legal_moves vs RT = {_sb(lm,rt)[0]:+.4f}\n")

    rows = []
    print(f"{'cost config':24s} {'OSS vs RT':>20s} {'OSS vs legal':>16s} {'ρ(OSS,Gain)':>12s} {'%OSS=0':>7s}")
    for cfg in CONFIGS:
        oss = df[f"oss__{cfg}"].to_numpy(float)
        rrt = _sb(oss, rt); rlm = _sb(oss, lm)
        rg = df[f"oss__{cfg}"].rank().corr(df["gain_costfree"].rank())
        f0 = (df[f"oss__{cfg}"]==0).mean()
        rows.append((cfg, rrt, rlm, rg))
        print(f"{cfg:24s} {rrt[0]:+.4f}[{rrt[1]:+.3f},{rrt[2]:+.3f}]  {rlm[0]:+.4f}[{rlm[1]:+.3f},{rlm[2]:+.3f}]  {rg:+.3f}  {f0:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, key, col, title in [(axes[0],1,MAIN,"OSS vs human log RT"),(axes[1],2,ACC,"OSS vs legal moves")]:
        pts=[r[key][0] for r in rows]; lo=[r[key][0]-r[key][1] for r in rows]; hi=[r[key][2]-r[key][0] for r in rows]
        ax.errorbar(range(len(rows)), pts, yerr=[lo,hi], fmt="o", color=col, capsize=3)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=45, ha="right", fontsize=7)
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].axhline(_sb(lm,rt)[0], color="#27AE60", ls="--", lw=1, label="legal_moves↔RT (ref)"); axes[0].legend(fontsize=8)
    fig.suptitle("Optimal stop step (OSS) under each cost regime vs human RT / decision-width (filtered elo2000)", fontsize=12)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(f"{FIGDIR}/oss_rt_costsweep.{ext}", dpi=200, bbox_inches="tight")
    print("\nsaved oss_rt_costsweep.png/pdf")


if __name__ == "__main__":
    main()
