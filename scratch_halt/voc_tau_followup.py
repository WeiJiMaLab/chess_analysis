"""τ-sweep RT follow-up: does a calibrated softmax temperature make the VOC track human RT
and decision-width? Joins voc_tau_sweep.parquet (+ oss/regret from voc_signals.parquet) to
human RT and reports Spearman(signal, log RT) + Spearman(VOC, legal_moves) per τ, bootstrap CIs.
"""
from __future__ import annotations
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_tau_sweep.parquet"
SIG = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIGDIR = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
MAIN, ACC = "#2E86C1", "#C0392B"
TAUS = [0.05, 0.1, 0.25, 0.5, 1.0]


def _sb(x, y, n_boot=500, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
        return float((a*b).sum()/d) if d > 0 else np.nan
    pt = pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def main():
    tau = pd.read_parquet(TAU)
    sig = pd.read_parquet(SIG)[["fen", "oss__power_law_p2.8_x1.0", "regret_alwaysstop__power_law_p2.8_x1.0"]]
    df0 = tau.merge(sig, on="fen", how="left")
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", df0[["fen"]])
    moves = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); conn.close()
    df0 = df0.drop(columns=[c for c in ["legal_moves"] if c in df0.columns])
    df = moves.merge(df0, on="fen", how="inner")
    rt = df["log_rt"].to_numpy(); lm = df["legal_moves"].to_numpy(float)
    print(f"matched moves: {len(df):,}", flush=True)

    print("\n=== reference vs log RT ===")
    for nm, col in [("legal_moves", lm), ("OSS(plaw2.8x1)", df["oss__power_law_p2.8_x1.0"].to_numpy(float)),
                    ("regret_alwaysstop(plaw2.8x1)", df["regret_alwaysstop__power_law_p2.8_x1.0"].to_numpy(float))]:
        p, lo, hi, n = _sb(col, rt); print(f"  {nm:30s} rho={p:+.4f} [{lo:+.4f},{hi:+.4f}]")

    res = {"cur": {}, "free": {}}
    print("\n=== softmax-VOC(τ) vs log RT  |  vs legal_moves ===")
    for sup in ("cur", "free"):
        print(f"  [{sup}]")
        for t in TAUS:
            c = f"softmax_voc_tau{t if t!=1.0 else 1}__{sup}"
            v = df[c].to_numpy(float)
            rrt = _sb(v, rt); rlm = _sb(v, lm)
            res[sup][t] = (rrt, rlm)
            print(f"    τ={t:<5} RT rho={rrt[0]:+.4f} [{rrt[1]:+.4f},{rrt[2]:+.4f}]   legal_moves rho={rlm[0]:+.4f} [{rlm[1]:+.4f},{rlm[2]:+.4f}]")

    # plot: rho vs tau, two panels (cur/free), RT + legal_moves curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, sup in zip(axes, ("cur", "free")):
        for key, col, lab in [(0, MAIN, "vs log RT"), (1, ACC, "vs legal moves")]:
            pts = [res[sup][t][key][0] for t in TAUS]
            lo = [res[sup][t][key][0]-res[sup][t][key][1] for t in TAUS]
            hi = [res[sup][t][key][2]-res[sup][t][key][0] for t in TAUS]
            ax.errorbar(TAUS, pts, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=lab)
        ax.axhline(0, color="#888", lw=0.8); ax.set_xscale("log")
        ax.set_xlabel("temperature τ (win-prob units)"); ax.set_title(f"supervisor = {sup}")
        ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Spearman ρ"); axes[0].legend(fontsize=9)
    fig.suptitle("Softmax-VOC vs human RT and decision-width, by temperature (filtered elo2000)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{FIGDIR}/voc_tau_sweep.{ext}", dpi=200, bbox_inches="tight")
    print("\nsaved voc_tau_sweep.png/pdf")


if __name__ == "__main__":
    main()
