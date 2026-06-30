"""NEW PLOT (b): evaluator strength is the SEARCH (N), not UCI_Elo.

Compares the ρ-vs-RT of the key decision-difficulty signals computed on two SF leaf
evaluators that differ ONLY in the leaf-eval node budget N:
  SF-1   (n1):  /scratch/.../sf_filtered/elo2000_n1/voc_signals.parquet
  SF-100 (n100):/scratch/.../sf_filtered/elo2000/voc_signals.parquet
Grouped horizontal bars (SF-1 vs SF-100), x-axis [-0.4, 0.4], percentile-bootstrap 95% CIs.
RT joined via DuckDB processed_moves_nonzero (one row per human move, move_time>0).
"""
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N1 = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000_n1/voc_signals.parquet"
N100 = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIG = "/home/hl4291/chess_analysis/figures/normative"

# theme palette
SIZE = "#475569"   # slate (structural)
SAT = "#e11d48"    # rose (satisfaction)
VOC = "#6366f1"    # indigo (value-of-computation)
XLIM = (-0.4, 0.4)


def sb(x, y, nb=400, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else np.nan
    pt = pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(nb))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def load(path):
    sig = pd.read_parquet(path)
    sig["frac_good_0.05"] = sig["n_good_0.05"] / sig["legal_moves"]
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", sig[["fen"]])
    mv = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt FROM processed_moves_nonzero m
        JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen WHERE m.move_time>0""").df(); conn.close()
    return mv.merge(sig, on="fen", how="inner")


def main():
    sigs = [("# legal moves (size)", "legal_moves", SIZE),
            ("action gap (sharpness)", "action_gap", SIZE),
            ("# good ≤0.05 (satisfaction)", "n_good_0.05", SAT),
            ("cost-free Gain", "gain_costfree", VOC)]
    res = {}
    for tag, path in [("SF-1", N1), ("SF-100", N100)]:
        df = load(path); rt = df["log_rt"].to_numpy()
        res[tag] = {col: sb(df[col].to_numpy(float), rt) for _, col, _ in sigs}
        print(f"\n=== {tag}  (n={len(df):,} moves) ===")
        for lab, col, _ in sigs:
            r = res[tag][col]; print(f"  {lab:30s} rho={r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(sigs)); cols = [c for _, _, c in sigs]
    for off, tag, hatch in [(-0.2, "SF-1", "//"), (0.2, "SF-100", None)]:
        pts = [res[tag][col][0] for _, col, _ in sigs]
        err = [[res[tag][col][0] - res[tag][col][1] for _, col, _ in sigs],
               [res[tag][col][2] - res[tag][col][0] for _, col, _ in sigs]]
        ax.barh(y + off, pts, height=0.36, color=cols, hatch=hatch,
                edgecolor="white" if hatch else None, label=f"{tag}  (N={'1' if tag=='SF-1' else '100'} nodes)")
        ax.errorbar(pts, y + off, xerr=err, fmt="none", ecolor="#222", capsize=2, lw=1)
    ax.axvline(0, color="#222", lw=0.9); ax.set_xlim(*XLIM)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _, _ in sigs], fontsize=10); ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ vs human RT", fontsize=13); ax.tick_params(axis="x", labelsize=11)
    ax.set_title("Evaluator strength is the SEARCH (N), not UCI_Elo: SF-1 vs SF-100", fontsize=15)
    # legend distinguishing the two series by hatch (color encodes the semantic role)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#cbd5e1", hatch="//", edgecolor="white", label="SF-1  (N=1 node)"),
               Patch(facecolor="#cbd5e1", label="SF-100  (N=100 nodes)")]
    ax.legend(handles=handles, fontsize=10, loc="lower right", frameon=False)
    ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(f"{FIG}/sf_n1_vs_n100.{e}", dpi=200, bbox_inches="tight")
    print("\nsaved sf_n1_vs_n100")


if __name__ == "__main__":
    main()
