"""STAGED for the 63k refresh — the saturation-model test.

Claim (construal/saturation model): RT vs # legal moves should be CONCAVE / saturating
(rises then plateaus, Hick-like), because a per-move inclusion cost caps the consideration
set (S* ~ min(n, S_unconstrained)). And the plateau should DROP with satisfaction (satisfice
early) and RISE with stakes (willing to pay for a bigger set).

Runs on whatever voc_signals.parquet is present (12k now → concavity only; 63k → adds the
plateau-shift panels, since that parquet carries n_good_* / action_gap).
"""
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARQ = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIG = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
MAIN, ACC, MID = "#475569", "#e11d48", "#6366f1"  # slate / rose / indigo (theme)


def r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]; b = np.polyfit(x, y, 1); yh = np.polyval(b, x)
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def curve(ax, n, rt, color, label):
    q = np.quantile(n, np.linspace(0, 1, 13)); q = np.unique(q)
    cen, med, lo, hi = [], [], [], []
    for a, b in zip(q[:-1], q[1:]):
        s = rt[(n >= a) & (n < b)]
        if s.size < 30: continue
        cen.append(np.median(n[(n >= a) & (n < b)])); med.append(np.median(s))
        bs = [np.median(np.random.choice(s, s.size)) for _ in range(200)]
        lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    cen, med = np.array(cen), np.array(med)
    ax.plot(cen, med, "o-", color=color, label=label)
    ax.fill_between(cen, lo, hi, color=color, alpha=0.15)


def main():
    sig = pd.read_parquet(PARQ)
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", sig[["fen"]])
    mv = conn.execute("""SELECT m.fen, m.move_time, m.n_possible_moves AS n
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); conn.close()
    mv["log_rt"] = np.log(mv["move_time"]); n = mv["n"].to_numpy(float)
    print(f"moves: {len(mv):,}", flush=True)
    rl, rlog = r2(n, mv["log_rt"]), r2(np.log(n), mv["log_rt"])
    print(f"R²(log_rt ~ n) = {rl:.4f}   R²(log_rt ~ log n) = {rlog:.4f}  -> "
          f"{'CONCAVE/saturating (log wins)' if rlog > rl else 'linear'}", flush=True)

    has_sat = "n_good_0.05" in sig.columns
    fig, axes = plt.subplots(1, 2 if has_sat else 1, figsize=(13 if has_sat else 7, 5), squeeze=False)
    ax = axes[0][0]
    curve(ax, n, mv["move_time"].to_numpy(float), MAIN, "all moves")
    ax.set_xscale("log"); ax.set_xlabel("# legal moves (log)"); ax.set_ylabel("median RT (s)")
    ax.set_title(f"RT vs legal moves — concave?\nlog-fit R²={rlog:.3f} vs linear R²={rl:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    if has_sat:
        j = mv.merge(sig[["fen", "n_good_0.05", "action_gap"]], on="fen", how="inner")
        j["frac_good"] = j["n_good_0.05"] / j["n"]
        ax2 = axes[0][1]
        # satisfaction tertiles: does the plateau drop?
        ter = pd.qcut(j["frac_good"], 3, labels=["low sat", "med sat", "high sat"])
        for lab, c in zip(["low sat", "med sat", "high sat"], [ACC, MID, MAIN]):
            s = j[ter == lab]
            curve(ax2, s["n"].to_numpy(float), s["move_time"].to_numpy(float), c, lab)
        ax2.set_xscale("log"); ax2.set_xlabel("# legal moves (log)"); ax2.set_ylabel("median RT (s)")
        ax2.set_title("Plateau shift by satisfaction\n(high satisfaction ⇒ lower plateau?)", fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_axisbelow(True)
        for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(f"{FIG}/rt_vs_n_concavity.{e}", dpi=200, bbox_inches="tight")
    print("saved rt_vs_n_concavity", flush=True)


if __name__ == "__main__":
    main()
