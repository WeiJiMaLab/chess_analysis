"""P1 — engine-strength litmus: does the sign-flip / decision-difficulty structure hold
(or strengthen) at the weaker engine? Compares SF-2000 vs SF-1350 on the 63k voc parquets.
The flip exists only if engine-good = human-good; if it's cleaner at one rung, that rung's
value function better matches the (≥2000 Lichess, time-pressured) human pool.
"""
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIG = "/home/hl4291/chess_analysis/figures/normative"
SIZE, SAT, GREY = "#475569", "#e11d48", "#6366f1"  # slate / rose / indigo (theme)


def sb(x, y, nb=400, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
        return float((a*b).sum()/d) if d > 0 else np.nan
    pt = pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(nb))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def load(elo):
    sig = pd.read_parquet(f"/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo{elo}/voc_signals.parquet")
    sig = sig.rename(columns={"legal_moves": "n_legal"})
    sig["frac_good_0.05"] = sig["n_good_0.05"] / sig["n_legal"]
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", sig[["fen"]])
    mv = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt FROM processed_moves_nonzero m
        JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen WHERE m.move_time>0""").df(); conn.close()
    return mv.merge(sig, on="fen", how="inner")


def main():
    rows = {}
    sigs = [("# legal (size)", "n_legal", SIZE), ("action gap (sharp)", "action_gap", SIZE),
            ("# good ≤0.1 (sat)", "n_good_0.1", SAT), ("fraction good (sat)", "frac_good_0.05", SAT),
            ("cost-free Gain", "gain_costfree", GREY)]
    for elo in (2000, 1350):
        try:
            df = load(elo)
        except Exception as e:
            print(f"elo{elo}: {e}"); continue
        rt = df["log_rt"].to_numpy()
        rows[elo] = {lab: sb(df[col].to_numpy(float), rt) for lab, col, _ in sigs}
        print(f"\n=== SF-{elo}  (n={len(df):,} moves) ===")
        for lab, col, _ in sigs:
            r = rows[elo][lab]; print(f"  {lab:22s} ρ={r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]")

    if len(rows) == 2:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        y = np.arange(len(sigs))
        for off, elo, hatch in [(-0.2, 2000, None), (0.2, 1350, "//")]:
            pts = [rows[elo][lab][0] for lab, _, _ in sigs]
            err = [[rows[elo][lab][0]-rows[elo][lab][1] for lab, _, _ in sigs],
                   [rows[elo][lab][2]-rows[elo][lab][0] for lab, _, _ in sigs]]
            cols = [c for _, _, c in sigs]
            ax.barh(y + off, pts, height=0.36, color=cols, hatch=hatch, edgecolor="white" if hatch else None,
                    label=f"SF-{elo}" + (" (weaker)" if elo == 1350 else ""))
            ax.errorbar(pts, y + off, xerr=err, fmt="none", ecolor="#222", capsize=2, lw=1)
        ax.axvline(0, color="#222", lw=0.9); ax.set_yticks(y); ax.set_yticklabels([l for l, _, _ in sigs], fontsize=10)
        ax.set_xlim(-0.4, 0.4)
        ax.invert_yaxis(); ax.set_xlabel("Spearman ρ vs human RT", fontsize=13)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_title("Strength litmus: decision-difficulty structure at SF-2000 vs SF-1350\n(solid=2000, hatched=1350)", fontsize=15)
        ax.legend(fontsize=10, loc="lower right"); ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        fig.tight_layout()
        for e in ("png", "pdf"): fig.savefig(f"{FIG}/litmus_strength.{e}", dpi=200, bbox_inches="tight")
        print("\nsaved litmus_strength")


if __name__ == "__main__":
    main()
