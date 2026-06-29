"""Light follow-up to the softmax-VOC TEMPERATURE sweep.

Joins the per-FEN tau-sweep signals (voc_tau_sweep.parquet) to human RT (one row per
human move, filtered elo2000), and reports:
  - Spearman(softmax_voc(tau), log RT)      with percentile-bootstrap 95% CIs, at cur & free cost
  - Spearman(softmax_voc(tau), legal_moves) with percentile-bootstrap 95% CIs, at cur & free cost
  - reference (current cost): Spearman(OSS, log RT), Spearman(regret_alwaysstop, log RT)
    [reused from the existing voc_signals.parquet]

Plot: rho vs tau, two curves (vs log RT, vs legal_moves) with 95% CI bands; panel A = current
cost, panel B = cost-free. Existing figure style. -> figures/lmcos_tiny/voc_tau_sweep.{png,pdf}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWEEP = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_tau_sweep.parquet"
SIG = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIGDIR = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
MAIN, ACC = "#2E86C1", "#C0392B"

TAUS = [0.05, 0.10, 0.25, 0.50, 1.0]
CUR = "power_law_p2.8_x1.0"  # reference cost in the existing voc_signals.parquet


def _tau_label(tau: float) -> str:
    return f"{tau:g}"


def _spearman_boot(x, y, n_boot=500, seed=0):
    """Point Spearman + percentile-bootstrap 95% CI (rank-pearson; no normality)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 50 or np.std(x) == 0:
        return (float("nan"), float("nan"), float("nan"), n)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()

    def pear(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else float("nan")

    pt = pear(rx, ry)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = pear(rx[idx], ry[idx])
    return (pt, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)), n)


def main():
    sweep = pd.read_parquet(SWEEP)
    print(f"sweep trees: {len(sweep)}", flush=True)

    # reference signals (OSS / regret_alwaysstop at current cost) from the existing parquet
    sig = pd.read_parquet(SIG)
    ref_cols = ["fen", f"oss__{CUR}", f"regret_alwaysstop__{CUR}"]
    sig = sig[[c for c in ref_cols if c in sig.columns]].copy()

    # drop sweep's tree-side legal_moves; the DB's n_possible_moves is the RT-population width
    sweep = sweep.drop(columns=[c for c in ["legal_moves"] if c in sweep.columns])

    conn = duckdb.connect(DB, read_only=True)
    conn.register("_sig", sweep[["fen"]])
    moves = conn.execute(
        """
        SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m
        JOIN (SELECT DISTINCT fen FROM _sig) f ON m.fen = f.fen
        WHERE m.move_time > 0
        """
    ).df()
    conn.close()

    df = moves.merge(sweep, on="fen", how="inner").merge(sig, on="fen", how="left")
    print(f"matched human moves: {len(df):,} across {df.fen.nunique():,} FENs", flush=True)

    rt = df["log_rt"].to_numpy(float)
    lm = df["legal_moves"].to_numpy(float)

    # --- tau sweep results -------------------------------------------------
    res = {"cur": {"rt": {}, "lm": {}}, "free": {"rt": {}, "lm": {}}}
    print("\n=== Spearman(softmax_voc(tau), log RT) and (tau, legal_moves) ===", flush=True)
    for ck in ("cur", "free"):
        print(f"\n  cost={ck}", flush=True)
        for tau in TAUS:
            col = f"softmax_voc_tau{_tau_label(tau)}__{ck}"
            v = df[col].to_numpy(float)
            res[ck]["rt"][tau] = _spearman_boot(v, rt)
            res[ck]["lm"][tau] = _spearman_boot(v, lm)
            a = res[ck]["rt"][tau]
            b = res[ck]["lm"][tau]
            print(
                f"    tau={tau:<4}  vsRT  rho={a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]   "
                f"vsLegalMoves rho={b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]  n={a[3]:,}",
                flush=True,
            )

    # --- reference 'how much to think' signals at current cost -------------
    print("\n=== reference (current cost) vs log RT ===", flush=True)
    refs = {}
    for name, col in [("OSS", f"oss__{CUR}"), ("regret_alwaysstop", f"regret_alwaysstop__{CUR}")]:
        if col in df:
            r = _spearman_boot(df[col].to_numpy(float), rt)
            refs[name] = r
            print(f"  {name:18s} rho={r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}] n={r[3]:,}", flush=True)
    # also legal_moves vs RT for context
    r = _spearman_boot(lm, rt)
    refs["legal_moves"] = r
    print(f"  {'legal_moves':18s} rho={r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}] n={r[3]:,}", flush=True)

    # --- plot: rho vs tau, two panels (cur, free), two curves each --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    panels = [("cur", "current pipeline cost\n(power-law p=2.8, λ=18.537)"),
              ("free", "cost-free (λ=0)")]
    x = np.array(TAUS)
    for ax, (ck, title) in zip(axes, panels):
        for which, color, lab in [("rt", MAIN, "vs human log-RT"),
                                  ("lm", ACC, "vs legal moves")]:
            pts = np.array([res[ck][which][t][0] for t in TAUS])
            lo = np.array([res[ck][which][t][1] for t in TAUS])
            hi = np.array([res[ck][which][t][2] for t in TAUS])
            ax.plot(x, pts, "o-", color=color, label=lab, lw=2)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        # reference line: regret_alwaysstop vs RT (current cost) as the bar to beat
        if "regret_alwaysstop" in refs:
            ax.axhline(refs["regret_alwaysstop"][0], color="#7F8C8D", ls="--", lw=1.2,
                       label="regret-alwaysstop vs RT (cur)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:g}" for t in TAUS])
        ax.set_xlabel("softmax temperature τ (win-prob units)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Softmax value-of-computation: temperature sweep (filtered elo2000)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGDIR}/voc_tau_sweep.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {FIGDIR}/voc_tau_sweep.(png|pdf)", flush=True)

    # emit a compact machine-readable summary for the report prose
    print("\n=== SUMMARY (for report) ===", flush=True)
    print(f"n_moves={len(df)}  n_fens={df.fen.nunique()}", flush=True)
    for ck in ("cur", "free"):
        for which in ("rt", "lm"):
            cells = "  ".join(
                f"tau{_tau_label(t)}:{res[ck][which][t][0]:+.3f}[{res[ck][which][t][1]:+.3f},{res[ck][which][t][2]:+.3f}]"
                for t in TAUS
            )
            print(f"{ck}/{which}: {cells}", flush=True)
    for name, r in refs.items():
        print(f"ref {name}: {r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
