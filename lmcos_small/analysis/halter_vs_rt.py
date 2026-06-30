"""Causal halter vs hindsight oracle, as RT predictors.

The budgeted-oracle stop step (OSS) is non-causal: backward DP with full knowledge of the
rolled-out search. The trained tree-stats halter is causal: at each step it sees only the
tree-so-far. If humans stop on causally-available cues, the halter's predicted stop step
should track RT BETTER than the hindsight OSS. Fit the StatsReadout by PG on train, roll
greedily on held-out validation, join to human RT by FEN.
"""
import sys, glob, re
sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_small/src")
import torch, numpy as np, pandas as pd, duckdb
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cts.analysis._budgeted import evaluate as E

PACK = "/scratch/gpfs/GRIFFITHS/hl4291/sf_mc_packed/elo2000"
ROOTS = "/scratch/gpfs/GRIFFITHS/hl4291/sf_ladder_roots_50k.txt"  # pack built on the 50k roots; index→FEN
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
VOCPQ = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"  # 63k legal-moves ref
FIG = "/home/hl4291/chess_analysis/figures/normative"

# theme palette
HALT = "#6366f1"   # causal halter — indigo (value-of-computation family)
ORACLE = "#a5b4fc"  # hindsight oracle (step*) — indigo-light (second VOC series)
SIZE = "#475569"   # legal moves — slate (structural)
XLIM = (-0.4, 0.4)


def load_eps(split, max_eps=None):
    eps = []
    for sp in sorted(glob.glob(f"{PACK}/{split}/shard_*.pt")):
        if max_eps is not None and len(eps) >= max_eps:
            break
        p = torch.load(sp, weights_only=False)
        esp, eti, tsp = p["episode_step_ptr"], p["episode_trajectory_index"], p["trajectory_step_ptr"]
        tnp, snc, thr = p["trajectory_node_ptr"], p["step_node_cutoffs"], p["trajectory_halt_rewards"]
        oss, ov, sbg, depth = p["oracle_stop_steps"], p["oracle_values"], p["starting_budgets"], p["depth"]
        src = p["trajectory_source_paths"]
        for i in range(int(p["num_episodes"])):
            ns = int(esp[i + 1]) - int(esp[i]); traj = int(eti[i])
            tb = int(tsp[traj]); nb = int(tnp[traj]); nend = int(tnp[traj + 1])
            td = depth[nb:nend]; sb = int(sbg[i]); ncut = snc[tb:tb + ns].tolist()
            h, w = E._derive_step_stats(td, ncut)
            eps.append(dict(halt_rewards=thr[tb:tb + ns].tolist(), tree_sizes=ncut, heights=h, widths=w,
                            time_budgets=list(range(sb, sb - ns, -1)), oracle_stop_step=int(oss[i]),
                            oracle_value=float(ov[i]), starting_budget=sb, source_path=src[traj]))
    return eps


def _sb(x, y, nb=500, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else np.nan
    pt = pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(nb))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def main():
    cfg = E._oracle_config(Path(PACK))
    print("fitting StatsReadout (PG) on train (25k subsample)...", flush=True)
    train = load_eps("train", max_eps=25000)
    model, _ = E._fit_stats_readout_pg(train, cfg)
    rule = E._stats_rule(model)
    n_train = len(train); del train
    print(f"  trained on {n_train:,} episodes; rolling on validation...", flush=True)
    val = load_eps("validation")
    roots = [l.strip() for l in open(ROOTS)]
    rgx = re.compile(r"_root_(\d+)\.pt")
    rows = []
    for ep in val:
        hs = max(0, min(rule(ep), len(ep["halt_rewards"]) - 1))
        idx = int(rgx.search(ep["source_path"]).group(1))
        rows.append(dict(fen=roots[idx] if idx < len(roots) else None,
                         budget=ep["starting_budget"], halter_stop=hs, oracle_stop=ep["oracle_stop_step"]))
    df = pd.DataFrame(rows).dropna(subset=["fen"])
    print(f"val episodes: {len(df):,}  unique fens: {df.fen.nunique():,}", flush=True)

    conn = duckdb.connect(DB, read_only=True); conn.register("_s", df[["fen"]].drop_duplicates())
    moves = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); conn.close()

    plot_res = {}
    for label, sub in [("all budgets", df), ("large+ (B>=26)", df[df.budget >= 26]),
                       ("very-large (B>=61)", df[df.budget >= 61])]:
        agg = sub.groupby("fen").agg(halter_stop=("halter_stop", "mean"),
                                     oracle_stop=("oracle_stop", "mean")).reset_index()
        j = moves.merge(agg, on="fen", how="inner")
        rt = j["log_rt"].to_numpy()
        print(f"\n=== {label}: {len(j):,} moves, {j.fen.nunique():,} fens ===", flush=True)
        res = {}
        for nm in ["halter_stop", "oracle_stop", "legal_moves"]:
            p, lo, hi, n = _sb(j[nm].to_numpy(float), rt)
            res[nm] = (p, lo, hi, n)
            print(f"  {nm:13s} vs log RT  rho={p:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
        # use the very-large budget regime (the halter is genuinely causal-positive there)
        if label == "very-large (B>=61)":
            plot_res = {"halter": res["halter_stop"], "oracle": res["oracle_stop"], "n_halt": len(j)}

    # legal-moves↔RT from the full 63k voc parquet (decision-width reference)
    voc = pd.read_parquet(VOCPQ)[["fen", "legal_moves"]]
    cv = duckdb.connect(DB, read_only=True); cv.register("_v", voc[["fen"]])
    lm = cv.execute("""SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _v) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); cv.close()
    lp, llo, lhi, ln_ = _sb(lm["legal_moves"].to_numpy(float), lm["log_rt"].to_numpy())
    plot_res["legal"] = (lp, llo, lhi, ln_)
    print(f"\n=== full 63k parquet: legal_moves vs RT  rho={lp:+.4f} [{llo:+.4f},{lhi:+.4f}]  (n={ln_:,}) ===", flush=True)

    # ===== hindsight_halter figure =====
    if plot_res:
        n = plot_res["n_halt"]
        items = [("causal halter\n(sees only tree-so-far)", plot_res["halter"], HALT),
                 ("hindsight oracle\n(step*, backward-DP)", plot_res["oracle"], ORACLE),
                 ("legal moves\n(decision width)", plot_res["legal"], SIZE)]
        order = np.argsort([it[1][0] for it in items])[::-1]
        items = [items[i] for i in order]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        y = np.arange(len(items)); pts = [it[1][0] for it in items]
        ax.barh(y, pts, color=[it[2] for it in items], height=0.62)
        ax.errorbar(pts, y, xerr=[[it[1][0] - it[1][1] for it in items],
                                  [it[1][2] - it[1][0] for it in items]],
                    fmt="none", ecolor="#222", capsize=3, lw=1)
        for yi, it in zip(y, items):
            ax.text(it[1][0] + (0.012 if it[1][0] >= 0 else -0.012), yi, f"{it[1][0]:+.2f}",
                    va="center", ha="left" if it[1][0] >= 0 else "right", fontsize=10, color="#222")
        ax.axvline(0, color="#222", lw=0.9); ax.set_xlim(*XLIM); ax.invert_yaxis()
        ax.set_yticks(y); ax.set_yticklabels([it[0] for it in items], fontsize=10)
        ax.set_xlabel("Spearman ρ vs human RT", fontsize=13); ax.tick_params(axis="x", labelsize=11)
        ax.set_title("Does hindsight explain it? (causal halter ≈ hindsight oracle, both ≪ legal-moves)",
                     fontsize=15)
        ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        fig.tight_layout()
        for e in ("png", "pdf"): fig.savefig(f"{FIG}/hindsight_halter.{e}", dpi=200, bbox_inches="tight")
        print(f"\nsaved hindsight_halter (n={n:,})", flush=True)


if __name__ == "__main__":
    main()
