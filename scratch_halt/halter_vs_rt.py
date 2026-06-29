"""Causal halter vs hindsight oracle, as RT predictors.

The budgeted-oracle stop step (OSS) is non-causal: backward DP with full knowledge of the
rolled-out search. The trained tree-stats halter is causal: at each step it sees only the
tree-so-far. If humans stop on causally-available cues, the halter's predicted stop step
should track RT BETTER than the hindsight OSS. Fit the StatsReadout by PG on train, roll
greedily on held-out validation, join to human RT by FEN.
"""
import sys, glob, re
sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")
import torch, numpy as np, pandas as pd, duckdb
from pathlib import Path
from cts.analysis._budgeted import evaluate as E

PACK = "/scratch/gpfs/GRIFFITHS/hl4291/sf_mc_packed/elo2000"
ROOTS = "/scratch/gpfs/GRIFFITHS/hl4291/sf_ladder_roots_50k.txt"  # pack built on the 50k roots; index→FEN
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"


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

    for label, sub in [("all budgets", df), ("large+ (B>=26)", df[df.budget >= 26]),
                       ("very-large (B>=61)", df[df.budget >= 61])]:
        agg = sub.groupby("fen").agg(halter_stop=("halter_stop", "mean"),
                                     oracle_stop=("oracle_stop", "mean")).reset_index()
        j = moves.merge(agg, on="fen", how="inner")
        rt = j["log_rt"].to_numpy()
        print(f"\n=== {label}: {len(j):,} moves, {j.fen.nunique():,} fens ===", flush=True)
        for nm in ["halter_stop", "oracle_stop", "legal_moves"]:
            p, lo, hi, n = _sb(j[nm].to_numpy(float), rt)
            print(f"  {nm:13s} vs log RT  rho={p:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
