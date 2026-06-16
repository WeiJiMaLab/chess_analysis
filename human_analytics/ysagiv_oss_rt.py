"""OSS (oracle stop step) from the ysagiv lc0 trees vs human reaction time.

For a sample of the ysagiv ``human_trees`` we compute the budgeted-DP oracle
``optimal_stop_step`` (OSS) per tree, join each tree's root FEN (4-field) to the
human move times in ``processed_moves_nonzero``, and plot OSS vs log(RT) in the
standard quantile-bin dashboard style.

This is the **real per-tree value-of-computation** signal from the lc0 search
trees — as opposed to ``voc_budget_analysis.py``, which only re-runs Stockfish at
a 96-node budget. OSS is high when the oracle keeps finding decision-changing
value deep into the search; the hypothesis is that humans also think longer there.

Usage (from chess_analysis/):
    PYTHONPATH=human_analytics python human_analytics/ysagiv_oss_rt.py --n-trees 10000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# cts oracle machinery lives in lmcos (imported as the `src` layout, like
# lmcos/analysis/human_oracle_comparison.py does).
_LMCOS = Path(__file__).resolve().parent.parent / "lmcos"
if str(_LMCOS) not in sys.path:
    sys.path.insert(0, str(_LMCOS))
from src.data.preprocess_mc.oracle import BudgetedOracleConfig  # noqa: E402
from src.data.preprocess_mc.pack import (  # noqa: E402
    budgeted_oracle_from_trajectory,
    build_compact_trajectory_from_payload,
)

# human_analytics dashboard style (quantile bins, ply tertiles, seconds axis).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import Variable, Analyzer  # noqa: E402

_TREES_DEFAULT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()
_BUDGET = 96  # set per-run in compute_oss before the worker pool forks


def _oss_worker(path: str):
    """Load one tree and return (root_fen, oss); None on failure. CPU-bound, 1 thread."""
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        traj = build_compact_trajectory_from_payload(t)
        if traj is None:
            return None
        oss = budgeted_oracle_from_trajectory(traj, _BUDGET, _CONFIG).optimal_stop_step
        return (t["root_position_spec"], int(oss))
    except Exception:  # noqa: BLE001
        return None


def compute_oss(trees_dir: str, n_trees: int, seed: int, budget: int, n_workers: int) -> pd.DataFrame:
    """Sample ``n_trees`` trees and return a DataFrame of (root fen, oss).

    OSS is *derived* from the stored oracle trace via the budgeted DP oracle (it is
    not a stored scalar — it depends on the cost model). The DP is cheap; the cost
    is loading individual ``.pt`` files, so we parallelize across ``n_workers``.
    """
    global _BUDGET
    _BUDGET = budget
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    rows = []
    with mp.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_oss_worker, paths, chunksize=64), total=len(paths), desc="OSS"):
            if r is not None:
                rows.append({"fen": r[0], "oss": r[1]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees-dir", default=_TREES_DEFAULT)
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--db", default=_DB_DEFAULT)
    parser.add_argument("--output", default=str(_FIGURES_DIR / "oss_vs_rt.png"))
    args = parser.parse_args(argv)

    print(f"Computing OSS on {args.n_trees:,} sampled trees from {args.trees_dir} ({args.n_workers} workers) …")
    oss_df = compute_oss(args.trees_dir, args.n_trees, args.seed, args.budget, args.n_workers)
    print(f"  OSS computed for {len(oss_df):,} trees (OSS range {oss_df['oss'].min()}–{oss_df['oss'].max()}).")

    conn = duckdb.connect(args.db, read_only=False)
    conn.register("_oss_df", oss_df)
    # One row per (tree FEN × human move at that FEN). Most fens.txt positions are
    # unique so this is ~1:1, but common positions contribute every instance.
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE oss_rt AS
        SELECT o.oss, o.fen, m.gid, m.move_ply, m.move_time
        FROM _oss_df o
        JOIN processed_moves_nonzero m ON m.fen = o.fen
        WHERE m.move_time > 0
    """)
    n_rows = conn.execute("SELECT count(*) FROM oss_rt").fetchone()[0]
    n_fen = conn.execute("SELECT count(DISTINCT fen) FROM oss_rt").fetchone()[0]
    print(f"  Joined {n_rows:,} human moves across {n_fen:,} matched FENs.")

    d = conn.execute("SELECT oss, ln(move_time) AS lrt FROM oss_rt").df()
    pear = float(d["oss"].corr(d["lrt"]))
    spear = float(d["oss"].corr(d["lrt"], method="spearman"))
    print(f"\n  r(OSS, log RT)  Pearson = {pear:+.4f}   Spearman = {spear:+.4f}   (n = {len(d):,})")

    analyzer = Analyzer(
        conn,
        "oss_rt",
        x_var=Variable(column="oss", is_log=False, name="Oracle stop step"),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        filter_query="move_time > 0",
        title=f"Oracle stop step (lc0 tree) vs. log(RT)   ρ = {spear:+.3f}",
    )
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    analyzer.save_dashboard(args.output)
    conn.close()


if __name__ == "__main__":
    main()
