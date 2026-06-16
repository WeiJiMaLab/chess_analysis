"""Tree-derived "generated values" vs human reaction time, on the lc0-tree subset.

For a sample of the ysagiv ``human_trees`` we derive THREE quantities from each
lc0 search tree, then join the tree's root FEN (4-field) to human ``move_time`` in
``processed_moves_nonzero`` and plot each vs log(RT) in the standard quantile-bin
dashboard style:

  * OSS         — oracle stop step, budgeted DP oracle on the expansion trace
                  (optimal_stop_step; cost model = canonical BudgetedOracleConfig).
  * VOC         — value of computation = final_Q(best) − final_Q(shallow choice),
                  where the shallow choice is the best move after the first
                  expansion step (lc0 analog of the Stockfish depth-VOC). ≥ 0.
  * Action Gap  — final_Q(best) − final_Q(second best) at the root (lc0 analog of
                  Stockfish ``toptwo``). ≥ 0.

All three are on the SAME subset (positions that have a generated lc0 tree), so
they are directly comparable. This replaces the Stockfish proxies: OSS supersedes
the node-budget VOC_budget, and VOC / Action Gap are read from the lc0 search
itself rather than re-run on Stockfish.

NOTE: run on the cluster via slurm/tree_values.slurm (it loads many .pt trees).
It is intentionally NOT part of the standard full-dataset analysis.

Usage (cluster):
    PYTHONPATH=human_analytics python human_analytics/tree_values_analysis.py \
        --n-trees 100000 --n-workers 40
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

# cts oracle machinery (lmcos), imported via the `src` layout like
# lmcos/analysis/human_oracle_comparison.py.
_LMCOS = Path(__file__).resolve().parent.parent / "lmcos"
if str(_LMCOS) not in sys.path:
    sys.path.insert(0, str(_LMCOS))
from src.data.preprocess_mc.oracle import BudgetedOracleConfig  # noqa: E402
from src.data.preprocess_mc.pack import (  # noqa: E402
    budgeted_oracle_from_trajectory,
    build_compact_trajectory_from_payload,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import Variable, Analyzer  # noqa: E402

_TREES_DEFAULT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()
_BUDGET = 96  # set per-run in compute_values before the worker pool forks


def _tree_voc_and_gap(payload) -> tuple[float, float]:
    """(VOC, Action Gap) from a tree's root oracle Q values.

    Action Gap = best − second-best of the final root Q values.
    VOC        = final_Q(final-best) − final_Q(first-step best); the shallow choice
                 is the argmax of the first expansion step's root Q trace.
    """
    final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
    if final_q.size < 2:
        return float("nan"), float("nan")
    order = np.argsort(final_q)[::-1]
    action_gap = float(final_q[order[0]] - final_q[order[1]])

    trace = np.asarray(payload["oracle_root_q_trace"], dtype=float)  # [steps, n_moves]
    voc = float("nan")
    if trace.ndim == 2 and trace.shape[0] >= 1 and trace.shape[1] == final_q.size:
        first = trace[0]
        if np.isfinite(first).any():
            a_shallow = int(np.nanargmax(first))
            a_deep = int(order[0])
            voc = float(final_q[a_deep] - final_q[a_shallow])
    return voc, action_gap


def _worker(path: str):
    """Load one tree; return (fen, oss, voc, action_gap) or None. CPU-bound, 1 thread."""
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        traj = build_compact_trajectory_from_payload(t)
        if traj is None:
            return None
        oss = int(budgeted_oracle_from_trajectory(traj, _BUDGET, _CONFIG).optimal_stop_step)
        voc, gap = _tree_voc_and_gap(t)
        return (t["root_position_spec"], oss, voc, gap)
    except Exception:  # noqa: BLE001
        return None


def compute_values(trees_dir: str, n_trees: int, seed: int, budget: int, n_workers: int) -> pd.DataFrame:
    """Sample trees and return DataFrame(fen, oss, voc, action_gap)."""
    global _BUDGET
    _BUDGET = budget
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    rows = []
    with mp.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, paths, chunksize=64), total=len(paths), desc="trees"):
            if r is not None:
                rows.append({"fen": r[0], "oss": r[1], "voc": r[2], "action_gap": r[3]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees-dir", default=_TREES_DEFAULT)
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--db", default=_DB_DEFAULT)
    args = parser.parse_args(argv)

    print(f"Deriving OSS/VOC/Action Gap on {args.n_trees:,} trees ({args.n_workers} workers) …")
    vals = compute_values(args.trees_dir, args.n_trees, args.seed, args.budget, args.n_workers)
    print(f"  {len(vals):,} trees (OSS {vals['oss'].min()}–{vals['oss'].max()}).")

    conn = duckdb.connect(args.db, read_only=False)
    conn.register("_vals", vals)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE tree_rt AS
        SELECT v.oss, v.voc, v.action_gap, v.fen, m.gid, m.move_ply, m.move_time
        FROM _vals v
        JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0
    """)
    n_rows = conn.execute("SELECT count(*) FROM tree_rt").fetchone()[0]
    n_fen = conn.execute("SELECT count(DISTINCT fen) FROM tree_rt").fetchone()[0]
    print(f"  joined {n_rows:,} human moves across {n_fen:,} FENs.")
    for col in ("oss", "voc", "action_gap"):
        r = conn.execute(f"SELECT corr({col}, ln(move_time)) FROM tree_rt").fetchone()[0]
        print(f"  r({col}, log RT) = {r:+.4f}")

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # OSS is not zero-inflated; VOC / Action Gap have a mass near 0 (search did not
    # change / barely separated the decision) → lump the near-zero point.
    specs = [
        ("oss", "Oracle stop step", "oss_vs_rt.png", False, 0.0),
        ("voc", "VOC (lc0 tree)", "voc_vs_rt.png", True, 0.05),
        ("action_gap", "Action Gap (lc0 tree)", "actiongap_vs_rt.png", True, 0.05),
    ]
    for col, label, fname, zinf, thr in specs:
        analyzer = Analyzer(
            conn,
            "tree_rt",
            x_var=Variable(column=col, is_log=False, name=label),
            y_var=Variable(column="move_time", is_log=True, name="RT"),
            filter_query="move_time > 0",
            title=f"{label} vs. log(RT)",
            zero_inflated=zinf,
            zero_threshold=thr,
        )
        analyzer.save_dashboard(str(_FIGURES_DIR / fname))
    conn.close()


if __name__ == "__main__":
    main()
