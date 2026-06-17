"""Tree-derived "generated values" vs human reaction time, on the lc0-tree subset.

For a sample of the ysagiv ``human_trees`` we derive FOUR quantities from each
lc0 search tree, then join the tree's root FEN (4-field) to human ``move_time`` in
``processed_moves_nonzero`` and plot each vs log(RT) in the standard quantile-bin
dashboard style:

  * OSS         — oracle stop step, budgeted DP oracle on the expansion trace
                  (optimal_stop_step; cost model = canonical BudgetedOracleConfig).
  * VOC         — value of computation = final_Q(best) − final_Q(shallow choice),
                  where the shallow choice is the best move after the first
                  expansion step (lc0 analog of the Stockfish depth-VOC). ≥ 0.
  * Action Gap  — MYOPIC gap between the root's best and second-best child by the
                  children's 1-ply value-head value (not a deep/converged gap). ≥ 0.
  * MQ          — move quality of the HUMAN's actual move = final_Q(move played)
                  − final_Q(best), the post-search (full-budget) Lc0 root value
                  loss. ≤ 0 (0 = the human played the engine-best move). This is
                  the Lc0 definition of MQ — it replaces the Stockfish ``mq``
                  (e_win_taken − e_win_best) entirely, and is the only per-move
                  quantity here: it needs the human's played UCI (``moves.move_uci``)
                  matched to the tree's ``oracle_root_moves`` (= all legal moves).

All four are on the SAME subset (positions that have a generated lc0 tree), so
they are directly comparable. This replaces the Stockfish proxies: OSS supersedes
the node-budget VOC_budget, and VOC / Action Gap / MQ are read from the lc0 search
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
_FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
_CONFIG = BudgetedOracleConfig()
_BUDGET = 96  # set per-run in compute_values before the worker pool forks


def _tree_voc_and_gap(payload) -> tuple[float, float]:
    """(VOC, Action Gap) for one tree.

    Action Gap = **myopic** gap between the root's best and second-best child by the
                 children's 1-ply value-head value (parent perspective = −child.value).
                 This is the immediate value separation, NOT a deep/converged gap.
    VOC        = final_Q(final-best) − final_Q(first-step best): the deep-converged
                 regret of the shallow (first-expansion) choice = value of computation.
    """
    # Action Gap — myopic, from the root children's value-head values.
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    vi = feature_names.index("value")
    action_gap = float("nan")
    roots = np.where(par < 0)[0]
    if roots.size:
        kids = np.where(par == int(roots[0]))[0]
        if kids.size >= 2:
            myopic_q = -nf[kids, vi]  # parent-perspective 1-ply value (negamax flip)
            top = np.sort(myopic_q)[::-1]
            action_gap = float(top[0] - top[1])

    # VOC — deep-converged regret of the first-step (shallow) choice.
    final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
    trace = np.asarray(payload["oracle_root_q_trace"], dtype=float)  # [steps, n_moves]
    voc = float("nan")
    if final_q.size >= 2 and trace.ndim == 2 and trace.shape[0] >= 1 \
            and trace.shape[1] == final_q.size and np.isfinite(trace[0]).any():
        a_shallow = int(np.nanargmax(trace[0]))
        a_deep = int(np.argmax(final_q))
        voc = float(final_q[a_deep] - final_q[a_shallow])
    return voc, action_gap


def _tree_root_mq(payload) -> tuple[list[str], list[float]]:
    """Per-root-move Lc0 move quality: MQ(move) = final_Q(move) − max final_Q ≤ 0.

    ``oracle_final_root_q_values`` is the post-search (full-budget) root Q per legal
    move from the side-to-move's perspective; the engine-best move is argmax (verified
    == ``oracle_best_move_index[-1]``), so MQ = 0 ⇔ the move *is* engine-best. The
    paired ``oracle_root_moves`` give the UCI of each move, which is matched downstream
    to the human's played move. Returns (uci_list, mq_list); empty on malformed trees.
    """
    moves = list(payload["oracle_root_moves"])
    fq = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
    if not moves or fq.size != len(moves) or not np.isfinite(fq).all():
        return [], []
    mq = (fq - float(np.max(fq))).tolist()  # ≤ 0, best move = 0
    return moves, mq


def _worker(path: str):
    """Load one tree; return (fen, oss, voc, action_gap, root_ucis, root_mqs) or None.

    The last two are parallel per-root-move lists (UCI, Lc0 MQ) used to attach MQ to
    whichever of those moves the human actually played. CPU-bound, 1 thread.
    """
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        traj = build_compact_trajectory_from_payload(t)
        if traj is None:
            return None
        oss = int(budgeted_oracle_from_trajectory(traj, _BUDGET, _CONFIG).optimal_stop_step)
        voc, gap = _tree_voc_and_gap(t)
        ucis, mqs = _tree_root_mq(t)
        return (t["root_position_spec"], oss, voc, gap, ucis, mqs)
    except Exception:  # noqa: BLE001
        return None


def compute_values(trees_dir: str, n_trees: int, seed: int, budget: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample trees; return (per-tree DataFrame(fen, oss, voc, action_gap),
    per-root-move DataFrame(fen, move_uci, mq)). The second is exploded one row per
    legal root move so the human's played UCI can be joined to its Lc0 MQ."""
    global _BUDGET
    _BUDGET = budget
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    tree_rows, move_rows = [], []
    with mp.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, paths, chunksize=64), total=len(paths), desc="trees"):
            if r is None:
                continue
            fen, oss, voc, gap, ucis, mqs = r
            tree_rows.append({"fen": fen, "oss": oss, "voc": voc, "action_gap": gap})
            for u, q in zip(ucis, mqs):
                move_rows.append({"fen": fen, "move_uci": u, "mq": q})
    return pd.DataFrame(tree_rows), pd.DataFrame(move_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees-dir", default=_TREES_DEFAULT)
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--db", default=_DB_DEFAULT)
    args = parser.parse_args(argv)

    print(f"Deriving OSS/VOC/Action Gap/MQ on {args.n_trees:,} trees ({args.n_workers} workers) …")
    vals, root_moves = compute_values(args.trees_dir, args.n_trees, args.seed, args.budget, args.n_workers)
    print(f"  {len(vals):,} trees (OSS {vals['oss'].min()}–{vals['oss'].max()}); "
          f"{len(root_moves):,} root moves for MQ.")

    conn = duckdb.connect(args.db, read_only=False)
    conn.register("_vals", vals)
    conn.register("_root_moves", root_moves)
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

    # MQ is per played move: attach each subset human move's actual UCI (moves.move_uci)
    # then match it to its Lc0 root-move MQ on (fen, move_uci). Build the human side
    # first (1 row/move) to avoid a fan-out over the ~30 root moves per FEN.
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE mq_rt AS
        WITH human AS (
            SELECT m.fen, m.gid, m.move_ply, m.move_time, mv.move_uci
            FROM (SELECT DISTINCT fen FROM _root_moves) f
            JOIN processed_moves_nonzero m ON m.fen = f.fen AND m.move_time > 0
            JOIN moves mv ON mv.gid = m.gid AND mv.move_ply = m.move_ply
        )
        SELECT rm.mq, h.fen, h.gid, h.move_ply, h.move_time
        FROM human h
        JOIN _root_moves rm ON rm.fen = h.fen AND rm.move_uci = h.move_uci
        WHERE h.move_time > 0
    """)
    n_human = conn.execute("""
        SELECT count(*) FROM (SELECT DISTINCT fen FROM _root_moves) f
        JOIN processed_moves_nonzero m ON m.fen = f.fen AND m.move_time > 0
    """).fetchone()[0]
    n_mq = conn.execute("SELECT count(*) FROM mq_rt").fetchone()[0]
    r_mq = conn.execute("SELECT corr(mq, ln(move_time)) FROM mq_rt").fetchone()[0]
    print(f"  MQ: matched {n_mq:,}/{n_human:,} human moves to a root move "
          f"({100.0 * n_mq / max(n_human, 1):.1f}%).")
    print(f"  r(mq, log RT) = {r_mq:+.4f}")

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # OSS is not zero-inflated; VOC / Action Gap have a mass near 0 (search did not
    # change / barely separated the decision); MQ has a large mass at exactly 0
    # (human played the engine-best move) → lump the near-zero point for those.
    specs = [
        ("tree_rt", "oss", "Oracle stop step", "oss_vs_rt.png", False, 0.0),
        ("tree_rt", "voc", "VOC (lc0 tree)", "voc_vs_rt.png", True, 0.05),
        ("tree_rt", "action_gap", "Action Gap (lc0 tree)", "actiongap_vs_rt.png", True, 0.05),
        ("mq_rt", "mq", "MQ (lc0 tree)", "mq_vs_rt.png", True, 0.05),
    ]
    for table, col, label, fname, zinf, thr in specs:
        analyzer = Analyzer(
            conn,
            table,
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
