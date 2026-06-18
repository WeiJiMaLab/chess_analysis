"""Tree-derived "generated values" vs human reaction time, on the lc0-tree subset.

For a sample of the canonical ``human_trees`` set (lc0 search trees on human FENs) we
derive FOUR quantities from each tree, then join the tree's root FEN — normalized to
4-field, since payloads may carry a 6-field root_position_spec — to human ``move_time``
in ``processed_moves_nonzero`` and plot each vs log(RT) in the standard quantile-bin
dashboard style:

  * GSS         — greedy stopping step: the first expansion at which the eventual-best
                  move is recommended (``oracle_best_move_index`` first equals its final
                  value) — the cost-free "search effort to find the best move." This is
                  the budgeted oracle's stop at zero cost; we use it instead of the
                  cost-aware DP optimal_stop_step (which, under the default time cost,
                  bails almost immediately and is not an interpretable difficulty proxy).
  * Gain        — value of computation = final_Q(deep best) − final_Q(1-ply best),
                  i.e. how much deep search improves on the shallow (1-ply value-head
                  lookahead) choice. ≥ 0. (Internally the column is still named ``voc``;
                  it is displayed as "Gain".)
  * Action Gap  — MYOPIC gap between the root's best and second-best move by the
                  children's 1-ply value-head backup (not a deep/converged gap). ≥ 0.
                  Gain and Action Gap share the same 1-ply value-head lookahead basis.
  * MQ          — move quality of the HUMAN's actual move = final_Q(move played)
                  − final_Q(best), the post-search (full-budget) Lc0 root value
                  loss. ≤ 0 (0 = the human played the engine-best move). This is
                  the Lc0 definition of MQ — it replaces the Stockfish ``mq``
                  (e_win_taken − e_win_best) entirely, and is the only per-move
                  quantity here: it needs the human's played UCI (``moves.move_uci``)
                  matched to the tree's ``oracle_root_moves`` (= all legal moves).

All four are on the SAME subset (positions that have a generated lc0 tree), so
they are directly comparable, and are read from the lc0 search itself rather than
re-run on Stockfish.

The per-tree/per-move values are deterministic in (trees, n_trees, seed) and are
CACHED to parquet (``--cache-dir``, key includes the trees-dir basename). The first run
loads all .pt trees (slow → run on the cluster); after that, tweaking the joins/plots
reloads the cache in seconds and can run locally — no slurm. Use ``--refresh`` to recompute.

Usage:
    # one-time compute (populates the cache), on the cluster:
    sbatch human_analytics/slurm/tree_values.slurm
    # iterate on plots later, locally, straight from the cache:
    PYTHONPATH=human_analytics python human_analytics/tree_values_analysis.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from utils import Variable, Analyzer  # noqa: E402
from utils.helpers import apply_poster_style  # noqa: E402
from utils.plots import highlight_corr_row  # noqa: E402

_TREES_DEFAULT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_CACHE_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache"
_FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def _tree_voc_and_gap(payload) -> tuple[float, float]:
    """(Gain, Action Gap) for one tree — both built on the SAME 1-ply value-head
    lookahead backup of the root's children (parent perspective = −child.value).
    (The returned first quantity is stored in the ``voc`` column and displayed as "Gain".)

    Action Gap = top1 − top2 of the children's 1-ply backups: the immediate value
                 separation between the best and second-best move at 1-ply (NOT a
                 deep/converged gap).
    Gain       = final_Q(deep best) − final_Q(1-ply best): how much deep search
                 improves on the shallow (1-ply value-head) choice = value of
                 computation. ≥ 0; perspective-invariant (both are root side-to-move
                 Qs in one node).

    Why the 1-ply backup and not the search trace for the shallow choice: every legal
    root move is present as an evaluated child (verified: n_children == n_legal, no
    uninitialized zeros), so −child.value is a clean one-step lookahead. The
    ``oracle_root_q_trace`` instead stores a move's Q as 0 until its child is first
    visited (trace[0] is all-zero pre-search), so a trace-based shallow choice is
    contaminated — in losing positions the unvisited zeros beat the visited negatives,
    which previously inflated Gain to a spurious ~1.0 mass.
    """
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    vi = feature_names.index("value")
    final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
    incoming = payload["incoming_moves"]
    root_moves = list(payload["oracle_root_moves"])

    action_gap = float("nan")
    voc = float("nan")
    roots = np.where(par < 0)[0]
    if roots.size:
        kids = np.where(par == int(roots[0]))[0]
        if kids.size >= 2:
            myopic_q = -nf[kids, vi]  # parent-perspective 1-ply value-head backup
            order = np.argsort(myopic_q)[::-1]
            action_gap = float(myopic_q[order[0]] - myopic_q[order[1]])
            # Gain — deep-search regret of the 1-ply best move. Map that child to its
            # root-move index (via its incoming UCI) to read the deep final_Q.
            if final_q.size >= 2:
                uci = incoming[int(kids[order[0]])]
                if uci in root_moves:
                    a_shallow = root_moves.index(uci)
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


def _tree_gss(payload) -> int:
    """Greedy Stopping Step: the first expansion at which the eventual-best move is
    recommended — ``oracle_best_move_index`` first equals its final value. This is the
    cost-free "search effort to find the best move" (== the budgeted oracle's stop at
    zero cost), NOT the cost-aware DP optimal_stop_step (which, under the default time
    cost, bails almost immediately). ∈ [0, n_steps-1]."""
    bmi = np.asarray(payload["oracle_best_move_index"]).ravel()
    if bmi.size == 0:
        return -1
    return int(np.argmax(bmi == bmi[-1]))


def _worker(path: str):
    """Load one tree; return (fen, gss, voc, action_gap, root_ucis, root_mqs) or None.
    (``voc`` is the Gain metric — value of computation; kept as the ``voc`` column name.)

    The last two are parallel per-root-move lists (UCI, Lc0 MQ) used to attach MQ to
    whichever of those moves the human actually played. CPU-bound, 1 thread.
    """
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        gss = _tree_gss(t)
        if gss < 0:
            return None
        voc, gap = _tree_voc_and_gap(t)
        ucis, mqs = _tree_root_mq(t)
        # Join key is the 4-field FEN (placement/stm/castling/ep). These trees may
        # store a 6-field root_position_spec (with move counters); processed_moves_nonzero.fen
        # is 4-field, so normalize or the FEN join silently misses everything.
        fen = " ".join(t["root_position_spec"].split()[:4])
        return (fen, gss, voc, gap, ucis, mqs)
    except Exception:  # noqa: BLE001
        return None


def compute_values(trees_dir: str, n_trees: int, seed: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample trees; return (per-tree DataFrame(fen, gss, voc, action_gap),
    per-root-move DataFrame(fen, move_uci, mq)). The second is exploded one row per
    legal root move so the human's played UCI can be joined to its Lc0 MQ."""
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    tree_rows, move_rows = [], []
    with mp.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, paths, chunksize=64), total=len(paths), desc="trees"):
            if r is None:
                continue
            fen, gss, voc, gap, ucis, mqs = r
            tree_rows.append({"fen": fen, "gss": gss, "voc": voc, "action_gap": gap})
            for u, q in zip(ucis, mqs):
                move_rows.append({"fen": fen, "move_uci": u, "mq": q})
    return pd.DataFrame(tree_rows), pd.DataFrame(move_rows)


def plot_lc0_correlation_matrix(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """Spearman ρ matrix over the lc0-tree metrics + structure/RT, one row per joined
    human move (``tree_rt`` ⋈ ``mq_rt`` ⋈ ``processed_moves_nonzero``). Spearman (rank)
    because the tree metrics (Gain, MQ, Action Gap) are zero-inflated/skewed with
    monotone-but-nonlinear relations that Pearson understates."""
    df = conn.execute("""
        SELECT ln(t.move_time) AS log_T, t.move_ply AS ply,
               p.n_possible_moves AS branching, p.n_self_pieces_exc_pawns AS own_material,
               m.mq, t.action_gap, t.voc, t.gss
        FROM tree_rt t
        JOIN mq_rt m ON m.gid = t.gid AND m.move_ply = t.move_ply
        JOIN processed_moves_nonzero p ON p.gid = t.gid AND p.move_ply = t.move_ply
        WHERE t.move_time > 0
    """).df()
    # Column/row order is fixed here: RT → structure → move quality → search depth.
    labels = {
        "log_T": "log(RT)", "ply": "Ply", "branching": "Branching",
        "own_material": "Own Material", "mq": "MQ", "action_gap": "Action Gap",
        "voc": "Gain", "gss": "GSS",
    }
    corr = df[list(labels)].corr(method="spearman").rename(columns=labels, index=labels)
    n = len(corr)
    apply_poster_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(False)
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, fontsize=20, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=20)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=16,
                    color="white" if abs(val) > 0.5 else "black",
                    fontweight="bold" if i == j else "normal")
    highlight_corr_row(ax, n)  # log(RT) row (index 0) — the response variable
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title(f"Spearman correlation — lc0 metrics (n = {len(df):,})", fontsize=20, pad=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees-dir", default=_TREES_DEFAULT)
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--db", default=_DB_DEFAULT)
    parser.add_argument("--cache-dir", default=_CACHE_DEFAULT,
                        help="where the per-tree/per-move values are cached (parquet)")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute from the trees even if a cache exists")
    args = parser.parse_args(argv)

    # The per-tree/per-move values are deterministic in (trees, n_trees, seed), so cache
    # them: tweaking the join or the plots then reloads the cache in seconds (run locally —
    # no slurm, no re-loading 100k .pt trees). Key includes the trees-dir basename so
    # different tree sets cache separately.
    key = f"{os.path.basename(args.trees_dir.rstrip('/'))}_{args.n_trees}_{args.seed}"
    cache = Path(args.cache_dir)
    vals_path, rm_path = cache / f"vals_{key}.parquet", cache / f"rootmoves_{key}.parquet"
    if not args.refresh and vals_path.exists() and rm_path.exists():
        print(f"Loading cached values from {cache} (key={key}; --refresh to recompute) …")
        vals, root_moves = pd.read_parquet(vals_path), pd.read_parquet(rm_path)
    else:
        print(f"Deriving GSS/Gain/Action Gap/MQ on {args.n_trees:,} trees ({args.n_workers} workers) …")
        vals, root_moves = compute_values(args.trees_dir, args.n_trees, args.seed, args.n_workers)
        cache.mkdir(parents=True, exist_ok=True)
        vals.to_parquet(vals_path)
        root_moves.to_parquet(rm_path)
        print(f"  cached → {cache} (key={key})")
    print(f"  {len(vals):,} trees (GSS {vals['gss'].min()}–{vals['gss'].max()}); "
          f"{len(root_moves):,} root moves for MQ.")

    conn = duckdb.connect(args.db, read_only=False)
    conn.register("_vals", vals)
    conn.register("_root_moves", root_moves)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE tree_rt AS
        SELECT v.gss, v.voc, v.action_gap, v.fen, m.gid, m.move_ply, m.move_time
        FROM _vals v
        JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0
    """)
    n_rows = conn.execute("SELECT count(*) FROM tree_rt").fetchone()[0]
    n_fen = conn.execute("SELECT count(DISTINCT fen) FROM tree_rt").fetchone()[0]
    print(f"  joined {n_rows:,} human moves across {n_fen:,} FENs.")
    for col in ("gss", "voc", "action_gap"):
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
    # GSS / Gain / Action Gap are properties of the position/search, so we plot human
    # RT (y) as a function of the value (x). Per-figure binning (see Analyzer docstring):
    #   * GSS is an INTEGER count (expansions until the best move is first found);
    #     ``ntile`` splits its heavy low-value mass across identical-mean bins → spurious
    #     swings, so bin in fixed groups of 5 (each plotted at its group mean). No cap:
    #     unlike the old cost-aware stop, GSS↔RT is positive across the full range.
    #   * Gain / Action Gap have a large near-zero mass: lump the near-zero rows and
    #     bin the interior TIE-SAFE with few (8) bins so a single value can't straddle
    #     adjacent bins. (Gain's former ~1.0 spike was a bug in the shallow-choice — now
    #     fixed at source in _tree_voc_and_gap — so no edge_mass is needed.)
    # min_bin_count drops the noisy sparse tails (high Gain / high GSS) from both panels.
    # spec: (table, col, label, fname, kwargs-for-Analyzer)
    specs = [
        ("tree_rt", "gss", "Greedy stop step", "gss_vs_rt.png",
         dict(bin_mode="integer", integer_bin_width=5,
              filter_query="move_time > 0", min_bin_count=100)),
        ("tree_rt", "voc", "Gain (lc0 tree)", "gain_vs_rt.png",
         dict(zero_inflated=True, zero_threshold=0.05, tie_safe=True, n_bins=8,
              filter_query="move_time > 0", min_bin_count=100)),
        ("tree_rt", "action_gap", "Action Gap (lc0 tree)", "actiongap_vs_rt.png",
         dict(zero_inflated=True, zero_threshold=0.05, tie_safe=True, n_bins=8,
              filter_query="move_time > 0", min_bin_count=100)),
    ]
    for table, col, label, fname, bin_kwargs in specs:
        analyzer = Analyzer(
            conn,
            table,
            x_var=Variable(column=col, is_log=False, name=label),
            y_var=Variable(column="move_time", is_log=True, name="RT"),
            title=f"{label} vs. log(RT)",
            **bin_kwargs,
        )
        analyzer.save_dashboard(str(_FIGURES_DIR / fname))

    # MQ is the OUTCOME of the human's decision, so plot mean move quality (y) as a
    # function of think time: log RT on x, MQ on y ("does thinking longer yield better
    # moves?"). Binning on the well-behaved RT axis also sidesteps the MQ point-masses
    # (the −1.0 / 0.0 spikes) entirely, so no zero-lump / tie-safe / edge-mass needed.
    Analyzer(
        conn,
        "mq_rt",
        x_var=Variable(column="move_time", is_log=True, name="RT (s)"),
        y_var=Variable(column="mq", is_log=False, name="MQ (lc0 tree)"),
        filter_query="move_time > 0",
        title="MQ (lc0 tree) vs. log(RT)",
        min_bin_count=100,
    ).save_dashboard(str(_FIGURES_DIR / "mq_vs_rt.png"))

    # Relationships among the lc0 metrics (+ structure/RT): Spearman ρ matrix.
    plot_lc0_correlation_matrix(conn, str(_FIGURES_DIR / "correlation_matrix.png"))
    conn.close()


if __name__ == "__main__":
    main()
