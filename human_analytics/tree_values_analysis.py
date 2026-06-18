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
  * Gain        — value of computation = final_Q(best @ 96 expansions) − final_Q(best @ 1
                  expansion), both scored on the converged 96-expansion root Q, read from
                  the SAME growing oracle tree (the 1-expansion best is the first move the
                  search expands; the 1-expansion tree is a subset of the 96-expansion one,
                  so they line up exactly). ≥ 0. (Internally the column is still named
                  ``voc``; it is displayed as "Gain".)
  * Action Gap  — MYOPIC gap between the root's best and second-best move by the
                  children's 1-ply value-head backup (not a deep/converged gap). ≥ 0.
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
from utils.helpers import apply_poster_style, FONT_SIZE_LABEL  # noqa: E402
from utils.plots import highlight_corr_row  # noqa: E402

_TREES_DEFAULT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_CACHE_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache"
_FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def _tree_voc_and_gap(payload) -> tuple[float, float]:
    """(Gain, Action Gap) for one tree. (The first is stored in the ``voc`` column and
    displayed as "Gain".)

    Gain = value of computation, read from the SINGLE growing oracle tree:

        Gain = final_Q(best move @ 96 expansions) − final_Q(best move @ 1 expansion)

    where BOTH moves are scored on the converged 96-expansion root Q
    (``oracle_final_root_q_values``). The 96-expansion best is ``argmax final_Q`` (==
    ``oracle_best_move_index[-1]``). The 1-expansion best is the first root move the search
    actually expands — the smallest first-visit step in ``oracle_root_q_trace`` — which,
    under PUCT's cold start, is the policy's top move. Because the search only ever *adds*
    nodes, the 1-expansion tree is a subset of the 96-expansion tree, so both moves live on
    the same board and Gain ≥ 0 by construction (a_96 = argmax). a_shallow is read from what
    the search *expanded* (any nonzero q-trace entry), so it is always a real, visited move —
    we never read the uninitialized 0.0 of a never-visited move, the bug two earlier
    definitions hit (the all-zero step-0 trace, and the unvisited-0.0 deep lookup of a
    1-ply-value-head choice) that spuriously pinned Gain at ≈ 1.0 in won positions. (Note:
    final_Q[a_shallow] may legitimately be ≈ 0 for a drawish move — that is its true
    converged value, not an unvisited placeholder, so it is used as-is.) A won position now
    gives Gain ≈ 0 (the policy's first pick is already the win; search adds nothing).

    Action Gap = top1 − top2 of the children's 1-ply value-head backup (parent perspective
    = −child.value): the immediate value separation at 1-ply (NOT a deep/converged gap, and
    no longer sharing Gain's basis — Gain now comes from the search trace, not the 1-ply
    head).
    """
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    vi = feature_names.index("value")
    final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()

    action_gap = float("nan")
    voc = float("nan")
    roots = np.where(par < 0)[0]
    if roots.size:
        kids = np.where(par == int(roots[0]))[0]
        if kids.size >= 2:
            myopic_q = -nf[kids, vi]  # parent-perspective 1-ply value-head backup
            order = np.argsort(myopic_q)[::-1]
            action_gap = float(myopic_q[order[0]] - myopic_q[order[1]])
            # Gain from the growing tree: a_deep = best @ 96 expansions; a_shallow = the
            # first root move the search expanded (smallest first-visit step in the q-trace
            # — under PUCT's cold start this is the policy's top move). A move is "expanded"
            # the moment it gets a real Q (any nonzero q-trace entry); both moves are scored
            # on the converged final_Q. No further filtering: the first-expanded move is by
            # definition in the tree, and Gain ≥ 0 holds anyway because a_deep = argmax(final_Q)
            # (final_Q[a_shallow] is its true converged Q — which may legitimately be ≈ 0 for a
            # drawish move; that is a real value, not the unvisited-0.0 the old defs misread).
            qt = np.asarray(payload["oracle_root_q_trace"], dtype=float)
            qt = qt.reshape(qt.shape[0], -1)
            if qt.shape[1] == final_q.size and final_q.size >= 2:
                visited = qt != 0.0
                if visited.any():
                    steps = np.where(visited, np.arange(qt.shape[0])[:, None], qt.shape[0])
                    first_visit = np.where(visited.any(axis=0), steps.min(axis=0), qt.shape[0] + 1)
                    a_shallow = int(np.argmin(first_visit))  # first move the search expanded
                    a_deep = int(np.argmax(final_q))         # 96-expansion best
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


def _tree_hpi(payload) -> float:
    """Shannon entropy H(π) (nats) of the lc0 POLICY PRIOR over the root's legal moves —
    the network's ex-ante uncertainty over which move is best ("prior argmax
    uncertainty"), i.e. an effective branching factor (raw move count weighted by
    plausibility). Read from ``node_features[:, prior]`` of the root's children — the
    policy head's move probabilities (they sum to ~1 over the legal moves; renormalized
    for safety), NOT the value head. This is the P1 test's predictor: deliberation time
    should grow with H(π). NaN if the root has < 2 children."""
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    pi_idx = feature_names.index("prior")
    roots = np.where(par < 0)[0]
    if not roots.size:
        return float("nan")
    kids = np.where(par == int(roots[0]))[0]
    if kids.size < 2:
        return float("nan")
    p = nf[kids, pi_idx].astype(float)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _spearman_partials(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> None:
    """Print raw + partial Spearman ρ(x, y) controlling for each of ``controls`` (and
    all jointly). Spearman = Pearson on ranks; partials by residualizing the ranks of
    x and y on the rank(s) of the control(s) via least squares, then correlating the
    residuals."""
    df = df[[x, y] + controls].dropna()  # Gain (voc) is NaN on degenerate trees → drop
    R = df.rank()

    def resid(col: str, ctrl: list[str]) -> np.ndarray:
        A = np.c_[np.ones(len(R)), R[ctrl].to_numpy()]
        beta, *_ = np.linalg.lstsq(A, R[col].to_numpy(), rcond=None)
        return R[col].to_numpy() - A @ beta

    print(f"\n  === P1: H(π) prior argmax-uncertainty vs {y} (Spearman, n={len(df):,}) ===")
    print(f"  raw ρ(H(π), {y})              = {R[x].corr(R[y]):+.4f}")
    for c in controls:
        rho = float(np.corrcoef(resid(x, [c]), resid(y, [c]))[0, 1])
        print(f"  partial ρ(H(π), {y} | {c:<9s}) = {rho:+.4f}")
    rho_all = float(np.corrcoef(resid(x, controls), resid(y, controls))[0, 1])
    print(f"  partial ρ(H(π), {y} | all)        = {rho_all:+.4f}")
    print(f"  [context] raw ρ(legal moves, {y})  = {df['legal_moves'].rank().corr(R[y]):+.4f}")
    print(f"  [context] raw ρ(H(π), legal moves) = {R[x].corr(df['legal_moves'].rank()):+.4f}")


def _worker(path: str):
    """Load one tree; return (fen, gss, voc, action_gap, h_pi, root_ucis, root_mqs) or None.
    (``voc`` is the Gain metric — value of computation; kept as the ``voc`` column name;
    ``h_pi`` is the root policy-prior entropy H(π).)

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
        hpi = _tree_hpi(t)
        ucis, mqs = _tree_root_mq(t)
        # Join key is the 4-field FEN (placement/stm/castling/ep). These trees may
        # store a 6-field root_position_spec (with move counters); processed_moves_nonzero.fen
        # is 4-field, so normalize or the FEN join silently misses everything.
        fen = " ".join(t["root_position_spec"].split()[:4])
        return (fen, gss, voc, gap, hpi, ucis, mqs)
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
            fen, gss, voc, gap, hpi, ucis, mqs = r
            tree_rows.append({"fen": fen, "gss": gss, "voc": voc, "action_gap": gap, "h_pi": hpi})
            for u, q in zip(ucis, mqs):
                move_rows.append({"fen": fen, "move_uci": u, "mq": q})
    return pd.DataFrame(tree_rows), pd.DataFrame(move_rows)


def save_mq_dashboard(analyzer: Analyzer, gss_analyzer: Analyzer, output_path: str) -> None:
    """MQ-vs-RT dashboard — a 1×3 poster figure of the SAME MQ-vs-log-RT relationship
    segmented three ways: [global | by ply tertile | by GSS difficulty stratum]. All three
    panels are rendered by the identical Analyzer machinery (same binning, axes, seconds
    ticks, styling); the right panel only swaps the segmentation column (ply → GSS), so it
    differs from the centre panel *only in its legend*. ``gss_analyzer`` is a second Analyzer
    on the same mq_rt table constructed with ``segment_column='gss'``."""
    apply_poster_style()
    fig, axes = plt.subplots(1, 3, figsize=(45, 13.72))
    analyzer.plot_quantile_bins(axes[0])
    analyzer.plot_quantile_bins_tertile_segmented(axes[1])
    gss_analyzer.plot_quantile_bins_tertile_segmented(axes[2])
    fig.subplots_adjust(top=0.80)
    suptitle = fig.suptitle(f"{analyzer.title}\nn = {analyzer.n_moves:,} moves",
                            fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    extra = [suptitle] + [ax.get_legend() for ax in axes if ax.get_legend() is not None]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.3)
    plt.close()
    print(f"✅ Dashboard saved to {output_path}")


def plot_lc0_correlation_matrix(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """Spearman ρ matrix over the lc0-tree metrics + structure/RT, one row per joined
    human move (``tree_rt`` ⋈ ``mq_rt`` ⋈ ``processed_moves_nonzero``). Spearman (rank)
    because the tree metrics (Gain, MQ, Action Gap) are zero-inflated/skewed with
    monotone-but-nonlinear relations that Pearson understates."""
    df = conn.execute("""
        SELECT ln(t.move_time) AS log_T, t.move_ply AS ply,
               p.n_possible_moves AS legal_moves, t.h_pi,
               p.n_self_pieces_exc_pawns AS own_material,
               m.mq, t.action_gap, t.voc, t.gss
        FROM tree_rt t
        JOIN mq_rt m ON m.gid = t.gid AND m.move_ply = t.move_ply
        JOIN processed_moves_nonzero p ON p.gid = t.gid AND p.move_ply = t.move_ply
        WHERE t.move_time > 0
    """).df()
    # Column/row order is fixed here: RT → structure (incl. policy entropy) → move
    # quality → value-of-search → search depth.
    labels = {
        "log_T": "log(RT)", "ply": "Ply", "legal_moves": "Legal moves", "h_pi": "H(π)",
        "own_material": "Own Material", "mq": "MQ", "action_gap": "Action Gap",
        "voc": "Gain", "gss": "GSS",
    }
    corr = df[list(labels)].corr(method="spearman").rename(columns=labels, index=labels)
    n = len(corr)
    apply_poster_style()
    fig, ax = plt.subplots(figsize=(11, 9))
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
    use_cache = (not args.refresh) and vals_path.exists() and rm_path.exists()
    if use_cache:
        print(f"Loading cached values from {cache} (key={key}; --refresh to recompute) …")
        vals, root_moves = pd.read_parquet(vals_path), pd.read_parquet(rm_path)
        if "h_pi" not in vals.columns:  # cache predates the H(π) column → recompute
            print("  cached values predate H(π); recomputing from the trees …")
            use_cache = False
    if not use_cache:
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
        SELECT v.gss, v.voc, v.action_gap, v.h_pi, v.fen, m.gid, m.move_ply, m.move_time
        FROM _vals v
        JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0
    """)
    n_rows = conn.execute("SELECT count(*) FROM tree_rt").fetchone()[0]
    n_fen = conn.execute("SELECT count(DISTINCT fen) FROM tree_rt").fetchone()[0]
    print(f"  joined {n_rows:,} human moves across {n_fen:,} FENs.")
    for col in ("gss", "voc", "action_gap", "h_pi"):
        r = conn.execute(f"SELECT corr({col}, ln(move_time)) FROM tree_rt WHERE {col} IS NOT NULL").fetchone()[0]
        print(f"  r({col}, log RT) = {r:+.4f}")

    # P1 — does the prior policy entropy H(π) predict RT, and does it survive controlling
    # for legal moves / Gain (voc) / GSS? legal_moves = n_possible_moves from the human table.
    p1 = conn.execute("""
        SELECT ln(t.move_time) AS log_T, t.h_pi,
               p.n_possible_moves AS legal_moves, t.voc, t.gss
        FROM tree_rt t
        JOIN processed_moves_nonzero p ON p.gid = t.gid AND p.move_ply = t.move_ply
        WHERE t.move_time > 0 AND t.h_pi IS NOT NULL
    """).df()
    _spearman_partials(p1, "h_pi", "log_T", ["legal_moves", "voc", "gss"])

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
        SELECT rm.mq, h.fen, h.gid, h.move_ply, h.move_time, v.gss
        FROM human h
        JOIN _root_moves rm ON rm.fen = h.fen AND rm.move_uci = h.move_uci
        JOIN _vals v ON v.fen = h.fen
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
    #     adjacent bins. (Gain is now the growing-tree definition in _tree_voc_and_gap —
    #     best@96 vs best@1 expansion on the converged Q — which has no spurious ~1.0 spike;
    #     two earlier definitions did, from reading an unvisited-0.0 shallow value.)
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

    # MQ is the OUTCOME of the human's decision, so plot mean move quality (y) as a function
    # of think time: log RT on x, MQ on y. THREE panels side by side — all MQ-vs-log-RT,
    # segmented differently: global | by ply tertile | by GSS difficulty stratum. The third
    # is the difficulty-confound test (MQ↔RT stays negative within every GSS stratum).
    # Binning on the well-behaved RT axis sidesteps the MQ point-masses (the −1.0 / 0.0 spikes).
    mq_kwargs = dict(
        x_var=Variable(column="move_time", is_log=True, name="RT (s)"),
        y_var=Variable(column="mq", is_log=False, name="MQ (lc0 tree)"),
        filter_query="move_time > 0",
        title="MQ (lc0 tree) vs. log(RT)",
        min_bin_count=100,
    )
    mq_analyzer = Analyzer(conn, "mq_rt", **mq_kwargs)
    # Same plot, segmented by GSS stratum instead of ply tertile (cuts a-priori from mq_rt's
    # gss). Rendered by the identical Analyzer panel → differs from the by-ply panel only in
    # the legend. (This is the difficulty-confound test: MQ↔RT slope within GSS strata.)
    gss_seg_analyzer = Analyzer(conn, "mq_rt", segment_column="gss", segment_source="mq_rt",
                                segment_label="GSS", **mq_kwargs)
    save_mq_dashboard(mq_analyzer, gss_seg_analyzer, str(_FIGURES_DIR / "mq_vs_rt.png"))

    # Relationships among the lc0 metrics (+ structure/RT): Spearman ρ matrix.
    plot_lc0_correlation_matrix(conn, str(_FIGURES_DIR / "correlation_matrix.png"))
    conn.close()


if __name__ == "__main__":
    main()
