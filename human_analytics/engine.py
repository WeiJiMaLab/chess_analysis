"""
Unified engine-level and stopping analyses (DuckDB and tree evaluation).
Consolidates engine metrics, timing benchmarks, difficulty confound stats,
and the large-scale tree-values join from reports/engine.md.

All generated plots are saved in PDF format under figures/engine/.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import duckdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Align imports with human_analytics structure
from utils import Variable, Analyzer
from utils.helpers import (
    apply_poster_style,
    db_connection,
    get_lc0_engine,
    get_stockfish_engine,
    STOCKFISH_SF14_DIR,
    STOCKFISH_SF14_PATH,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
)
from utils.plots import (
    highlight_corr_row,
    save_figure,
)
from utils.selected_db import (
    SELECTED_DB_DEFAULT,
    TABLE_PROCESSED_MOVES_NONZERO,
)

# Cache defaults
_TREES_DEFAULT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_CACHE_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache"
_KEY_DEFAULT = "human_trees_200000_7"


# =============================================================================
# Core Engine Metrics (collapsed from engine_analysis.py)
# =============================================================================

def _win_prob(info: chess.engine.InfoDict, board: chess.Board) -> float | None:
    """Win probability for the side to move from a single engine InfoDict."""
    if board.is_checkmate():
        return 0.0
    if board.is_game_over():
        return 0.5
    score = info.get("score")
    if score is None:
        return None
    wdl = score.pov(board.turn).wdl()
    return (wdl.wins + 0.5 * wdl.draws) / wdl.total()


@dataclass
class PositionEval:
    """All engine-derived quantities for one (position, move) pair."""
    e_win_best: float | None
    e_win_second_best: float | None
    e_win_taken: float | None
    voc: float | None

    @property
    def mq(self) -> float | None:
        """MQ = e_win_taken - e_win_best  (≤ 0; 0 = optimal move)."""
        if self.e_win_taken is None or self.e_win_best is None:
            return None
        return min(0.0, self.e_win_taken - self.e_win_best)

    @property
    def toptwo(self) -> float | None:
        """toptwo = e_win_best - e_win_second_best  (≥ 0)."""
        if self.e_win_best is None or self.e_win_second_best is None:
            return None
        return max(0.0, self.e_win_best - self.e_win_second_best)


def evaluate_position(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    depth_deep: int = 5,
    depth_shallow: int = 1,
    nodes_deep: int | None = None,
    nodes_shallow: int | None = None,
) -> PositionEval:
    """Evaluate one (position, move) pair and return all engine quantities."""
    if board.is_game_over():
        p = 0.5 if not board.is_checkmate() else 0.0
        return PositionEval(e_win_best=p, e_win_second_best=p, e_win_taken=p, voc=0.0)

    use_nodes = nodes_deep is not None
    limit_shallow = chess.engine.Limit(nodes=nodes_shallow or 1) if use_nodes else chess.engine.Limit(depth=depth_shallow)
    limit_deep = chess.engine.Limit(nodes=nodes_deep) if use_nodes else chess.engine.Limit(depth=depth_deep)

    # --- Step 1: shallow search → a_shallow ---
    shallow_info = engine.analyse(board, limit_shallow, multipv=1)
    if not shallow_info:
        return PositionEval(None, None, None, None)
    pv_shallow = shallow_info[0].get("pv")
    a_shallow = pv_shallow[0] if pv_shallow else None

    # --- Step 2: deep search multipv=2 ---
    deep_info = engine.analyse(board, limit_deep, multipv=2)
    if not deep_info:
        return PositionEval(None, None, None, None)

    e_win_best = _win_prob(deep_info[0], board)
    e_win_second_best = _win_prob(deep_info[1], board) if len(deep_info) > 1 else e_win_best

    pv_deep = deep_info[0].get("pv")
    a_deep = pv_deep[0] if pv_deep else None

    # Scan top-2 lines for e_win_taken and V_deep(a_shallow)
    e_win_taken = None
    v_deep_shallow = None
    for info in deep_info:
        pv = info.get("pv")
        if not pv:
            continue
        first_move = pv[0]
        val = _win_prob(info, board)
        if first_move == move:
            e_win_taken = val
        if a_shallow is not None and first_move == a_shallow:
            v_deep_shallow = val

    # --- Step 3: fallback for e_win_taken ---
    if e_win_taken is None:
        fb = engine.analyse(board, limit_deep, root_moves=[move])
        e_win_taken = _win_prob(fb, board) if fb else None

    # --- Step 4: fallback for V_deep(a_shallow) ---
    if a_shallow is not None and v_deep_shallow is None:
        fb = engine.analyse(board, limit_deep, root_moves=[a_shallow])
        v_deep_shallow = _win_prob(fb, board) if fb else None

    # VOC
    if e_win_best is None or a_shallow is None or v_deep_shallow is None:
        voc_val = None
    elif a_shallow == a_deep:
        voc_val = 0.0
    else:
        voc_val = max(0.0, e_win_best - v_deep_shallow)

    return PositionEval(
        e_win_best=e_win_best,
        e_win_second_best=e_win_second_best,
        e_win_taken=e_win_taken,
        voc=voc_val,
    )


def move_quality(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    depth: int = 5,
) -> float | None:
    """MQ = e_win_taken - e_win_best (≤ 0)."""
    return evaluate_position(board, move, engine, depth_deep=depth).mq


def voc(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth_deep: int = 5,
    depth_shallow: int = 1,
) -> float | None:
    """VOC = V_deep(a_deep) - V_deep(a_shallow) (≥ 0)."""
    dummy_move = next(iter(board.legal_moves), None)
    if dummy_move is None:
        return None
    return evaluate_position(board, dummy_move, engine, depth_deep=depth_deep,
                             depth_shallow=depth_shallow).voc


# =============================================================================
# Timing Comparisons (collapsed from timing_comparison.py)
# =============================================================================

def _sample_positions(db_path: str, n: int, seed: int = 7) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT fen, move_uci, player_clock_time, move_time
        FROM (
            SELECT fen, move_uci, player_clock_time, move_time
            FROM (
                SELECT fen, move_uci, player_clock_time, move_time, gid,
                       ROW_NUMBER() OVER (PARTITION BY fen ORDER BY gid) AS _rn
                FROM (
                    SELECT pm.fen, pm.player_clock_time, pm.move_time, pm.gid, m.move_uci
                    FROM {TABLE_PROCESSED_MOVES_NONZERO} pm
                    JOIN moves m ON pm.gid = m.gid AND pm.move_ply = m.move_ply
                    WHERE pm.move_ply > 10
                ) filtered
            ) dedup
            WHERE _rn = 1
        ) deduped
        USING SAMPLE {n} ROWS (RESERVOIR, {seed})
    """).df()
    conn.close()
    return df


def _open_stockfish() -> chess.engine.SimpleEngine:
    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH_SF14_PATH, cwd=STOCKFISH_SF14_DIR)
    e.configure({"Threads": 1, "Hash": 32})
    return e


def _time_depth(
    rows: list[dict],
    engine: chess.engine.SimpleEngine,
    engine_name: str,
    depth: int,
) -> dict:
    times = []
    for row in rows:
        try:
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(row["move_uci"])
            if board.is_game_over() or move not in board.legal_moves:
                continue
            if engine_name == "stockfish":
                engine.configure({"Clear Hash": None})
            t0 = time.perf_counter()
            mq_val = move_quality(board, move, engine, depth=depth)
            voc_val = voc(board, engine, depth_deep=depth, depth_shallow=1)
            elapsed = time.perf_counter() - t0
            if mq_val is not None and voc_val is not None:
                times.append(elapsed)
        except Exception:
            pass
    if not times:
        return {"engine": engine_name, "depth": depth, "n": 0,
                "mean_s": float("nan"), "total_s": float("nan")}
    import statistics
    return {
        "engine": engine_name,
        "depth": depth,
        "n": len(times),
        "mean_s": statistics.mean(times),
        "total_s": sum(times),
    }


def _extrapolate(mean_s: float, parallelism: int = 10) -> tuple[str, str]:
    t10k = mean_s * 10_000 / parallelism
    t100k = mean_s * 100_000 / parallelism
    def _fmt(s: float) -> str:
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"
    return _fmt(t10k), _fmt(t100k)


def run_timing_comparison(db_path: str, n_positions: int, depths: list[int], seed: int, parallelism: int):
    """Run timing benchmark comparisons between Stockfish and lc0."""
    print(f"Sampling {n_positions} middlegame positions…")
    df = _sample_positions(db_path, n_positions, seed=seed)
    print(f"  Got {len(df)} unique positions.")
    rows = df.to_dict("records")

    results = []
    for depth in depths:
        print(f"\n--- Stockfish depth={depth} ---")
        try:
            sf = _open_stockfish()
            try:
                r = _time_depth(rows, sf, "stockfish", depth)
                results.append(r)
                print(f"  mean {r['mean_s']:.2f}s/pos  (n={r['n']})")
            finally:
                sf.quit()
        except Exception as e:
            print(f"  Failed to run Stockfish depth={depth}: {e}")

    for depth in depths:
        print(f"\n--- lc0 depth={depth} ---")
        try:
            lc0 = get_lc0_engine(threads=1)
            try:
                r = _time_depth(rows, lc0, "lc0", depth)
                results.append(r)
                print(f"  mean {r['mean_s']:.2f}s/pos  (n={r['n']})")
            finally:
                lc0.quit()
        except Exception as e:
            print(f"  Failed to run lc0 depth={depth}: {e}")

    print("\n" + "=" * 72)
    print(f"{'Engine':<12} {'Depth':<8} {'n':<6} {'s/pos':<10} "
          f"{'10K ({:d}×par)':<16} {'100K ({:d}×par)':<16}".format(parallelism, parallelism))
    print("-" * 72)
    for r in results:
        t10k, t100k = _extrapolate(r["mean_s"], parallelism)
        print(f"{r['engine']:<12} {r['depth']:<8} {r['n']:<6} "
              f"{r['mean_s']:<10.2f} {t10k:<16} {t100k:<16}")
    print("=" * 72)


# =============================================================================
# Difficulty Confound Statistics (collapsed from mq_vs_rt_by_gss.py)
# =============================================================================

def load_played_moves(cache_dir: Path, key: str, db_path: str) -> pd.DataFrame:
    """Load and join human played moves with tree-derived variables."""
    vals = pd.read_parquet(cache_dir / f"vals_{key}.parquet")[["fen", "gss"]]
    root_moves = pd.read_parquet(cache_dir / f"rootmoves_{key}.parquet")[["fen", "move_uci", "mq"]]

    conn = duckdb.connect(db_path, read_only=True)
    conn.register("_vals", vals)
    conn.register("_root_moves", root_moves)
    df = conn.execute("""
        WITH human AS (
            SELECT m.fen, m.gid, m.move_ply, m.move_time,
                   m.n_possible_moves AS legal_moves, mv.move_uci
            FROM (SELECT DISTINCT fen FROM _root_moves) f
            JOIN processed_moves_nonzero m ON m.fen = f.fen AND m.move_time > 0
            JOIN moves mv ON mv.gid = m.gid AND mv.move_ply = m.move_ply
        )
        SELECT rm.mq, ln(h.move_time) AS log_rt, v.gss, h.legal_moves
        FROM human h
        JOIN _root_moves rm ON rm.fen = h.fen AND rm.move_uci = h.move_uci
        JOIN _vals v ON v.fen = h.fen
    """).df()
    conn.close()
    return df.dropna(subset=["mq", "log_rt", "gss", "legal_moves"]).reset_index(drop=True)


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    """Spearman ρ(x, y) controlling for covariates."""
    R = df[[x, y] + controls].rank()
    A = np.c_[np.ones(len(R)), R[controls].to_numpy()]

    def resid(col: str) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, R[col].to_numpy(), rcond=None)
        return R[col].to_numpy() - A @ beta

    return float(np.corrcoef(resid(x), resid(y))[0, 1])


def gss_strata(df: pd.DataFrame) -> pd.Series:
    """Quantile-bin GSS difficulty strata."""
    edges = df["gss"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    edges = np.unique(edges)
    labels = ["Easy (low GSS)", "Medium GSS", "Hard (high GSS)"][: len(edges) - 1]
    return pd.cut(df["gss"], bins=edges, labels=labels, include_lowest=True)


def run_difficulty_confound_stats(cache_dir: Path, key: str, db_path: str):
    """Print the difficulty-confound partial Spearman correlation reports."""
    df = load_played_moves(cache_dir, key, db_path)
    print(f"\n=== MQ↔RT, conditioning on difficulty (n = {len(df):,} played moves) ===")
    print("    MQ ≤ 0 (more negative = worse); a NEGATIVE ρ(MQ, logRT) = worse moves on longer thinks.\n")

    overall = df["mq"].rank().corr(df["log_rt"].rank())
    print(f"  overall  ρ(MQ, logRT)              = {overall:+.4f}")

    df = df.assign(stratum=gss_strata(df))
    print("\n  within-GSS-stratum ρ(MQ, logRT):")
    for name, sub in df.groupby("stratum", observed=True):
        rho = sub["mq"].rank().corr(sub["log_rt"].rank())
        print(f"    {name:<18s} GSS {int(sub['gss'].min()):>2d}–{int(sub['gss'].max()):<2d} "
              f"n={len(sub):>7,}  ρ={rho:+.4f}")

    print("\n  partial Spearman ρ(MQ, logRT | …):")
    print(f"    | GSS                = {partial_spearman(df, 'mq', 'log_rt', ['gss']):+.4f}")
    print(f"    | legal moves        = {partial_spearman(df, 'mq', 'log_rt', ['legal_moves']):+.4f}")
    print(f"    | GSS + legal moves  = {partial_spearman(df, 'mq', 'log_rt', ['gss', 'legal_moves']):+.4f}")


# =============================================================================
# Tree Values Pipeline and Dashboards (collapsed from tree_values_analysis.py)
# =============================================================================

def _tree_voc_and_gap(payload) -> tuple[float, float]:
    """Extract Gain and Action Gap from tree payload."""
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    vi = feature_names.index("value")
    final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()

    action_gap = float("nan")
    voc_val = float("nan")
    roots = np.where(par < 0)[0]
    if roots.size:
        kids = np.where(par == int(roots[0]))[0]
        if kids.size >= 2:
            myopic_q = -nf[kids, vi]
            order = np.argsort(myopic_q)[::-1]
            action_gap = float(myopic_q[order[0]] - myopic_q[order[1]])
            qt = np.asarray(payload["oracle_root_q_trace"], dtype=float)
            qt = qt.reshape(qt.shape[0], -1)
            if qt.shape[1] == final_q.size and final_q.size >= 2:
                visited = qt != 0.0
                if visited.any():
                    steps = np.where(visited, np.arange(qt.shape[0])[:, None], qt.shape[0])
                    first_visit = np.where(visited.any(axis=0), steps.min(axis=0), qt.shape[0] + 1)
                    a_shallow = int(np.argmin(first_visit))
                    a_deep = int(np.argmax(final_q))
                    voc_val = float(final_q[a_deep] - final_q[a_shallow])
    return voc_val, action_gap


def _tree_root_mq(payload) -> tuple[list[str], list[float]]:
    """Extract Lc0 root-move MQ from tree payload."""
    moves = list(payload["oracle_root_moves"])
    fq = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
    if not moves or fq.size != len(moves) or not np.isfinite(fq).all():
        return [], []
    mq = (fq - float(np.max(fq))).tolist()
    return moves, mq


def _tree_gss(payload) -> int:
    """Extract Greedy Stopping Step from tree payload."""
    bmi = np.asarray(payload["oracle_best_move_index"]).ravel()
    if bmi.size == 0:
        return -1
    return int(np.argmax(bmi == bmi[-1]))


def _tree_hpi(payload) -> float:
    """Extract policy prior entropy H(π)."""
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


def _worker(path: str):
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        gss = _tree_gss(t)
        if gss < 0:
            return None
        voc_val, gap = _tree_voc_and_gap(t)
        hpi = _tree_hpi(t)
        ucis, mqs = _tree_root_mq(t)
        fen = " ".join(t["root_position_spec"].split()[:4])
        return (fen, gss, voc_val, gap, hpi, ucis, mqs)
    except Exception:
        return None


def compute_values(trees_dir: str, n_trees: int, seed: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    tree_rows, move_rows = [], []
    with mp.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, paths, chunksize=64), total=len(paths), desc="trees"):
            if r is None:
                continue
            fen, gss, voc_val, gap, hpi, ucis, mqs = r
            tree_rows.append({"fen": fen, "gss": gss, "voc": voc_val, "action_gap": gap, "h_pi": hpi})
            for u, q in zip(ucis, mqs):
                move_rows.append({"fen": fen, "move_uci": u, "mq": q})
    return pd.DataFrame(tree_rows), pd.DataFrame(move_rows)


def _spearman_partials(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> None:
    df = df[[x, y] + controls].dropna()
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


def save_mq_dashboard(analyzer: Analyzer, gss_analyzer: Analyzer, out_dir: str) -> None:
    apply_poster_style()
    fig, axes = plt.subplots(1, 3, figsize=(45, 13.72))
    analyzer.plot_quantile_bins(axes[0])
    analyzer.plot_quantile_bins_tertile_segmented(axes[1])
    gss_analyzer.plot_quantile_bins_tertile_segmented(axes[2])
    fig.subplots_adjust(top=0.80)
    suptitle = fig.suptitle(f"{analyzer.title} — quantile bins\nn = {analyzer.n_moves:,} moves",
                            fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    extra = [suptitle] + [ax.get_legend() for ax in axes if ax.get_legend() is not None]
    
    out_path_pdf = os.path.join(out_dir, "mq.pdf")
    out_path_png = os.path.join(out_dir, "mq.png")
    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.3)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.3)
    plt.close()
    print(f"Saved figures: {out_path_pdf} and {out_path_png}")


def plot_lc0_correlation_matrix(conn: duckdb.DuckDBPyConnection, out_path: str) -> None:
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
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, fontsize=20, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=20)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=16,
                    color="white" if abs(val) > 0.5 else "black",
                    fontweight="bold" if i == j else "normal")
    highlight_corr_row(ax, n)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title(f"Spearman correlation — lc0 metrics (n = {len(df):,})", fontsize=20, pad=12)
    plt.tight_layout()
    
    base, _ = os.path.splitext(out_path)
    out_path_pdf = f"{base}.pdf"
    out_path_png = f"{base}.png"
    fig.savefig(out_path_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figures: {out_path_pdf} and {out_path_png}")


def run_tree_values_pipeline(
    trees_dir: str, n_trees: int, n_workers: int, seed: int, db_path: str, cache_dir: str, refresh: bool
):
    """Run the complete tree-values extraction, human-join, and dashboard generation pipeline."""
    key = f"{os.path.basename(trees_dir.rstrip('/'))}_{n_trees}_{seed}"
    cache = Path(cache_dir)
    vals_path, rm_path = cache / f"vals_{key}.parquet", cache / f"rootmoves_{key}.parquet"
    
    use_cache = (not refresh) and vals_path.exists() and rm_path.exists()
    if use_cache:
        print(f"Loading cached values from {cache} (key={key}; --refresh to recompute) …")
        vals, root_moves = pd.read_parquet(vals_path), pd.read_parquet(rm_path)
        if "h_pi" not in vals.columns:
            print("  cached values predate H(π); recomputing from the trees …")
            use_cache = False
            
    if not use_cache:
        print(f"Deriving GSS/Gain/Action Gap/MQ on {n_trees:,} trees ({n_workers} workers) …")
        vals, root_moves = compute_values(trees_dir, n_trees, seed, n_workers)
        cache.mkdir(parents=True, exist_ok=True)
        vals.to_parquet(vals_path)
        root_moves.to_parquet(rm_path)
        print(f"  cached → {cache} (key={key})")

    print(f"  {len(vals):,} trees (GSS {vals['gss'].min()}–{vals['gss'].max()}); {len(root_moves):,} root moves for MQ.")

    # Locate output directory
    ha_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(ha_dir)
    out_dir = os.path.join(repo_root, "figures", "engine")
    os.makedirs(out_dir, exist_ok=True)

    with db_connection(db_path, read_only=True) as conn:
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

        p1 = conn.execute("""
            SELECT ln(t.move_time) AS log_T, t.h_pi,
                   p.n_possible_moves AS legal_moves, t.voc, t.gss
            FROM tree_rt t
            JOIN processed_moves_nonzero p ON p.gid = t.gid AND p.move_ply = t.move_ply
            WHERE t.move_time > 0 AND t.h_pi IS NOT NULL
        """).df()
        _spearman_partials(p1, "h_pi", "log_T", ["legal_moves", "voc", "gss"])

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

        # Define tree values analyses using a clean, unified dictionary structure
        analyses = {
            "gss": {
                "table": "tree_rt",
                "column": "gss",
                "name": "Greedy stop step",
                "filename": "gss.pdf",
                "filter_query": "move_time > 0",
                "min_bin_count": 100,
                "tie_safe": True,
            },
            "gain": {
                "table": "tree_rt",
                "column": "voc",
                "name": "Gain (lc0 tree)",
                "filename": "gain.pdf",
                "filter_query": "move_time > 0",
                "min_bin_count": 100,
                "tie_safe": True,
                "zero_inflated": True,
                "zero_threshold": 0.0,
            },
            "action_gap": {
                "table": "tree_rt",
                "column": "action_gap",
                "name": "Action Gap (lc0 tree)",
                "filename": "action_gap.pdf",
                "filter_query": "move_time > 0",
                "min_bin_count": 100,
                "tie_safe": True,
                "zero_inflated": True,
                "zero_threshold": 0.05,
            },
            "mq": lambda conn, out_dir: save_mq_dashboard(
                Analyzer(
                    conn,
                    "mq_rt",
                    x_var=Variable(column="move_time", is_log=True, name="RT (s)"),
                    y_var=Variable(column="mq", is_log=False, name="MQ (lc0 tree)"),
                    filter_query="move_time > 0",
                    title="MQ (lc0 tree) vs. log(RT)",
                    min_bin_count=100,
                    n_bins=10,
                ),
                Analyzer(
                    conn,
                    "mq_rt",
                    x_var=Variable(column="move_time", is_log=True, name="RT (s)"),
                    y_var=Variable(column="mq", is_log=False, name="MQ (lc0 tree)"),
                    filter_query="move_time > 0",
                    title="MQ (lc0 tree) vs. log(RT)",
                    min_bin_count=100,
                    n_bins=10,
                    segment_column="gss",
                    segment_source="mq_rt",
                    segment_label="GSS",
                ),
                out_dir,
            ),
            "correlation_matrix": lambda conn, out_dir: plot_lc0_correlation_matrix(
                conn, os.path.join(out_dir, "correlation_matrix.pdf")
            ),
        }

        for name, config in analyses.items():
            print(f"Executing engine tree analysis: {name}...")
            if callable(config):
                config(conn, out_dir)
            else:
                analyzer = Analyzer(
                    db_conn=conn,
                    table_name=config["table"],
                    x_var=Variable(column=config["column"], is_log=False, name=config["name"]),
                    y_var=Variable(column="move_time", is_log=True, name="RT"),
                    title=f"{config['name']} vs. log(RT)",
                    filter_query=config.get("filter_query"),
                    min_bin_count=config.get("min_bin_count", 100),
                    tie_safe=config.get("tie_safe", True),
                    zero_inflated=config.get("zero_inflated", False),
                    zero_threshold=config.get("zero_threshold", 0.0),
                )
                base, _ = os.path.splitext(config["filename"])
                analyzer.save_dashboard(os.path.join(out_dir, f"{base}.pdf"))
                analyzer.save_dashboard(os.path.join(out_dir, f"{base}.png"))


# =============================================================================
# CLI Entry Point
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified engine-level and stopping analyses")
    parser.add_argument(
        "--mode", choices=["all", "timing", "mq_gss", "tree_values"], default="all",
        help="Execution mode: all (default), timing benchmarks, difficulty confound stats, or full tree-values pipeline."
    )
    # Timing args
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--n-positions", type=int, default=10)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--parallelism", type=int, default=10)
    
    # Tree values args
    parser.add_argument("--trees-dir", default=_TREES_DEFAULT)
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--cache-dir", default=_CACHE_DEFAULT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--key", default=_KEY_DEFAULT)
    
    args = parser.parse_args(argv)

    modes = {
        "timing": {
            "func": run_timing_comparison,
            "args": [args.db, args.n_positions, args.depths, args.seed, args.parallelism]
        },
        "mq_gss": {
            "func": run_difficulty_confound_stats,
            "args": [Path(args.cache_dir), args.key, args.db]
        },
        "tree_values": {
            "func": run_tree_values_pipeline,
            "args": [args.trees_dir, args.n_trees, args.n_workers, args.seed, args.db, args.cache_dir, args.refresh]
        }
    }

    if args.mode == "all":
        print("Running all engine analyses...")
        for name, config in modes.items():
            print(f"\n=== Running {name} ===")
            config["func"](*config["args"])
    else:
        modes[args.mode]["func"](*modes[args.mode]["args"])


if __name__ == "__main__":
    main()
