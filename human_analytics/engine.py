"""
Unified engine-level and stopping analyses (DuckDB and tree evaluation).
Consolidates engine metrics, timing benchmarks, difficulty confound stats,
and the large-scale tree-values join from reports/engine.md.

All generated plots are saved in PDF and PNG formats under figures/engine/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

# Align imports with human_analytics structure
from utils import Variable, Analyzer
from utils.helpers import (
    apply_poster_style,
    db_connection,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    partial_spearman,
    CONFIG,
)
from utils.plots import (
    highlight_corr_row,
    save_figure,
)
from utils.selected_db import (
    SELECTED_DB_DEFAULT,
    TABLE_PROCESSED_MOVES_NONZERO,
)

# Imported from newly extracted modular utilities
from utils.tree_loader import compute_values, _tree_voc_and_gap
from utils.engine_eval import move_quality, voc
import matplotlib.pyplot as plt





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
# Tree Values Pipeline and Dashboards
# =============================================================================

def _spearman_partials(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> None:
    print(f"\n  === P1: H(π) prior argmax-uncertainty vs {y} (Spearman, n={len(df):,}) ===")
    raw = df[x].rank().corr(df[y].rank())
    print(f"  raw ρ(H(π), {y})              = {raw:+.4f}")
    for c in controls:
        rho = partial_spearman(df, x, y, [c])
        print(f"  partial ρ(H(π), {y} | {c:<9s}) = {rho:+.4f}")
    rho_all = partial_spearman(df, x, y, controls)
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
        "--mode", choices=["all", "mq_gss", "tree_values"], default="all",
        help="Execution mode: all (default), difficulty confound stats, or full tree-values pipeline."
    )
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--seed", type=int, default=7)
    
    # Tree values args
    parser.add_argument("--trees-dir", default=CONFIG["trees_default"])
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--cache-dir", default=CONFIG["cache_default"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--key", default=CONFIG["key_default"])
    
    args = parser.parse_args(argv)

    modes = {
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

