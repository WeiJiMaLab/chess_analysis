"""
MQ and VOC analysis: reads pos_with_engine_eval and produces figures.

All figures go to figures/ (flat):
    voc_histogram.png      — distribution of VOC
    mq_histogram.png       — distribution of MQ
    toptwo_histogram.png   — distribution of top-2 gap
    voc_vs_movetime.png    — log RT as a function of VOC (2×2 dashboard)
    mq_vs_clock.png        — MQ as a function of player clock (2×2 dashboard)

Usage (from chess_analysis/):
    python human_analytics/voc_mq_analysis.py
"""

from __future__ import annotations

import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HA = os.path.dirname(os.path.abspath(__file__))
if _HA not in sys.path:
    sys.path.insert(0, _HA)

from utils import Variable, Analyzer
from utils.helpers import apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR
from utils.selected_db import SELECTED_DB_DEFAULT

_FIGURES_DIR = os.path.join(_HA, "figures")
_N_BINS = 50
_TABLE = "pos_with_engine_eval"
_VIEW = "_pwe_view"  # temp view with ply_tertiles


def _open_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path, read_only=False)
    conn.execute("SET enable_progress_bar = false")
    # Add ply_tertiles if not already a column in the table
    cols = [r[0] for r in conn.execute(f"DESCRIBE {_TABLE}").fetchall()]
    if "ply_tertiles" in cols:
        conn.execute(f"CREATE OR REPLACE TEMP VIEW {_VIEW} AS SELECT * FROM {_TABLE}")
    else:
        conn.execute(f"""
            CREATE OR REPLACE TEMP VIEW {_VIEW} AS
            SELECT *, ntile(3) OVER (ORDER BY move_ply) AS ply_tertiles
            FROM {_TABLE}
        """)
    return conn


# ---------------------------------------------------------------------------
# Histogram helper
# ---------------------------------------------------------------------------

def _histogram(ax, values: np.ndarray, x_label: str, n_moves: int, n_bins: int = _N_BINS) -> None:
    """50-bar histogram with dashed mean and median lines."""
    ax.hist(values, bins=n_bins, color=MAIN_COLOR, alpha=0.65, edgecolor=MAIN_COLOR, linewidth=0.5)
    mean = float(np.mean(values))
    median = float(np.median(values))
    ax.axvline(mean, color="black", linestyle="--", lw=2.5, label=f"Mean = {mean:.4f}")
    ax.axvline(median, color="dimgray", linestyle=":", lw=2.5, label=f"Median = {median:.4f}")
    ax.legend(fontsize=FONT_SIZE_TICKS)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"n = {n_moves:,} moves", fontsize=FONT_SIZE_LABEL, pad=12)


def _save(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {path}")


# ---------------------------------------------------------------------------
# Histogram plots (from pandas df — fast, no engine)
# ---------------------------------------------------------------------------

def plot_voc_histogram(df: pd.DataFrame, output_path: str) -> None:
    apply_poster_style()
    v = df["voc"].dropna().to_numpy(float)
    fig, ax = plt.subplots(figsize=(18, 10))
    _histogram(ax, v, "VOC  (V_deep(a_deep) − V_deep(a_shallow))", len(v))
    _save(output_path)


def plot_mq_histogram(df: pd.DataFrame, output_path: str) -> None:
    apply_poster_style()
    m = df["mq"].dropna().to_numpy(float)
    fig, ax = plt.subplots(figsize=(18, 10))
    _histogram(ax, m, "MQ  (e_win_taken − e_win_best,  ≤ 0)", len(m))
    _save(output_path)


def plot_toptwo_histogram(df: pd.DataFrame, output_path: str) -> None:
    apply_poster_style()
    t = df["toptwo"].dropna().to_numpy(float)
    fig, ax = plt.subplots(figsize=(18, 10))
    _histogram(ax, t, "toptwo  (e_win_best − e_win_second_best,  ≥ 0)", len(t))
    _save(output_path)


# ---------------------------------------------------------------------------
# Bivariate dashboards via Analyzer (2×2: raw trend | qbin | × ply tertile)
# ---------------------------------------------------------------------------

def plot_voc_vs_movetime(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """log(RT) vs VOC — 2×2 dashboard stratified by ply tertile."""
    analyzer = Analyzer(
        db_conn=conn,
        table_name=_VIEW,
        x_var=Variable(column="voc", is_log=False, name="VOC"),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        filter_query="move_time > 0",
        title="VOC vs. log(RT)",
    )
    analyzer.save_dashboard(output_path)


def plot_correlation_matrix(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """Pearson correlation matrix: ply, branching, own material, VOC, toptwo, MQ, log(RT)."""
    df = conn.execute(f"""
        SELECT p.move_ply, p.n_possible_moves, pm.n_self_pieces_exc_pawns,
               p.voc, p.toptwo, p.mq, ln(p.move_time) AS log_T
        FROM {_TABLE} p
        JOIN processed_moves_nonzero pm ON p.gid = pm.gid AND p.move_ply = pm.move_ply
        WHERE p.move_time > 0
    """).df()

    labels = {
        "move_ply": "Ply",
        "n_possible_moves": "Branching",
        "n_self_pieces_exc_pawns": "Own material",
        "voc": "VOC",
        "toptwo": "toptwo",
        "mq": "MQ",
        "log_T": "log(RT)",
    }
    corr = df[list(labels)].corr().rename(columns=labels, index=labels)
    n_vars = len(corr)

    apply_poster_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(corr.columns, fontsize=11, rotation=30, ha="right")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(corr.index, fontsize=11)
    for i in range(n_vars):
        for j in range(n_vars):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color,
                    fontweight="bold" if i == j else "normal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=11)
    ax.set_title(f"Correlation matrix  (n = {len(df):,})", fontsize=13, pad=12)
    plt.tight_layout()
    _save(output_path)


def plot_toptwo_vs_movetime(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """log(RT) vs toptwo — 2×2 dashboard stratified by ply tertile."""
    analyzer = Analyzer(
        db_conn=conn,
        table_name=_VIEW,
        x_var=Variable(column="toptwo", is_log=False, name="toptwo"),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        filter_query="move_time > 0",
        title="toptwo vs. log(RT)",
    )
    analyzer.save_dashboard(output_path)


def plot_mq_vs_clock(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """MQ vs player clock time — 2×2 dashboard stratified by ply tertile."""
    analyzer = Analyzer(
        db_conn=conn,
        table_name=_VIEW,
        x_var=Variable(column="player_clock_time", is_log=False, name="Player Clock (s)"),
        y_var=Variable(column="mq", is_log=False, name="MQ"),
        filter_query="player_clock_time < 600",
        title="MQ vs. Clock Time",
    )
    analyzer.save_dashboard(output_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"pos_with_engine_eval  n = {len(df):,}")
    print(f"{'='*60}")
    for col, label in [
        ("e_win_best", "e_win_best"),
        ("e_win_second_best", "e_win_second_best"),
        ("e_win_taken", "e_win_taken"),
        ("voc", "VOC"),
        ("mq", "MQ"),
        ("toptwo", "toptwo"),
    ]:
        if col not in df.columns:
            continue
        v = df[col].dropna()
        print(f"  {label:<22} mean={v.mean():+.4f}  std={v.std():.4f}  "
              f"p50={v.median():+.4f}  |>0.005|={( v.abs()>0.005).mean():.1%}")

    sub = df[df["move_time"] > 0].dropna(subset=["voc", "move_time"])
    if len(sub) > 2:
        r = float(np.corrcoef(sub["voc"], np.log(sub["move_time"]))[0, 1])
        print(f"\n  r(log RT, VOC)   = {r:+.4f}  (n={len(sub):,})")

    sub2 = df.dropna(subset=["mq", "player_clock_time"])
    if len(sub2) > 2:
        r2 = float(np.corrcoef(sub2["player_clock_time"], sub2["mq"])[0, 1])
        print(f"  r(MQ, clock)     = {r2:+.4f}  (n={len(sub2):,})  [negative = more clock → worse MQ]")

    print("\n  Within-ply-tertile r(MQ, clock):")
    df2 = df.copy()
    df2["ply_tertile"] = pd.qcut(df2["move_ply"], q=3, labels=[1, 2, 3]).astype(int)
    for t in [1, 2, 3]:
        ts = df2[df2["ply_tertile"] == t].dropna(subset=["mq", "player_clock_time"])
        r3 = float(np.corrcoef(ts["player_clock_time"], ts["mq"])[0, 1])
        ply_min, ply_max = int(ts["move_ply"].min()), int(ts["move_ply"].max())
        print(f"    Tertile {t} (ply {ply_min}–{ply_max}): r = {r3:+.4f}  (n={len(ts):,})")

    print("\n  Interpretation: r(MQ, clock) is negative within every ply tertile.")
    print("  Players with more clock remaining have played quickly through low-VOC")
    print("  (book/simple) positions; depth-5 penalises their strategic choices.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--table", default=_TABLE)
    parser.add_argument("--output-dir", default=_FIGURES_DIR)
    args = parser.parse_args(argv)

    print(f"Loading {args.table} from {args.db}…")
    conn = _open_conn(args.db)
    df = conn.execute(f"SELECT * FROM {_TABLE}").df()
    print(f"  {len(df):,} rows loaded.")
    _print_summary(df)

    out = args.output_dir
    plot_voc_histogram(df, os.path.join(out, "voc_histogram.png"))
    plot_mq_histogram(df, os.path.join(out, "mq_histogram.png"))
    plot_toptwo_histogram(df, os.path.join(out, "toptwo_histogram.png"))
    plot_voc_vs_movetime(conn, os.path.join(out, "voc_vs_movetime.png"))
    plot_toptwo_vs_movetime(conn, os.path.join(out, "toptwo_vs_movetime.png"))
    plot_mq_vs_clock(conn, os.path.join(out, "mq_vs_clock.png"))
    plot_correlation_matrix(conn, os.path.join(out, "correlation_matrix.png"))

    conn.close()


if __name__ == "__main__":
    main()
