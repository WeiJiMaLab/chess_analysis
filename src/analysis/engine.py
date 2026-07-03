"""
Unified engine-level (no-normative) RT analyses (DuckDB + SF-1 tree evaluation).
Derives the SF-1 (n1md36) tree signals — Gain, MQ, greedy action gap, greedy
stop step (GSS), greedy frac-good, optimal stop step (OSS) — joins them to human
RTs, and emits the per-signal RT dashboards plus a Spearman correlation matrix
over all engine signals AND the board features (ply / legal moves / clock).

Trees are read from human_analysis.trees_default in the active config (the SF-1
n1md36 set). All processed-move reads — including the backwards tree->move join —
go through a ply-windowed view (move_ply in [min_ply, max_ply]) so the ply filter
is applied consistently on arrival and on the way back. All plots are saved
(PDF + PNG) under <figures_dir>/engine/ (namespaced by run_name).
"""

from __future__ import annotations

import argparse
import os

# Resolve --config BEFORE importing analysis.utils so helpers loads the right
# config file (its CONFIG global is built at import from the CONFIG env var).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config")
_cfg, _ = _pre.parse_known_args()
if _cfg.config:
    os.environ["CONFIG"] = _cfg.config

from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

# Align imports with the src/analysis package structure
from analysis.utils import Variable, Analyzer
from analysis.utils.helpers import (
    apply_poster_style,
    db_connection,
    create_ply_windowed_views,
    WIN_PROCESSED_MOVES_NONZERO,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    partial_spearman,
    CONFIG,
)
from analysis.utils.plots import (
    highlight_corr_row,
    save_figure,
)
from analysis.utils.selected_db import (
    SELECTED_DB_DEFAULT,
)

# Imported from newly extracted modular utilities
from analysis.utils.tree_loader import UNITS, compute_values
import matplotlib.pyplot as plt





# =============================================================================
# Difficulty Confound Statistics (collapsed from mq_vs_rt_by_gss.py)
# =============================================================================

def load_played_moves(cache_dir: Path, key: str, db_path: str) -> pd.DataFrame:
    """Load and join human played moves with tree-derived variables."""
    vals = pd.read_parquet(cache_dir / f"vals_{key}.parquet")[["fen", "gss"]]
    root_moves = pd.read_parquet(cache_dir / f"rootmoves_{key}.parquet")[["fen", "move_uci", "mq"]]

    conn = duckdb.connect(db_path, read_only=True)
    create_ply_windowed_views(conn)   # ply filter applied on arrival (pmnz_win)
    conn.register("_vals", vals)
    conn.register("_root_moves", root_moves)
    df = conn.execute("""
        WITH human AS (
            SELECT m.fen, m.gid, m.move_ply, m.move_time,
                   m.n_possible_moves AS legal_moves, mv.move_uci
            FROM (SELECT DISTINCT fen FROM _root_moves) f
            JOIN pmnz_win m ON m.fen = f.fen AND m.move_time > 0
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


def save_tree_dashboard(analyzer_ply: Analyzer, out_dir: str, base_name: str) -> None:
    """1x2 engine dashboard: overall, by ply tertile.

    ``analyzer_ply`` supplies the base panel + ply segmentation."""
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(30, 13.72))
    analyzer_ply.plot_quantile_bins(axes[0])                    # overall
    analyzer_ply.plot_quantile_bins_tertile_segmented(axes[1])  # color: ply
    for ax, t in zip(axes, ("overall", "by ply")):
        ax.set_title(t, fontsize=FONT_SIZE_TICKS)
    # wider figure + extra horizontal gap so the two panels' y-labels/legends
    # don't crowd each other.
    fig.subplots_adjust(top=0.80, wspace=0.28)
    suptitle = fig.suptitle(f"{analyzer_ply.title}\nn = {analyzer_ply.n_moves:,} moves",
                            fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    extra = [suptitle] + [ax.get_legend() for ax in axes if ax.get_legend() is not None]

    out_path_pdf = os.path.join(out_dir, f"{base_name}.pdf")
    out_path_png = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.3)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.3)
    plt.close()
    print(f"Saved figures: {out_path_pdf} and {out_path_png}")


def plot_lc0_correlation_matrix(conn: duckdb.DuckDBPyConnection, out_path: str, unit: str = "pwin") -> None:
    """Spearman AND Pearson matrices (trust protocol: report both; the Pearson
    matrix saves with a ``_pearson`` suffix)."""
    df = conn.execute("""
        SELECT ln(t.move_time) AS log_T, t.move_ply AS ply,
               p.n_possible_moves AS legal_moves, p.player_clock_time AS player_clock,
               m.mq, t.voc, t.action_gap, t.gss, t.greedy_frac_good, t.frac_acceptable, t.oss
        FROM tree_rt t
        JOIN mq_rt m ON m.gid = t.gid AND m.move_ply = t.move_ply
        JOIN pmnz_win p ON p.gid = t.gid AND p.move_ply = t.move_ply
        WHERE t.move_time > 0
    """).df()
    labels = {
        "log_T": "log(RT)",
        # board features
        "ply": "Ply", "legal_moves": "Legal moves", "player_clock": "Clock left",
        # engine signals — order: Gain, MQ, ActionGap, FracGood, FracAcceptable, GSS, OSS
        "voc": "Gain", "mq": "MQ", "action_gap": "Action Gap",
        "greedy_frac_good": "Greedy frac-good", "frac_acceptable": "Frac-acceptable",
        "gss": "GSS", "oss": "OSS",
    }
    base, _ = os.path.splitext(out_path)
    for method, mlabel, suffix in (("spearman", "Spearman ρ", ""), ("pearson", "Pearson r", "_pearson")):
        corr = df[list(labels)].corr(method=method).rename(columns=labels, index=labels)
        n = len(corr)
        apply_poster_style()
        fig, ax = plt.subplots(figsize=(12, 10))
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
        cbar.set_label(mlabel, fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        ax.set_title(f"{mlabel} — SF-1 BeFS metrics, unit={unit} (n = {len(df):,})",
                     fontsize=20, pad=12)
        plt.tight_layout()
        fig.savefig(f"{base}{suffix}.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(f"{base}{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figures: {base}{suffix}.pdf and {base}{suffix}.png")


def run_trust_tests(conn, unit: str, n_boot: int = 1000, seed: int = 7) -> None:
    """P1 trust tests (trust_protocol.md §5): the pre-registered SIGN-MIRROR —
    net of legal moves, action_gap (decisiveness) and greedy_frac_good
    (ambiguity share) must have OPPOSITE-signed partials with CIs excluding 0
    and not overlapping each other. Run per-FEN (the headline unit); also
    reports frac_acceptable and the hurdle decomposition for zero-inflated
    signals. Fail is a finding — printed either way."""
    df = conn.execute("""
        SELECT t.fen, avg(ln(t.move_time)) AS log_rt,
               any_value(t.action_gap) AS action_gap,
               any_value(t.greedy_frac_good) AS frac_good,
               any_value(t.frac_acceptable) AS frac_acceptable,
               any_value(p.n_possible_moves) AS legal_moves
        FROM tree_rt t
        JOIN pmnz_win p ON p.gid = t.gid AND p.move_ply = t.move_ply
        WHERE t.move_time > 0
        GROUP BY t.fen
    """).df().dropna()
    rng = np.random.default_rng(seed)
    n = len(df)

    def _partial_rho(d, xcol):
        rx = d[xcol].rank().to_numpy(float)
        ry = d["log_rt"].rank().to_numpy(float)
        rz = d["legal_moves"].rank().to_numpy(float)
        zc = np.column_stack([np.ones_like(rz), rz])
        bx, *_ = np.linalg.lstsq(zc, rx, rcond=None)
        by, *_ = np.linalg.lstsq(zc, ry, rcond=None)
        return float(np.corrcoef(rx - zc @ bx, ry - zc @ by)[0, 1])

    print(f"\n  === P1 trust tests (unit={unit}, per-FEN, n={n:,}) ===")
    results = {}
    for xcol in ("action_gap", "frac_good", "frac_acceptable"):
        point = _partial_rho(df, xcol)
        boots = [_partial_rho(df.iloc[rng.integers(0, n, n)], xcol) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[xcol] = (point, lo, hi)
        print(f"  partial ρ({xcol:16s}, logRT | legal) = {point:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")
    (ag, ag_lo, ag_hi), (fg, fg_lo, fg_hi) = results["action_gap"], results["frac_good"]
    mirror = (ag * fg < 0) and (ag_lo * ag_hi > 0) and (fg_lo * fg_hi > 0) \
        and (min(ag_hi, fg_hi) < max(ag_lo, fg_lo))
    print(f"  SIGN-MIRROR (action_gap vs frac_good): {'PASS' if mirror else 'FAIL'} "
          f"(opposite signs, CIs exclude 0, CIs disjoint)")

    # Hurdle decomposition for the zero-inflated signals (per-unit threshold).
    thr = {"pwin": 0.05, "cp": 5.0}[unit]
    for xcol, t in (("action_gap", thr),):
        above = df[df[xcol] > t]
        p_above = float((df[xcol] > t).mean())
        rho_mag = float(above[xcol].rank().corr(above["log_rt"].rank())) if len(above) > 100 else float("nan")
        print(f"  hurdle {xcol}: P(>{t}) = {p_above:.3f}; ρ(magnitude | >{t}) = {rho_mag:+.4f} "
              f"(n={len(above):,})")


def run_tree_values_pipeline(
    trees_dir: str, n_trees: int, n_workers: int, seed: int, db_path: str, cache_dir: str,
    refresh: bool, unit: str = "pwin"
):
    """Tree-values extraction, human-join, dashboards, and P1 trust tests — in one
    VALUE UNIT ("pwin" | "cp"; see tree_loader.UNITS). Figures land in
    figures_dir/engine_<unit>/ so both variants coexist."""
    key = f"{os.path.basename(trees_dir.rstrip('/'))}_{n_trees}_{seed}_{unit}"
    cache = Path(cache_dir)
    vals_path, rm_path = cache / f"vals_{key}.parquet", cache / f"rootmoves_{key}.parquet"

    use_cache = (not refresh) and vals_path.exists() and rm_path.exists()
    if use_cache:
        print(f"Loading cached values from {cache} (key={key}; --refresh to recompute) …")
        vals, root_moves = pd.read_parquet(vals_path), pd.read_parquet(rm_path)
        _required = ("h_pi", "greedy_frac_good", "frac_acceptable", "oss")
        _missing = [c for c in _required if c not in vals.columns]
        if _missing:
            print(f"  cached values predate columns {_missing}; recomputing from the trees …")
            use_cache = False

    if not use_cache:
        print(f"Deriving GSS/Gain/Action Gap/MQ on {n_trees:,} trees ({n_workers} workers, unit={unit}) …")
        vals, root_moves = compute_values(trees_dir, n_trees, seed, n_workers, unit=unit)
        cache.mkdir(parents=True, exist_ok=True)
        vals.to_parquet(vals_path)
        root_moves.to_parquet(rm_path)
        print(f"  cached → {cache} (key={key})")

    print(f"  {len(vals):,} trees (GSS {vals['gss'].min()}–{vals['gss'].max()}); {len(root_moves):,} root moves for MQ.")

    # Output dir is wired from config (human_analysis.figures_dir); engine figures
    # live in a per-unit subdir so the pwin and cp variants coexist.
    out_dir = os.path.join(CONFIG["figures_dir"], f"engine_{unit}")
    os.makedirs(out_dir, exist_ok=True)

    with db_connection(db_path, read_only=True) as conn:
        create_ply_windowed_views(conn)   # ply filter on arrival; tree_rt/mq_rt build off pmnz_gf
        # game_fraction = move_ply / TRUE total plies in the game. The total must
        # come from the UNWINDOWED move table: max(move_ply) over the ply-windowed
        # pmnz_win would cap the denominator at max_ply (=75), pinning every game
        # longer than the window to the same denominator and inflating game_fraction
        # toward 1.0 for the majority of moves whose game exceeds 75 plies.
        conn.execute(
            "CREATE OR REPLACE TEMP VIEW pmnz_gf AS "
            "SELECT w.*, w.move_ply * 1.0 / g.game_len AS game_fraction "
            "FROM pmnz_win w "
            "JOIN (SELECT gid, max(move_ply) AS game_len "
            f"      FROM {CONFIG['table_processed_moves']} GROUP BY gid) g USING (gid)"
        )
        conn.register("_vals", vals)
        conn.register("_root_moves", root_moves)
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE tree_rt AS
            SELECT v.gss, v.voc, v.action_gap, v.h_pi, v.greedy_frac_good,
                   v.frac_acceptable, v.oss,
                   v.fen, m.gid, m.move_ply, m.move_time, m.game_fraction
            FROM _vals v
            JOIN pmnz_gf m ON m.fen = v.fen
            WHERE m.move_time > 0
        """)
        n_rows = conn.execute("SELECT count(*) FROM tree_rt").fetchone()[0]
        n_fen = conn.execute("SELECT count(DISTINCT fen) FROM tree_rt").fetchone()[0]
        print(f"  joined {n_rows:,} human moves across {n_fen:,} FENs.")

        for col in ("gss", "voc", "action_gap", "h_pi", "greedy_frac_good", "frac_acceptable", "oss"):
            r = conn.execute(f"SELECT corr({col}, ln(move_time)) FROM tree_rt WHERE {col} IS NOT NULL").fetchone()[0]
            print(f"  r({col}, log RT) = {r:+.4f}")

        p1 = conn.execute("""
            SELECT ln(t.move_time) AS log_T, t.h_pi,
                   p.n_possible_moves AS legal_moves, t.voc, t.gss
            FROM tree_rt t
            JOIN pmnz_win p ON p.gid = t.gid AND p.move_ply = t.move_ply
            WHERE t.move_time > 0 AND t.h_pi IS NOT NULL
        """).df()
        _spearman_partials(p1, "h_pi", "log_T", ["legal_moves", "voc", "gss"])

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE mq_rt AS
            WITH human AS (
                SELECT m.fen, m.gid, m.move_ply, m.move_time, m.game_fraction, mv.move_uci
                FROM (SELECT DISTINCT fen FROM _root_moves) f
                JOIN pmnz_gf m ON m.fen = f.fen AND m.move_time > 0
                JOIN moves mv ON mv.gid = m.gid AND mv.move_ply = m.move_ply
            )
            SELECT rm.mq, h.fen, h.gid, h.move_ply, h.move_time, h.game_fraction, v.gss
            FROM human h
            JOIN _root_moves rm ON rm.fen = h.fen AND rm.move_uci = h.move_uci
            JOIN _vals v ON v.fen = h.fen
            WHERE h.move_time > 0
        """)
        
        n_human = conn.execute("""
            SELECT count(*) FROM (SELECT DISTINCT fen FROM _root_moves) f
            JOIN pmnz_win m ON m.fen = f.fen AND m.move_time > 0
        """).fetchone()[0]
        n_mq = conn.execute("SELECT count(*) FROM mq_rt").fetchone()[0]
        r_mq = conn.execute("SELECT corr(mq, ln(move_time)) FROM mq_rt").fetchone()[0]
        print(f"  MQ: matched {n_mq:,}/{n_human:,} human moves to a root move "
              f"({100.0 * n_mq / max(n_human, 1):.1f}%).")
        print(f"  r(mq, log RT) = {r_mq:+.4f}")

        # Per-signal tree dashboards. Each renders a 1x3 (overall / by ply / by game
        # fraction) via two Analyzers over tree_rt that differ only in segment column.
        # gss/oss are integers → bin per-integer (ntile over a discrete right-skewed
        # variable manufactures fake non-monotonicity); tail above the cut is merged.
        # Value-unit thresholds scale with the unit (pwin ↔ cp, ~×100 near balance).
        _zero_thr = {"pwin": 0.05, "cp": 5.0}[unit]
        _u = f" ({unit})"
        tree_signals = {
            "gss": {"column": "gss", "name": "Greedy stop step" + _u, "filter_query": "move_time > 0",
                    "bin_mode": "integer", "integer_bin_width": 5, "integer_tail_cut": 35},
            "gain": {"column": "voc", "name": "Gain (SF-1)" + _u, "filter_query": "move_time > 0",
                     "zero_inflated": True, "zero_threshold": 0.0},
            "action_gap": {"column": "action_gap", "name": "Action Gap (SF-1)" + _u,
                           "filter_query": "move_time > 0",
                           "zero_inflated": True, "zero_threshold": _zero_thr},
            "frac_good": {"column": "greedy_frac_good", "name": "Greedy frac-good (SF-1)" + _u,
                          "filter_query": "move_time > 0 AND greedy_frac_good IS NOT NULL"},
            "frac_acceptable": {"column": "frac_acceptable", "name": "Frac-acceptable (SF-1)" + _u,
                                "filter_query": "move_time > 0 AND frac_acceptable IS NOT NULL"},
            "oss": {"column": "oss", "name": "Optimal stop step (SF-1)" + _u,
                    "filter_query": "move_time > 0 AND oss IS NOT NULL",
                    "bin_mode": "integer", "integer_bin_width": 5, "integer_tail_cut": 35},
        }

        def _make_analyzer(table, x_var, y_var, title, **opts):
            """Ply-segmented Analyzer for the overall + by-ply dashboard."""
            return Analyzer(conn, table, segment_column="move_ply",
                            segment_source=table, segment_label="Ply",
                            x_var=x_var, y_var=y_var, title=title,
                            ply_tertile_source=table, **opts)

        for name, cfg in tree_signals.items():
            print(f"Executing engine tree analysis: {name}...")
            a_ply = _make_analyzer(
                "tree_rt",
                Variable(column=cfg["column"], is_log=False, name=cfg["name"]),
                Variable(column="move_time", is_log=True, name="RT"),
                f"{cfg['name']} vs. log(RT)",
                filter_query=cfg["filter_query"], min_bin_count=300, tie_safe=True,
                zero_inflated=cfg.get("zero_inflated", False),
                zero_threshold=cfg.get("zero_threshold", 0.0),
                bin_mode=cfg.get("bin_mode", "ntile"),
                integer_bin_width=cfg.get("integer_bin_width", 1),
                integer_tail_cut=cfg.get("integer_tail_cut"),
            )
            save_tree_dashboard(a_ply, out_dir, name)

        print("Executing engine tree analysis: mq...")
        mq_ply = _make_analyzer(
            "mq_rt",
            Variable(column="move_time", is_log=True, name="RT (s)"),
            Variable(column="mq", is_log=False, name=f"MQ (SF-1) ({unit})"),
            f"MQ (SF-1) ({unit}) vs. log(RT)",
            filter_query="move_time > 0", min_bin_count=300, n_bins=10,
        )
        save_tree_dashboard(mq_ply, out_dir, "mq")

        print("Executing engine tree analysis: correlation_matrix...")
        plot_lc0_correlation_matrix(conn, os.path.join(out_dir, "correlation_matrix.pdf"), unit=unit)

        print("Executing engine tree analysis: P1 trust tests...")
        run_trust_tests(conn, unit)


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
    parser.add_argument("--config", help="Path to the run config (else $CONFIG or the default).")
    parser.add_argument("--seed", type=int, default=7)

    # Tree values args
    parser.add_argument("--trees-dir", default=CONFIG["trees_default"])
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--cache-dir", default=CONFIG["cache_default"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--key", default=None,
                        help="cache key for mq_gss (default: <key_default>_<unit>)")
    parser.add_argument("--unit", choices=sorted(UNITS), default="pwin",
                        help="value unit for the tree signals (pwin | cp); figures go to engine_<unit>/")

    args = parser.parse_args(argv)
    key = args.key or f"{CONFIG['key_default']}_{args.unit}"

    # tree_values BUILDS the cache parquet that mq_gss READS, so it must run first
    # in --mode all (otherwise a fresh cache, e.g. the first SF-2000 run, fails).
    modes = {
        "tree_values": {
            "func": run_tree_values_pipeline,
            "args": [args.trees_dir, args.n_trees, args.n_workers, args.seed, args.db,
                     args.cache_dir, args.refresh, args.unit]
        },
        "mq_gss": {
            "func": run_difficulty_confound_stats,
            "args": [Path(args.cache_dir), key, args.db]
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

