from __future__ import annotations

import argparse
import os

from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

# Align imports with the src/analysis package structure
from analysis.utils import Variable, Analyzer
from analysis.utils.helpers import (
    apply_poster_style,
    db_connection,
    partial_spearman,
    CONFIG,
)
from analysis.utils.plots import (
    highlight_corr_row,
    save_figure,
    _annotate_n,
    _draw_feature_histogram,
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

    with db_connection(db_path, read_only=True) as conn:
        conn.execute(f"CREATE OR REPLACE TEMP VIEW filtered AS SELECT * FROM {CONFIG['table_filtered']}")
        conn.register("_vals", vals)
        conn.register("_root_moves", root_moves)
        df = conn.execute("""
            WITH human AS (
                SELECT m.fen, m.gid, m.move_ply, m.move_time,
                       m.n_possible_moves AS legal_moves, mv.move_uci
                FROM (SELECT DISTINCT fen FROM _root_moves) f
                JOIN filtered m ON m.fen = f.fen AND m.move_time > 0
                JOIN moves mv ON mv.gid = m.gid AND mv.move_ply = m.move_ply
            )
            SELECT rm.mq, ln(h.move_time) AS log_rt, v.gss, h.legal_moves
            FROM human h
            JOIN _root_moves rm ON rm.fen = h.fen AND rm.move_uci = h.move_uci
            JOIN _vals v ON v.fen = h.fen
        """).df()
    return df.dropna(subset=["mq", "log_rt", "gss", "legal_moves"]).reset_index(drop=True)


def gss_strata(df: pd.DataFrame) -> pd.Series:
    """Quantile-bin GSS difficulty strata."""
    edges = df["gss"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    edges = np.unique(edges)
    labels = ["Easy (low GSS)", "Medium GSS", "Hard (high GSS)"][: len(edges) - 1]
    return pd.cut(df["gss"], bins=edges, labels=labels, include_lowest=True)


def difficulty_confound_stats(cache_dir: Path, key: str, db_path: str):
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


def save_tree_dashboard(
    analyzer_ply: Analyzer, out_dir: str, base_name: str, *,
    hist_column: str | None = None, hist_label: str | None = None,
    hist_kind: str = "cont", hist_clip=None, hist_where: str | None = None,
) -> None:
    """1x3 engine dashboard: marginal histogram (left), overall trend (middle), by
    ply tertile (right) — the same layout as board.py's ``bivariate_analysis``
    (house plot standard, ``outputs/reports/reference.md``). ``analyzer_ply``
    supplies the trend panels + ply segmentation; the histogram reads
    ``hist_column`` (default: ``analyzer_ply.x.column`` — override for a caller
    like the MQ dashboard, whose Analyzer has RT on x and the signal on y).
    ``analyzer_ply.zero_inflated``/``zero_threshold`` drive the histogram's
    isolated zero-mass bar too, so it stays in sync with the trend panels'
    ✕-marker mass point."""
    apply_poster_style()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(42, 13), constrained_layout=True)
    _draw_feature_histogram(
        analyzer_ply.conn, analyzer_ply.table,
        hist_column or analyzer_ply.x.column, hist_kind, hist_clip, ax0,
        name=hist_label or analyzer_ply.x.label,
        zero_inflated=analyzer_ply.zero_inflated, zero_threshold=analyzer_ply.zero_threshold,
        extra_where=hist_where,
    )
    analyzer_ply.plot_quantile_bins(ax1)                    # overall
    analyzer_ply.plot_quantile_bins_tertile_segmented(ax2)  # color: ply
    _annotate_n(fig, analyzer_ply.n_moves)

    out_path_pdf = os.path.join(out_dir, f"{base_name}.pdf")
    out_path_png = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved figures: {out_path_pdf} and {out_path_png}")


def plot_engine_correlation_matrix(conn: duckdb.DuckDBPyConnection, out_path: str, unit: str = "pwin") -> None:
    """Spearman AND Pearson matrices (trust protocol: report both; the Pearson
    matrix saves with a ``_pearson`` suffix)."""
    df = conn.execute("""
        SELECT ln(t.move_time) AS log_T, t.move_ply AS ply,
               p.n_possible_moves AS legal_moves, p.player_clock_time AS player_clock,
               m.mq, t.voc, t.action_gap, t.gss, t.n_within_epsilon, t.n_acceptable, t.oss
        FROM tree_rt t
        JOIN mq_rt m ON m.gid = t.gid AND m.move_ply = t.move_ply
        JOIN filtered p ON p.gid = t.gid AND p.move_ply = t.move_ply
        WHERE t.move_time > 0
    """).df()
    labels = {
        "log_T": "log(RT)",
        # board features
        "ply": "Ply", "legal_moves": "Legal moves", "player_clock": "Clock left",
        # engine signals — order: Gain, MQ, ActionGap, n-within-ε, n-acceptable, GSS, OSS
        "voc": "Gain", "mq": "MQ", "action_gap": "Action Gap",
        "n_within_epsilon": "n within ε", "n_acceptable": "n acceptable",
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
        plt.tight_layout()
        _annotate_n(fig, len(df))
        fig.savefig(f"{base}{suffix}.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(f"{base}{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figures: {base}{suffix}.pdf and {base}{suffix}.png")


def trust_tests(conn, unit: str, n_boot: int = 1000, seed: int = 7) -> None:
    """P1 trust tests (trust_protocol.md §5): the pre-registered SIGN-MIRROR —
    net of legal moves, action_gap (decisiveness) and n_within_epsilon (ambiguity
    COUNT) must have OPPOSITE-signed partials with CIs excluding 0 and not
    overlapping each other. Run per-FEN (the headline unit); also reports
    n_acceptable and the hurdle decomposition for the zero-inflated signals
    (action_gap, and n_acceptable whose zero = 'no ≥-equal move'). The ambiguity
    signal is now a raw count (no legal-move denominator confound); the mirror
    logic is unchanged. Fail is a finding — printed either way."""
    df = conn.execute("""
        SELECT t.fen, avg(ln(t.move_time)) AS log_rt,
               any_value(t.action_gap) AS action_gap,
               any_value(t.n_within_epsilon) AS n_within_epsilon,
               any_value(t.n_acceptable) AS n_acceptable,
               any_value(p.n_possible_moves) AS legal_moves
        FROM tree_rt t
        JOIN filtered p ON p.gid = t.gid AND p.move_ply = t.move_ply
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
    for xcol in ("action_gap", "n_within_epsilon", "n_acceptable"):
        point = _partial_rho(df, xcol)
        boots = [_partial_rho(df.iloc[rng.integers(0, n, n)], xcol) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[xcol] = (point, lo, hi)
        print(f"  partial ρ({xcol:16s}, logRT | legal) = {point:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")
    (ag, ag_lo, ag_hi), (fg, fg_lo, fg_hi) = results["action_gap"], results["n_within_epsilon"]
    mirror = (ag * fg < 0) and (ag_lo * ag_hi > 0) and (fg_lo * fg_hi > 0) \
        and (min(ag_hi, fg_hi) < max(ag_lo, fg_lo))
    print(f"  SIGN-MIRROR (action_gap vs n_within_epsilon): {'PASS' if mirror else 'FAIL'} "
          f"(opposite signs, CIs exclude 0, CIs disjoint)")

    # Hurdle decomposition for the zero-inflated signals. action_gap uses a per-unit
    # magnitude threshold; n_acceptable is a count whose zero ('no ≥-equal move' =
    # losing/forced) is the natural hurdle, so its threshold is 0.
    ag_thr = {"pwin": 0.05, "cp": 5.0}[unit]
    for xcol, t in (("action_gap", ag_thr), ("n_acceptable", 0)):
        above = df[df[xcol] > t]
        p_above = float((df[xcol] > t).mean())
        rho_mag = float(above[xcol].rank().corr(above["log_rt"].rank())) if len(above) > 100 else float("nan")
        print(f"  hurdle {xcol}: P(>{t}) = {p_above:.3f}; ρ(magnitude | >{t}) = {rho_mag:+.4f} "
              f"(n={len(above):,})")


def _ensure_n_root_children(vals: pd.DataFrame, root_moves: pd.DataFrame) -> pd.DataFrame:
    """Backfill ``n_root_children`` (the legal-move denominator) for a cache that
    predates it, from the per-FEN root-move row count. The count signals themselves
    are guaranteed by the recompute gate, so no fraction reconstruction is needed."""
    if "n_root_children" not in vals.columns:
        counts = root_moves.groupby("fen").size().rename("n_root_children")
        vals = vals.merge(counts, left_on="fen", right_index=True, how="left")
    return vals


def cache_key(trees_dir: str, n_trees: int, seed: int, unit: str) -> str:
    """The tree-values cache key. Built identically wherever the cache is written
    (tree_values_pipeline) or read (mq_gss), so the two never disagree."""
    return f"{os.path.basename(trees_dir.rstrip('/'))}_{n_trees}_{seed}_{unit}"


def tree_values_pipeline(
    trees_dir: str, n_trees: int, n_workers: int, seed: int, db_path: str, cache_dir: str,
    refresh: bool, unit: str = "pwin"
):
    """Tree-values extraction, human-join, dashboards, and P1 trust tests — in one
    VALUE UNIT ("pwin" | "cp"; see tree_loader.UNITS). Figures land in
    figures_dir/engine_<unit>/ so both variants coexist."""
    key = cache_key(trees_dir, n_trees, seed, unit)
    cache = Path(cache_dir)
    vals_path, root_moves_path = cache / f"vals_{key}.parquet", cache / f"rootmoves_{key}.parquet"

    use_cache = (not refresh) and vals_path.exists() and root_moves_path.exists()
    if use_cache:
        print(f"Loading cached values from {cache} (key={key}; --refresh to recompute) …")
        vals, root_moves = pd.read_parquet(vals_path), pd.read_parquet(root_moves_path)
        # A cache missing any of these is recomputed from the trees. The count signals
        # are included because they encode the acceptable/ε thresholds: a legacy cache's
        # fractions were computed at the OLD threshold, so reconstructing counts from
        # them would silently report the wrong bar.
        _required = ("h_pi", "oss", "n_acceptable", "n_within_epsilon")
        _missing = [c for c in _required if c not in vals.columns]
        if _missing:
            print(f"  cached values predate columns {_missing}; recomputing from the trees …")
            use_cache = False

    if not use_cache:
        print(f"Deriving GSS/Gain/Action Gap/MQ on {n_trees:,} trees ({n_workers} workers, unit={unit}) …")
        vals, root_moves = compute_values(trees_dir, n_trees, seed, n_workers, unit=unit)
        cache.mkdir(parents=True, exist_ok=True)
        vals.to_parquet(vals_path)
        root_moves.to_parquet(root_moves_path)
        print(f"  cached → {cache} (key={key})")

    vals = _ensure_n_root_children(vals, root_moves)
    print(f"  {len(vals):,} trees (GSS {vals['gss'].min()}–{vals['gss'].max()}); {len(root_moves):,} root moves for MQ.")

    # Output dir is wired from config (human_analysis.figures_dir); engine figures
    # live in a per-unit subdir so the pwin and cp variants coexist.
    out_dir = os.path.join(CONFIG["figures_dir"], f"engine_{unit}")
    os.makedirs(out_dir, exist_ok=True)

    with db_connection(db_path, read_only=True) as conn:
        # The run's canonical windowed table (game_fraction already baked in by the
        # filter stage); tree_rt/mq_rt build off it.
        conn.execute(f"CREATE OR REPLACE TEMP VIEW filtered AS SELECT * FROM {CONFIG['table_filtered']}")
        conn.register("_vals", vals)
        conn.register("_root_moves", root_moves)
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE tree_rt AS
            SELECT v.gss, v.voc, v.action_gap, v.h_pi, v.n_within_epsilon,
                   v.n_acceptable, v.n_root_children, v.oss,
                   v.fen, m.gid, m.move_ply, m.move_time, m.game_fraction,
                   m.player_clock_time
            FROM _vals v
            JOIN filtered m ON m.fen = v.fen
            WHERE m.move_time > 0
        """)
        n_rows = conn.execute("SELECT count(*) FROM tree_rt").fetchone()[0]
        n_fen = conn.execute("SELECT count(DISTINCT fen) FROM tree_rt").fetchone()[0]
        print(f"  joined {n_rows:,} human moves across {n_fen:,} FENs.")

        for col in ("gss", "voc", "action_gap", "h_pi", "n_within_epsilon", "n_acceptable", "oss"):
            r = conn.execute(f"SELECT corr({col}, ln(move_time)) FROM tree_rt WHERE {col} IS NOT NULL").fetchone()[0]
            print(f"  r({col}, log RT) = {r:+.4f}")

        p1 = conn.execute("""
            SELECT ln(t.move_time) AS log_T, t.h_pi,
                   p.n_possible_moves AS legal_moves, t.voc, t.gss
            FROM tree_rt t
            JOIN filtered p ON p.gid = t.gid AND p.move_ply = t.move_ply
            WHERE t.move_time > 0 AND t.h_pi IS NOT NULL
        """).df()
        _spearman_partials(p1, "h_pi", "log_T", ["legal_moves", "voc", "gss"])

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE mq_rt AS
            WITH human AS (
                SELECT m.fen, m.gid, m.move_ply, m.move_time, m.game_fraction, mv.move_uci
                FROM (SELECT DISTINCT fen FROM _root_moves) f
                JOIN filtered m ON m.fen = f.fen AND m.move_time > 0
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
            JOIN filtered m ON m.fen = f.fen AND m.move_time > 0
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
        # hist_kind/hist_clip drive ``_draw_feature_histogram``'s left panel (the
        # shared board.py/engine.py histogram helper): "disc" -> one bar per integer
        # value, clipped to the same tail cut as the trend panel's integer binning;
        # "cont" -> a 50-bin continuous histogram, auto-ranged (0.5th/99.5th pctile)
        # since the pwin/cp value units have very different scales; zero_inflated
        # continuous signals additionally split their zero mass into its own
        # isolated bar (consistent with the ✕-marker mass point on the trend panels).
        tree_signals = {
            "gss": {"column": "gss", "name": "Greedy stop step" + _u, "filter_query": "move_time > 0",
                    "bin_mode": "integer", "integer_bin_width": 5, "integer_tail_cut": 35,
                    "hist_kind": "disc", "hist_clip": (0, 35)},
            "gain": {"column": "voc", "name": "Gain (SF-1)" + _u, "filter_query": "move_time > 0",
                     "zero_inflated": True, "zero_threshold": 0.0,
                     "hist_kind": "cont", "hist_clip": None},
            "action_gap": {"column": "action_gap", "name": "Action Gap (SF-1)" + _u,
                           "filter_query": "move_time > 0",
                           "zero_inflated": True, "zero_threshold": _zero_thr,
                           "hist_kind": "cont", "hist_clip": None},
            # COUNTS, not fractions (no legal-move denominator confound). n_within_epsilon
            # is a right-skewed count with min 1 (best move is always within ε of itself),
            # binned per-integer with a merged tail. n_acceptable is zero-inflated (~47% of
            # positions have NO ≥-equal move = losing/forced): the zero mass is split off as
            # its own point (hurdle) so it can't straddle bins, then the >0 magnitude is
            # tie-safe binned.
            "n_good": {"column": "n_within_epsilon", "name": "n within ε (SF-1)" + _u,
                       "filter_query": "move_time > 0 AND n_within_epsilon IS NOT NULL",
                       "bin_mode": "integer", "integer_bin_width": 3, "integer_tail_cut": 30,
                       "hist_kind": "disc", "hist_clip": (0, 30)},
            "n_acceptable": {"column": "n_acceptable", "name": "n acceptable (SF-1)" + _u,
                             "filter_query": "move_time > 0 AND n_acceptable IS NOT NULL",
                             "zero_inflated": True, "zero_threshold": 0.0,
                             "hist_kind": "cont", "hist_clip": None},
            "oss": {"column": "oss", "name": "Optimal stop step (SF-1)" + _u,
                    "filter_query": "move_time > 0 AND oss IS NOT NULL",
                    "bin_mode": "integer", "integer_bin_width": 5, "integer_tail_cut": 35,
                    "hist_kind": "disc", "hist_clip": (0, 35)},
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
            save_tree_dashboard(a_ply, out_dir, name,
                                hist_kind=cfg["hist_kind"], hist_clip=cfg["hist_clip"],
                                hist_where=cfg["filter_query"])

        # Supplementary confound-removed dashboards (board.py's ``supplements``
        # pattern, generalized here since engine.py had no equivalent): identify a
        # confound, build a filtered view excluding/isolating it, and RE-RUN THE SAME
        # dashboard function — the dip vanishing on the plot is the evidence, not a
        # prose claim. GSS/OSS's mid-range dip (trough ~20-27) is a clock/time-
        # pressure composition-shift artifact: higher-GSS/OSS trees skew toward
        # earlier game stage with MORE clock remaining, and remaining clock has its
        # own (board.py-documented) non-monotonic relationship with RT — mixing
        # clock regimes together reproduces a dip that is absent (or far weaker)
        # within any single clock tertile. Verified empirically (not asserted): the
        # pooled GSS trend dips from 6.60s (qbin~12) to 6.31s (qbin~22) then rises to
        # 8.1s (tail); restricted to the MIDDLE clock tertile alone the same bins run
        # 7.24 -> 7.23 -> 7.24s (essentially flat) before the same late rise — the
        # dip's amplitude collapses by roughly an order of magnitude. (n_acceptable's
        # and n_within_epsilon's own mid-range troughs were ALSO tested against this
        # clock-tertile split and persisted within every tertile — clock does NOT
        # explain those two; left as an open, honestly-unresolved wrinkle.)
        clock_c1, clock_c2 = conn.execute(
            "SELECT quantile_disc(player_clock_time, 1.0/3), quantile_disc(player_clock_time, 2.0/3) "
            "FROM tree_rt WHERE player_clock_time IS NOT NULL"
        ).fetchone()
        clock_mid_sql = f"player_clock_time > {clock_c1} AND player_clock_time <= {clock_c2}"
        supplements = {"supp_gss_clockmid": "gss", "supp_oss_clockmid": "oss"}
        for supp_name, sig_name in supplements.items():
            cfg = tree_signals[sig_name]
            print(f"Executing engine tree analysis: {supp_name} (mid clock tertile [{clock_c1:.0f}, {clock_c2:.0f}]s)...")
            supp_filter = f"({cfg['filter_query']}) AND ({clock_mid_sql})"
            a_supp = _make_analyzer(
                "tree_rt",
                # NOT suffixed with "(mid clock tertile)" on the axis label itself: at
                # FONT_SIZE_LABEL that text is wider than a 1x3 panel and bleeds into
                # its neighbor (the same class of overlap bug fixed in _x_axis_label
                # for zero_inflated/edge_mass — here the fix is simply "keep it short"
                # since bin_mode="integer" carries no qualifier suffix at all). The
                # supplement's nature is conveyed by the filename/title, not the axis.
                Variable(column=cfg["column"], is_log=False, name=cfg["name"]),
                Variable(column="move_time", is_log=True, name="RT"),
                f"{cfg['name']} vs. log(RT), mid clock tertile",
                filter_query=supp_filter, min_bin_count=300, tie_safe=True,
                zero_inflated=cfg.get("zero_inflated", False),
                zero_threshold=cfg.get("zero_threshold", 0.0),
                bin_mode=cfg.get("bin_mode", "ntile"),
                integer_bin_width=cfg.get("integer_bin_width", 1),
                integer_tail_cut=cfg.get("integer_tail_cut"),
            )
            save_tree_dashboard(a_supp, out_dir, supp_name,
                                hist_kind=cfg["hist_kind"], hist_clip=cfg["hist_clip"],
                                hist_where=supp_filter)

        # Axes are intentionally inverted vs the other signal dashboards: RT on X,
        # MQ on Y. This panel asks "do longer thinks yield better moves?", so RT is
        # the predictor and MQ the outcome — not signal-on-X-vs-RT like the rest.
        # The histogram panel still shows MQ's own marginal (not RT's, already
        # covered by move_time_summary), so hist_column/hist_label override the
        # Analyzer's (inverted) x_var.
        print("Executing engine tree analysis: mq...")
        mq_ply = _make_analyzer(
            "mq_rt",
            Variable(column="move_time", is_log=True, name="RT (s)"),
            Variable(column="mq", is_log=False, name=f"MQ (SF-1) ({unit})"),
            f"MQ (SF-1) ({unit}) vs. log(RT)",
            filter_query="move_time > 0", min_bin_count=300, n_bins=10,
        )
        save_tree_dashboard(mq_ply, out_dir, "mq",
                            hist_column="mq", hist_label=f"MQ (SF-1) ({unit})",
                            hist_kind="cont", hist_clip=None, hist_where="move_time > 0")

        print("Executing engine tree analysis: correlation_matrix...")
        plot_engine_correlation_matrix(conn, os.path.join(out_dir, "correlation_matrix.pdf"), unit=unit)

        print("Executing engine tree analysis: P1 trust tests...")
        trust_tests(conn, unit)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified engine-level and stopping analyses")
    parser.add_argument(
        "--mode", choices=["all", "mq_gss", "tree_values"], default="all",
        help="Execution mode: all (default), difficulty confound stats, or full tree-values pipeline."
    )
    parser.add_argument("--db", default=CONFIG["selected_db_default"])
    parser.add_argument("--seed", type=int, default=7)

    # Tree values args
    parser.add_argument("--trees-dir", default=CONFIG["trees_default"])
    parser.add_argument("--n-trees", type=int, default=100000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--cache-dir", default=CONFIG["cache_default"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--key", default=None,
                        help="cache key for mq_gss (default: derived from --trees-dir/--n-trees/--seed/--unit)")
    parser.add_argument("--unit", choices=sorted(UNITS), default="pwin",
                        help="value unit for the tree signals (pwin | cp); figures go to engine_<unit>/")

    args = parser.parse_args(argv)
    key = args.key or cache_key(args.trees_dir, args.n_trees, args.seed, args.unit)

    # tree_values BUILDS the cache parquet that mq_gss READS, so it must run first
    # in --mode all (otherwise a fresh cache, e.g. the first SF-2000 run, fails).
    modes = {
        "tree_values": {
            "func": tree_values_pipeline,
            "args": [args.trees_dir, args.n_trees, args.n_workers, args.seed, args.db,
                     args.cache_dir, args.refresh, args.unit]
        },
        "mq_gss": {
            "func": difficulty_confound_stats,
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

