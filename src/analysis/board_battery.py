"""P0 board-correlate battery — analysis over the engine-free features
(plan_final_push.md P0; rules per outputs/reports/trust_protocol.md).

Consumes the per-FEN features written by ``analysis.featurize_board`` plus the
DB, and produces the staged deliverables:

  hist   P0.a — per-FEN feature distributions (no RT; sanity before any trend)
  corr   P0.b — Spearman+Pearson vs ln(RT) with FEN-clustered bootstrap CIs and
                partials vs n_possible_moves, on BOTH units (per-instance,
                per-FEN), from a game-level sample
  shape  P0.b — the pre-registered ∩-shape test for material_imbalance
  dash   P0.c — 1x3 dashboards (overall / by ply / by game fraction) per feature
                over the full windowed join (run via SLURM; heavy)

Usage:
  python -m analysis.board_battery --config config_minply15_maxply75.yaml --stage hist
"""

import argparse
import os

# --config early-peek: set $CONFIG before helpers import reads it (same shim as engine.py).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config")
_args, _ = _pre.parse_known_args()
if _args.config:
    os.environ["CONFIG"] = _args.config

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.utils.analysis import Analyzer, Variable  # noqa: E402
from analysis.utils.helpers import (  # noqa: E402
    CONFIG,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    GAME_FRAC_CUTS,
    GAME_FRAC_LABELS,
    MAIN_COLOR,
    apply_poster_style,
    create_ply_windowed_views,
)
from analysis.utils.selected_db import SELECTED_DB_DEFAULT  # noqa: E402

# Feature -> (pretty name, pre-registered hypothesis tag) — order fixes figure/matrix order.
FEATURES = {
    "n_captures_avail": "Captures available",
    "n_checks_avail": "Checks available",
    "self_material": "Self material (non-pawn)",
    "material_imbalance": "Material imbalance",
    "in_check": "In check",
    "prev_move_was_capture": "Prev move was capture",
}
_BOOT_HEADLINE = 1000  # trust protocol: B=1000 headline
_BOOT_CLUSTER = 200    # B=200 for the heavier clustered/per-instance runs


def features_glob(out_dir: str) -> str:
    return os.path.join(out_dir, "features_*.parquet")


def setup_views(conn, feat_glob: str) -> None:
    """Windowed views + game_fraction + features view + prev-capture flag."""
    create_ply_windowed_views(conn)
    conn.execute(
        "CREATE OR REPLACE TEMP VIEW pmnz_gf AS "
        "SELECT *, move_ply * 1.0 / max(move_ply) OVER (PARTITION BY gid) AS game_fraction "
        "FROM pmnz_win"
    )
    conn.execute(f"CREATE OR REPLACE TEMP VIEW feats AS SELECT * FROM read_parquet('{feat_glob}')")
    # prev_move_was_capture: opponent's last move shrank the piece count. Lag over
    # the FULL moves table (not the window) so ply-15 rows still see ply-14.
    conn.execute(
        "CREATE OR REPLACE TEMP VIEW prev_cap AS "
        "SELECT gid, move_ply, "
        "       (n_pieces < lag(n_pieces) OVER (PARTITION BY gid ORDER BY move_ply)) "
        "         AS prev_move_was_capture "
        f"FROM {CONFIG['table_moves']}"
    )


def battery_join_sql(sample_gid_table: str | None = None) -> str:
    """The battery join: windowed instances x features x prev-capture flag."""
    gid_filter = f"JOIN {sample_gid_table} sg ON sg.gid = m.gid" if sample_gid_table else ""
    return f"""
        SELECT m.gid, m.move_ply, m.fen, m.move_time, m.game_fraction,
               m.n_possible_moves, m.player_clock_time,
               f.in_check, f.n_captures_avail, f.n_checks_avail,
               f.self_material, f.material_imbalance,
               p.prev_move_was_capture
        FROM pmnz_gf m
        {gid_filter}
        JOIN feats f ON f.fen = m.fen
        LEFT JOIN prev_cap p ON p.gid = m.gid AND p.move_ply = m.move_ply
        WHERE m.move_time > 0
    """


# ---------------------------------------------------------------------------
# hist — P0.a
# ---------------------------------------------------------------------------

def run_hist(conn, feat_glob: str, out_dir: str) -> None:
    """Per-FEN feature distributions straight off the parquet (unit = position)."""
    apply_poster_style()
    fig, axes = plt.subplots(2, 3, figsize=(30, 16))
    specs = [
        ("n_captures_avail", None), ("n_checks_avail", None), ("material_imbalance", None),
        ("self_material", None), ("in_check", (0, 1)),
    ]
    for ax, (col, _) in zip(axes.flat, specs):
        counts = conn.execute(
            f"SELECT {col} AS v, count(*) AS n FROM read_parquet('{feat_glob}') "
            f"GROUP BY 1 ORDER BY 1"
        ).df()
        total = counts.n.sum()
        ax.bar(counts.v.astype(float), counts.n / total, width=0.9,
               color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
        top = counts.sort_values("n", ascending=False).iloc[0]
        ax.set_title(f"{FEATURES.get(col, col)}\nmode {top.v} ({100 * top.n / total:.1f}%)",
                     fontsize=FONT_SIZE_TICKS)
        ax.set_ylabel("fraction of FENs", fontsize=FONT_SIZE_LABEL - 6)
    axes.flat[-1].axis("off")
    n_fens = conn.execute(f"SELECT count(*) FROM read_parquet('{feat_glob}')").fetchone()[0]
    fig.suptitle(f"P0 feature distributions — per distinct FEN (n = {n_fens:,})",
                 fontsize=FONT_SIZE_LABEL)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"battery_distributions.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved P0.a histogram panel -> {out_dir}/battery_distributions.png")


# ---------------------------------------------------------------------------
# corr — P0.b
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _partial(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, rank: bool) -> float:
    """Partial corr of x,y | z (rank=True -> partial Spearman via ranked vars)."""
    if rank:
        x = pd.Series(x).rank().to_numpy()
        y = pd.Series(y).rank().to_numpy()
        z = pd.Series(z).rank().to_numpy()
    zc = np.column_stack([np.ones_like(z), z])
    bx, *_ = np.linalg.lstsq(zc, x, rcond=None)
    by, *_ = np.linalg.lstsq(zc, y, rcond=None)
    return float(np.corrcoef(x - zc @ bx, y - zc @ by)[0, 1])


def _boot_ci(fn, df: pd.DataFrame, b: int, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap CI resampling ROWS of df (rows = FENs for per-FEN unit)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    stats = [fn(df.iloc[rng.integers(0, n, n)]) for _ in range(b)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_corr(conn, out_dir: str, sample_games: int) -> None:
    """Spearman+Pearson (+partials vs legal moves) on both units, bootstrap CIs."""
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE sample_gids AS "
        f"SELECT DISTINCT gid FROM pmnz_win USING SAMPLE {sample_games}"
    )
    inst = conn.execute(
        battery_join_sql("sample_gids") + " AND m.move_time > 0"
    ).df()
    inst["log_rt"] = np.log(inst.move_time)
    inst["prev_move_was_capture"] = inst.prev_move_was_capture.fillna(False).astype(int)
    inst["in_check"] = inst.in_check.astype(int)
    # per-FEN unit: mean log RT per position (features are constant within FEN)
    per_fen = inst.groupby("fen").agg(
        log_rt=("log_rt", "mean"), n_possible_moves=("n_possible_moves", "first"),
        **{c: (c, "first") for c in FEATURES},
    ).reset_index(drop=True)

    rows = []
    for col in FEATURES:
        for unit, df, b in (("instance", inst, _BOOT_CLUSTER), ("fen", per_fen, _BOOT_HEADLINE)):
            x, y, z = df[col].to_numpy(float), df.log_rt.to_numpy(), df.n_possible_moves.to_numpy(float)
            rec = {
                "feature": col, "unit": unit, "n": len(df),
                "spearman": _spearman(x, y), "pearson": float(np.corrcoef(x, y)[0, 1]),
                "partial_spearman_legal": _partial(x, y, z, rank=True),
                "partial_pearson_legal": _partial(x, y, z, rank=False),
            }
            lo, hi = _boot_ci(
                lambda d, c=col: _partial(d[c].to_numpy(float), d.log_rt.to_numpy(),
                                          d.n_possible_moves.to_numpy(float), rank=True),
                df, b)
            rec["partial_spearman_ci_lo"], rec["partial_spearman_ci_hi"] = lo, hi
            rows.append(rec)
    res = pd.DataFrame(rows)
    res_path = os.path.join(out_dir, "battery_correlations.csv")
    res.to_csv(res_path, index=False)
    print(res.round(4).to_string(index=False))
    print(f"Saved -> {res_path}")


# ---------------------------------------------------------------------------
# shape — P0.b (∩ test for material_imbalance, per trust protocol)
# ---------------------------------------------------------------------------

def run_shape(conn, out_dir: str, sample_games: int) -> None:
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE sample_gids AS "
        f"SELECT DISTINCT gid FROM pmnz_win USING SAMPLE {sample_games}"
    )
    df = conn.execute(battery_join_sql("sample_gids")).df()
    df["log_rt"] = np.log(df.move_time)
    # integer bins, tails merged at |imb| >= 6 (protocol: peak {-1,0,1}, tails |imb|>=3)
    imb = df.material_imbalance.clip(-6, 6)
    g = df.assign(imb=imb).groupby("imb").log_rt
    agg = g.agg(["mean", "median", "count"])
    rng = np.random.default_rng(7)
    ci = {}
    for v, grp in df.assign(imb=imb).groupby("imb"):
        vals = grp.log_rt.to_numpy()
        boots = [np.median(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(_BOOT_HEADLINE)]
        ci[v] = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    agg["ci_lo"] = [ci[v][0] for v in agg.index]
    agg["ci_hi"] = [ci[v][1] for v in agg.index]

    # Test 1: peak bins' CI lower bounds above at least one bin's CI upper in EACH tail
    peak_lo = agg.loc[[-1, 0, 1], "ci_lo"].min()
    left_tail = agg.loc[agg.index <= -3, "ci_hi"]
    right_tail = agg.loc[agg.index >= 3, "ci_hi"]
    test1 = bool(len(left_tail) and len(right_tail)
                 and (peak_lo > left_tail.min()) and (peak_lo > right_tail.min()))
    # Test 2: quadratic term negative with CI excluding 0, quadratic dR2 >= 0.001
    x = df.material_imbalance.to_numpy(float)
    y = df.log_rt.to_numpy()
    X1 = np.column_stack([np.ones_like(x), x])
    X2 = np.column_stack([np.ones_like(x), x, x ** 2])
    b1, res1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    b2, res2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    sst = ((y - y.mean()) ** 2).sum()
    dr2 = float((res1[0] - res2[0]) / sst) if len(res1) and len(res2) else float("nan")
    boots_b2 = []
    n = len(df)
    for _ in range(_BOOT_CLUSTER):
        idx = rng.integers(0, n, n)
        bb, *_ = np.linalg.lstsq(X2[idx], y[idx], rcond=None)
        boots_b2.append(bb[2])
    b2_lo, b2_hi = np.percentile(boots_b2, [2.5, 97.5])
    test2 = bool(b2[2] < 0 and b2_hi < 0 and dr2 >= 0.001)

    verdict = {True: {True: "CONFIRMED (both tests)", False: "suggestive (binned only)"},
               False: {True: "suggestive (quadratic only)", False: "NOT confirmed"}}[test1][test2]
    print(f"∩-shape test: binned={test1}, quadratic={test2} (β2={b2[2]:+.5f} "
          f"[{b2_lo:+.5f},{b2_hi:+.5f}], ΔR²={dr2:.5f}) -> {verdict}")

    apply_poster_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.errorbar(agg.index, agg["median"],
                yerr=[agg["median"] - agg.ci_lo, agg.ci_hi - agg["median"]],
                marker="o", lw=3, capsize=4, color=MAIN_COLOR, label="median log RT (95% CI)")
    ax.plot(agg.index, agg["mean"], marker="s", lw=1.5, ls="--", color="gray", label="mean log RT")
    ax.set_xlabel("Material imbalance (mover POV, non-pawn; tails merged at ±6)",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("log RT", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"∩-shape test — {verdict}\n(n = {len(df):,} instances)", fontsize=FONT_SIZE_TICKS + 4)
    ax.legend(fontsize=FONT_SIZE_TICKS)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"imbalance_shape.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    agg.to_csv(os.path.join(out_dir, "imbalance_shape_bins.csv"))
    print(f"Saved -> {out_dir}/imbalance_shape.png")


# ---------------------------------------------------------------------------
# dash — P0.c (full join; heavy -> SLURM)
# ---------------------------------------------------------------------------

def run_dash(conn, out_dir: str) -> None:
    from analysis.engine import save_tree_dashboard  # reuse the 1x3 renderer

    conn.execute("CREATE OR REPLACE TEMP TABLE battery_rt AS " + battery_join_sql())
    n = conn.execute("SELECT count(*) FROM battery_rt").fetchone()[0]
    print(f"battery_rt: {n:,} instances")

    dash_specs = {
        "n_captures_avail": {"bin_mode": "integer", "integer_bin_width": 1, "integer_tail_cut": 8},
        "n_checks_avail": {"bin_mode": "integer", "integer_bin_width": 1, "integer_tail_cut": 4,
                           "zero_inflated": True, "zero_threshold": 0.0},
        "self_material": {"bin_mode": "integer", "integer_bin_width": 3},
        "material_imbalance": {"bin_mode": "integer", "integer_bin_width": 1, "integer_tail_cut": 6},
    }
    for col, opts in dash_specs.items():
        print(f"Battery dashboard: {col}...")
        common = dict(
            x_var=Variable(column=col, is_log=False, name=FEATURES[col]),
            y_var=Variable(column="move_time", is_log=True, name="RT"),
            title=f"{FEATURES[col]} vs. log(RT)",
            filter_query="move_time > 0", min_bin_count=300, tie_safe=True,
            ply_tertile_source="battery_rt", **opts,
        )
        a_ply = Analyzer(conn, "battery_rt", segment_column="move_ply",
                         segment_source="battery_rt", segment_label="Ply", **common)
        a_gf = Analyzer(conn, "battery_rt", segment_column="game_fraction",
                        segment_source="battery_rt", segment_label="Game fraction",
                        segment_cuts=GAME_FRAC_CUTS, segment_range_labels=GAME_FRAC_LABELS,
                        **common)
        save_tree_dashboard(a_ply, a_gf, out_dir, f"battery_{col}")

    # Binary features: grouped means + CI, not dashboards.
    for col in ("in_check", "prev_move_was_capture"):
        stats = conn.execute(
            f"SELECT {col} AS v, count(*) AS n, avg(ln(move_time)) AS mean_lrt, "
            f"       stddev(ln(move_time)) / sqrt(count(*)) AS sem "
            f"FROM battery_rt WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY 1"
        ).df()
        print(f"\n{col}:")
        print(stats.round(4).to_string(index=False))
        stats.to_csv(os.path.join(out_dir, f"battery_{col}_groups.csv"), index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0 board-correlate battery analysis")
    parser.add_argument("--config", help="Path to the run config (else $CONFIG or the default).")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--stage", choices=("hist", "corr", "shape", "dash", "all"), default="all")
    parser.add_argument(
        "--features-dir",
        default=os.path.join(os.path.dirname(CONFIG["cache_default"]), "board_features"),
    )
    parser.add_argument("--sample-games", type=int, default=25000,
                        help="game-level sample for corr/shape (~1.2M instances)")
    args = parser.parse_args(argv)

    out_dir = os.path.join(CONFIG["figures_dir"], "board")
    os.makedirs(out_dir, exist_ok=True)
    feat_glob = features_glob(args.features_dir)

    conn = duckdb.connect(args.db, read_only=True)
    try:
        setup_views(conn, feat_glob)
        if args.stage in ("hist", "all"):
            run_hist(conn, feat_glob, out_dir)
        if args.stage in ("corr", "all"):
            run_corr(conn, out_dir, args.sample_games)
        if args.stage in ("shape", "all"):
            run_shape(conn, out_dir, args.sample_games)
        if args.stage in ("dash", "all"):
            run_dash(conn, out_dir)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
