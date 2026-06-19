"""Diagnostic (diagnose_gain.md Part 2): confirm the fixed Gain has no ≈1.0 point-mass,
and audit the plotting CI (analytic SEM) against a percentile bootstrap on all four
error-bar figures.

Background. ``figures/{gain,mq,gss,actiongap}_vs_rt.png`` carry per-bin error bars that are
the **analytic normal SEM** on the displayed mean: ``1.96 · stddev(y) / sqrt(n)`` computed
per quantile bin in DuckDB (utils/analysis.py:366-386). The y the panels display is the
*transformed* quantity — ``move_time`` is plotted with ``is_log=True``, so the bin point is
mean **logT = ln(move_time)** and its SEM is on mean logT. (mq_vs_rt is the mirror: x=logT,
y=MQ; there the displayed/binned quantity is mean MQ.) We bootstrap the SAME displayed mean
([Q2]).

The old Gain bug (pre-c903646) put a spurious point-mass at Gain≈1.0 from reading an
UNVISITED-0.0 shallow Q. That mass (a) inflated the top tie-safe bin's n → analytic SEM → 0,
and (b) being low-RT, pulled the bin mean down — a tight band around a wrong value. The
hypothesis to TEST: the narrow CI was that contaminated-bin artifact, NOT a broken SEM
formula. If so, on the FIXED data the bootstrap and analytic half-widths agree (ratio ≈ 1)
and no bin carries a degenerate point-mass.

This script (a) re-confirms the spike is gone on the fixed Gain (reuses gain_dip_diagnostic's
characterization), (b) replicates each figure's exact binning, and for every bin reports
analytic vs bootstrap half-widths + ratio + a contamination triple (n, within-bin variance,
point-mass score), and (c) flags any point-mass bin.

    PYTHONPATH=human_analytics python human_analytics/gain_ci_audit.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from utils.ci import analytic_ci_half, bootstrap_ci_half, pointmass_score, bin_contamination

# Largest available full-n cache (no _96 suffix == the current growing-tree format).
# A Part-1B agent may write a larger sentinel; prefer it if present.
_CACHE = Path("/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache")
_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"


def _pick_vals() -> tuple[Path, Path]:
    """Return (vals_path, rootmoves_path) for the largest full-n (non-_96) human_trees cache,
    preferring a Part-1B 'sentinel'/'full' cache if one has appeared."""
    vals = sorted(_CACHE.glob("vals_human_trees_*.parquet"))
    sentinel = [p for p in vals if "sentinel" in p.name or "full" in p.name]
    pool = sentinel or [p for p in vals if not p.name.endswith("_96.parquet")]
    # largest n by the integer in the name
    def _n(p: Path) -> int:
        for tok in p.stem.split("_"):
            if tok.isdigit():
                return int(tok)
        return 0
    vals_path = max(pool, key=_n)
    rm_path = vals_path.parent / vals_path.name.replace("vals_", "rootmoves_")
    return vals_path, rm_path


def build_tables(vals_path: Path, rm_path: Path) -> duckdb.DuckDBPyConnection:
    """Reproduce tree_rt and mq_rt EXACTLY as tree_values_analysis.main does, so the bins
    match the published figures."""
    vals = pd.read_parquet(vals_path)
    root_moves = pd.read_parquet(rm_path)
    conn = duckdb.connect(_DB, read_only=True)
    conn.register("_vals", vals)
    conn.register("_root_moves", root_moves)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE tree_rt AS
        SELECT v.gss, v.voc, v.action_gap, v.h_pi, v.fen, m.gid, m.move_ply, m.move_time
        FROM _vals v
        JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0
    """)
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
    return conn


# --- per-figure binning (mirrors utils.analysis Analyzer interior logic) ------------------

def _zero_tiesafe_bins(x: np.ndarray, y: np.ndarray, zero_thr: float, n_bins: int):
    """zero_inflated + tie_safe interior, as gain_vs_rt / actiongap_vs_rt use. Yields
    (bin_label, y_in_bin). Bin 0 = near-zero lump; bins 1..n_bins = value-bucketed interior."""
    is_zero = np.abs(x) <= zero_thr
    yield "zero-lump", y[is_zero]
    xi, yi = x[~is_zero], y[~is_zero]
    if xi.size == 0:
        return
    edges = np.quantile(xi, [i / n_bins for i in range(1, n_bins)])
    # bin index = #cut-points exceeded (matches list_filter in analysis.py)
    idx = (xi[:, None] > edges[None, :]).sum(axis=1)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            lo = xi[m].min(); hi = xi[m].max()
            yield f"interior[{lo:.3f},{hi:.3f}]", yi[m]


def _integer_bins(x: np.ndarray, y: np.ndarray, width: int):
    """integer mode (gss_vs_rt: width 5). Yields (label, y_in_bin)."""
    grp = np.floor(x / width).astype(int)
    for g in np.unique(grp):
        m = grp == g
        yield f"gss[{g*width},{g*width+width-1}]", y[m]


def _ntile_bins(x: np.ndarray, y: np.ndarray, n_bins: int):
    """plain equal-count ntile (mq_vs_rt: x=logT ntile(20), y=MQ). Yields (label, y_in_bin)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(x.size)
    idx = (ranks * n_bins // x.size)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            yield f"logT-q{b+1}", y[m]


# --- the four figures, each as (name, fetch_xy, bin_fn, min_bin_count) --------------------

def _figure_specs(conn):
    def fetch(col, table, ycol, ylog):
        df = conn.execute(
            f"SELECT {col} AS x, {ycol} AS yraw FROM {table} "
            f"WHERE {col} IS NOT NULL AND {ycol} IS NOT NULL AND move_time > 0"
        ).df() if "move_time" in (col, ycol) or table else None
        return df
    # explicit fetches (y displayed quantity already transformed)
    gain = conn.execute("SELECT voc AS x, ln(move_time) AS y FROM tree_rt WHERE voc IS NOT NULL AND move_time>0").df()
    agap = conn.execute("SELECT action_gap AS x, ln(move_time) AS y FROM tree_rt WHERE action_gap IS NOT NULL AND move_time>0").df()
    gss = conn.execute("SELECT gss AS x, ln(move_time) AS y FROM tree_rt WHERE gss IS NOT NULL AND move_time>0").df()
    mq = conn.execute("SELECT ln(move_time) AS x, mq AS y FROM mq_rt WHERE mq IS NOT NULL AND move_time>0").df()
    return [
        ("gain_vs_rt", gain, lambda x, y: _zero_tiesafe_bins(x, y, 0.05, 8), 100, "mean logT"),
        ("actiongap_vs_rt", agap, lambda x, y: _zero_tiesafe_bins(x, y, 0.05, 8), 100, "mean logT"),
        ("gss_vs_rt", gss, lambda x, y: _integer_bins(x, y, 5), 100, "mean logT"),
        ("mq_vs_rt", mq, lambda x, y: _ntile_bins(x, y, 20), 100, "mean MQ"),
    ]


def confirm_spike_gone(conn) -> None:
    print("\n" + "=" * 78)
    print("=== TASK 1: is the Gain≈1.0 point-mass gone on the FIXED gain? ===")
    print("=" * 78)
    g = conn.execute("SELECT voc FROM tree_rt WHERE voc IS NOT NULL").df()["voc"].to_numpy()
    print(f"  n={len(g):,}  min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}")
    for lo, hi, lbl in [(1.0, 1.0, "exactly 1.0"), (0.99, 1.01, "[0.99,1.01]"),
                        (0.95, 2.0, ">=0.95"), (0.0, 0.0, "exactly 0")]:
        m = (g >= lo) & (g <= hi)
        print(f"    {lbl:<12s}: {m.mean()*100:7.4f}%  (n={m.sum():,})")
    print(f"    Gain<0 (should be 0): {(g < -1e-9).sum()}")
    ps = pointmass_score(g[np.abs(g) > 0.05])  # interior point-mass
    print(f"  interior (Gain>0.05) point-mass score (max single-value freq): {ps:.4f}")
    edges = np.arange(0.90, 1.06, 0.01)
    h, _ = np.histogram(g, bins=edges)
    print("  fine histogram near top (per 0.01):")
    for i, c in enumerate(h):
        bar = "#" * int(40 * c / max(h.max(), 1))
        print(f"    [{edges[i]:.2f},{edges[i+1]:.2f}) {c:6d} {bar}")
    spike = ((g >= 0.99) & (g <= 1.01)).mean()
    verdict = "GONE" if spike < 0.005 else "STILL PRESENT"
    print(f"  >>> ≈1.0 point-mass: {verdict} ({spike*100:.4f}% of mass in [0.99,1.01])")


def audit_figure(name, df, bin_fn, min_n, disp, B=2000, seed=7):
    print("\n" + "-" * 78)
    print(f"FIGURE {name}  (displayed: {disp})   n_total={len(df):,}")
    print("-" * 78)
    x = df["x"].to_numpy(); y = df["y"].to_numpy()
    rng = np.random.default_rng(seed)
    print(f"  {'bin':<26s} {'n':>7s} {'analytic±':>10s} {'boot±':>10s} "
          f"{'ratio':>6s} {'var':>8s} {'pmass':>6s}  flags")
    rows = []
    n_flag = 0
    ratios = []
    for label, yb in bin_fn(x, y):
        n = yb.size
        if n < min_n:
            continue  # plotted bins honor min_bin_count
        a = analytic_ci_half(yb)
        bhalf, blo, bhi = bootstrap_ci_half(yb, n_boot=B, rng=rng)
        cont = bin_contamination(yb)
        ratio = bhalf / a if a > 0 else float("nan")
        ratios.append(ratio)
        flags = []
        if cont["pointmass_score"] > 0.5:
            flags.append("POINTMASS")
            n_flag += 1
        if n < 30:
            flags.append("LOW-N")
        print(f"  {label:<26s} {n:>7d} {a:>10.5f} {bhalf:>10.5f} {ratio:>6.3f} "
              f"{cont['variance']:>8.4f} {cont['pointmass_score']:>6.3f}  {' '.join(flags)}")
        rows.append(dict(figure=name, bin=label, n=n, analytic=a, boot=bhalf,
                         ratio=ratio, variance=cont["variance"],
                         pointmass=cont["pointmass_score"]))
    if ratios:
        r = np.array(ratios)
        print(f"  >>> ratio bootstrap/analytic: median={np.median(r):.3f} "
              f"mean={r.mean():.3f} range=[{r.min():.3f},{r.max():.3f}]  "
              f"|ratio-1|>0.10 in {np.mean(np.abs(r-1)>0.10)*100:.0f}% of bins  "
              f"pointmass-flagged bins={n_flag}")
    return rows


def main() -> None:
    vals_path, rm_path = _pick_vals()
    print(f"Using cache: {vals_path.name}")
    if not rm_path.exists():
        print(f"  (rootmoves cache {rm_path.name} missing — mq_vs_rt will be skipped)")
    conn = build_tables(vals_path, rm_path)

    confirm_spike_gone(conn)

    print("\n" + "=" * 78)
    print("=== TASKS 2+3: analytic-vs-bootstrap CI + per-bin contamination, 4 figures ===")
    print("=" * 78)
    all_rows = []
    for name, df, bin_fn, min_n, disp in _figure_specs(conn):
        if df is None or len(df) == 0:
            print(f"\nFIGURE {name}: no data (cache missing) — skipped.")
            continue
        all_rows += audit_figure(name, df, bin_fn, min_n, disp)
    conn.close()

    print("\n" + "=" * 78)
    print("=== HEADLINE: bootstrap/analytic half-width ratio across ALL bins ===")
    print("=" * 78)
    rdf = pd.DataFrame(all_rows)
    for name, sub in rdf.groupby("figure"):
        r = sub["ratio"].to_numpy()
        print(f"  {name:<16s} bins={len(sub):2d}  median ratio={np.median(r):.3f}  "
              f"mean={r.mean():.3f}  max|ratio-1|={np.max(np.abs(r-1)):.3f}")
    rall = rdf["ratio"].to_numpy()
    print(f"  {'ALL':<16s} bins={len(rdf):2d}  median ratio={np.median(rall):.3f}  "
          f"mean={rall.mean():.3f}  max|ratio-1|={np.max(np.abs(rall-1)):.3f}")
    agree = np.mean(np.abs(rall - 1) <= 0.10) * 100
    print(f"\n  HYPOTHESIS (estimator sound; old narrow CI was contamination): "
          f"{'HELD' if agree >= 90 and (rdf['pointmass']>0.5).sum()==0 else 'CHECK'} "
          f"— {agree:.0f}% of bins within 10% ratio; "
          f"{(rdf['pointmass']>0.5).sum()} point-mass-flagged bins.")


if __name__ == "__main__":
    main()
