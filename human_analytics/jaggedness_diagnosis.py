"""Jaggedness diagnosis for gain_vs_rt.png and gss_vs_rt.png (diagnose_gain.md Part 2 ext).

The puzzle: the binned mean-RT curve is visibly JAGGED bin-to-bin, yet the analytic per-bin
SEM bars are razor-thin. This script resolves WHY, running five mechanisms on the FULL-n
human_trees cache and emitting figures/jaggedness_diagnosis.png:

  1. Curve-level (shape) bootstrap vs per-bin SEM: does the honest band widen, and does the
     jaggedness fall within it? Reports fraction of adjacent jumps exceeding (i) per-bin SEM
     vs (ii) curve-bootstrap band.
  2. Bin-count robustness: K in {5,8,12,20,40}, quantile vs equal-width; roughness(K).
  3. Predictor discreteness: GSS histogram; native-integer binning vs quantile.
  4. Estimator choice: arithmetic-mean RT vs geometric-mean (mean logT) vs median.
  5. Confound spot-check: ply / legal_moves / own_material across adjacent jagged bins.

IMPORTANT: the published figures plot **mean logT** (geometric-mean RT after exp), NOT
arithmetic mean RT, with SEM on mean logT. We reproduce that displayed quantity.

    PYTHONPATH=human_analytics python human_analytics/jaggedness_diagnosis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.jaggedness import (
    quantile_bin_means, equal_width_bin_means, integer_bin_means,
    curve_bootstrap, roughness, adjacent_jumps, frac_jumps_exceeding,
)

_CACHE = Path("/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache")
_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_FIG = Path("/home/hl4291/chess_analysis/figures/jaggedness_diagnosis.png")
_VALS = _CACHE / "vals_human_trees_10000000_7.parquet"

B = 1000
SEED = 7


def _try_load() -> pd.DataFrame:
    """Join the full-n vals cache to processed_moves_nonzero by fen. Confounders:
    n_possible_moves (legal moves) and n_self_pieces_exc_pawns (own-material proxy)."""
    vals = pd.read_parquet(_VALS)
    conn = duckdb.connect(_DB, read_only=True)
    conn.register("_vals", vals)
    df = conn.execute("""
        SELECT v.gss, v.voc, ln(m.move_time) AS logT, m.move_time AS rt,
               m.move_ply AS ply,
               m.n_possible_moves AS legal_moves,
               m.n_self_pieces_exc_pawns AS own_material
        FROM _vals v
        JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0
    """).df()
    conn.close()
    return df


def mech1_curve_bootstrap(name, x, y, k, binning, rng):
    """Mechanism 1: per-bin SEM vs curve-level bootstrap band."""
    print("\n" + "=" * 78)
    print(f"[M1] {name}: per-bin SEM vs curve-level (shape) bootstrap  (k={k}, {binning})")
    print("=" * 78)
    if binning == "integer":
        ref = integer_bin_means(x, y, min_n=100)
    elif binning == "quantile":
        ref = quantile_bin_means(x, y, k, min_n=100)
    res = curve_bootstrap(x, y, k, binning=binning, n_boot=B, min_n=100, rng=rng)
    ref = res["ref"]
    sem = ref.sems_y
    band = res["half"]
    f_sem = frac_jumps_exceeding(ref, sem)
    f_band = frac_jumps_exceeding(ref, band)
    print(f"  bins kept: {ref.means_y.size}")
    print(f"  median per-bin SEM half-width  : {np.median(sem):.4f}")
    print(f"  median curve-bootstrap half-w  : {np.median(band):.4f}  "
          f"(band/SEM ratio median={np.median(band/np.where(sem>0,sem,np.nan)):.2f})")
    print(f"  RMS adjacent jump |Δmean|      : {roughness(ref):.4f}")
    print(f"  frac adjacent jumps > combined per-bin SEM   : {f_sem*100:.0f}%")
    print(f"  frac adjacent jumps > combined bootstrap band: {f_band*100:.0f}%")
    return ref, sem, band, f_sem, f_band


def mech2_bincount(name, x, y, Ks=(5, 8, 12, 20, 40)):
    print("\n" + "=" * 78)
    print(f"[M2] {name}: roughness vs bin-count K (quantile vs equal-width)")
    print("=" * 78)
    print(f"  {'K':>4s} {'rough_quantile':>16s} {'rough_eqwidth':>14s}")
    out = {}
    for K in Ks:
        cq = quantile_bin_means(x, y, K, min_n=100)
        cw = equal_width_bin_means(x, y, K, min_n=100)
        out[K] = (cq, cw)
        print(f"  {K:>4d} {roughness(cq):>16.4f} {roughness(cw):>14.4f}")
    return out


def mech3_discreteness(name, x, y):
    print("\n" + "=" * 78)
    print(f"[M3] {name}: predictor discreteness (native-integer vs quantile binning)")
    print("=" * 78)
    xi = np.round(x).astype(int)
    vals, counts = np.unique(xi, return_counts=True)
    print(f"  distinct integer values of x: {vals.size}  (range {vals.min()}..{vals.max()})")
    # show how quantile bins straddle integer sets
    q = quantile_bin_means(x, y, 8, min_n=100)
    edges = q.edges
    print(f"  8-quantile interior edges (note non-integer cuts straddling integer mass): "
          f"{np.round(edges, 2).tolist()}")
    cn = integer_bin_means(x, y, min_n=100)
    print(f"  native-integer binning: {cn.means_y.size} points, "
          f"roughness={roughness(cn):.4f} vs 8-quantile roughness={roughness(q):.4f}")
    return cn, vals, counts


def mech4_estimator(name, x, rt, logT, k, binning):
    print("\n" + "=" * 78)
    print(f"[M4] {name}: estimator choice (arith mean RT vs geom mean vs median), k={k}")
    print("=" * 78)
    if binning == "integer":
        c_arith = integer_bin_means(x, rt, min_n=100, center="mean")
        c_geo = integer_bin_means(x, logT, min_n=100, center="mean")
        c_med = integer_bin_means(x, rt, min_n=100, center="median")
    else:
        c_arith = quantile_bin_means(x, rt, k, min_n=100, center="mean")
        c_geo = quantile_bin_means(x, logT, k, min_n=100, center="mean")
        c_med = quantile_bin_means(x, rt, k, min_n=100, center="median")
    # normalize roughness by the curve's own scale (std of centers) for cross-center compare
    def nr(c):
        s = np.std(c.means_y)
        return roughness(c) / s if s > 0 else float("nan")
    print(f"  normalized roughness (RMS jump / std of curve):")
    print(f"    arithmetic mean RT  : {nr(c_arith):.3f}")
    print(f"    geometric mean (logT, the FIGURE's center): {nr(c_geo):.3f}")
    print(f"    median RT           : {nr(c_med):.3f}")
    return c_arith, c_geo, c_med


def mech5_confound(name, df, xcol, k, binning):
    print("\n" + "=" * 78)
    print(f"[M5] {name}: confounder composition across adjacent jagged bins")
    print("=" * 78)
    x = df[xcol].to_numpy(float)
    logT = df["logT"].to_numpy(float)
    if binning == "integer":
        c = integer_bin_means(x, logT, min_n=100)
        xi = np.round(x).astype(int)
        vals = np.round(c.centers_x).astype(int)
        def mask_for(b):
            return xi == vals[b], f"{xcol}={vals[b]}"
    else:
        c = quantile_bin_means(x, logT, k, min_n=100)
        # Re-derive each kept bin's membership from edges midway between consecutive kept
        # centers (robust to min_n having dropped bins, unlike a raw qbin index).
        centers = c.centers_x
        cuts = (centers[:-1] + centers[1:]) / 2.0
        assign = np.searchsorted(cuts, x, side="left")
        def mask_for(b):
            return assign == b, f"bin@x~{centers[b]:.3f}"
    j = adjacent_jumps(c)
    if j.size == 0:
        print("  (no adjacent pairs)")
        return
    ji = int(np.argmax(j))  # biggest jump between kept bins ji and ji+1
    print(f"  biggest adjacent jump between kept bins {ji} and {ji+1}: Δmean_logT={j[ji]:.3f}")
    cols = [cc for cc in ("ply", "legal_moves", "own_material")
            if cc in df.columns and df[cc].notna().any()]
    for b in (ji, ji + 1):
        m, tag = mask_for(b)
        sub = df[m]
        stat = "  ".join(f"{cc}={sub[cc].mean():.2f}" for cc in cols)
        ml = logT[m].mean() if m.any() else float("nan")
        print(f"    {tag:<16s} n={m.sum():6d}  meanlogT={ml:.3f}  {stat}")
    print("  → if adjacent bins differ systematically in a confounder, the jump is REAL "
          "conditional structure (bars are 'right'), not sampling noise.")


# ------------------------------------------------------------------- figure
def make_figure(gain, gss, rng):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ---- GAIN row -------------------------------------------------------
    x = gain["voc"].to_numpy(float); logT = gain["logT"].to_numpy(float)
    # (a) multi-K quantile overlay
    ax = axes[0, 0]
    for K in (5, 8, 12, 20, 40):
        c = quantile_bin_means(x, logT, K, min_n=100)
        ax.plot(c.centers_x, c.means_y, "-o", ms=3, alpha=0.7, label=f"K={K}")
    ax.set_title("gain_vs_rt: multi-K quantile curves (mean logT)")
    ax.set_xlabel("Gain"); ax.set_ylabel("mean logT"); ax.legend(fontsize=7)
    # (b) per-bin SEM vs curve bootstrap band
    ax = axes[0, 1]
    res = curve_bootstrap(x, logT, 8, binning="quantile", n_boot=B, min_n=100, rng=rng)
    c = res["ref"]
    ax.errorbar(c.centers_x, c.means_y, yerr=c.sems_y, fmt="-o", ms=4, color="C0",
                capsize=2, label="per-bin SEM (figure)")
    ax.fill_between(c.centers_x, c.means_y - res["half"], c.means_y + res["half"],
                    color="C1", alpha=0.3, label="curve-bootstrap band")
    ax.set_title("gain_vs_rt: per-bin SEM vs curve-bootstrap (K=8)")
    ax.set_xlabel("Gain"); ax.set_ylabel("mean logT"); ax.legend(fontsize=8)
    # (c) estimator comparison
    ax = axes[0, 2]
    rt = gain["rt"].to_numpy(float)
    ca = quantile_bin_means(x, rt, 8, min_n=100, center="mean")
    cm = quantile_bin_means(x, rt, 8, min_n=100, center="median")
    cg = quantile_bin_means(x, logT, 8, min_n=100, center="mean")
    ax.plot(ca.centers_x, ca.means_y / ca.means_y.mean(), "-o", ms=3, label="arith mean RT")
    ax.plot(cm.centers_x, cm.means_y / cm.means_y.mean(), "-s", ms=3, label="median RT")
    ax.plot(cg.centers_x, np.exp(cg.means_y) / np.exp(cg.means_y).mean(), "-^", ms=3,
            label="geo mean (figure)")
    ax.set_title("gain_vs_rt: center choice (normalized)")
    ax.set_xlabel("Gain"); ax.set_ylabel("RT center / its mean"); ax.legend(fontsize=8)

    # ---- GSS row --------------------------------------------------------
    x = gss["gss"].to_numpy(float); logT = gss["logT"].to_numpy(float)
    ax = axes[1, 0]
    for K in (5, 8, 12, 20, 40):
        c = quantile_bin_means(x, logT, K, min_n=100)
        ax.plot(c.centers_x, c.means_y, "-o", ms=3, alpha=0.7, label=f"K={K}")
    cn = integer_bin_means(x, logT, min_n=100)
    ax.plot(cn.centers_x, cn.means_y, "-", color="k", lw=1.5, alpha=0.6, label="native int")
    ax.set_title("gss_vs_rt: multi-K quantile + native-integer (mean logT)")
    ax.set_xlabel("GSS"); ax.set_ylabel("mean logT"); ax.legend(fontsize=7)
    # (b) per-bin SEM vs curve bootstrap, native integer
    ax = axes[1, 1]
    res = curve_bootstrap(x, logT, 0, binning="integer", n_boot=B, min_n=100, rng=rng)
    c = res["ref"]
    ax.errorbar(c.centers_x, c.means_y, yerr=c.sems_y, fmt="-o", ms=3, color="C0",
                capsize=2, label="per-bin SEM (native int)")
    ax.fill_between(c.centers_x, c.means_y - res["half"], c.means_y + res["half"],
                    color="C1", alpha=0.3, label="curve-bootstrap band")
    ax.set_title("gss_vs_rt: per-bin SEM vs curve-bootstrap (native int)")
    ax.set_xlabel("GSS"); ax.set_ylabel("mean logT"); ax.legend(fontsize=8)
    # (c) GSS histogram
    ax = axes[1, 2]
    xi = np.round(x).astype(int)
    vals, counts = np.unique(xi, return_counts=True)
    ax.bar(vals, counts, width=1.0, color="C2", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("GSS value histogram (integer, heavy 0/1 mass)")
    ax.set_xlabel("GSS"); ax.set_ylabel("count (log)")

    fig.tight_layout()
    _FIG.parent.mkdir(exist_ok=True)
    fig.savefig(_FIG, dpi=110)
    print(f"\nSaved figure → {_FIG}")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"Loading FULL-n cache: {_VALS.name}")
    df = _try_load()
    print(f"  joined rows (move_time>0): {len(df):,}")
    rng = np.random.default_rng(SEED)

    gain = df[df["voc"].notna()].copy()
    gss = df[df["gss"].notna()].copy()
    print(f"  gain rows: {len(gain):,}   gss rows: {len(gss):,}")

    xg = gain["voc"].to_numpy(float); lg = gain["logT"].to_numpy(float)
    xs = gss["gss"].to_numpy(float); ls = gss["logT"].to_numpy(float)

    # GAIN: quantile K=8 (figure uses zero-inflated+tie-safe 8; here plain quantile-8 for the
    # mechanism contrast — the discreteness/edge effects are the same family).
    mech1_curve_bootstrap("gain_vs_rt", xg, lg, 8, "quantile", rng)
    mech2_bincount("gain_vs_rt", xg, lg)
    mech3_discreteness("gain_vs_rt", xg, lg)
    mech4_estimator("gain_vs_rt", xg, gain["rt"].to_numpy(float), lg, 8, "quantile")
    mech5_confound("gain_vs_rt", gain, "voc", 8, "quantile")

    # GSS: native integer (the figure uses integer_bin_width=5; native-1 is the discreteness probe)
    mech1_curve_bootstrap("gss_vs_rt", xs, ls, 0, "integer", rng)
    mech2_bincount("gss_vs_rt", xs, ls)
    mech3_discreteness("gss_vs_rt", xs, ls)
    mech4_estimator("gss_vs_rt", xs, gss["rt"].to_numpy(float), ls, 0, "integer")
    mech5_confound("gss_vs_rt", gss, "gss", 0, "integer")

    make_figure(gain, gss, rng)


if __name__ == "__main__":
    main()
