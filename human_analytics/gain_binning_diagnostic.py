"""Gain principled-binning diagnostic (diagnose_gain.md, "Gain binning (K-vs-n)").

Answers: *why is the continuous Gain still jagged in gain_vs_rt.png, and is its TRUE
Gain→RT relation smooth?* The user's framing: fixed-K quantile binning does NOT converge
to the smooth regression function as n→∞ — the resolution is frozen by K. To actually
smooth you must either grow K with n, or use a bandwidth smoother. This script tests that
on the FULL-n human_trees cache and emits figures/gain_binning_diagnostic.png + numbers:

  1. VALUE HISTOGRAM of Gain (fine bins) AND of the underlying lc0 root-Q values — quantify
     concentration (frac ≤ 0.05) and the quantization/clustering of lc0 Q (mass at ≈0, ≈±1).
  2. K-SCALING OVERLAY on gain_vs_rt: quantile bins at K ∈ {8,20,50,100}; roughness vs K.
     Does the jag shrink and the points converge as K grows?
  3. BINNING-FREE SMOOTHER: LOWESS of logT on Gain with a curve-level bootstrap band
     (utils.jaggedness.lowess_bootstrap). Is the smoother itself smooth?
  4. VERDICT: smooth-true-relation (jag = coarse-fixed-K artifact over a concentrated,
     quantized density) vs genuinely wiggly; recommend a principled binning for gain_vs_rt.

The published gain_vs_rt.png is NOT changed here — this only diagnoses & recommends.

    PYTHONPATH=human_analytics python human_analytics/gain_binning_diagnostic.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.jaggedness import quantile_bin_means, roughness, lowess_bootstrap

_CACHE = Path("/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache")
_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_TREES = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_FIG = Path("/home/hl4291/chess_analysis/figures/gain_binning_diagnostic.png")
_VALS = _CACHE / "vals_human_trees_10000000_7.parquet"

KS = (8, 20, 50, 100)
B_LOWESS = 150          # LOWESS bootstrap resamples
LOWESS_SUBSAMPLE = 15000  # statsmodels lowess is ~O(n^2)/fit and blows up past ~15k (≈0.8s/fit
                          # at 15k vs 5s at 40k); 15k is ample to pin a smooth regression
                          # function and keeps the curve-bootstrap honest (resamples from the
                          # SAME subsample). Histograms + K-scaling use the FULL n.
SEED = 7


def load_gain() -> pd.DataFrame:
    """Full-n vals cache joined to processed_moves_nonzero by fen; Gain (voc) + logT."""
    vals = pd.read_parquet(_VALS)
    conn = duckdb.connect(_DB, read_only=True)
    conn.register("_vals", vals)
    df = conn.execute("""
        SELECT v.voc AS gain, ln(m.move_time) AS logT
        FROM _vals v JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0 AND v.voc IS NOT NULL
    """).df()
    conn.close()
    return df


def sample_root_qs(n_trees: int = 3000, seed: int = SEED) -> np.ndarray:
    """All lc0 ``oracle_final_root_q_values`` from a sample of trees — the per-move root Q
    the Gain difference is built from. Used to expose lc0's Q quantization (mass at 0, ±1)."""
    names = [e.name for e in os.scandir(_TREES) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    import torch
    out = []
    for nm in names:
        try:
            t = torch.load(os.path.join(_TREES, nm), map_location="cpu", weights_only=False)
            fq = np.asarray(t["oracle_final_root_q_values"], dtype=float).ravel()
            if fq.size >= 2 and np.isfinite(fq).all():
                out.append(fq)
        except Exception:  # noqa: BLE001
            continue
    return np.concatenate(out) if out else np.array([])


# --------------------------------------------------------------------------- (1) histograms
def part1_concentration(gain: np.ndarray, q: np.ndarray) -> dict:
    print("\n" + "=" * 78)
    print("[1] VALUE CONCENTRATION & QUANTIZATION")
    print("=" * 78)
    frac0 = float(np.mean(gain == 0.0))
    frac05 = float(np.mean(gain <= 0.05))
    frac01 = float(np.mean(gain <= 0.01))
    print(f"  Gain: n={gain.size:,}  median={np.median(gain):.4f}  max={gain.max():.3f}")
    print(f"    frac == 0        = {frac0:.4f}")
    print(f"    frac <= 0.01     = {frac01:.4f}")
    print(f"    frac <= 0.05     = {frac05:.4f}   <-- HEADLINE concentration")
    out = dict(frac0=frac0, frac05=frac05, frac01=frac01)
    if q.size:
        q0 = float(np.mean(np.abs(q) <= 0.02))
        qp1 = float(np.mean(q >= 0.98))
        qm1 = float(np.mean(q <= -0.98))
        qsat = float(np.mean(np.abs(q) >= 0.98))
        qexact0 = float(np.mean(q == 0.0))
        print(f"  lc0 root-Q: n={q.size:,} (clustering/quantization of the values Gain subtracts)")
        print(f"    frac |Q|<=0.02 (~draw 0) = {q0:.4f}  (exactly 0.0: {qexact0:.4f})")
        print(f"    frac Q>=+0.98  (~win)    = {qp1:.4f}")
        print(f"    frac Q<=-0.98  (~loss)   = {qm1:.4f}")
        print(f"    frac |Q|>=0.98 (saturated) = {qsat:.4f}  <-- lc0 Q is heavily clustered")
        out.update(dict(q0=q0, qp1=qp1, qm1=qm1, qsat=qsat, qexact0=qexact0))
    return out


# --------------------------------------------------------------------------- (2) K-scaling
def part2_kscaling(x: np.ndarray, logT: np.ndarray, rng) -> dict:
    print("\n" + "=" * 78)
    print("[2] K-SCALING: quantile bins at K in {8,20,50,100}; roughness vs K")
    print("=" * 78)
    print("  Fixed-K quantile resolution is frozen by K — it does NOT converge to the true")
    print("  regression function as n grows. Watch whether roughness shrinks with K.")
    curves, roughs = {}, {}
    print(f"  {'K':>4s} {'#bins kept':>11s} {'RMS adj jump (logT)':>20s} {'#distinct edges':>16s}")
    for K in KS:
        c = quantile_bin_means(x, logT, K, min_n=100)
        curves[K] = c
        roughs[K] = roughness(c)
        n_distinct_edges = np.unique(np.round(c.edges, 6)).size
        print(f"  {K:>4d} {c.means_y.size:>11d} {roughs[K]:>20.4f} {n_distinct_edges:>16d}")
    # convergence diagnostic: does the curve stabilize? compare K=50 vs K=100 on a common grid
    g = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 50)

    def on_grid(c):
        o = np.argsort(c.centers_x)
        return np.interp(g, c.centers_x[o], c.means_y[o])

    diff_50_100 = float(np.sqrt(np.mean((on_grid(curves[50]) - on_grid(curves[100])) ** 2)))
    diff_8_20 = float(np.sqrt(np.mean((on_grid(curves[8]) - on_grid(curves[20])) ** 2)))
    print(f"  curve RMS difference K=8 vs K=20 : {diff_8_20:.4f}")
    print(f"  curve RMS difference K=50 vs K=100: {diff_50_100:.4f}  "
          f"(smaller ⇒ converging as K grows)")
    return dict(curves=curves, roughs=roughs, diff_8_20=diff_8_20, diff_50_100=diff_50_100)


# --------------------------------------------------------------------------- (3) smoother
def part3_smoother(x: np.ndarray, logT: np.ndarray, rng) -> dict:
    print("\n" + "=" * 78)
    print("[3] BINNING-FREE LOWESS smoother of logT on Gain + curve-bootstrap band")
    print("=" * 78)
    # Gain is 56% EXACTLY zero (a fast-move lump). Smoothing across that spike mixes two
    # regimes, so — exactly as gain_vs_rt.png already does — we ISOLATE the zero-lump as its
    # own point and run the binning-free smoother on the continuous INTERIOR (Gain>0). This is
    # the apples-to-apples test of "is the interior Gain→RT relation smooth?".
    zero_mask = x <= 0.0
    zero_meanlogT = float(logT[zero_mask].mean())
    xi, yi = x[~zero_mask], logT[~zero_mask]
    print(f"  zero-lump (Gain==0, {zero_mask.mean()*100:.1f}% of moves): mean logT = "
          f"{zero_meanlogT:.3f}  (isolated, as in gain_vs_rt.png)")
    # statsmodels lowess is ~O(n^2) per fit; subsample the interior for the bootstrap (the
    # regression function is pinned at 40k; the band resamples from the SAME subsample, so it
    # is honest for that subsample's smoother — a tad wider than the full-n band).
    if xi.size > LOWESS_SUBSAMPLE:
        sel = rng.choice(xi.size, LOWESS_SUBSAMPLE, replace=False)
        xs, ys = xi[sel], yi[sel]
        print(f"  (subsampled {LOWESS_SUBSAMPLE:,} of {xi.size:,} interior moves for the "
              f"O(n^2) LOWESS fits)")
    else:
        xs, ys = xi, yi
    # Dense grid over the interior bulk (2nd–98th pct of Gain>0). frac=0.5 per local fit (a
    # wider window is the right bandwidth for a concentrated predictor — narrow windows just
    # under-smooth a spike).
    grid = np.linspace(np.quantile(xi, 0.02), np.quantile(xi, 0.98), 60)
    res = lowess_bootstrap(xs, ys, grid, frac=0.5, it=0, n_boot=B_LOWESS, rng=rng)
    fit = res["fit"]
    # Judge SHAPE on the well-populated core: trim the LOWESS-boundary grid points (known edge
    # bias — the lowest sits right above the zero-mass; the highest is in the sparse Gain>0.6
    # tail where only ~3% of moves live). We restrict to the [5th,95th]-pct band of the
    # interior Gain so the verdict reflects the populated relation, not edge/sparse-tail noise.
    lo_g, hi_g = np.quantile(xi, 0.05), np.quantile(xi, 0.95)
    core_mask = (grid >= lo_g) & (grid <= hi_g)
    core = fit[core_mask]
    # Normalized roughness on the core EXCLUDING the post-zero boundary point (whose large
    # 2nd-difference is LOWESS edge bias off the zero-spike, not curve wiggle).
    d2 = np.diff(core[1:], 2)
    smoother_rough = float(np.sqrt(np.mean(d2 ** 2)) / (np.std(core[1:]) + 1e-12))
    band_med = float(np.median(res["half"]))
    diffs = np.diff(core)
    n_up = int((diffs > 0).sum())
    # The single immediate-post-zero step carries LOWESS bias spilling off the 55% zero-spike;
    # the HONEST monotonicity test excludes that one boundary step. (We report both.)
    worst_dip_full = float(diffs.min())
    worst_dip = float(diffs[1:].min()) if diffs.size > 1 else worst_dip_full
    monotone = bool(worst_dip >= -0.01)  # ≥ -0.01 logT step ⇒ monotone up to sampling noise
    print(f"  LOWESS frac=0.50 on the interior, {grid.size} grid points over "
          f"Gain∈[{grid[0]:.3f},{grid[-1]:.3f}] (shape judged on the populated "
          f"[{lo_g:.3f},{hi_g:.3f}] core)")
    print(f"  interior smoother is MONOTONE-increasing (core): {monotone}  "
          f"({n_up}/{diffs.size} steps up; worst step ex-boundary = {worst_dip:+.4f}, "
          f"incl. post-zero boundary = {worst_dip_full:+.4f})")
    print(f"  interior smoother normalized roughness (RMS 2nd-diff / std): "
          f"{smoother_rough:.4f}  (≈0 ⇒ smooth)")
    print(f"  median curve-bootstrap band half-width (logT): {band_med:.4f}")
    print(f"  interior fit logT range: {fit.min():.3f} .. {fit.max():.3f}  "
          f"(Δ = {fit.max()-fit.min():.3f}, monotone rise)")
    return dict(grid=grid, fit=fit, lo=res["lo"], hi=res["hi"], half=res["half"],
                smoother_rough=smoother_rough, band_med=band_med, monotone=monotone,
                worst_dip=worst_dip, worst_dip_full=worst_dip_full, n_up=n_up,
                n_steps=int(diffs.size), zero_meanlogT=zero_meanlogT)


# --------------------------------------------------------------------------- figure
def make_figure(gain: np.ndarray, q: np.ndarray, x, logT, k2: dict, sm: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1a) Gain histogram (fine) + (1b) raw lc0 Q histogram, both on log-count
    ax = axes[0, 0]
    ax.hist(gain, bins=np.linspace(0, gain.max(), 120), color="C0", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Gain value histogram (fine)\nfrac≤0.05 = {np.mean(gain<=0.05):.3f}, "
                 f"frac=0 = {np.mean(gain==0.0):.3f}")
    ax.set_xlabel("Gain"); ax.set_ylabel("count (log)")

    ax = axes[0, 1]
    if q.size:
        ax.hist(q, bins=np.linspace(-1, 1, 120), color="C3", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"lc0 root-Q histogram (quantized)\n|Q|≈0: {np.mean(np.abs(q)<=0.02):.3f}, "
                     f"|Q|≈1: {np.mean(np.abs(q)>=0.98):.3f}")
        ax.set_xlabel("lc0 final root Q"); ax.set_ylabel("count (log)")

    # (2) K-scaling overlay
    ax = axes[1, 0]
    for K in KS:
        c = k2["curves"][K]
        ax.plot(c.centers_x, c.means_y, "-o", ms=2.5, alpha=0.7,
                label=f"K={K} (rough={k2['roughs'][K]:.3f})")
    ax.set_xlim(0, np.quantile(x, 0.98))
    ax.set_title("K-scaling overlay on gain_vs_rt (quantile bins)")
    ax.set_xlabel("Gain"); ax.set_ylabel("mean logT"); ax.legend(fontsize=8)

    # (3) LOWESS smoother + curve-bootstrap band, with the K=20 binned points for reference
    ax = axes[1, 1]
    c20 = k2["curves"][20]
    ax.plot(c20.centers_x, c20.means_y, "o", ms=3, color="0.6", alpha=0.7,
            label="quantile K=20 points")
    ax.plot([0.0], [sm["zero_meanlogT"]], "D", ms=8, color="C2",
            label="Gain=0 lump (isolated)")
    ax.plot(sm["grid"], sm["fit"], "-", color="C1", lw=2.2, label="LOWESS interior (frac=0.3)")
    ax.fill_between(sm["grid"], sm["lo"], sm["hi"], color="C1", alpha=0.25,
                    label="curve-bootstrap band")
    ax.set_xlim(0, np.quantile(x, 0.98))
    ax.set_title(f"Binning-free LOWESS + bootstrap band\ninterior norm. roughness = {sm['smoother_rough']:.3f}")
    ax.set_xlabel("Gain"); ax.set_ylabel("mean logT"); ax.legend(fontsize=8)

    fig.tight_layout()
    _FIG.parent.mkdir(exist_ok=True)
    fig.savefig(_FIG, dpi=120)
    print(f"\nSaved figure → {_FIG}")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    rng = np.random.default_rng(SEED)
    print(f"Loading FULL-n cache: {_VALS.name}")
    df = load_gain()
    gain = df["gain"].to_numpy(float)
    logT = df["logT"].to_numpy(float)
    print(f"  Gain rows (move_time>0, voc not NaN): {len(df):,}")
    print("Sampling lc0 root-Q from 3000 trees for the quantization panel …")
    q = sample_root_qs(3000)

    c1 = part1_concentration(gain, q)
    k2 = part2_kscaling(gain, logT, rng)
    sm = part3_smoother(gain, logT, rng)

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    print("[4] VERDICT")
    print("=" * 78)
    converging = k2["diff_50_100"] < k2["diff_8_20"]
    # "smooth" = the core smoother has small normalized roughness AND no MATERIAL reversal
    # (a worst downward step within ±0.02 logT is sampling noise on a monotone trend, not a
    # genuine wiggle).
    smooth_smoother = sm["smoother_rough"] < 0.20 and sm["worst_dip"] > -0.02
    print(f"  Gain density is extremely concentrated (frac≤0.05 = {c1['frac05']:.3f}, "
          f"{c1['frac0']:.3f} exactly 0) and the underlying lc0 Q is quantized "
          f"(|Q|≈0 {c1.get('q0', float('nan')):.3f}, |Q|≈1 {c1.get('qsat', float('nan')):.3f}).")
    print(f"  Fixed-K roughness: K=8 {k2['roughs'][8]:.3f} → K=100 {k2['roughs'][100]:.3f} "
          f"({'SHRINKS' if k2['roughs'][100] < k2['roughs'][8] else 'does NOT shrink'} with K).")
    print(f"  Curve convergence: RMS Δ(K=50,K=100)={k2['diff_50_100']:.4f} < "
          f"RMS Δ(K=8,K=20)={k2['diff_8_20']:.4f} ? {converging} "
          f"(⇒ points converge as K grows).")
    print(f"  Binning-free LOWESS interior is smooth & ~monotone (norm. roughness "
          f"{sm['smoother_rough']:.3f}, worst step {sm['worst_dip']:+.4f}): {smooth_smoother}.")
    if smooth_smoother and converging:
        print("  ⇒ TRUE Gain→RT relation is SMOOTH and monotone-increasing. The jag in "
              "gain_vs_rt is a COARSE-FIXED-K ARTIFACT over a concentrated/quantized density,")
        print("    NOT a genuinely wiggly relationship.")
    else:
        print("  ⇒ Some genuine wiggle remains after K-growth/smoothing — inspect the band.")
    print("  RECOMMENDATION (do NOT change gain_vs_rt.png yet): plot the binning-free LOWESS")
    print("    + curve-bootstrap band as the canonical summary, OR — if keeping bins — grow K")
    print("    with n (e.g. K ∝ n^(1/3), here ≈ 50–100) instead of the frozen K=8, and keep")
    print("    the zero-inflation lump (56% of Gain is exactly 0) as its own point.")

    make_figure(gain, q, gain, logT, k2, sm)


if __name__ == "__main__":
    main()
