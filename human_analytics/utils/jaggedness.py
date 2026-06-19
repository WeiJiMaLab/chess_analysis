"""Jaggedness-diagnosis helpers (diagnose_gain.md Part 2 extension).

The puzzle: in ``gain_vs_rt.png`` / ``gss_vs_rt.png`` the binned mean-RT curve is visibly
JAGGED bin-to-bin, yet the analytic per-bin SEM bars are razor-thin. This module provides the
estimators/bootstraps needed to resolve why thin bars coexist with jaggedness:

  * ``curve_bootstrap`` — the curve-level (shape) bootstrap: resample the moves with
    replacement, RE-COMPUTE the quantile edges, RE-BIN, recompute every bin's mean. The
    per-bin SEM is CONDITIONAL on bin membership (fixed edges, fixed assignment); this honest
    band also carries edge-jitter + reassignment uncertainty.
  * ``adjacent_jumps`` / ``roughness`` — quantify bin-to-bin jaggedness and how it scales
    with bin count K.
  * ``quantile_bin_means`` / ``equal_width_bin_means`` / ``integer_bin_means`` — the three
    binnings, returning per-bin (center_x, mean_y, sem_y, n).
  * centers: ``arith_mean`` (mean RT), ``geo_center`` (mean logT == geometric-mean RT) and
    ``median`` — to test whether a robust center smooths a heavy-tailed RT.

All functions operate on plain numpy arrays (no DB / big-file dependency) so they are unit
testable on tiny synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- centers
def arith_mean(y: np.ndarray) -> float:
    return float(np.mean(y))


def geo_center(y: np.ndarray) -> float:
    """Mean of log y == log of the geometric mean. Expects y already in the displayed
    transform (e.g. logT). For raw RT pass ``np.log(rt)``."""
    return float(np.mean(y))


def median_center(y: np.ndarray) -> float:
    return float(np.median(y))


def _sem(y: np.ndarray) -> float:
    n = y.size
    if n < 2:
        return 0.0
    return float(Z95 * y.std(ddof=1) / np.sqrt(n))


# --------------------------------------------------------------------------- binnings
@dataclass
class BinCurve:
    centers_x: np.ndarray   # mean x per kept bin
    means_y: np.ndarray     # center statistic of y per bin
    sems_y: np.ndarray      # analytic SEM half-width per bin (on the mean)
    counts: np.ndarray      # n per bin
    edges: np.ndarray       # the x cut-points used (interior edges)


def _assign(x, edges):
    """Bin index = number of interior edges strictly exceeded by x (matches the figures'
    tie-safe ``list_filter(edges, e -> x > e)``). ``searchsorted(..., side='left')`` counts
    edges < x; for strict ``x > e`` with float data ties are negligible and this matches."""
    if edges.size == 0:
        return np.zeros(x.size, dtype=np.intp)
    return np.searchsorted(edges, x, side="left").astype(np.intp)


def quantile_bin_means(x, y, k: int, min_n: int = 1, center="mean") -> BinCurve:
    """Equal-COUNT (quantile) bins: k bins via the (k-1) interior quantile edges. Bin index =
    #edges exceeded (matches the figures' tie-safe ``list_filter`` assignment)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    qs = [i / k for i in range(1, k)]
    edges = np.quantile(x, qs) if k > 1 else np.array([])
    idx = _assign(x, edges)
    return _assemble(x, y, idx, k, edges, min_n, center)


def equal_width_bin_means(x, y, k: int, min_n: int = 1, center="mean") -> BinCurve:
    """Equal-WIDTH bins over [min(x), max(x)]."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    lo, hi = x.min(), x.max()
    edges = np.linspace(lo, hi, k + 1)[1:-1]
    idx = _assign(x, edges)
    return _assemble(x, y, idx, k, edges, min_n, center)


def integer_bin_means(x, y, min_n: int = 1, center="mean") -> BinCurve:
    """Native integer binning: one point per distinct integer value of x (e.g. GSS)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xi = np.round(x).astype(int)
    vals = np.unique(xi)
    cx, my, se, cn = [], [], [], []
    for v in vals:
        m = xi == v
        if m.sum() < min_n:
            continue
        cx.append(float(v)); my.append(_center(y[m], center)); se.append(_sem(y[m])); cn.append(int(m.sum()))
    return BinCurve(np.array(cx), np.array(my), np.array(se), np.array(cn), vals.astype(float))


def _center(y, center):
    if center == "mean":
        return float(np.mean(y))
    if center == "median":
        return float(np.median(y))
    raise ValueError(center)


def _assemble(x, y, idx, k, edges, min_n, center) -> BinCurve:
    cx, my, se, cn = [], [], [], []
    for b in range(k):
        m = idx == b
        if m.sum() < min_n:
            continue
        cx.append(float(x[m].mean())); my.append(_center(y[m], center))
        se.append(_sem(y[m])); cn.append(int(m.sum()))
    order = np.argsort(cx)
    cx = np.array(cx)[order]; my = np.array(my)[order]
    se = np.array(se)[order]; cn = np.array(cn)[order]
    return BinCurve(cx, my, se, cn, edges)


# --------------------------------------------------------------------------- roughness
def adjacent_jumps(curve: BinCurve) -> np.ndarray:
    """|Δ mean_y| between adjacent (sorted-by-x) bins."""
    return np.abs(np.diff(curve.means_y))


def roughness(curve: BinCurve) -> float:
    """RMS of adjacent jumps — a scalar jaggedness metric, comparable across K."""
    j = adjacent_jumps(curve)
    return float(np.sqrt(np.mean(j ** 2))) if j.size else 0.0


def frac_jumps_exceeding(curve: BinCurve, half: np.ndarray) -> float:
    """Fraction of adjacent-bin jumps that EXCEED the combined band of the two bins, where
    ``half`` is the per-bin half-width to compare against (per-bin SEM, or curve-bootstrap
    band half-width). Combined band = sqrt(h_i^2 + h_{i+1}^2)."""
    j = adjacent_jumps(curve)
    if j.size == 0:
        return 0.0
    comb = np.sqrt(half[:-1] ** 2 + half[1:] ** 2)
    with np.errstate(invalid="ignore"):
        return float(np.mean(j > comb))


# --------------------------------------------------------------------------- curve bootstrap
def curve_bootstrap(x, y, k: int, binning="quantile", center="mean",
                    n_boot: int = 1000, min_n: int = 1, rng=None):
    """Curve-level (shape) bootstrap. Resample (x, y) PAIRS with replacement, RE-COMPUTE the
    bin edges on the resample, RE-BIN, recompute each bin's center. Returns per-bin
    (lo, hi, half) of the center across resamples, aligned to a FIXED reference grid of bin
    indices computed on the full data (so band[i] corresponds to reference bin i).

    The key contrast with the per-bin analytic SEM: that SEM holds the edges and the point→bin
    assignment FIXED; this band lets the edges jitter and points reassign — the honest
    shape-uncertainty of the whole curve.

    Returns dict with: ref (BinCurve on full data), boot_means (n_boot x n_ref array, NaN where
    a bin was empty in a resample), lo, hi, half (per ref bin), band_half (alias of half)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.default_rng() if rng is None else rng
    n = x.size

    if binning == "integer":
        # Native-integer bootstrap, vectorized by bincount over a fixed reference value grid.
        ref = integer_bin_means(x, y, min_n=min_n, center=center)
        ref_vals = np.round(ref.centers_x).astype(int)
        vmin = int(np.round(x).min())
        offset = vmin
        width = int(np.round(x).max()) - vmin + 1
        xi = (np.round(x).astype(int) - offset)
        # map each reference value to its slot in the bincount grid
        ref_slots = ref_vals - offset
        boot = np.full((n_boot, ref_vals.size), np.nan)
        for b in range(n_boot):
            s = rng.integers(0, n, n)
            ib = xi[s]
            sums = np.bincount(ib, weights=y[s], minlength=width)
            cnts = np.bincount(ib, minlength=width)
            with np.errstate(invalid="ignore", divide="ignore"):
                means_full = sums / cnts
            boot[b] = means_full[ref_slots]   # NaN where that value was absent in resample
        return _band(ref, boot)
    elif binning == "quantile":
        ref = quantile_bin_means(x, y, k, min_n=min_n, center=center)
    elif binning == "equal_width":
        ref = equal_width_bin_means(x, y, k, min_n=min_n, center=center)
    else:
        raise ValueError(binning)

    nref = ref.means_y.size
    boot = np.full((n_boot, nref), np.nan)
    eqwidth = (binning == "equal_width")
    if eqwidth:
        lo, hi = x.min(), x.max()
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        xb = x[s]; yb = y[s]
        # RE-COMPUTE edges on the resample (the crux of the shape bootstrap)
        if eqwidth:
            edges = np.linspace(lo, hi, k + 1)[1:-1]
        else:
            edges = np.quantile(xb, [i / k for i in range(1, k)]) if k > 1 else np.array([])
        idx = _assign(xb, edges)
        sums = np.bincount(idx, weights=yb, minlength=k)
        cnts = np.bincount(idx, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            means_full = sums / cnts            # k-length, NaN where empty
        # keep only bins that meet min_n in the resample; align by sorted bin index (0..k-1
        # is already x-sorted because edges are increasing). Reference kept the same bins.
        keep = cnts >= 1
        m = min(keep.sum(), nref)
        kept_means = means_full[keep]
        boot[b, :m] = kept_means[:m]
    return _band(ref, boot)


def _band(ref: BinCurve, boot: np.ndarray, ci: float = 0.95) -> dict:
    a = (1 - ci) / 2
    lo = np.nanquantile(boot, a, axis=0)
    hi = np.nanquantile(boot, 1 - a, axis=0)
    half = (hi - lo) / 2
    return dict(ref=ref, boot_means=boot, lo=lo, hi=hi, half=half, band_half=half)


# --------------------------------------------------------------------------- smoother
def lowess_curve(x, y, grid, frac: float = 0.3, it: int = 0, delta: float = 0.0) -> np.ndarray:
    """Binning-FREE LOWESS smoother (statsmodels) of y on x, evaluated at ``grid``.

    Unlike fixed-K quantile binning — whose resolution is frozen by K and does NOT
    converge to the regression function as n→∞ — a bandwidth smoother's resolution is set
    by ``frac`` (a fraction of the data per local fit) and DOES converge as n grows. We
    delegate to ``statsmodels.lowess`` (robust, tricube-weighted local linear) and then
    interpolate its fit onto a fixed ``grid`` so the curve and its bootstrap band share an
    x-axis. ``it=0`` = no robustifying iterations (plain local linear); raise for outlier
    resistance. Returns y_hat at each grid point (linear interpolation/extrapolation off the
    fitted support)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)
    # tied x-values (e.g. a value point-mass) can make a local window's weights sum to ~0,
    # which statsmodels handles but warns about; the result is finite after our unique-collapse.
    with np.errstate(invalid="ignore", divide="ignore"):
        # ``delta>0`` (statsmodels' large-n speedup: do full local fits only at points
        # spaced >delta apart, linearly interpolate between) — pass 0.01*range for big n.
        fitted = lowess(y, x, frac=frac, it=it, delta=delta, return_sorted=True)
    fx, fy = fitted[:, 0], fitted[:, 1]
    # collapse duplicate x (lowess returns one row per input x; ties → average) so np.interp
    # has a strictly increasing support.
    ux, inv = np.unique(fx, return_inverse=True)
    uy = np.zeros_like(ux)
    np.add.at(uy, inv, fy)
    uy /= np.bincount(inv)
    return np.interp(grid, ux, uy)


def lowess_bootstrap(x, y, grid, frac: float = 0.3, it: int = 0,
                     n_boot: int = 200, ci: float = 0.95, rng=None, delta: float = 0.0) -> dict:
    """Curve-level (shape) bootstrap band for ``lowess_curve``. Resample (x, y) PAIRS with
    replacement, RE-FIT the LOWESS each time, evaluate on the SAME fixed ``grid``, and take
    per-grid-point percentiles across resamples — the binning-free analogue of
    ``curve_bootstrap``. Returns dict(grid, fit (full-data LOWESS), lo, hi, half, boot
    (n_boot×len(grid))). The band answers "is the SMOOTHER itself smooth, and how uncertain
    is its shape?" without any K to freeze the resolution."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)
    rng = np.random.default_rng() if rng is None else rng
    n = x.size
    fit = lowess_curve(x, y, grid, frac=frac, it=it, delta=delta)
    boot = np.full((n_boot, grid.size), np.nan)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        boot[b] = lowess_curve(x[s], y[s], grid, frac=frac, it=it, delta=delta)
    a = (1 - ci) / 2
    lo = np.nanquantile(boot, a, axis=0)
    hi = np.nanquantile(boot, 1 - a, axis=0)
    half = (hi - lo) / 2
    return dict(grid=grid, fit=fit, lo=lo, hi=hi, half=half, boot=boot)
