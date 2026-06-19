"""Unit tests for the jaggedness estimators/bootstrap (utils/jaggedness.py).

Synthetic data only — no DB / parquet loads. Pins:
  * curve-level bootstrap band brackets a known clean curve and is WIDER than the per-bin SEM
    on a heavy-tailed / discreteness-stressed predictor;
  * native-integer binning recovers a clean monotone curve on an integer predictor where
    quantile binning is jagged;
  * a robust center (geometric mean / median) reduces roughness on heavy-tailed RT.
"""
import numpy as np

from utils.jaggedness import (
    quantile_bin_means, equal_width_bin_means, integer_bin_means,
    curve_bootstrap, roughness, adjacent_jumps, frac_jumps_exceeding,
    lowess_curve, lowess_bootstrap,
)


def test_curve_bootstrap_brackets_true_curve_and_widens_vs_sem():
    """On a noisy linear trend, the curve-level bootstrap band must (a) contain the per-bin
    means and (b) be >= the per-bin SEM (it carries extra edge/reassignment uncertainty)."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    y = 2.0 * x + rng.normal(0, 1.0, n)
    res = curve_bootstrap(x, y, 8, binning="quantile", n_boot=400, min_n=50, rng=rng)
    c = res["ref"]
    # band brackets the reference means by construction (means are inside their own band)
    assert np.all(c.means_y >= res["lo"] - 1e-9)
    assert np.all(c.means_y <= res["hi"] + 1e-9)
    # band half-width is at least ~ the per-bin SEM (allow small slack for MC noise)
    assert np.median(res["half"]) >= 0.9 * np.median(c.sems_y)


def test_integer_binning_cleaner_than_quantile_on_integer_predictor():
    """Integer predictor with a clean monotone y(value): native-integer binning yields a
    smooth monotone curve; quantile binning straddles integer sets and is rougher (per its
    own scale)."""
    rng = np.random.default_rng(1)
    # heavy mass at small values, monotone mean
    vals = rng.choice(np.arange(0, 10), size=8000,
                      p=np.array([5, 4, 3, 2, 1, 1, 1, 1, 1, 1]) / 20)
    y = 0.3 * vals + rng.normal(0, 0.3, vals.size)
    x = vals.astype(float)
    cn = integer_bin_means(x, y, min_n=20)
    cq = quantile_bin_means(x, y, 8, min_n=20)
    # native-integer curve is monotone increasing
    assert np.all(np.diff(cn.means_y) > -0.05)
    # normalized roughness: quantile >= native-integer
    nr = lambda c: roughness(c) / np.std(c.means_y)
    assert nr(cq) >= nr(cn)


def test_robust_center_reduces_roughness_on_heavy_tailed_rt():
    """Heavy-tailed RT (lognormal): arithmetic mean is jumpy bin-to-bin; geometric mean
    (mean of log) and median are smoother (lower normalized roughness)."""
    rng = np.random.default_rng(2)
    n = 6000
    x = rng.uniform(0, 1, n)
    logT = 0.5 * x + rng.normal(0, 1.0, n)   # smooth in log
    rt = np.exp(logT)                         # heavy-tailed in raw units
    ca = quantile_bin_means(x, rt, 12, min_n=30, center="mean")
    cg = quantile_bin_means(x, logT, 12, min_n=30, center="mean")
    cm = quantile_bin_means(x, rt, 12, min_n=30, center="median")
    # Geometric mean (mean logT) is smoother per its own scale.
    nr = lambda c: roughness(c) / np.std(c.means_y)
    assert nr(cg) <= nr(ca)
    # In the SAME raw-RT units, the median curve is less jumpy than the arithmetic mean,
    # because the heavy tail destabilizes the mean per bin (compare unnormalized roughness).
    assert roughness(cm) <= roughness(ca)


def test_frac_jumps_and_band_consistency():
    """frac_jumps_exceeding falls (weakly) when comparing against a wider band: a curve has
    fewer jumps exceeding the (wider) bootstrap band than the (narrower) per-bin SEM."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.uniform(0, 1, n)
    y = rng.normal(0, 1.0, n)  # pure noise → jagger than its bars
    res = curve_bootstrap(x, y, 20, binning="quantile", n_boot=300, min_n=20, rng=rng)
    c = res["ref"]
    f_sem = frac_jumps_exceeding(c, c.sems_y)
    f_band = frac_jumps_exceeding(c, res["half"])
    assert 0.0 <= f_band <= f_sem <= 1.0


def test_roughness_decreases_with_smoothing():
    """A constant curve has zero roughness; a noisy one is positive."""
    flat = quantile_bin_means(np.linspace(0, 1, 500), np.ones(500), 5, min_n=1)
    assert roughness(flat) == 0.0
    rng = np.random.default_rng(4)
    noisy = quantile_bin_means(rng.uniform(0, 1, 500), rng.normal(0, 1, 500), 5, min_n=1)
    assert roughness(noisy) > 0.0


def test_lowess_recovers_smooth_monotone_trend_on_concentrated_predictor():
    """The binning-free smoother must recover a SMOOTH, monotone-increasing regression
    function even on a concentrated/zero-inflated predictor (the Gain case): mass at 0 plus a
    thin positive tail, with logT rising in x. This is the estimator the Gain diagnostic uses
    to show the true relation is smooth (jag = fixed-K artifact)."""
    rng = np.random.default_rng(0)
    n = 6000
    # zero-inflated x: 55% exactly 0, the rest exponential — mirrors Gain's density.
    x = np.where(rng.random(n) < 0.55, 0.0, rng.exponential(0.12, n))
    y = 1.5 + 1.0 * x + rng.normal(0, 0.4, n)  # smooth monotone truth + noise
    interior = x > 0
    grid = np.linspace(np.quantile(x[interior], 0.05), np.quantile(x[interior], 0.95), 30)
    fit = lowess_curve(x[interior], y[interior], grid, frac=0.5)
    assert np.isfinite(fit).all()
    # monotone-increasing up to small sampling noise on the smoothed fit
    assert np.min(np.diff(fit)) > -0.02
    # and it tracks the truth (slope ~1): the fit should rise materially across the grid
    assert fit[-1] - fit[0] > 0.5 * (grid[-1] - grid[0]) * 0.5


def test_lowess_bootstrap_band_brackets_fit_and_is_tight_at_large_n():
    """The curve-level LOWESS bootstrap band must (a) bracket the full-data fit at every grid
    point and (b) be tight (small half-width) at large n — the binning-free analogue of the
    curve_bootstrap band used elsewhere."""
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.uniform(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.5, n)
    grid = np.linspace(0.1, 0.9, 25)
    res = lowess_bootstrap(x, y, grid, frac=0.4, n_boot=120, rng=rng)
    assert np.all(res["fit"] >= res["lo"] - 1e-9)
    assert np.all(res["fit"] <= res["hi"] + 1e-9)
    assert np.all(res["half"] >= 0.0)
    # at n=5000 with σ=0.5 the band half-width should be well under the curve's total rise (~2)
    assert np.median(res["half"]) < 0.2
