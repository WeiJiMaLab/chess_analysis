"""Tests for the analytic-with-guards CI helpers (utils/ci.py), the [Q1] resolution for the
binned RT figures (diagnose_gain.md Part 2 §3).

We KEEP the analytic SEM (validated against bootstrap) but add a min-n guard and a point-mass
warning so a future contaminated bin (the old Gain≈1.0 0.0-lookup signature: huge n, ~zero
within-bin variance, tight SEM around a wrong mean) can't pass silently.

All synthetic — no real-tree / 100k load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ci import (  # noqa: E402
    analytic_ci_half,
    bootstrap_ci_half,
    pointmass_score,
    bin_contamination,
    analytic_ci_with_guards,
)


def test_analytic_vs_bootstrap_agree():
    """On synthetic Gaussian bins, the analytic 1.96·std/√n half-width and the percentile
    bootstrap half-width agree within ~5% for n ≥ 500. This is the head-to-head the audit
    runs on real bins — pinned here so the estimator equivalence is guaranteed."""
    rng = np.random.default_rng(7)
    for n in (500, 2000, 10000):
        for mu, sd in [(0.0, 1.0), (3.0, 0.5), (-2.0, 2.0)]:
            y = rng.normal(mu, sd, size=n)
            a = analytic_ci_half(y)
            b, _, _ = bootstrap_ci_half(y, n_boot=3000, rng=rng)
            assert b / a == pytest.approx(1.0, abs=0.05), \
                f"n={n} mu={mu} sd={sd}: ratio {b/a:.3f} off by >5%"


def test_ci_shrinks_with_n():
    """The 'size-invariance' intuition made concrete: per-sample variability is fixed, but
    the MEAN's CI half-width shrinks ∝ 1/√n. Quadrupling n halves the half-width."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, size=4_000_000)
    h1 = analytic_ci_half(base[:10_000])
    h4 = analytic_ci_half(base[:40_000])    # 4x n
    h16 = analytic_ci_half(base[:160_000])  # 16x n
    assert h4 / h1 == pytest.approx(0.5, abs=0.03)
    assert h16 / h1 == pytest.approx(0.25, abs=0.02)
    # bootstrap shows the same scaling
    b1, _, _ = bootstrap_ci_half(base[:10_000], n_boot=2000, rng=rng)
    b4, _, _ = bootstrap_ci_half(base[:40_000], n_boot=2000, rng=rng)
    assert b4 / b1 == pytest.approx(0.5, abs=0.06)


def test_pointmass_bin_flagged():
    """A bin that is 90% a single value triggers the point-mass guard; a clean Gaussian bin
    does not. This is the exact contamination the old Gain≈1.0 mass produced."""
    rng = np.random.default_rng(2)
    n = 1000
    contaminated = np.concatenate([np.zeros(900), rng.normal(0, 1, 100)])
    clean = rng.normal(0, 1, n)

    assert pointmass_score(contaminated) == pytest.approx(0.9, abs=1e-9)
    assert pointmass_score(clean) < 0.05

    g_bad = analytic_ci_with_guards(contaminated, min_n=100, pointmass_threshold=0.5)
    g_ok = analytic_ci_with_guards(clean, min_n=100, pointmass_threshold=0.5)
    assert g_bad.pointmass and g_bad.warn
    assert not g_ok.pointmass and not g_ok.warn


def test_min_n_guard():
    """A bin with n below min_n is flagged low_n (its SEM is unreliable) regardless of shape."""
    rng = np.random.default_rng(3)
    small = rng.normal(0, 1, size=20)
    big = rng.normal(0, 1, size=500)
    assert analytic_ci_with_guards(small, min_n=100).low_n
    assert not analytic_ci_with_guards(big, min_n=100).low_n


def test_bin_contamination_triple():
    """bin_contamination returns n, within-bin variance, and point-mass score consistently."""
    rng = np.random.default_rng(4)
    y = rng.normal(5, 2, size=2000)
    c = bin_contamination(y)
    assert c["n"] == 2000
    assert c["variance"] == pytest.approx(np.var(y, ddof=1), rel=1e-9)
    assert c["pointmass_score"] < 0.02
    # degenerate (all one value): zero variance, point-mass 1.0
    z = bin_contamination(np.full(50, 3.0))
    assert z["variance"] == 0.0 and z["pointmass_score"] == 1.0


def test_degenerate_inputs():
    """n<2 inputs return 0 half-width, not a crash / NaN propagation issue."""
    assert analytic_ci_half([]) == 0.0
    assert analytic_ci_half([1.0]) == 0.0
    h, lo, hi = bootstrap_ci_half([1.0], n_boot=100)
    assert h == 0.0 and lo == hi == 1.0
    assert pointmass_score([]) == 0.0
