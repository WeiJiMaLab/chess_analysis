"""Sanity tests for the E2 stats-decodability probe (plan.md).

TDD-lite per plan.md's "Operating principles": a couple of well-chosen sanity checks
on the REUSED probe primitives (``_linear_r2`` / ``_mlp_r2``, ``cts.analysis.zt_probe``),
not an exhaustive suite. These are the same functions
``cts.analysis.zt_stats_decodability.compute_stats_decodability`` calls for the real
{n_nodes, height, width} vs z_t probe — real packed/materialized data isn't available in
a unit-test fixture, so we validate the probes directly against synthetic data:

  1. target == feature (the "trivial upper bound" case in the real probe) -> R^2 ~= 1.
  2. feature is independent random noise (the "shuffle floor" case) -> R^2 ~= 0.
"""
from __future__ import annotations

import numpy as np

from cts.analysis.zt_probe import _linear_r2, _mlp_r2


def _split(n: int, seed: int = 0, train_frac: float = 0.7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_frac * n))
    return idx[:n_train], idx[n_train:]


def test_linear_r2_self_prediction_near_one():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.normal(size=n) * 10 + 5
    z = y.reshape(-1, 1)  # feature == target exactly
    tr, te = _split(n)
    r2 = _linear_r2(z[tr], y[tr], z[te], y[te])
    assert r2 > 0.999, f"expected near-perfect self-prediction R^2, got {r2}"


def test_mlp_r2_self_prediction_near_one():
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.normal(size=n) * 10 + 5
    z = y.reshape(-1, 1)
    tr, te = _split(n)
    r2 = _mlp_r2(z[tr], y[tr], z[te], y[te], seed=0)
    assert r2 > 0.95, f"expected near-perfect self-prediction R^2, got {r2}"


def test_linear_r2_random_noise_near_zero():
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.normal(size=n) * 10 + 5
    z = rng.normal(size=(n, 32))  # independent of y, matches d_embed=32 shape
    tr, te = _split(n)
    r2 = _linear_r2(z[tr], y[tr], z[te], y[te])
    assert abs(r2) < 0.05, f"expected near-zero R^2 for independent noise, got {r2}"


def test_linear_r2_shuffled_feature_destroys_correspondence():
    """Row-shuffling a genuinely predictive feature (the real probe's shuffle-floor
    construction) should collapse a near-perfect R^2 down toward the noise floor."""
    rng = np.random.default_rng(4)
    n = 2000
    y = rng.normal(size=n) * 10 + 5
    z = y.reshape(-1, 1)
    z_shuf = z[rng.permutation(n)]
    tr, te = _split(n)
    r2_aligned = _linear_r2(z[tr], y[tr], z[te], y[te])
    r2_shuffled = _linear_r2(z_shuf[tr], y[tr], z_shuf[te], y[te])
    assert r2_aligned > 0.999
    assert abs(r2_shuffled) < 0.05, f"expected shuffled feature to destroy R^2, got {r2_shuffled}"
