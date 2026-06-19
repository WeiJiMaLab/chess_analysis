"""Confidence-interval helpers for the binned RT figures (diagnose_gain.md Part 2, [Q1]).

The published figures (gain/mq/gss/actiongap_vs_rt) draw per-bin error bars as the
**analytic normal SEM** on the displayed mean: ``1.96 · stddev(y) / sqrt(n)``. The Part-2
audit validated that, on the FIXED Gain data, this analytic SEM matches a percentile
bootstrap (ratio ≈ 1) — so we KEEP the analytic SEM rather than switch to bootstrap, but add
two guards so a future contaminated/degenerate bin can't pass silently as a confidently tight
band:

  * ``min_n`` guard — a bin with too few rows has an unreliable SEM (and a Gaussian-SEM
    assumption that hasn't kicked in); flag it.
  * point-mass warning — if one y-value dominates the bin (the exact failure of the old
    Gain≈1.0 artifact: a degenerate 0.0-lookup mass that made n huge and SEM tiny around the
    wrong mean), flag it; its low within-bin variance makes the SEM tight for a non-cognitive
    reason.

``analytic_ci_with_guards`` returns the same half-width as the figures plus the guard flags.
``bootstrap_ci_half`` is the percentile-bootstrap check used by the audit and tests. All
operate on the SAME transformed quantity the panel displays (e.g. mean logT), per [Q2].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054  # 1.96-ish; two-sided 95% normal quantile


def analytic_ci_half(y, z: float = Z95) -> float:
    """Analytic normal SEM half-width on the mean of ``y`` (== the figures' error bar):
    ``z · stddev(y) / sqrt(n)``. Uses sample stddev (ddof=1), matching DuckDB ``stddev``.
    Returns 0.0 for n<2 (degenerate)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        return 0.0
    return float(z * y.std(ddof=1) / np.sqrt(n))


def bootstrap_ci_half(y, n_boot: int = 2000, ci: float = 0.95, rng=None):
    """Percentile-bootstrap CI of the MEAN of ``y`` (the displayed mean, per [Q2]).

    Returns (half_width, lo, hi). ``half_width`` = (hi − lo) / 2 so it is directly comparable
    to ``analytic_ci_half``. Resamples ``y`` with replacement ``n_boot`` times."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        m = float(y.mean()) if n else float("nan")
        return 0.0, m, m
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.integers(0, n, size=(n_boot, n))
    means = y[idx].mean(axis=1)
    a = (1 - ci) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float((hi - lo) / 2), float(lo), float(hi)


def pointmass_score(y) -> float:
    """Fraction of the bin held by its single most-frequent value (max single-value
    frequency). 1.0 ⇒ a perfect point-mass (zero within-bin variance); the signature of the
    old Gain≈1.0 0.0-lookup contamination. Empty ⇒ 0.0."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    # round lightly so float ties (e.g. an exact 0.0 lump) collapse to one value
    vals, counts = np.unique(np.round(y, 10), return_counts=True)
    return float(counts.max() / y.size)


def bin_contamination(y) -> dict:
    """Per-bin contamination triple used by the audit and the plotting guard:
    ``n``, within-bin ``variance``, and ``pointmass_score``."""
    y = np.asarray(y, dtype=float)
    return {
        "n": int(y.size),
        "variance": float(y.var(ddof=1)) if y.size >= 2 else 0.0,
        "pointmass_score": pointmass_score(y),
    }


@dataclass
class GuardedCI:
    half: float          # analytic SEM half-width (the plotted error bar)
    n: int
    pointmass_score: float
    low_n: bool          # n < min_n guard tripped
    pointmass: bool      # one value dominates > pointmass_threshold of the bin
    @property
    def warn(self) -> bool:
        return self.low_n or self.pointmass


def analytic_ci_with_guards(
    y,
    min_n: int = 100,
    pointmass_threshold: float = 0.5,
    z: float = Z95,
) -> GuardedCI:
    """Analytic-with-guards CI ([Q1] resolution): the analytic SEM half-width PLUS guard
    flags. Keeps the cheap, validated analytic error bar but surfaces the two ways a tight
    band can be spurious — too few rows (``min_n``) or a dominating point-mass
    (``pointmass_threshold``). Does NOT alter the half-width; callers decide whether to widen,
    annotate, or drop a flagged bin.

    Default ``min_n=100`` matches the figures' ``min_bin_count``."""
    y = np.asarray(y, dtype=float)
    n = y.size
    half = analytic_ci_half(y, z=z)
    pm = pointmass_score(y)
    return GuardedCI(
        half=half,
        n=n,
        pointmass_score=pm,
        low_n=(n < min_n),
        pointmass=(pm > pointmass_threshold),
    )
