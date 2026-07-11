"""All CIs are percentile bootstrap, never normal-theory."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(fn, *cols, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile-bootstrap CI for a statistic of paired columns.

    Returns ``(point, lo, hi)`` where ``point = fn(*cols)`` and ``lo/hi`` are the
    ``alpha/2`` / ``1-alpha/2`` percentiles over ``n_boot`` paired resamples (one shared
    index per resample, so all columns stay aligned). RNG stream matches the former
    ``_boot`` copies exactly.
    """
    rng = np.random.default_rng(seed)
    n = len(cols[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)              # one paired resample shared across columns
        vals[b] = fn(*[c[idx] for c in cols])
    return (float(fn(*cols)),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
