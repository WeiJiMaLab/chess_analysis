"""All CIs are percentile bootstrap, never normal-theory."""
from __future__ import annotations

import numpy as np


def spearman(a, b) -> float:
    """Spearman rank correlation of two paired sequences."""
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def partial_spearman(y, x, z) -> float:
    """Spearman correlation of ``x`` and ``y`` with ``z`` linearly partialled out (on ranks)."""
    from scipy.stats import rankdata
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    rz1 = np.c_[np.ones_like(rz), rz]
    ex = rx - rz1 @ np.linalg.lstsq(rz1, rx, rcond=None)[0]
    ey = ry - rz1 @ np.linalg.lstsq(rz1, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


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


def bootstrap_mean_ci(values, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile-bootstrap ``(lo, hi)`` CI on the **mean** of a 1-D list of scalars.

    Torch-based, bit-identical to the former ``analysis.evaluate._bootstrap_ci``
    (used for mean-regret CIs in the controller dashboard).
    """
    import torch
    if not len(values):
        return (float("nan"), float("nan"))
    t = torch.tensor(values, dtype=torch.float64)
    n = t.numel()
    gen = torch.Generator().manual_seed(seed)
    boot, done = [], 0
    while done < n_boot:
        b = min(256, n_boot - done)
        boot.append(t[torch.randint(0, n, (b, n), generator=gen)].mean(dim=1))
        done += b
    boot = torch.cat(boot)
    return (float(torch.quantile(boot, alpha / 2)), float(torch.quantile(boot, 1 - alpha / 2)))
