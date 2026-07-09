"""TDD-lite sanity tests for ``analysis.cp_recalibration`` (CP node, plan.md).

Two pure functions get direct synthetic coverage (``leaf_mask``, ``headroom_summary``);
the rest of the module is I/O-heavy (loads real ``.pt`` trees) and is exercised instead
by the actual sweep run (see ``cp_recalibration.md``), consistent with the TDD-lite
convention used by ``test_quiescence_pruning_sweep.py``.
"""
from __future__ import annotations

import numpy as np

from analysis.cp_recalibration import headroom_summary, leaf_mask


def test_leaf_mask_basic():
    # Same synthetic tree shape as test_relabel_replay's _synthetic_tree: 0->1,2,3; 1->4,5; 4->6,7.
    child_ptr = np.array([0, 3, 5, 5, 5, 7, 7, 7, 7], dtype=np.int64)
    mask = leaf_mask(child_ptr)
    # Nodes 2,3,5,6,7 are leaves (no children); 0,1,4 are expanded.
    assert mask.tolist() == [False, False, True, True, False, True, True, True]


def test_headroom_summary_singlehalt_beats_always_stop_on_toy_curves():
    """Toy curves where continuing strictly helps: SingleHalt* at its own fit-selected k
    should recover MORE than 0% of the AlwaysStop-relative achievable regret (sanity that
    the fraction isn't trivially 0 or negative on an unambiguous case)."""
    # Each curve rises then flattens -- stopping at k=0 is strictly worse than continuing.
    curves = [np.array([0.0, 0.3, 0.6, 0.6, 0.6]) for _ in range(20)]
    summary = headroom_summary(curves, k_singlehalt=2)
    assert summary["regret_singlehalt_mean"] == 0.0  # k=2 hits the max exactly on every curve
    assert summary["regret_always_stop_mean"] > 0.0
    assert summary["fraction_recovered_mean"] == 1.0


def test_headroom_summary_ci_bounds_bracket_point_estimate():
    rng = np.random.default_rng(0)
    curves = [np.cumsum(np.abs(rng.normal(size=6))) for _ in range(50)]
    summary = headroom_summary(curves, k_singlehalt=3)
    assert summary["regret_singlehalt_lo"] <= summary["regret_singlehalt_mean"] <= summary["regret_singlehalt_hi"]
    assert summary["fraction_recovered_lo"] <= summary["fraction_recovered_mean"] <= summary["fraction_recovered_hi"]
