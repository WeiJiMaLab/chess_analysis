"""TDD-lite sanity tests for ``cts.analysis.stats_ablation`` (plan.md S1, 2026-07-08).

A couple of well-chosen checks on the new pure per-step feature builder --
this is a diagnostic-script helper, not production training code, so we don't
re-test the reused ``analysis.evaluate`` fitting machinery here (already
covered indirectly by that module's own usage in the real eval pipeline).
"""
from __future__ import annotations

import numpy as np
import pytest

from cts.analysis.stats_ablation import steps_subset_tensor, _ALL_STATS


def _toy_episode():
    return {
        "halt_rewards": [0.1, 0.2, 0.3],
        "tree_sizes": [3, 5, 9],
        "heights": [1, 2, 2],
        "widths": [1, 2, 4],
    }


def test_steps_subset_tensor_all_stats_shape_and_values():
    ep = _toy_episode()
    t = steps_subset_tensor(ep, _ALL_STATS)
    assert t.shape == (3, 1 + len(_ALL_STATS))  # steps_taken + n_nodes + height + width + current_value
    # column 0 is steps_taken (0-indexed within the episode)
    assert t[:, 0].tolist() == [0.0, 1.0, 2.0]
    # remaining columns follow _ALL_STATS order: n_nodes, height, width, current_value
    assert t[:, 1].tolist() == ep["tree_sizes"]
    assert t[:, 2].tolist() == ep["heights"]
    assert t[:, 3].tolist() == ep["widths"]
    assert t[:, 4].tolist() == pytest.approx(ep["halt_rewards"], abs=1e-6)


def test_steps_subset_tensor_empty_stats_is_steps_only():
    ep = _toy_episode()
    t = steps_subset_tensor(ep, ())
    assert t.shape == (3, 1)
    assert t[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_steps_subset_tensor_single_stat_drop():
    ep = _toy_episode()
    subset = tuple(s for s in _ALL_STATS if s != "height")
    t = steps_subset_tensor(ep, subset)
    assert t.shape == (3, 1 + len(subset))
    # "height" excluded -- remaining stats appear in _ALL_STATS order: n_nodes, width, current_value
    assert t[:, 1].tolist() == ep["tree_sizes"]
    assert t[:, 2].tolist() == ep["widths"]
    assert t[:, 3].tolist() == pytest.approx(ep["halt_rewards"], abs=1e-6)
