"""
Tests for engine-level analyses in lmcos_small/human.
Consolidates integration tests on real saved trees and mathematical unit tests on mock trees.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import pytest

_HA = Path(__file__).resolve().parent.parent
if str(_HA) not in sys.path:
    sys.path.insert(0, str(_HA))

from utils.helpers import CONFIG
from utils.tree_loader import _tree_voc_and_gap, _tree_root_mq, _tree_gss, _tree_hpi

# Check if the saved trees directory exists and contains any .pt files
TREES_DIR = CONFIG["trees_default"]
HAS_SAVED_TREES = os.path.exists(TREES_DIR) and any(
    f.endswith(".pt") for f in os.listdir(TREES_DIR)
) if os.path.exists(TREES_DIR) else False


@unittest.skipUnless(
    HAS_SAVED_TREES,
    f"No saved Leela search trees (.pt) found in {TREES_DIR}",
)
class TestSavedTreeAnalysis(unittest.TestCase):
    """Integration tests based on real saved Leela MCTS search trees."""

    def setUp(self) -> None:
        # Load the first available saved tree
        self.tree_files = [
            os.path.join(TREES_DIR, f)
            for f in os.listdir(TREES_DIR)
            if f.endswith(".pt")
        ]
        self.assertTrue(len(self.tree_files) > 0)
        self.payload = torch.load(self.tree_files[0], map_location="cpu", weights_only=False)

    def test_tree_voc_and_gap(self) -> None:
        """Verify that tree-based VOC (Gain) and Action Gap are valid float/NaN values."""
        voc_val, gap = _tree_voc_and_gap(self.payload)
        
        if not (self.payload["parent_index"].numpy() == 0).sum() < 2:
            self.assertFalse(np.isnan(gap), "Action Gap should not be NaN for a valid tree")
            self.assertFalse(np.isnan(voc_val), "Tree-based VOC should not be NaN for a valid tree")
            self.assertGreaterEqual(voc_val, -1e-9, "Tree-based VOC must be non-negative")
            self.assertGreaterEqual(gap, -1e-9, "Action Gap must be non-negative")

    def test_tree_root_mq(self) -> None:
        """Verify that tree-based root MQ is correctly extracted and behaves as expected."""
        moves, mqs = _tree_root_mq(self.payload)
        
        self.assertEqual(len(moves), len(mqs))
        if moves:
            self.assertAlmostEqual(max(mqs), 0.0, places=6)
            for mq in mqs:
                self.assertLessEqual(mq, 1e-9, "Move Quality must be non-positive")

    def test_tree_gss(self) -> None:
        """Verify that Greedy Stopping Step is non-negative."""
        gss = _tree_gss(self.payload)
        self.assertGreaterEqual(gss, 0)

    def test_tree_hpi(self) -> None:
        """Verify that prior policy entropy H(pi) is valid and non-negative."""
        hpi = _tree_hpi(self.payload)
        if not np.isnan(hpi):
            self.assertGreaterEqual(hpi, 0.0)


# --- Mathematical Unit Tests on Mock Trees (Gain/VOC) ---
def _make_tree(*, child_values, final_q, q_trace, root_moves=None):
    """Minimal payload dict matching what _tree_voc_and_gap reads."""
    n_kids = len(child_values)
    node_features = torch.tensor([[0.0]] + [[v] for v in child_values], dtype=torch.float64)
    parent_index = torch.tensor([-1] + [0] * n_kids, dtype=torch.long)
    if root_moves is None:
        root_moves = [f"m{i}" for i in range(len(final_q))]
    return {
        "feature_names": ["value"],
        "node_features": node_features,
        "parent_index": parent_index,
        "oracle_final_root_q_values": np.asarray(final_q, dtype=float),
        "oracle_root_q_trace": np.asarray(q_trace, dtype=float),
        "oracle_root_moves": list(root_moves),
        "incoming_moves": {i + 1: f"m{i}" for i in range(n_kids)},
    }


def test_gain_shallow_is_visited():
    """Verify that Gain picks a_shallow from the visited moves in the q-trace."""
    child_values = [-1.0, -0.2, -0.9]
    final_q = [1.0, 0.1, 0.0]
    q_trace = [
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
    ]
    payload = _make_tree(child_values=child_values, final_q=final_q, q_trace=q_trace)
    voc, _ = _tree_voc_and_gap(payload)
    assert voc == pytest.approx(0.0, abs=1e-9)
    assert voc < 0.95


def test_gain_nonnegative():
    """Gain >= 0 for any well-formed growing tree."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = rng.integers(2, 8)
        final_q = rng.uniform(-1, 1, size=k)
        q_trace = np.zeros((k, k))
        order = rng.permutation(k)
        for step, mv in enumerate(order):
            q_trace[step:, mv] = final_q[mv] if final_q[mv] != 0 else 0.01
        child_values = (-final_q).tolist()
        payload = _make_tree(child_values=child_values, final_q=final_q.tolist(),
                             q_trace=q_trace.tolist())
        voc, _ = _tree_voc_and_gap(payload)
        assert not np.isnan(voc)
        assert voc >= -1e-9


def test_gain_matches_growing_tree_identity():
    """Gain == final_Q[argmax] - final_Q[first-expanded]."""
    final_q = [0.3, 0.8, -0.2, 0.5]
    q_trace = [
        [0.0, 0.0, -0.2, 0.0],
        [0.0, 0.8, -0.2, 0.0],
        [0.0, 0.8, -0.2, 0.5],
    ]
    child_values = [-v for v in final_q]
    payload = _make_tree(child_values=child_values, final_q=final_q, q_trace=q_trace)
    voc, _ = _tree_voc_and_gap(payload)
    a_deep = int(np.argmax(final_q))
    a_shallow = 2
    assert voc == pytest.approx(final_q[a_deep] - final_q[a_shallow], abs=1e-12)
    assert voc == pytest.approx(0.8 - (-0.2), abs=1e-12)


def test_gain_nan_on_degenerate_tree():
    """A root with < 2 children yields NaN, not a fabricated number."""
    payload = _make_tree(child_values=[-0.5], final_q=[0.5], q_trace=[[0.5]])
    voc, gap = _tree_voc_and_gap(payload)
    assert np.isnan(voc)
    assert np.isnan(gap)


if __name__ == "__main__":
    unittest.main()

