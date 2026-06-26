"""
Integration tests for saved-tree engine analyses.
Loads a real saved Leela Chess Zero search tree from the configured directory
and verifies that tree-based VOC (Gain) and tree-based MQ are correctly computed.
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
        
        # If the tree is large enough to contain at least 2 children, these should be valid floats
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
            # The best move must have MQ exactly 0, and all other moves must have MQ <= 0
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


if __name__ == "__main__":
    unittest.main()
