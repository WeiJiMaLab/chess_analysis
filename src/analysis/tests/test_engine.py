from __future__ import annotations

import os
import unittest

import numpy as np
import torch
import pytest

from analysis.utils.helpers import CONFIG
from analysis.utils.tree_loader import _tree_signals, _deep_values_prefix, compute_values, UNITS


def _make_tree(node_static, parents, *, feature="value", moves=None, extra_features=None):
    """Build a minimal payload dict for _tree_signals.

    node_static[i]  = the per-node static value (node 0 is the root; its value is
                      unused). Values are in the node's OWN side-to-move POV.
    parents[i]      = parent index (-1 for the root).
    extra_features  = optional {feature_name: [per-node values]} for multi-unit trees.
    """
    feats = {feature: node_static}
    if extra_features:
        feats.update(extra_features)
    names = list(feats)
    cols = np.array([feats[n] for n in names], dtype=np.float64).T  # [n_nodes, n_feat]
    n_kids = sum(1 for p in parents if p == 0)
    return {
        "feature_names": names,
        "node_features": torch.tensor(cols, dtype=torch.float64),
        "parent_index": torch.tensor(parents, dtype=torch.long),
        "oracle_root_moves": list(moves) if moves is not None else [f"m{i}" for i in range(n_kids)],
        "root_position_spec": "8/8/8/8/8/8/8/8 w - - 0 1",
    }


# --------------------------------------------------------------------------
# negamax deep-value replay
# --------------------------------------------------------------------------

def test_deep_values_negamax_leaf():
    """A flat tree: every node is a leaf, so deep value == static value."""
    static = np.array([0.0, 0.3, -0.5, 0.1])
    par = np.array([-1, 0, 0, 0])
    v = _deep_values_prefix(static, par, upto=3)
    assert v[1:] == pytest.approx([0.3, -0.5, 0.1])


def test_deep_values_negamax_one_level():
    """Parent takes -min(children): node1 has a child node3 (static 0.8), so
    v[1] = -0.8; the root's other child node2 (leaf) keeps its static value."""
    #        0(root)
    #       /       \
    #     1(0.9)    2(-0.3)
    #      |
    #     3(0.8)
    static = np.array([0.0, 0.9, -0.3, 0.8])
    par = np.array([-1, 0, 0, 1])
    v = _deep_values_prefix(static, par, upto=3)
    assert v[3] == pytest.approx(0.8)     # leaf
    assert v[1] == pytest.approx(-0.8)    # -min(child) = -0.8
    assert v[2] == pytest.approx(-0.3)    # leaf
    assert v[0] == pytest.approx(-min(-0.8, -0.3))  # root negamax = 0.8


# --------------------------------------------------------------------------
# signals on a flat (depth-1) tree — no search can change anything -> VOC == 0
# --------------------------------------------------------------------------

def test_flat_tree_signals():
    # root-mover myopic value of a child = -static[child]
    static = np.array([0.0, -0.5, 0.3, -0.1])   # myo = [0.5, -0.3, 0.1]
    par = np.array([-1, 0, 0, 0])
    sig = _tree_signals(_make_tree(static.tolist(), par.tolist()), "pwin")
    myo = np.array([0.5, -0.3, 0.1])
    assert sig["voc"] == pytest.approx(0.0, abs=1e-12)          # leaves: deep == myopic
    assert sig["action_gap"] == pytest.approx(0.5 - 0.1)       # top1 - top2
    # n_within_epsilon: COUNT within 0.1 of best (0.5) -> only the 0.5 move
    assert sig["n_within_epsilon"] == 1
    # n_acceptable: COUNT of myo >= 0 -> {0.5, 0.1} = 2
    assert sig["n_acceptable"] == 2
    assert sig["n_root_children"] == 3                          # legal-move denominator
    # mq: final deep values (== myo) minus best; best is 0
    assert max(sig["mq"]) == pytest.approx(0.0)
    assert all(q <= 1e-12 for q in sig["mq"])


def test_oss_gss_in_range_and_not_degenerate():
    """GSS/OSS must be valid step indices; OSS must not collapse to 0 when a
    later commit is strictly better net of cost (regression: the -1 fallback
    used to make t=0 the trivial argmax)."""
    # myopic best = node2 (0.3); deep best = node1 (0.8, revealed at node3).
    static = [0.0, 0.9, -0.3, 0.8]
    par = [-1, 0, 0, 1]
    sig = _tree_signals(_make_tree(static, par), "pwin")
    assert 0 <= sig["gss"] < 4
    assert 0 <= sig["oss"] < 4
    # committing later (after node3 reveals node1) yields the better final value,
    # so with a small node cost the optimal stop is not step 0.
    assert sig["oss"] >= sig["gss"] - 1  # oss tracks the informative commit, not t=0


def test_voc_positive_when_search_flips_best():
    """Search reveals node1 is actually best. Myopically node2 wins (0.3 vs -0.9);
    after expanding node1's subtree its deep value is 0.8, so VOC = 0.8 - 0.3."""
    #        0
    #      /   \
    #    1(0.9) 2(-0.3)      myo = [-0.9, 0.3]  -> myopic best = node2
    #    |
    #    3(0.8)              deep node1 = 0.8   -> deep best = node1
    static = [0.0, 0.9, -0.3, 0.8]
    par = [-1, 0, 0, 1]
    sig = _tree_signals(_make_tree(static, par), "pwin")
    assert sig["voc"] == pytest.approx(0.8 - 0.3, abs=1e-9)


def test_degenerate_tree_returns_none():
    """A root with < 2 children yields None, not a fabricated number."""
    assert _tree_signals(_make_tree([0.0, -0.5], [-1, 0]), "pwin") is None


# --------------------------------------------------------------------------
# THE unit-matters test: pwin saturation vs cp resolution on the SAME tree
# --------------------------------------------------------------------------

def test_unit_changes_action_gap_under_saturation():
    """Two children both saturate in pwin (value -1.0 -> root-mover myo 1.0), so
    action_gap_pwin == 0; but their cp_order differs (root-mover myo cp 400 vs
    900), so action_gap_cp == 500. This is the whole reason cp exists."""
    static_value = [0.0, -1.0, -1.0]          # both saturated at the win wall
    static_cp = [0.0, -400.0, -900.0]         # root-mover cp = 400 vs 900
    tree = _make_tree(static_value, [-1, 0, 0],
                      extra_features={"cp_order": static_cp})
    sig_pwin = _tree_signals(tree, "pwin")
    sig_cp = _tree_signals(tree, "cp")
    assert sig_pwin["action_gap"] == pytest.approx(0.0, abs=1e-12)
    assert sig_cp["action_gap"] == pytest.approx(500.0, abs=1e-9)


def test_cp_unit_reads_cp_order_feature():
    """The cp unit must read cp_order, not value."""
    tree = _make_tree([0.0, 0.1, 0.2], [-1, 0, 0],
                      extra_features={"cp_order": [0.0, -50.0, -120.0]})
    sig = _tree_signals(tree, "cp")
    # myo cp = [50, 120]; action_gap = 70
    assert sig["action_gap"] == pytest.approx(70.0)


# --------------------------------------------------------------------------
# integration on real saved trees (skipped until befs1cp_md36 exists)
# --------------------------------------------------------------------------

TREES_DIR = CONFIG["trees_default"]
HAS_TREES = os.path.isdir(TREES_DIR) and any(f.endswith(".pt") for f in os.listdir(TREES_DIR))


@unittest.skipUnless(HAS_TREES, f"No saved trees in {TREES_DIR}")
class TestSavedTrees(unittest.TestCase):
    def test_compute_values_both_units(self):
        for unit in UNITS:
            vals, moves = compute_values(TREES_DIR, n_trees=50, seed=7, n_workers=2, unit=unit)
            self.assertGreater(len(vals), 0)
            self.assertIn("n_acceptable", vals.columns)
            self.assertTrue((vals["gss"] >= 0).all())
            # counts are bounded by the legal-move denominator, and >= 0 (acceptable)
            # / >= 1 (within-epsilon: the best move is always within epsilon of itself).
            self.assertTrue((vals["n_within_epsilon"].dropna() >= 1).all())
            self.assertTrue((vals["n_acceptable"] <= vals["n_root_children"]).all())
            if len(moves):
                # MQ is a loss: <= 0, best == 0 within each fen
                self.assertLessEqual(moves["mq"].max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
