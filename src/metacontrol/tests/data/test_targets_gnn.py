"""Tests for metacontrol.data.targets_gnn."""

import math
import pytest
from typing import Dict, Tuple

from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import SearchNode, EdgeStats
from metacontrol.data.targets_gnn import (
    compute_gnn_targets,
    normalize_wdl,
)


def test_normalize_wdl():
    w, d, l = normalize_wdl((0.5, 0.3, 0.2))
    assert math.isclose(w, 0.5)
    assert math.isclose(d, 0.3)
    assert math.isclose(l, 0.2)

    w, d, l = normalize_wdl((10.0, 5.0, 5.0))
    assert math.isclose(w, 0.5)
    assert math.isclose(d, 0.25)
    assert math.isclose(l, 0.25)

    with pytest.raises(ValueError):
        normalize_wdl((0.0, 0.0, 0.0))


def test_compute_gnn_targets_node_value():
    root = SearchNode(node_id=0, fen="root", features={"value": 0.1})
    tree = SearchTree(root)

    c1 = SearchNode(node_id=1, fen="c1", features={"value": -0.3})
    tree.add_node(0, "e2e4", c1)

    c2 = SearchNode(node_id=2, fen="c2", features={"value": -0.7})
    tree.add_node(0, "d2d4", c2)

    edge_stats = {
        (0, 1): EdgeStats(visit_count=3, q_value=0.5),
        (0, 2): EdgeStats(visit_count=1, q_value=0.1),
    }

    targets = compute_gnn_targets(tree, edge_stats)

    # Root value should be weighted avg of Q-values: (3*0.5 + 1*0.1) / 4 = 1.6 / 4 = 0.4
    assert math.isclose(targets.node_values[0], 0.4)

    # Leaves c1, c2 have no children, so they fallback to their own static values
    assert math.isclose(targets.node_values[1], -0.3)
    assert math.isclose(targets.node_values[2], -0.7)


def test_compute_gnn_targets_edge_wdl():
    root = SearchNode(node_id=0, fen="root", features={"value": 0.1})
    tree = SearchTree(root)

    # c1 has no edge stats but has static WDL, should fallback to flipped static WDL
    c1 = SearchNode(
        node_id=1,
        fen="c1",
        features={"value": -0.3, "wdl_win": 0.2, "wdl_draw": 0.5, "wdl_loss": 0.3}
    )
    tree.add_node(0, "e2e4", c1)

    # c2 has edge stats with mean_wdl
    c2 = SearchNode(node_id=2, fen="c2", features={"value": -0.7})
    tree.add_node(0, "d2d4", c2)

    edge_stats = {
        (0, 1): EdgeStats(visit_count=0), # no visits, triggers fallback
        (0, 2): EdgeStats(visit_count=5, mean_wdl=(0.7, 0.2, 0.1)),
    }

    targets = compute_gnn_targets(tree, edge_stats)

    # (0, 1) fallback to flipped c1 static WDL: (0.3, 0.5, 0.2)
    w1, d1, l1 = targets.edge_wdls[(0, 1)]
    assert math.isclose(w1, 0.3)
    assert math.isclose(d1, 0.5)
    assert math.isclose(l1, 0.2)

    # (0, 2) uses stats mean WDL directly (no flipping since mean_wdl is already computed from parent perspective during backprop)
    w2, d2, l2 = targets.edge_wdls[(0, 2)]
    assert math.isclose(w2, 0.7)
    assert math.isclose(d2, 0.2)
    assert math.isclose(l2, 0.1)

def test_compute_gnn_targets_missing_wdl():
    root = SearchNode(node_id=0, fen="root", features={"value": 0.1})
    tree = SearchTree(root)

    c1 = SearchNode(node_id=1, fen="c1", features={"value": -0.3}) # No WDL
    tree.add_node(0, "e2e4", c1)

    edge_stats = {(0, 1): EdgeStats(visit_count=0)}

    targets = compute_gnn_targets(tree, edge_stats)
    assert (0, 1) not in targets.edge_wdls
