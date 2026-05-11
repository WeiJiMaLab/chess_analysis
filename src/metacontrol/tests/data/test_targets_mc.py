"""Tests for metacontrol.data.targets — Stage 3 of the migration plan."""

import math
import pytest
from typing import Dict, List, Tuple

from metacontrol.core.tree import SearchNode, SearchTree
from metacontrol.data.generator import EdgeStats
from metacontrol.data.targets_mc import (
    SearchSnapshot,
    calculate_dp_values,
    compute_halt_rewards,
    derive_snapshots,
)


# ---------------------------------------------------------------------------
# Helpers — build small hand-crafted trees
# ---------------------------------------------------------------------------

def _build_linear_tree() -> Tuple[SearchTree, Dict[Tuple[int, int], EdgeStats]]:
    """Build a 3-node linear tree: root → c1 → c2.

    Expansion history: [c1, c2] (root expanded first adding c1,
    then c1 expanded adding c2).

    Edge stats simulate a simple search:
      (0,1) visited once with Q = 0.3  (root→c1)
      (1,2) visited once with Q = 0.5  (c1→c2)
    """
    root = SearchNode(node_id=0, fen="root", features={"value": 0.1})
    tree = SearchTree(root)

    c1 = SearchNode(node_id=1, fen="c1", features={"value": -0.3, "prior": 1.0})
    tree.add_node(0, "e2e4", c1)

    c2 = SearchNode(node_id=2, fen="c2", features={"value": 0.5, "prior": 1.0})
    tree.add_node(1, "e7e5", c2)

    edge_stats = {
        (0, 1): EdgeStats(visit_count=1, total_value=0.3, q_value=0.3),
        (1, 2): EdgeStats(visit_count=1, total_value=0.5, q_value=0.5),
    }
    return tree, edge_stats


def _build_branching_tree() -> Tuple[SearchTree, Dict[Tuple[int, int], EdgeStats]]:
    """Build a tree:  root → {c1, c2}, c1 → c3.

    Expansion history: [c1, c2, c3]
    (root expanded adding c1 and c2, then c1 expanded adding c3).

    Edge stats:
      (0,1) Q = 0.4 (best root child)
      (0,2) Q = 0.2
      (1,3) Q = 0.6
    """
    root = SearchNode(node_id=0, fen="root", features={"value": 0.1})
    tree = SearchTree(root)

    c1 = SearchNode(node_id=1, fen="c1", features={"value": -0.4, "prior": 0.7})
    c2 = SearchNode(node_id=2, fen="c2", features={"value": -0.2, "prior": 0.3})
    tree.add_node(0, "e2e4", c1)
    tree.add_node(0, "d2d4", c2)

    c3 = SearchNode(node_id=3, fen="c3", features={"value": 0.6, "prior": 1.0})
    tree.add_node(1, "e7e5", c3)

    edge_stats = {
        (0, 1): EdgeStats(visit_count=2, total_value=0.8, q_value=0.4),
        (0, 2): EdgeStats(visit_count=1, total_value=0.2, q_value=0.2),
        (1, 3): EdgeStats(visit_count=1, total_value=0.6, q_value=0.6),
    }
    return tree, edge_stats


# ---------------------------------------------------------------------------
# DP math tests
# ---------------------------------------------------------------------------

def test_dp_single_step():
    """With one halt reward, DP should just return that value."""
    dp = calculate_dp_values([0.5], continue_cost=0.1)
    assert len(dp) == 1
    assert math.isclose(dp[0], 0.5)


def test_dp_two_steps_should_continue():
    """When continuing is clearly better, DP value should exceed halt reward."""
    # halt_rewards = [0.2, 0.9], cost = 0.1
    # V[1] = 0.9
    # V[0] = max(0.2, 0.9 - 0.1) = max(0.2, 0.8) = 0.8
    dp = calculate_dp_values([0.2, 0.9], continue_cost=0.1)
    assert math.isclose(dp[0], 0.8)
    assert math.isclose(dp[1], 0.9)


def test_dp_two_steps_should_halt():
    """When halting is better, DP value should equal halt reward."""
    # halt_rewards = [0.8, 0.5], cost = 0.1
    # V[1] = 0.5
    # V[0] = max(0.8, 0.5 - 0.1) = max(0.8, 0.4) = 0.8
    dp = calculate_dp_values([0.8, 0.5], continue_cost=0.1)
    assert math.isclose(dp[0], 0.8)
    assert math.isclose(dp[1], 0.5)


def test_linear_dp_math():
    """Manual 3-step DP verification."""
    # halt_rewards = [0.1, 0.3, 0.7], cost = 0.05
    # V[2] = 0.7
    # V[1] = max(0.3, 0.7 - 0.05) = max(0.3, 0.65) = 0.65
    # V[0] = max(0.1, 0.65 - 0.05) = max(0.1, 0.60) = 0.60
    dp = calculate_dp_values([0.1, 0.3, 0.7], continue_cost=0.05)
    assert math.isclose(dp[0], 0.60)
    assert math.isclose(dp[1], 0.65)
    assert math.isclose(dp[2], 0.70)


def test_dp_empty():
    assert calculate_dp_values([], continue_cost=0.1) == []


# ---------------------------------------------------------------------------
# Halt reward tests
# ---------------------------------------------------------------------------

def test_halt_rewards_linear_tree():
    tree, edge_stats = _build_linear_tree()
    rewards = compute_halt_rewards(tree, edge_stats)

    # 3 snapshots: prefix-0 (root only), prefix-1 (root+c1), prefix-2 (root+c1+c2)
    assert len(rewards) == 3

    # prefix-0: root only, no children → halt = root.value = 0.1
    assert math.isclose(rewards[0], 0.1, abs_tol=1e-6)

    # prefix-1: root has child c1 with edge Q = 0.3
    assert math.isclose(rewards[1], 0.3, abs_tol=1e-6)

    # prefix-2: same root child, same Q
    assert math.isclose(rewards[2], 0.3, abs_tol=1e-6)


def test_halt_rewards_branching_tree():
    tree, edge_stats = _build_branching_tree()
    rewards = compute_halt_rewards(tree, edge_stats)

    # 4 snapshots: prefix-0, prefix-1 (c1+c2 added), prefix-2, prefix-3 (c3 added)
    # But expansion_history has c1, c2, c3 → len = 3+1 = 4 snapshots
    assert len(rewards) == len(tree.expansion_history) + 1


# ---------------------------------------------------------------------------
# Optimal stop logic
# ---------------------------------------------------------------------------

def test_optimal_stop_logic():
    """With high enough cost, the DP should prefer halting immediately."""
    halt_rewards = [0.5, 0.55, 0.6]
    dp = calculate_dp_values(halt_rewards, continue_cost=0.5)
    # Cost is so high that continuing never helps
    # V[2] = 0.6
    # V[1] = max(0.55, 0.6 - 0.5) = max(0.55, 0.1) = 0.55
    # V[0] = max(0.5, 0.55 - 0.5) = max(0.5, 0.05) = 0.5
    assert math.isclose(dp[0], 0.5)
    assert math.isclose(dp[1], 0.55)
    assert math.isclose(dp[2], 0.6)


# ---------------------------------------------------------------------------
# Snapshot derivation tests
# ---------------------------------------------------------------------------

def test_derive_snapshots_count():
    tree, edge_stats = _build_linear_tree()
    snapshots = derive_snapshots(tree, edge_stats, continue_cost=0.05)
    assert len(snapshots) == len(tree.expansion_history) + 1


def test_derive_snapshots_last_advantage_nonpositive():
    """The last snapshot should have advantage ≤ 0 (forced halt)."""
    tree, edge_stats = _build_linear_tree()
    snapshots = derive_snapshots(tree, edge_stats, continue_cost=0.05)
    assert snapshots[-1].advantage <= 1e-10


def test_derive_snapshots_advantage_sign():
    """With zero cost and improving halt rewards, advantages should be non-negative."""
    halt_rewards = [0.1, 0.2, 0.3]
    dp = calculate_dp_values(halt_rewards, continue_cost=0.0)

    # Manually check: with zero cost, continuing is always at least as good
    # V[2] = 0.3
    # V[1] = max(0.2, 0.3) = 0.3, continue_value = 0.3, adv = 0.3 - 0.2 = 0.1
    # V[0] = max(0.1, 0.3) = 0.3, continue_value = 0.3, adv = 0.3 - 0.1 = 0.2
    assert math.isclose(dp[0], 0.3)
    assert math.isclose(dp[1], 0.3)


def test_monotonicity():
    """Halt reward should be non-decreasing when more nodes are added
    (in a well-behaved tree where each expansion improves the best move).
    """
    # Construct a tree where each expansion strictly improves the best root child
    root = SearchNode(node_id=0, fen="root", features={"value": 0.0})
    tree = SearchTree(root)

    # First expansion: add one child
    c1 = SearchNode(node_id=1, fen="c1", features={"value": -0.3, "prior": 1.0})
    tree.add_node(0, "e2e4", c1)

    # Second expansion: add a better child
    c2 = SearchNode(node_id=2, fen="c2", features={"value": -0.5, "prior": 1.0})
    tree.add_node(0, "d2d4", c2)

    edge_stats = {
        (0, 1): EdgeStats(visit_count=1, total_value=0.3, q_value=0.3),
        (0, 2): EdgeStats(visit_count=1, total_value=0.5, q_value=0.5),
    }

    rewards = compute_halt_rewards(tree, edge_stats)
    # prefix-0: root only → 0.0
    # prefix-1: c1 present, Q = 0.3
    # prefix-2: c1 + c2 present, best Q = 0.5
    assert rewards[0] <= rewards[1] <= rewards[2] + 1e-10


def test_build_snapshots_integrity():
    """End-to-end: snapshots should be consistent with each other."""
    tree, edge_stats = _build_branching_tree()
    snapshots = derive_snapshots(tree, edge_stats, continue_cost=0.05)

    for i, snap in enumerate(snapshots):
        assert snap.expansion_index == i
        # advantage = continue_value - halt_reward
        assert math.isclose(snap.advantage, snap.continue_value - snap.halt_reward, abs_tol=1e-10)

    # Last snapshot advantage should be 0 (halt = continue at terminal)
    assert math.isclose(snapshots[-1].advantage, 0.0, abs_tol=1e-10)
