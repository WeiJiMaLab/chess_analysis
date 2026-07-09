"""TDD-lite sanity tests for ``analysis.tree_diagnostics`` (plan.md T1/T2, 2026-07-08).

A few well-chosen checks per function, not an exhaustive suite -- these are diagnostic-script
helpers, not production training code.
"""
from __future__ import annotations

import numpy as np

from analysis.tree_diagnostics import (
    duplicate_fen_check,
    illegal_move_check,
    linear_chain_check,
    prune_topk_subtree,
    root_dominance_check,
    select_percentile_episodes,
)

FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_select_percentile_episodes_picks_closest_real_values():
    values = np.array([1, 3, 3, 6, 11, 19, 40, 89])
    idx = select_percentile_episodes(values, [3, 6, 11, 19, 89])
    assert [int(values[i]) for i in idx] == [3, 6, 11, 19, 89]


def test_duplicate_fen_check_flags_repeat_position():
    fens = [FEN_START, FEN_AFTER_E4, FEN_START]  # node 0 and node 2 share a position
    result = duplicate_fen_check(fens)
    assert result["pass"] is False
    assert result["n_duplicates"] == 1
    assert result["example_pairs"] == [(0, 2)]


def test_duplicate_fen_check_passes_on_distinct_positions():
    result = duplicate_fen_check([FEN_START, FEN_AFTER_E4])
    assert result["pass"] is True
    assert result["n_duplicates"] == 0


def test_linear_chain_check_detects_zero_branching():
    # root -> child -> grandchild, nobody has 2+ children.
    assert linear_chain_check(np.array([1, 1, 0])) == {
        "pass": False, "max_children": 1, "n_branching_nodes": 0,
    }


def test_linear_chain_check_passes_with_real_branching():
    result = linear_chain_check(np.array([3, 0, 0, 0]))
    assert result["pass"] is True
    assert result["max_children"] == 3


def test_root_dominance_check_fires_on_starved_siblings():
    # one child owns 95/100 nodes, two siblings never expanded (size 1 each).
    result = root_dominance_check([95, 1, 1], total_nodes=100)
    assert result["pass"] is False
    assert result["n_unexpanded_siblings"] == 2


def test_root_dominance_check_passes_when_small_or_balanced():
    assert root_dominance_check([5, 4], total_nodes=10)["pass"] is True  # too small to judge
    balanced = root_dominance_check([30, 25, 25], total_nodes=100, min_total_nodes=10)
    assert balanced["pass"] is True


def test_illegal_move_check_flags_impossible_move():
    # node 1's parent is node 0 (FEN_START); "e2e5" is not a legal pawn move from the start pos.
    result = illegal_move_check([FEN_START, FEN_AFTER_E4], np.array([-1, 0]), [None, "e2e5"])
    assert result["pass"] is False
    assert result["n_bad"] == 1


def test_illegal_move_check_passes_on_real_move():
    result = illegal_move_check([FEN_START, FEN_AFTER_E4], np.array([-1, 0]), [None, "e2e4"])
    assert result["pass"] is True


def test_prune_topk_subtree_stays_connected_and_within_budget():
    # a small star-ish tree: root(0) -> 1,2,3; 1 -> 4,5; 2 -> 6
    parent_index = np.array([-1, 0, 0, 0, 1, 1, 2])
    selected = prune_topk_subtree(parent_index, budget=4)
    assert 0 in selected
    assert len(selected) <= 4
    # every selected non-root node's parent must also be selected (connectivity).
    for node_id in selected:
        if node_id != 0:
            assert int(parent_index[node_id]) in selected


def test_prune_topk_subtree_keeps_whole_tree_if_budget_allows():
    parent_index = np.array([-1, 0, 0, 1])
    selected = prune_topk_subtree(parent_index, budget=10)
    assert selected == {0, 1, 2, 3}
