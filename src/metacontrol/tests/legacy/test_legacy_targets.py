import chess
import sys
import os
import pytest

# Add src to path
sys.path.append(os.path.abspath("src"))

from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import SearchNode, EdgeStats, ChildInfo
from metacontrol.data.targets_gnn import compute_gnn_targets

# Mock legacy backup logic
def legacy_backup_root_value(tree, edge_stats):
    total_val = 0
    total_visits = 0
    root_id = tree.root.node_id
    for (p_id, c_id), stats in edge_stats.items():
        if p_id == root_id:
            total_val += stats.visit_count * stats.q_value
            total_visits += stats.visit_count
    return total_val / total_visits if total_visits > 0 else 0.0

def test_target_parity_full_tree():
    """
    Simulate a small search tree and compare Root Target Value.
    Root: White to move.
    Moves: e2e4 (Good), d2d4 (Neutral)
    Engine Q (Parent perspective): e2e4=0.6, d2d4=0.2
    """
    
    # --- New Pipeline Setup ---
    root = SearchNode(node_id=0, fen=chess.STARTING_FEN)
    tree = SearchTree(root)
    
    # Child A: e2e4. We store Child perspective value (-0.6)
    child_a = SearchNode(node_id=1, fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    child_a.features["value"] = -0.6
    child_a.features["wdl_win"] = 0.1
    child_a.features["wdl_draw"] = 0.2
    child_a.features["wdl_loss"] = 0.7
    tree.add_node(0, "e2e4", child_a)
    
    # Child B: d2d4. We store Child perspective value (-0.2)
    child_b = SearchNode(node_id=2, fen="rnbqkbnr/pppp1ppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1")
    child_b.features["value"] = -0.2
    child_b.features["wdl_win"] = 0.3
    child_b.features["wdl_draw"] = 0.2
    child_b.features["wdl_loss"] = 0.5
    tree.add_node(0, "d2d4", child_b)
    
    # Simulating edge stats after backprop
    # In new pipeline, backprop flips Child Value (-0.6) to get Edge Q (0.6)
    edge_stats = {
        (0, 1): EdgeStats(visit_count=3, total_value=1.8, q_value=0.6),
        (0, 2): EdgeStats(visit_count=1, total_value=0.2, q_value=0.2),
    }
    
    new_targets = compute_gnn_targets(tree, edge_stats)
    # Root target is usually at index 0 or we can check by node_id
    new_root_v = new_targets.node_values[0]
    
    # --- Legacy Pipeline Setup ---
    # Legacy stores Parent perspective value (0.6) in child features.
    # Then backprop flips it to -0.6 for the edge.
    leg_edge_stats = {
        (0, 1): EdgeStats(visit_count=3, total_value=-1.8, q_value=-0.6),
        (0, 2): EdgeStats(visit_count=1, total_value=-0.2, q_value=-0.2),
    }
    
    leg_root_v = legacy_backup_root_value(tree, leg_edge_stats)
    
    print(f"\n=== TARGET PARITY ANALYSIS ===")
    print(f"New Root Target V:    {new_root_v:.3f} (Correct: Positive for winning)")
    print(f"Legacy Root Target V: {leg_root_v:.3f} (Inverted: Negative for winning)")
    
    print(f"Exact New Root Target V: {new_root_v}")
    assert new_root_v == pytest.approx(0.5)
    assert leg_root_v == pytest.approx(-0.5)

if __name__ == "__main__":
    test_target_parity_full_tree()
