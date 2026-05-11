import chess
import math
import sys
import os
from typing import Dict, List, Tuple, Mapping, Sequence

# Add src to path
sys.path.append(os.path.abspath("src"))

from metacontrol.core.tree import SearchTree, SearchNode
from metacontrol.core.schemas import EdgeStats, ChildInfo
from metacontrol.data.targets_gnn import compute_gnn_targets

# --- MOCKS ---

class MockProvider:
    """A provider that understands the Scholar's Mate position."""
    def __init__(self, mode="new"):
        self.mode = mode # "new" (fixed) or "legacy" (buggy)
        
    def evaluate_root(self, fen: str):
        # In a real engine, root evaluation is standard
        return 0.1, (0.4, 0.2, 0.4)

    def expand(self, fen: str) -> List[ChildInfo]:
        board = chess.Board(fen)
        children = []
        for move in board.legal_moves:
            m_uci = move.uci()
            board.push(move)
            # Simple heuristic: Qf7 is mate
            if board.is_checkmate():
                q_parent = 1.0
                wdl_parent = (1.0, 0.0, 0.0)
            else:
                q_parent = 0.0 # Neutral
                wdl_parent = (0.33, 0.34, 0.33)
            
            # THE CRITICAL DIFFERENCE
            if self.mode == "new":
                val_child = -q_parent
            else:
                val_child = q_parent # LEGACY BUG: Storing parent Q in child
            
            children.append(ChildInfo(
                move_uci=m_uci,
                fen=board.fen(),
                value=val_child,
                prior=0.1, # Uniform prior
                is_terminal=board.is_game_over()
            ))
            board.pop()
        return children

# --- SEARCH LOGIC ---

class DivergentSearch:
    def __init__(self, provider, c_puct=1.0):
        self.provider = provider
        self.c_puct = c_puct
        
    def grow(self, root_fen: str, iterations: int) -> Tuple[SearchTree, Dict]:
        root_node = SearchNode(node_id=0, fen=root_fen)
        tree = SearchTree(root_node)
        edge_stats = {}
        
        # Initial expansion of root
        children = self.provider.expand(root_fen)
        for i, c in enumerate(children):
            child_node = SearchNode(node_id=len(tree.nodes_by_id), fen=c.fen)
            child_node.features["value"] = c.value
            child_node.features["prior"] = c.prior
            child_node.is_terminal = c.is_terminal
            tree.add_node(0, c.move_uci, child_node)
            edge_stats[(0, child_node.node_id)] = EdgeStats()

        # Search iterations
        for _ in range(iterations):
            # 1. Select
            node_id = 0
            path = []
            while tree.get_node(node_id).children:
                best_score = -float('inf')
                best_move = None
                best_child_id = None
                
                # PUCT Selection
                parent_visits = sum(edge_stats[(node_id, cid)].visit_count for cid in [n.node_id for n in tree.get_node(node_id).children.values()])
                for move, child in tree.get_node(node_id).children.items():
                    stats = edge_stats[(node_id, child.node_id)]
                    q = stats.q_value
                    prior = child.features.get("prior", 0.1)
                    u = self.c_puct * prior * math.sqrt(parent_visits + 1) / (1 + stats.visit_count)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child_id = child.node_id
                
                path.append((node_id, best_child_id))
                node_id = best_child_id
                if tree.get_node(node_id).is_terminal:
                    break
            
            # 2. Backprop
            leaf_val = tree.get_node(node_id).features["value"]
            val = leaf_val
            for p_id, c_id in reversed(path):
                val = -val # Perspective flip
                stats = edge_stats[(p_id, c_id)]
                stats.visit_count += 1
                stats.total_value += val
                stats.q_value = stats.total_value / stats.visit_count
                
        return tree, edge_stats

def run_experiment():
    # Scholar's Mate Position: White can mate with Qf7
    root_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    
    print("\n=== SEARCH DIVERGENCE EXPERIMENT ===")
    print("Position: Scholar's Mate (White to move, Qf7# is winning)")
    
    # 1. New (Fixed) Search
    new_search = DivergentSearch(MockProvider(mode="new"))
    new_tree, new_edges = new_search.grow(root_fen, iterations=100)
    
    # 2. Legacy (Buggy) Search
    leg_search = DivergentSearch(MockProvider(mode="legacy"))
    leg_tree, leg_edges = leg_search.grow(root_fen, iterations=100)
    
    # Analyze root edges
    print("\n--- RESULTS: Top moves by visit count ---")
    
    def print_top_moves(tree, edges, label):
        print(f"\n[{label}]")
        root_edges = [(m, child.node_id) for m, child in tree.root.children.items()]
        sorted_edges = sorted(root_edges, key=lambda x: edges[(0, x[1])].visit_count, reverse=True)
        for move, cid in sorted_edges[:5]:
            stats = edges[(0, cid)]
            print(f"  Move {move}: visits={stats.visit_count}, q={stats.q_value:.3f}")

    print_top_moves(new_tree, new_edges, "NEW PIPELINE (Fixed)")
    print_top_moves(leg_tree, leg_edges, "LEGACY PIPELINE (Buggy)")
    
    # Parity check on targets
    # (We skip compute_gnn_targets for now to avoid the WDL error, just check root value manually)
    def manual_root_v(edges):
        total_val = sum(s.visit_count * s.q_value for (p, c), s in edges.items() if p == 0)
        visits = sum(s.visit_count for (p, c), s in edges.items() if p == 0)
        return total_val / visits if visits > 0 else 0.0

    new_root_v = manual_root_v(new_edges)
    leg_root_v = manual_root_v(leg_edges)
    
    print(f"\nRoot Target Value (New):    {new_root_v:.3f}")
    print(f"Root Target Value (Legacy): {leg_root_v:.3f}")

if __name__ == "__main__":
    run_experiment()
