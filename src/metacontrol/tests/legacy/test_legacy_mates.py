import chess
import math
import sys
import os
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(os.path.abspath("src"))

from metacontrol.core.tree import SearchTree, SearchNode
from metacontrol.core.schemas import EdgeStats, ChildInfo

class MockProvider:
    """A provider that understands mating positions and can toggle between 'new' (fixed) and 'legacy' (buggy) modes."""
    def __init__(self, mode="new"):
        self.mode = mode 
        
    def expand(self, fen: str) -> List[ChildInfo]:
        board = chess.Board(fen)
        children = []
        for move in board.legal_moves:
            m_uci = move.uci()
            board.push(move)
            
            # Simple heuristic: Identify checkmate
            if board.is_checkmate():
                q_parent = 1.0
            else:
                q_parent = 0.0 # Neutral or other
            
            if self.mode == "new":
                val_child = -q_parent # Correct: Side to move in child is losing
            else:
                val_child = q_parent # BUGGY: Storing parent perspective in child
            
            children.append(ChildInfo(
                move_uci=m_uci,
                fen=board.fen(),
                value=val_child,
                prior=0.1, 
                is_terminal=board.is_game_over()
            ))
            board.pop()
        return children

class SearchRunner:
    def __init__(self, provider, c_puct=1.0):
        self.provider = provider
        self.c_puct = c_puct
        
    def run(self, root_fen: str, iterations: int):
        root_node = SearchNode(node_id=0, fen=root_fen)
        tree = SearchTree(root_node)
        edge_stats = {}
        
        # Root expansion
        children = self.provider.expand(root_fen)
        for c in children:
            child_node = SearchNode(node_id=len(tree.nodes_by_id), fen=c.fen)
            child_node.features["value"] = c.value
            child_node.features["prior"] = c.prior
            child_node.is_terminal = c.is_terminal
            tree.add_node(0, c.move_uci, child_node)
            edge_stats[(0, child_node.node_id)] = EdgeStats()

        for _ in range(iterations):
            # Select
            node_id = 0
            path = []
            while tree.get_node(node_id).children:
                best_score = -float('inf')
                best_child_id = None
                
                parent_visits = sum(edge_stats[(node_id, cid)].visit_count for cid in [n.node_id for n in tree.get_node(node_id).children.values()])
                for move, child in tree.get_node(node_id).children.items():
                    stats = edge_stats[(node_id, child.node_id)]
                    q = stats.q_value
                    prior = child.features.get("prior", 0.1)
                    u = self.c_puct * prior * math.sqrt(parent_visits + 1) / (1 + stats.visit_count)
                    score = q + u
                    if score > best_score:
                        best_score, best_child_id = score, child.node_id
                
                path.append((node_id, best_child_id))
                node_id = best_child_id
                if tree.get_node(node_id).is_terminal: break
            
            # Backprop
            leaf_val = tree.get_node(node_id).features["value"]
            val = leaf_val
            for p_id, c_id in reversed(path):
                val = -val 
                stats = edge_stats[(p_id, c_id)]
                stats.visit_count += 1
                stats.total_value += val
                stats.q_value = stats.total_value / stats.visit_count
                
        return tree, edge_stats

def analyze_position(name, fen, mate_move):
    print(f"\n--- Analyzing {name} ---")
    print(f"FEN: {fen}")
    print(f"Expected Mate Move: {mate_move}")
    
    for mode in ["legacy", "new"]:
        provider = MockProvider(mode=mode)
        runner = SearchRunner(provider)
        tree, edges = runner.run(fen, iterations=100)
        
        # Find mate move stats
        mate_cid = [node.node_id for move, node in tree.root.children.items() if move == mate_move][0]
        stats = edges[(0, mate_cid)]
        
        # Find best move by visits
        root_edges = [(m, child.node_id) for m, child in tree.root.children.items()]
        best_move, best_cid = max(root_edges, key=lambda x: edges[(0, x[1])].visit_count)
        best_stats = edges[(0, best_cid)]
        
        label = "LEGACY (Buggy)" if mode == "legacy" else "NEW (Fixed)"
        print(f"[{label}]")
        print(f"  Mate Move ({mate_move}): visits={stats.visit_count}, Q={stats.q_value:.3f}")
        print(f"  Best Move ({best_move}): visits={best_stats.visit_count}, Q={best_stats.q_value:.3f}")
        
        if mode == "new" and best_move == mate_move:
            print("  Result: SUCCESS (Identified Mate)")
        elif mode == "legacy" and best_move != mate_move:
            print("  Result: CONFIRMED BUG (Ignored Mate)")
        elif mode == "legacy" and best_move == mate_move:
            print("  Result: UNEXPECTED (Legacy found mate? Check logic)")

if __name__ == "__main__":
    positions = [
        ("Scholar's Mate", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "f3f7"),
        ("Back-rank Mate", "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8"),
        ("King+Rook Mate", "k7/8/K1R5/8/8/8/8/8 w - - 0 1", "c6c8"),
        ("Anastasia's Mate (modified)", "r4rk1/pp3ppp/2n5/2b1p3/2B1P3/5N2/PPP1NnPP/R1BQ1R1K w - - 1 10", "f1f2") # Not mate, but important move
    ]
    
    for name, fen, move in positions:
        analyze_position(name, fen, move)
