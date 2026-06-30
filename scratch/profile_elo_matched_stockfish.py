import sys
import time
from pathlib import Path
import chess
import chess.engine

# Set up workspace paths
repo_path = Path("/home/hl4291/chess_analysis")
sys.path.insert(0, str(repo_path))
sys.path.insert(0, str(repo_path / "lmcos_small" / "human"))

from utils.helpers import get_engine
from cts.core.providers.base import TreeExpansionProvider
from cts.core.tree import ExpansionChild
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    TeacherSearchConfig,
    generate_partial_tree_from_provider
)

class EloMatchedStockfishProvider(TreeExpansionProvider):
    def __init__(self, engine, search_limit):
        self.engine = engine
        self.limit = search_limit
        self._cache = {}

    def _eval_fen(self, fen: str) -> dict:
        if fen in self._cache:
            return self._cache[fen]
            
        board = chess.Board(fen)
        if board.is_game_over():
            if board.is_checkmate():
                w, d, l = 0.0, 0.0, 1.0
            else:
                w, d, l = 0.0, 1.0, 0.0
            val = w - l
            var = (w + l) - val * val
            res = {"value": val, "wdl_win": w, "wdl_draw": d, "wdl_loss": l, "wdl_var": var}
            self._cache[fen] = res
            return res
            
        # Analyze using the configured search limit (e.g. nodes=100 or depth=5)
        info = self.engine.analyse(board, self.limit)
        score = info.get("score")
        if score is None:
            w, d, l = 0.33, 0.34, 0.33
        else:
            wdl = score.pov(board.turn).wdl()
            total = wdl.total()
            w, d, l = wdl.wins / total, wdl.draws / total, wdl.losses / total
            
        val = w - l
        var = (w + l) - val * val
        res = {"value": val, "wdl_win": w, "wdl_draw": d, "wdl_loss": l, "wdl_var": var}
        self._cache[fen] = res
        return res

    def root_features(self, fen: str):
        feats = self._eval_fen(fen)
        return {**feats, "prior": 1.0}

    def expand_node(self, fen: str, depth: int, max_children=None):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if max_children is not None:
            legal_moves = legal_moves[:max_children]
            
        children = []
        n_moves = len(legal_moves)
        prior = 1.0 / n_moves if n_moves > 0 else 1.0
        
        for move in legal_moves:
            child_board = board.copy(stack=False)
            child_board.push(move)
            child_fen = child_board.fen()
            
            child_feats = self._eval_fen(child_fen)
            feats = {**child_feats, "prior": prior}
            
            children.append(
                ExpansionChild(
                    move_uci=move.uci(),
                    fen=child_fen,
                    scalar_features=feats,
                    is_terminal=child_board.is_game_over()
                )
            )
        return children

def run_elo_benchmark():
    print("=== Elo-Matched Stockfish MCTS Benchmarks ===")
    root_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    node_budget = NodeBudgetDistribution(10, 10)
    search_config = TeacherSearchConfig(
        max_depth=4,
        search_budget=10,
        c_puct=1.0,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id="benchmark"
    )

    # Start Stockfish
    engine = get_engine("stockfish", threads=1)
    
    # Enable Elo Matching to 1800 Elo
    print("[Configuring Stockfish to 1800 Elo...]")
    engine.configure({
        "UCI_LimitStrength": "true",
        "UCI_Elo": "1800"
    })

    # Test Case 1: nodes=100
    print("\n--- Test Case 1: Elo-Matched 1800 Stockfish (nodes=100) ---")
    provider_100 = EloMatchedStockfishProvider(engine, chess.engine.Limit(nodes=100))
    t0 = time.perf_counter()
    res_100 = generate_partial_tree_from_provider(root_fen, provider_100, search_config, node_budget)
    t1 = time.perf_counter()
    time_100 = t1 - t0
    print(f"Nodes: {res_100.tree.num_nodes()} | Time: {time_100:.4f} seconds.")

    # Test Case 2: depth=5
    print("\n--- Test Case 2: Elo-Matched 1800 Stockfish (depth=5) ---")
    provider_d5 = EloMatchedStockfishProvider(engine, chess.engine.Limit(depth=5))
    t0 = time.perf_counter()
    res_d5 = generate_partial_tree_from_provider(root_fen, provider_d5, search_config, node_budget)
    t1 = time.perf_counter()
    time_d5 = t1 - t0
    print(f"Nodes: {res_d5.tree.num_nodes()} | Time: {time_d5:.4f} seconds.")

    engine.close()
    
    print("\n" + "="*50)
    print("MCTS TREE GENERATION SPEED SUMMARY")
    print(f"Stockfish (Raw, nodes=10)    : 0.1814 seconds")
    print(f"Stockfish (1800 Elo, nodes=100): {time_100:.4f} seconds")
    print(f"Stockfish (1800 Elo, depth=5)  : {time_d5:.4f} seconds")
    print(f"Lc0 MCTS (CPU, Eigen)          : 209.5515 seconds")
    print("="*50)

if __name__ == "__main__":
    run_elo_benchmark()
