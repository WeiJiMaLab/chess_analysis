import sys
import time
import os
from pathlib import Path
import chess
import chess.engine
import numpy as np

# Set up workspace paths
repo_path = Path("/home/hl4291/chess_analysis")
sys.path.insert(0, str(repo_path))
sys.path.insert(0, str(repo_path / "human_analytics"))

from utils.helpers import get_engine
from cts.core.providers.base import TreeExpansionProvider
from cts.core.providers.lc0 import Lc0DirectEvalProvider
from cts.core.providers.process import UciEngineConfig, UciEngineProcess
from cts.core.tree import ExpansionChild
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    TeacherSearchConfig,
    generate_partial_tree_from_provider
)

# -----------------------------------------------------------------------------
# 1. Custom Stockfish Tree Expansion Provider
# -----------------------------------------------------------------------------
class StockfishDirectEvalProvider(TreeExpansionProvider):
    """MCTS expansion provider backed by Stockfish.
    Queries Stockfish to obtain evaluations (value, WDL) at each expanded node.
    """
    def __init__(self, engine: chess.engine.SimpleEngine):
        self.engine = engine
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
            
        # Run a very fast 10-node search to get WDL. Stockfish NNUE/eval handles this instantly.
        info = self.engine.analyse(board, chess.engine.Limit(nodes=10))
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

# -----------------------------------------------------------------------------
# 2. Benchmark Driver
# -----------------------------------------------------------------------------
def run_benchmark():
    print("=== MCTS Rollout/Tree Generation Benchmark ===")
    root_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"  # Open Game
    
    # 10 expansions budget for the MCTS tree
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

    # --- Part A: Stockfish MCTS ---
    print("\n[Stockfish MCTS] Starting Stockfish engine...")
    sf_engine = get_engine("stockfish", threads=1)
    sf_provider = StockfishDirectEvalProvider(sf_engine)
    
    print("[Stockfish MCTS] Running PUCT MCTS rollout (10 expansions)...")
    t0 = time.perf_counter()
    sf_tree_result = generate_partial_tree_from_provider(
        root_fen,
        sf_provider,
        search_config,
        node_budget
    )
    t1 = time.perf_counter()
    sf_time = t1 - t0
    sf_nodes = sf_tree_result.tree.num_nodes()
    print(f"[Stockfish MCTS] Done. Tree size: {sf_nodes} nodes. Time: {sf_time:.4f} seconds.")
    sf_engine.close()

    # --- Part B: Lc0 MCTS ---
    print("\n[Lc0 MCTS] Starting Lc0 engines (priors and value)...")
    lc0_path = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
    weights_path = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
    
    # Configure Lc0 to use CPU 'eigen' backend and enable WDL output
    uci_opts = {"Backend": "eigen", "UCI_ShowWDL": "true"}
    
    prior_config = UciEngineConfig(
        engine_path=lc0_path,
        engine_kind="lc0",
        engine_mode="classic",
        movetime_ms=0,
        multipv=8,
        depth=None,
        nodes=1,
        weights_path=weights_path,
        uci_options=uci_opts,
        set_multipv=False,
        enable_verbose_move_stats=True,
    )
    value_config = UciEngineConfig(
        engine_path=lc0_path,
        engine_kind="lc0",
        engine_mode="valuehead",
        movetime_ms=0,
        multipv=1,
        depth=None,
        nodes=1,
        weights_path=weights_path,
        uci_options=uci_opts,
        set_multipv=False,
        enable_verbose_move_stats=False,
    )
    
    with UciEngineProcess(prior_config) as prior_engine, UciEngineProcess(value_config) as value_engine:
        lc0_provider = Lc0DirectEvalProvider(prior_engine, value_engine)
        print("[Lc0 MCTS] Running PUCT MCTS rollout (10 expansions)...")
        t0 = time.perf_counter()
        lc0_tree_result = generate_partial_tree_from_provider(
            root_fen,
            lc0_provider,
            search_config,
            node_budget
        )
        t1 = time.perf_counter()
        lc0_time = t1 - t0
        lc0_nodes = lc0_tree_result.tree.num_nodes()
        print(f"[Lc0 MCTS] Done. Tree size: {lc0_nodes} nodes. Time: {lc0_time:.4f} seconds.")

    # --- Part C: Summary ---
    print("\n" + "="*50)
    print("MCTS TREE GENERATION COMPARISON (10 expansions)")
    print(f"Stockfish MCTS Time : {sf_time:.4f} seconds ({sf_nodes} nodes)")
    print(f"Lc0 MCTS Time       : {lc0_time:.4f} seconds ({lc0_nodes} nodes)")
    if sf_time > 0 and lc0_time > 0:
        ratio = lc0_time / sf_time
        print(f"Stockfish is {ratio:.2f}x FASTER at MCTS tree generation than Lc0 on CPU.")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
