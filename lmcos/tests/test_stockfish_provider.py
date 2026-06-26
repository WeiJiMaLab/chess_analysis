import unittest
import os
import chess
import chess.engine
from cts.core.providers.process import UciEngineConfig, UciEngineProcess
from cts.core.providers.stockfish import StockfishDirectEvalProvider
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    TeacherSearchConfig,
    generate_partial_tree_from_provider
)

class StockfishProviderTests(unittest.TestCase):
    def setUp(self):
        # Locate the stockfish binary
        stockfish_home = os.path.expanduser("~/stockfish")
        self.engine_path = os.path.join(stockfish_home, "src", "stockfish")
        
        # Base config for Stockfish
        self.config = UciEngineConfig(
            engine_path=self.engine_path,
            engine_kind="stockfish",
            movetime_ms=0,
            multipv=1,
            depth=None,
            nodes=10,
        )
        self.process = UciEngineProcess(self.config)
        self.process.start()

    def tearDown(self):
        self.process.close()

    def test_provider_initialization_and_metadata(self):
        provider = StockfishDirectEvalProvider(self.process, elo=1800, metadata={"tag": "test"})
        meta = provider.provider_metadata()
        self.assertEqual(meta["engine_kind"], "stockfish")
        self.assertEqual(meta["elo"], "1800")
        self.assertEqual(meta["tag"], "test")

    def test_root_features_extraction(self):
        provider = StockfishDirectEvalProvider(self.process)
        # Starting position FEN
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = provider.root_features(fen)
        
        self.assertIn("value", features)
        self.assertIn("wdl_win", features)
        self.assertIn("wdl_draw", features)
        self.assertIn("wdl_loss", features)
        self.assertIn("wdl_var", features)
        self.assertAlmostEqual(features["prior"], 1.0)
        # The WDL values must sum to 1.0
        self.assertAlmostEqual(features["wdl_win"] + features["wdl_draw"] + features["wdl_loss"], 1.0)

    def test_node_expansion(self):
        provider = StockfishDirectEvalProvider(self.process)
        # Starting position FEN
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        children = provider.expand_node(fen, depth=0)
        
        # Starting position has exactly 20 legal moves
        self.assertEqual(len(children), 20)
        for child in children:
            self.assertIsNotNone(child.move_uci)
            self.assertIsNotNone(child.fen)
            self.assertIn("prior", child.scalar_features)
            # Uniform priors: 1/20 = 0.05
            self.assertAlmostEqual(child.scalar_features["prior"], 0.05)
            self.assertIn("value", child.scalar_features)
            self.assertFalse(child.is_terminal)

    def test_mcts_tree_rollout_compatibility(self):
        provider = StockfishDirectEvalProvider(self.process, elo=1600)
        root_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        node_budget = NodeBudgetDistribution(5, 5)
        search_config = TeacherSearchConfig(
            max_depth=3,
            search_budget=5,
            c_puct=1.0,
            prior_feature="prior",
            value_feature="value",
            target_normalization_version="v1",
            search_config_id="test"
        )
        
        # Running the actual MCTS rollout using the Stockfish provider
        result = generate_partial_tree_from_provider(
            root_fen,
            provider,
            search_config,
            node_budget
        )
        
        # Verify MCTS tree was successfully built
        self.assertEqual(result.num_expansions, 5)
        self.assertGreater(result.tree.num_nodes(), 5)
        self.assertTrue(result.tree.root_id == 0)

if __name__ == "__main__":
    unittest.main()
