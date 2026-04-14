import unittest
from unittest.mock import patch

import uci_provider
from tree import ExpansionChild
from uci_provider import (
    Lc0DirectEvalProvider,
    Lc0AnalysisParser,
    Lc0NoSearchAnalysisParser,
    StockfishAnalysisParser,
    UciAnalysis,
    UciTreeExpansionProvider,
    parse_root_value_from_lines,
    append_move_to_position_spec,
    position_spec_to_uci_command,
)


class FakeEngineProcess:
    def __init__(self, lines):
        self.lines = list(lines)
        self.calls = []

    def analyse(self, fen):
        self.calls.append(fen)
        return list(self.lines)


class MappingEngineProcess:
    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.calls = []

    def analyse(self, fen):
        self.calls.append(fen)
        return list(self.mapping[fen])


class UciProviderTests(unittest.TestCase):
    def test_position_spec_helpers(self):
        position_spec = append_move_to_position_spec("root-fen", "e2e4")
        position_spec = append_move_to_position_spec(position_spec, "e7e5")

        self.assertEqual(position_spec, "root-fen ||moves|| e2e4 e7e5")
        self.assertEqual(
            position_spec_to_uci_command(position_spec),
            "position fen root-fen moves e2e4 e7e5",
        )

    def test_stockfish_parser_extracts_root_and_children(self):
        lines = [
            "info depth 12 seldepth 18 multipv 1 score cp 34 pv e2e4 e7e5",
            "info depth 12 seldepth 18 multipv 2 score cp 12 pv d2d4 d7d5",
            "bestmove e2e4 ponder e7e5",
        ]
        parser = StockfishAnalysisParser()

        analysis = parser.parse(lines, "fen")

        self.assertIsInstance(analysis, UciAnalysis)
        self.assertAlmostEqual(analysis.root_value, 0.034)
        self.assertEqual([child.move_uci for child in analysis.children], ["e2e4", "d2d4"])
        self.assertEqual(analysis.children[0].fen, "fen ||moves|| e2e4")

    def test_lc0_parser_extracts_priors_and_q_values(self):
        lines = [
            "info depth 8 score cp 21 pv e2e4 e7e5",
            "info string e2e4 P: 0.62 Q: 0.44 N: 100",
            "info string d2d4 P: 0.38 Q: 0.18 N: 60",
            "bestmove e2e4 ponder e7e5",
        ]
        parser = Lc0AnalysisParser()

        analysis = parser.parse(lines, "fen")

        self.assertAlmostEqual(analysis.root_value, 0.021)
        self.assertEqual([child.move_uci for child in analysis.children], ["e2e4", "d2d4"])
        self.assertAlmostEqual(analysis.children[0].scalar_features["prior"], 0.62)
        self.assertAlmostEqual(analysis.children[0].scalar_features["value"], 0.44)

    def test_provider_caches_fen_analyses(self):
        lines = [
            "info depth 8 score cp 21 pv e2e4 e7e5",
            "info string e2e4 P: 0.62 Q: 0.44 N: 100",
            "info string d2d4 P: 0.38 Q: 0.18 N: 60",
            "bestmove e2e4 ponder e7e5",
        ]
        engine = FakeEngineProcess(lines)
        provider = UciTreeExpansionProvider(engine, Lc0AnalysisParser(), metadata={"engine": "lc0"})

        root_features = provider.root_features("fen")
        children = provider.expand_node("fen", 0)

        self.assertEqual(engine.calls, ["fen"])
        self.assertAlmostEqual(root_features["value"], 0.021)
        self.assertEqual([child.move_uci for child in children], ["e2e4", "d2d4"])
        self.assertEqual(provider.provider_metadata()["engine"], "lc0")

    def test_lc0_no_search_parser_extracts_only_priors(self):
        lines = [
            "info depth 1 score cp 6 pv f2f4",
            "info string f2f4  (351 ) N:       0 (+ 0) (P: 13.94%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
            "info string g2g4  (378 ) N:       0 (+ 0) (P: 12.79%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
            "bestmove f2f4",
        ]
        parser = Lc0NoSearchAnalysisParser()

        analysis = parser.parse(lines, "fen")

        self.assertAlmostEqual(analysis.root_value, 0.006)
        self.assertEqual([child.move_uci for child in analysis.children], ["f2f4", "g2g4"])
        self.assertAlmostEqual(analysis.children[0].scalar_features["prior"], 0.1394)
        self.assertNotIn("value", analysis.children[0].scalar_features)

    def test_parse_root_value_from_lines(self):
        lines = [
            "info depth 1 seldepth 1 nodes 46 score cp 6",
            "bestmove f2f4",
        ]
        self.assertAlmostEqual(parse_root_value_from_lines(lines), 0.006)

    def test_lc0_direct_eval_provider_uses_prior_and_value_engines(self):
        prior_engine = MappingEngineProcess(
            {
                "fen": [
                    "info depth 1 score cp 6 pv f2f4",
                    "info string f2f4  (351 ) N:       0 (+ 0) (P: 13.94%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
                    "info string g2g4  (378 ) N:       0 (+ 0) (P: 12.79%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
                    "bestmove f2f4",
                ]
            }
        )
        value_engine = MappingEngineProcess(
            {
                "fen": ["info depth 1 seldepth 1 nodes 46 score cp 6", "bestmove f2f4"],
                "fen ||moves|| f2f4": ["info depth 1 seldepth 1 nodes 46 score cp -43", "bestmove e7e5"],
                "fen ||moves|| g2g4": ["info depth 1 seldepth 1 nodes 46 score cp 12", "bestmove e7e5"],
            }
        )
        provider = Lc0DirectEvalProvider(
            prior_engine,
            value_engine,
            metadata={"engine": "lc0"},
        )

        root_features = provider.root_features("fen")
        children = provider.expand_node("fen", 0)

        self.assertAlmostEqual(root_features["value"], 0.006)
        self.assertEqual([child.move_uci for child in children], ["f2f4", "g2g4"])
        self.assertAlmostEqual(children[0].scalar_features["prior"], 0.1394)
        self.assertAlmostEqual(children[0].scalar_features["value"], -0.043)
        self.assertEqual(prior_engine.calls, ["fen"])
        self.assertEqual(value_engine.calls, ["fen", "fen ||moves|| f2f4", "fen ||moves|| g2g4"])
        self.assertEqual(provider.provider_metadata()["engine"], "lc0")

    def test_lc0_direct_eval_provider_skips_value_engine_for_terminal_children(self):
        prior_engine = MappingEngineProcess(
            {
                "fen": [
                    "info depth 1 score cp 6 pv f2f4",
                    "info string f2f4  (351 ) N:       0 (+ 0) (P: 13.94%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
                    "info string g2g4  (378 ) N:       0 (+ 0) (P: 12.79%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
                    "bestmove f2f4",
                ]
            }
        )
        value_engine = MappingEngineProcess(
            {
                "fen": ["info depth 1 seldepth 1 nodes 46 score cp 6", "bestmove f2f4"],
                "fen ||moves|| g2g4": ["info depth 1 seldepth 1 nodes 46 score cp 12", "bestmove e7e5"],
            }
        )
        provider = Lc0DirectEvalProvider(prior_engine, value_engine)

        with patch.object(
            uci_provider,
            "terminal_value_from_position_spec",
            side_effect=lambda fen: -1.0 if fen == "fen ||moves|| f2f4" else None,
        ):
            children = provider.expand_node("fen", 0)

        self.assertEqual([child.move_uci for child in children], ["f2f4", "g2g4"])
        self.assertTrue(children[0].is_terminal)
        self.assertAlmostEqual(children[0].scalar_features["value"], -1.0)
        self.assertAlmostEqual(children[1].scalar_features["value"], 0.012)
        self.assertEqual(value_engine.calls, ["fen ||moves|| g2g4"])

    def test_lc0_direct_eval_provider_caches_terminal_checks(self):
        prior_engine = MappingEngineProcess(
            {
                "fen": [
                    "info depth 1 score cp 6 pv f2f4",
                    "info string f2f4  (351 ) N:       0 (+ 0) (P: 13.94%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q: -0.28929)",
                    "bestmove f2f4",
                ]
            }
        )
        value_engine = MappingEngineProcess(
            {
                "fen": ["info depth 1 seldepth 1 nodes 46 score cp 6", "bestmove f2f4"],
            }
        )
        provider = Lc0DirectEvalProvider(prior_engine, value_engine)

        with patch.object(
            uci_provider,
            "terminal_value_from_position_spec",
            side_effect=lambda fen: None,
        ) as mocked_terminal_check:
            provider.root_features("fen")
            provider.root_features("fen")

        self.assertEqual(mocked_terminal_check.call_count, 1)

    def test_lc0_direct_eval_provider_uses_parent_board_for_child_terminal_check(self):
        provider = Lc0DirectEvalProvider(MappingEngineProcess({}), MappingEngineProcess({}))
        child = ExpansionChild("e2e4", "fen ||moves|| e2e4", {"prior": 0.5})

        class FakeBoard:
            def copy(self, stack=False):
                return self

            def push_uci(self, move):
                self.move = move

        parent_board = FakeBoard()

        with patch.object(uci_provider, "chess", object()), patch.object(
            uci_provider,
            "terminal_value_from_board",
            return_value=-1.0,
        ) as mocked_terminal_from_board, patch.object(
            uci_provider,
            "terminal_value_from_position_spec",
            side_effect=AssertionError("should not reparse full child spec"),
        ):
            terminal_value = provider._terminal_value_for_child(child, parent_board)

        self.assertEqual(terminal_value, -1.0)
        self.assertEqual(provider._terminal_cache[child.fen], -1.0)
        self.assertEqual(parent_board.move, "e2e4")
        self.assertEqual(mocked_terminal_from_board.call_count, 1)


if __name__ == "__main__":
    unittest.main()
