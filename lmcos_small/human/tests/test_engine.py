"""
Tests for engine-level analyses in lmcos_small/human.
Consolidates integration tests on real saved trees and mathematical unit tests on mock trees.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import pytest

_HA = Path(__file__).resolve().parent.parent
if str(_HA) not in sys.path:
    sys.path.insert(0, str(_HA))

from utils.helpers import CONFIG
from utils.tree_loader import _tree_voc_and_gap, _tree_root_mq, _tree_gss, _tree_hpi

# Check if the saved trees directory exists and contains any .pt files
TREES_DIR = CONFIG["trees_default"]
HAS_SAVED_TREES = os.path.exists(TREES_DIR) and any(
    f.endswith(".pt") for f in os.listdir(TREES_DIR)
) if os.path.exists(TREES_DIR) else False


@unittest.skipUnless(
    HAS_SAVED_TREES,
    f"No saved Leela search trees (.pt) found in {TREES_DIR}",
)
class TestSavedTreeAnalysis(unittest.TestCase):
    """Integration tests based on real saved Leela MCTS search trees."""

    def setUp(self) -> None:
        # Load the first available saved tree
        self.tree_files = [
            os.path.join(TREES_DIR, f)
            for f in os.listdir(TREES_DIR)
            if f.endswith(".pt")
        ]
        self.assertTrue(len(self.tree_files) > 0)
        self.payload = torch.load(self.tree_files[0], map_location="cpu", weights_only=False)

    def test_tree_voc_and_gap(self) -> None:
        """Verify that tree-based VOC (Gain) and Action Gap are valid float/NaN values."""
        voc_val, gap = _tree_voc_and_gap(self.payload)
        
        if not (self.payload["parent_index"].numpy() == 0).sum() < 2:
            self.assertFalse(np.isnan(gap), "Action Gap should not be NaN for a valid tree")
            self.assertFalse(np.isnan(voc_val), "Tree-based VOC should not be NaN for a valid tree")
            self.assertGreaterEqual(voc_val, -1e-9, "Tree-based VOC must be non-negative")
            self.assertGreaterEqual(gap, -1e-9, "Action Gap must be non-negative")

    def test_tree_root_mq(self) -> None:
        """Verify that tree-based root MQ is correctly extracted and behaves as expected."""
        moves, mqs = _tree_root_mq(self.payload)
        
        self.assertEqual(len(moves), len(mqs))
        if moves:
            self.assertAlmostEqual(max(mqs), 0.0, places=6)
            for mq in mqs:
                self.assertLessEqual(mq, 1e-9, "Move Quality must be non-positive")

    def test_tree_gss(self) -> None:
        """Verify that Greedy Stopping Step is non-negative."""
        gss = _tree_gss(self.payload)
        self.assertGreaterEqual(gss, 0)

    def test_tree_hpi(self) -> None:
        """Verify that prior policy entropy H(pi) is valid and non-negative."""
        hpi = _tree_hpi(self.payload)
        if not np.isnan(hpi):
            self.assertGreaterEqual(hpi, 0.0)


# --- Mathematical Unit Tests on Mock Trees (Gain/VOC) ---
def _make_tree(*, child_values, final_q, q_trace, root_moves=None):
    """Minimal payload dict matching what _tree_voc_and_gap reads."""
    n_kids = len(child_values)
    node_features = torch.tensor([[0.0]] + [[v] for v in child_values], dtype=torch.float64)
    parent_index = torch.tensor([-1] + [0] * n_kids, dtype=torch.long)
    if root_moves is None:
        root_moves = [f"m{i}" for i in range(len(final_q))]
    return {
        "feature_names": ["value"],
        "node_features": node_features,
        "parent_index": parent_index,
        "oracle_final_root_q_values": np.asarray(final_q, dtype=float),
        "oracle_root_q_trace": np.asarray(q_trace, dtype=float),
        "oracle_root_moves": list(root_moves),
        "incoming_moves": {i + 1: f"m{i}" for i in range(n_kids)},
    }


def test_gain_shallow_is_visited():
    """Verify that Gain picks a_shallow from the visited moves in the q-trace."""
    child_values = [-1.0, -0.2, -0.9]
    final_q = [1.0, 0.1, 0.0]
    q_trace = [
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
    ]
    payload = _make_tree(child_values=child_values, final_q=final_q, q_trace=q_trace)
    voc, _ = _tree_voc_and_gap(payload)
    assert voc == pytest.approx(0.0, abs=1e-9)
    assert voc < 0.95


def test_gain_nonnegative():
    """Gain >= 0 for any well-formed growing tree."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = rng.integers(2, 8)
        final_q = rng.uniform(-1, 1, size=k)
        q_trace = np.zeros((k, k))
        order = rng.permutation(k)
        for step, mv in enumerate(order):
            q_trace[step:, mv] = final_q[mv] if final_q[mv] != 0 else 0.01
        child_values = (-final_q).tolist()
        payload = _make_tree(child_values=child_values, final_q=final_q.tolist(),
                             q_trace=q_trace.tolist())
        voc, _ = _tree_voc_and_gap(payload)
        assert not np.isnan(voc)
        assert voc >= -1e-9


def test_gain_matches_growing_tree_identity():
    """Gain == final_Q[argmax] - final_Q[first-expanded]."""
    final_q = [0.3, 0.8, -0.2, 0.5]
    q_trace = [
        [0.0, 0.0, -0.2, 0.0],
        [0.0, 0.8, -0.2, 0.0],
        [0.0, 0.8, -0.2, 0.5],
    ]
    child_values = [-v for v in final_q]
    payload = _make_tree(child_values=child_values, final_q=final_q, q_trace=q_trace)
    voc, _ = _tree_voc_and_gap(payload)
    a_deep = int(np.argmax(final_q))
    a_shallow = 2
    assert voc == pytest.approx(final_q[a_deep] - final_q[a_shallow], abs=1e-12)
    assert voc == pytest.approx(0.8 - (-0.2), abs=1e-12)


def test_gain_nan_on_degenerate_tree():
    """A root with < 2 children yields NaN, not a fabricated number."""
    payload = _make_tree(child_values=[-0.5], final_q=[0.5], q_trace=[[0.5]])
    voc, gap = _tree_voc_and_gap(payload)
    assert np.isnan(voc)
    assert np.isnan(gap)


if __name__ == "__main__":
    unittest.main()


# --- Unit and Integration Tests for Active Engine Evaluations ---
import chess
import chess.engine
from utils.engine_eval import _win_prob, evaluate_position, move_quality, voc
from utils.helpers import get_engine


class TestActiveEngineEval(unittest.TestCase):
    """Unit and integration tests for engine_eval.py functions on key board positions."""

    # --- Unit tests using chess.Board directly (game-over/terminal states) ---
    def test_win_prob_checkmate(self) -> None:
        """Verify win probability in checkmate is exactly 0.0 for the mated side."""
        # 1.f3 e5 2.g4 Qh4# (Fool's mate) - Black checkmates White. White is in checkmate.
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        board.push(chess.Move.from_uci("d8h4")) # Qh4#
        # White to move is in checkmate; win probability should be 0.0
        info = {}
        self.assertTrue(board.is_checkmate())
        self.assertEqual(_win_prob(info, board), 0.0)

    def test_win_prob_stalemate(self) -> None:
        """Verify win probability in stalemate is exactly 0.5."""
        # Valid stalemate FEN: White King on a1, Black King on c3, Black Queen on c2. White to move.
        board = chess.Board("k7/8/8/8/8/2k5/2q5/K7 w - - 0 1")
        info = {}
        self.assertTrue(board.is_game_over())
        self.assertTrue(board.is_stalemate())
        self.assertEqual(_win_prob(info, board), 0.5)

    def test_evaluate_position_checkmate(self) -> None:
        """Verify evaluate_position handles checkmate without engine queries."""
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        board.push(chess.Move.from_uci("d8h4"))
        res = evaluate_position(board, chess.Move.from_uci("g1f3"), engine=None)
        self.assertEqual(res.e_win_best, 0.0)
        self.assertEqual(res.e_win_second_best, 0.0)
        self.assertEqual(res.e_win_taken, 0.0)
        self.assertEqual(res.voc, 0.0)
        self.assertEqual(res.mq, 0.0)

    def test_evaluate_position_stalemate(self) -> None:
        """Verify evaluate_position handles stalemate without engine queries."""
        # Valid stalemate FEN
        board = chess.Board("k7/8/8/8/8/2k5/2q5/K7 w - - 0 1")
        res = evaluate_position(board, chess.Move.from_uci("a1a2"), engine=None)
        self.assertEqual(res.e_win_best, 0.5)
        self.assertEqual(res.e_win_second_best, 0.5)
        self.assertEqual(res.e_win_taken, 0.5)
        self.assertEqual(res.voc, 0.0)
        self.assertEqual(res.mq, 0.0)

    # --- Integration tests using Stockfish (if available) ---
    def test_opening_position_eval(self) -> None:
        """Verify that opening position evaluations yield balanced win probabilities (~0.5)."""
        try:
            engine = get_engine("stockfish", threads=1, hash_mb=32)
        except Exception:
            self.skipTest("Stockfish not available for active eval tests")
            
        try:
            board = chess.Board()
            move = chess.Move.from_uci("e2e4")
            res = evaluate_position(board, move, engine, depth_deep=5, depth_shallow=1)
            self.assertIsNotNone(res.e_win_best)
            self.assertIsNotNone(res.e_win_taken)
            self.assertIsNotNone(res.voc)
            # In opening, White's win prob is close to 0.5 (typically 0.51 - 0.55)
            self.assertTrue(0.4 <= res.e_win_best <= 0.65)
            self.assertTrue(res.voc >= 0.0)
            self.assertTrue(res.mq <= 0.0)
        finally:
            engine.close()

    def test_obvious_win_loss_eval(self) -> None:
        """Verify that a winning position yields a high win probability, and blunders yield negative MQ."""
        try:
            engine = get_engine("stockfish", threads=1, hash_mb=32)
        except Exception:
            self.skipTest("Stockfish not available for active eval tests")

        try:
            # White is completely winning (King + Queen vs King)
            board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
            best_move = chess.Move.from_uci("e2e7")
            res_win = evaluate_position(board, best_move, engine, depth_deep=5, depth_shallow=1)
            # White's win prob should be extremely high (close to 1.0)
            self.assertGreater(res_win.e_win_best, 0.85)
            
            # Blundering the King (e1e2) instead of playing the winning Queen move yields a negative MQ
            bad_move = chess.Move.from_uci("e1e2")
            mq_bad = move_quality(board, bad_move, engine, depth=5)
            self.assertLessEqual(mq_bad, 0.0)
        finally:
            engine.close()

