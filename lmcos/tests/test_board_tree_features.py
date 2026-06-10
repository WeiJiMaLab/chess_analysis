"""
Tests for board_tree_features.py.

All tree-feature tests use synthetic stub payloads (no disk access).
Board-feature tests use FEN strings only.

Run from lmcos/: pytest tests/test_board_tree_features.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.board_tree_features import extract_board_features, extract_tree_features


# ---------------------------------------------------------------------------
# Stub tree helpers
# ---------------------------------------------------------------------------

def _stub(q_final: list[float], best_idx: list[int]) -> dict:
    """Minimal fake payload for extract_tree_features — no disk access required."""
    qf = torch.tensor(q_final, dtype=torch.float32)
    return {
        "oracle_final_root_q_values": qf,
        "oracle_best_move_index": torch.tensor(best_idx, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# extract_tree_features — toptwo_equiv
# ---------------------------------------------------------------------------

class TestTopTwo:
    def test_two_children_clear_gap(self):
        t = _stub([0.9, 0.1], [0, 0, 0])
        assert abs(extract_tree_features(t)["toptwo_equiv"] - 0.8) < 1e-5

    def test_two_equal_children(self):
        t = _stub([0.5, 0.5], [0, 0, 0])
        assert extract_tree_features(t)["toptwo_equiv"] == pytest.approx(0.0, abs=1e-5)

    def test_three_children_zero_excluded(self):
        # Only two nonzero: 0.9 and 0.1; zero child is skipped
        t = _stub([0.9, 0.0, 0.1], [0, 0, 0])
        assert abs(extract_tree_features(t)["toptwo_equiv"] - 0.8) < 1e-5

    def test_single_nonzero_returns_nan(self):
        # Only 1 child has been evaluated; gap is undefined
        t = _stub([0.7, 0.0, 0.0], [0, 0, 0])
        assert math.isnan(extract_tree_features(t)["toptwo_equiv"])

    def test_all_zero_returns_nan(self):
        t = _stub([0.0, 0.0, 0.0], [0, 0, 0])
        assert math.isnan(extract_tree_features(t)["toptwo_equiv"])

    def test_toptwo_always_nonneg_when_defined(self):
        # topk is descending so v[0] >= v[1] — difference is always >= 0
        cases = [
            [0.3, 0.8],
            [0.8, 0.3],
            [0.5, 0.5, 0.5],
            [0.1, 0.9, 0.4],
        ]
        for qf in cases:
            t = _stub(qf, [0] * 3)
            val = extract_tree_features(t)["toptwo_equiv"]
            assert val >= 0, f"toptwo negative for q_final={qf}: {val}"

    def test_toptwo_uses_magnitude_ordering(self):
        # The two largest nonzero values are 0.8 and 0.4; 0.3 is third
        t = _stub([0.3, 0.8, 0.4], [0, 0, 0])
        assert abs(extract_tree_features(t)["toptwo_equiv"] - 0.4) < 1e-5


# ---------------------------------------------------------------------------
# extract_tree_features — gain_depth_equiv
# ---------------------------------------------------------------------------

class TestGainDepth:
    def test_stable_recommendation_gain_is_zero(self):
        # best_idx never changes from child 2 → deep == shallow, gain = 0
        t = _stub([0.1, 0.5, 0.9], [0, 2, 2, 2, 2])
        assert extract_tree_features(t)["gain_depth_equiv"] == pytest.approx(0.0, abs=1e-5)

    def test_gain_positive_when_deep_better(self):
        # step-1 recommends child 1 (q=0.5); final recommends child 2 (q=0.9)
        # gain = q_final[2] - q_final[1] = 0.4
        t = _stub([0.1, 0.5, 0.9], [0, 1, 2, 2])
        assert extract_tree_features(t)["gain_depth_equiv"] == pytest.approx(0.4, abs=1e-5)

    def test_gain_negative_when_shallow_accidentally_better(self):
        # step-1 recommends child 2 (q=0.9); final recommends child 1 (q=0.5)
        # gain = q_final[1] - q_final[2] = -0.4
        t = _stub([0.1, 0.5, 0.9], [0, 2, 1, 1])
        assert extract_tree_features(t)["gain_depth_equiv"] == pytest.approx(-0.4, abs=1e-5)

    def test_gain_depth_single_step_returns_nan(self):
        # len(best_idx) < 2 — no step-1 recommendation exists
        t = _stub([0.7], [0])
        assert math.isnan(extract_tree_features(t)["gain_depth_equiv"])

    def test_gain_depth_two_steps_uses_step1(self):
        # best_idx = [0, 1] → shallow=1, deep=1, gain = q[1] - q[1] = 0
        t = _stub([0.4, 0.8], [0, 1])
        assert extract_tree_features(t)["gain_depth_equiv"] == pytest.approx(0.0, abs=1e-5)

    def test_gain_depth_line_tree(self):
        # Line tree: only one child ever (index 0). Step 1 and final both recommend 0.
        t = _stub([0.6, 0.0, 0.0], [0, 0, 0, 0, 0])
        assert extract_tree_features(t)["gain_depth_equiv"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# extract_tree_features — keys and types
# ---------------------------------------------------------------------------

class TestTreeFeatureKeys:
    def test_returns_expected_keys(self):
        t = _stub([0.9, 0.1], [0, 0, 0])
        feats = extract_tree_features(t)
        assert "toptwo_equiv" in feats
        assert "gain_depth_equiv" in feats

    def test_values_are_floats(self):
        t = _stub([0.9, 0.1], [0, 0, 1])
        feats = extract_tree_features(t)
        assert isinstance(feats["toptwo_equiv"], float)
        assert isinstance(feats["gain_depth_equiv"], float)


# ---------------------------------------------------------------------------
# extract_board_features
# ---------------------------------------------------------------------------

class TestBoardFeatures:
    # ---- piece counts ----

    def test_starting_position_white_pieces(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        f = extract_board_features(fen)
        # King, Queen, 2 Rooks, 2 Bishops, 2 Knights = 8 (pawns excluded)
        assert f["n_self_pieces_exc_pawns"] == 8

    def test_starting_position_move_count(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        f = extract_board_features(fen)
        assert f["n_possible_moves"] == 20  # 16 pawn + 4 knight moves

    def test_black_to_move_counts_black_pieces(self):
        # Black King + Pawn; white King only
        # n_self_pieces_exc_pawns should count black non-pawn pieces (just the king)
        fen = "8/8/8/3k4/3p4/3K4/8/8 b - - 0 1"
        f = extract_board_features(fen)
        assert f["n_self_pieces_exc_pawns"] == 1  # black king only (pawn excluded)

    def test_white_to_move_counts_white_pieces(self):
        # Same position, white to move — counts white king only
        fen = "8/8/8/3k4/3p4/3K4/8/8 w - - 0 1"
        f = extract_board_features(fen)
        assert f["n_self_pieces_exc_pawns"] == 1  # white king only

    def test_full_white_army_without_pawns(self):
        # Position with all white pieces but no pawns
        # 1K + 1Q + 2R + 2B + 2N = 8 (no pawns on board)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/8/RNBQKBNR w KQ - 0 1"
        f = extract_board_features(fen)
        assert f["n_self_pieces_exc_pawns"] == 8

    # ---- move_ply formula ----

    def test_move_ply_white_move_1(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert extract_board_features(fen)["move_ply"] == 0

    def test_move_ply_black_move_1(self):
        # After 1.e4: fullmove=1, black to move → ply = 1
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        assert extract_board_features(fen)["move_ply"] == 1

    def test_move_ply_white_move_2(self):
        # fullmove=2, white → ply = 2
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        assert extract_board_features(fen)["move_ply"] == 2

    def test_move_ply_black_move_10(self):
        # fullmove=10, black → ply = (10-1)*2 + 1 = 19
        fen = "8/8/8/3k4/8/3K4/8/8 b - - 0 10"
        assert extract_board_features(fen)["move_ply"] == 19

    # ---- legal moves ----

    def test_legal_moves_after_e4(self):
        # After 1.e4 it's black's turn; black has the symmetric 20 initial moves
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        f = extract_board_features(fen)
        assert f["n_possible_moves"] == 20

    def test_legal_moves_endgame_nonzero(self):
        fen = "8/8/8/3k4/8/3K4/8/8 w - - 0 1"
        assert extract_board_features(fen)["n_possible_moves"] > 0

    # ---- keys ----

    def test_returns_expected_keys(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        f = extract_board_features(fen)
        for key in ("n_possible_moves", "n_self_pieces_exc_pawns", "move_ply"):
            assert key in f
