"""
Tests for oracle_stop_step_features.py (Analysis 0a).
Run from lmcos/: pytest tests/test_oracle_stop_step_features.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.board_tree_features import extract_board_features, extract_tree_features
from src.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    predicted_stop_from_advantages,
)

_CONFIG = BudgetedOracleConfig()


class TestComputeBudgetedOracleStopStep:
    def _run(self, halt_rewards, budget=20):
        tree_sizes = [1] * len(halt_rewards)
        return compute_budgeted_oracle(halt_rewards, tree_sizes, budget, _CONFIG).optimal_stop_step

    def test_in_range(self):
        rewards = list(np.linspace(0.3, 0.7, 20))
        step = self._run(rewards, budget=20)
        assert 0 <= step <= 20

    def test_constant_rewards_halt_immediately(self):
        rewards = [0.5] * 30
        step = self._run(rewards, budget=30)
        assert step == 0

    def test_monotone_increasing_low_cost(self):
        config = BudgetedOracleConfig(time_lambda=0.001)
        rewards = list(np.linspace(0.1, 0.9, 60))
        tree_sizes = [1] * 60
        step = compute_budgeted_oracle(rewards, tree_sizes, 60, config).optimal_stop_step
        assert step > 40, f"Expected stop near budget, got {step}"

    def test_high_cost_always_halt(self):
        config = BudgetedOracleConfig(time_lambda=1000.0)
        rewards = list(np.linspace(0.3, 0.9, 50))
        tree_sizes = [1] * 50
        step = compute_budgeted_oracle(rewards, tree_sizes, 50, config).optimal_stop_step
        assert step == 0, f"High cost should force immediate halt, got {step}"

    def test_deterministic(self):
        rewards = list(np.random.default_rng(0).uniform(0.3, 0.7, 40))
        s1 = self._run(rewards, budget=40)
        s2 = self._run(rewards, budget=40)
        assert s1 == s2

    def test_budget_truncation(self):
        rewards = list(np.linspace(0, 1, 96))
        for budget in [10, 30, 60]:
            step = self._run(rewards[:budget], budget=budget)
            assert step <= budget


class TestPredictedStopFromAdvantages:
    def test_first_nonpositive_step(self):
        assert predicted_stop_from_advantages([0.5, 0.3, -0.1, -0.5]) == 2

    def test_never_crosses_uses_last_step(self):
        assert predicted_stop_from_advantages([0.5, 0.3, 0.1]) == 2


class TestExtractBoardFeatures:
    def test_starting_position(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        feats = extract_board_features(fen)
        assert feats["n_possible_moves"] == 20
        assert feats["n_self_pieces_exc_pawns"] == 8

    def test_endgame_position(self):
        fen = "8/8/8/3k4/8/3K4/8/8 w - - 0 1"
        feats = extract_board_features(fen)
        assert feats["n_self_pieces_exc_pawns"] == 1
        assert feats["n_possible_moves"] > 0

    def test_returns_expected_keys(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        feats = extract_board_features(fen)
        for key in ("n_possible_moves", "n_self_pieces_exc_pawns", "move_ply"):
            assert key in feats

    def test_move_ply_white_first_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        feats = extract_board_features(fen)
        assert feats["move_ply"] == 0


class TestExtractTreeFeatures:
    @pytest.fixture
    def example_tree(self):
        path = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined/filtered_shard_00000/000000_root_0.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_toptwo_nonnegative(self, example_tree):
        feats = extract_tree_features(example_tree)
        if feats.get("toptwo_equiv") is not None:
            assert feats["toptwo_equiv"] >= 0

    def test_gain_depth_finite(self, example_tree):
        feats = extract_tree_features(example_tree)
        if feats.get("gain_depth_equiv") is not None:
            assert np.isfinite(feats["gain_depth_equiv"])

    def test_returns_expected_keys(self, example_tree):
        feats = extract_tree_features(example_tree)
        for key in ("toptwo_equiv", "gain_depth_equiv"):
            assert key in feats
