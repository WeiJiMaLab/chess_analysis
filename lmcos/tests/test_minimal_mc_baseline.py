"""
Tests for minimal_mc_baseline.py (Analysis 0b).
Run from lmcos/: pytest tests/test_minimal_mc_baseline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.minimal_mc_baseline import (
    MinimalMC,
    compute_sign_accuracy,
    extract_snapshot_features,
)
from src.data.preprocess_mc.pack import build_compact_trajectory_from_payload


class TestMinimalMC:
    def test_output_shape(self):
        model = MinimalMC(input_dim=5, hidden_dim=64, hidden_layers=2)
        x = torch.randn(32, 5)
        out = model(x)
        assert out.shape == (32, 1), f"Expected (32,1), got {out.shape}"

    def test_parameter_count_small(self):
        model = MinimalMC(input_dim=5, hidden_dim=64, hidden_layers=2)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 50_000, f"Minimal MC has {n_params} params — unexpectedly large"

    def test_gradient_flows(self):
        model = MinimalMC(input_dim=5, hidden_dim=64, hidden_layers=2)
        x = torch.randn(16, 5, requires_grad=False)
        y = torch.randn(16, 1)
        loss = torch.nn.MSELoss()(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan_on_small_input(self):
        model = MinimalMC(input_dim=5, hidden_dim=16, hidden_layers=1)
        x = torch.randn(8, 5)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_deterministic_inference(self):
        model = MinimalMC(input_dim=5, hidden_dim=32, hidden_layers=2)
        model.eval()
        x = torch.randn(10, 5)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2)


class TestExtractSnapshotFeatures:
    @pytest.fixture
    def example_tree(self):
        path = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined/filtered_shard_00000/000000_root_0.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_output_shape(self, example_tree):
        X, y, stop = extract_snapshot_features(example_tree, budget=43)
        trajectory = build_compact_trajectory_from_payload(example_tree)
        assert trajectory is not None
        expected_rows = min(len(trajectory["halt_rewards"]), 43)
        assert X.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {X.shape[0]}"
        assert X.shape[1] == 4
        assert y.shape == (X.shape[0],)
        assert isinstance(stop, int)

    def test_no_nans(self, example_tree):
        X, y, _ = extract_snapshot_features(example_tree, budget=43)
        assert not np.isnan(X).any(), "NaN in features"
        assert not np.isnan(y).any(), "NaN in labels"

    def test_halt_rewards_use_pack_trajectory(self, example_tree):
        trajectory = build_compact_trajectory_from_payload(example_tree)
        assert trajectory is not None
        num_steps = min(len(trajectory["halt_rewards"]), 43)
        halt_slice = [float(v) for v in trajectory["halt_rewards"][:num_steps]]
        q_final = example_tree["oracle_final_root_q_values"]
        best_idx = example_tree["oracle_best_move_index"]
        root_rank = int(trajectory["first_decision_expansion_count"]) - 1
        for s in range(num_steps):
            expected = float(q_final[best_idx[root_rank + s].item()].item())
            assert halt_slice[s] == expected, f"Step {s}: halt_reward != Q_final[best_idx]"

    def test_best_q_feature_matches_trajectory_halt_rewards(self, example_tree):
        X, _, _ = extract_snapshot_features(example_tree, budget=43)
        trajectory = build_compact_trajectory_from_payload(example_tree)
        num_steps = min(len(trajectory["halt_rewards"]), 43)
        halt_slice = [float(v) for v in trajectory["halt_rewards"][:num_steps]]
        np.testing.assert_allclose(X[:, 0], halt_slice, rtol=0, atol=1e-6)

    def test_best_q_in_unit_interval(self, example_tree):
        X, _, _ = extract_snapshot_features(example_tree, budget=43)
        best_q_col = X[:, 0]
        assert np.all((best_q_col >= 0) & (best_q_col <= 1 + 1e-6)), "best_q outside [0,1]"


class TestComputeSignAccuracy:
    def test_perfect(self):
        y = np.array([1.0, -1.0, 1.0])
        preds = np.array([0.5, -0.5, 0.1])
        assert compute_sign_accuracy(y, preds) == 1.0

    def test_chance(self):
        y = np.array([0.5, -0.5, 0.5, -0.5])
        preds = np.array([0.5, 0.5, -0.5, -0.5])
        assert compute_sign_accuracy(y, preds) == 0.5


# ---------------------------------------------------------------------------
# Stub payload helpers (no disk access)
# ---------------------------------------------------------------------------

def _stub_payload(q_final: list[float], best_idx: list[int]) -> dict:
    """Minimal fake .pt payload for extract_snapshot_features — no scratch disk needed.

    q_trace rows: all zeros except the last row which equals q_final.
    This is the simplest valid shape; wdl_var will be 0 at all steps except the last.
    """
    T = len(best_idx)
    n_children = len(q_final)
    q_trace = torch.zeros(T, n_children, dtype=torch.float32)
    q_trace[-1] = torch.tensor(q_final, dtype=torch.float32)
    return {
        "oracle_root_q_trace": q_trace,
        "oracle_best_move_index": torch.tensor(best_idx, dtype=torch.long),
        "oracle_final_root_q_values": torch.tensor(q_final, dtype=torch.float32),
    }


class TestExtractSnapshotFeaturesStub:
    """Pure-stub tests for extract_snapshot_features — no scratch disk access."""

    def test_output_shapes(self):
        t = _stub_payload([0.8, 0.3, 0.1], [0, 0, 1, 0, 0])
        X, y, stop = extract_snapshot_features(t, budget=5)
        assert X.shape == (5, 4), f"Expected (5,4), got {X.shape}"
        assert y.shape == (5,)
        assert isinstance(stop, int)

    def test_budget_truncates_steps(self):
        # T=10 steps but budget=4 → only 4 rows
        t = _stub_payload([0.8, 0.2], [0] * 10)
        X, y, stop = extract_snapshot_features(t, budget=4)
        assert X.shape[0] == 4

    def test_t_norm_in_unit_interval(self):
        t = _stub_payload([0.8, 0.2], [0] * 6)
        X, _, _ = extract_snapshot_features(t, budget=6)
        assert np.all(X[:, 2] >= 0) and np.all(X[:, 2] <= 1)

    def test_remaining_budget_norm_in_half_open_unit(self):
        t = _stub_payload([0.8, 0.2], [0] * 6)
        X, _, _ = extract_snapshot_features(t, budget=6)
        # (num_steps - s) / num_steps is in (0, 1] — never zero, at most 1
        assert np.all(X[:, 3] > 0) and np.all(X[:, 3] <= 1)

    def test_wdl_var_nonneg(self):
        t = _stub_payload([0.8, 0.3, 0.1], [0, 0, 1, 0])
        X, _, _ = extract_snapshot_features(t, budget=4)
        assert np.all(X[:, 1] >= 0)

    def test_best_q_uses_final_q_values(self):
        # halt_rewards[s] = q_final[best_idx[s]], so best_q column must equal that
        q_final = [0.9, 0.2, 0.4]
        best_idx = [0, 2, 1, 0, 0]
        t = _stub_payload(q_final, best_idx)
        X, _, _ = extract_snapshot_features(t, budget=5)
        expected = [q_final[i] for i in best_idx]
        np.testing.assert_allclose(X[:, 0], expected, atol=1e-5)

    def test_oracle_stop_step_in_valid_range(self):
        t = _stub_payload([0.8, 0.2], [0] * 8)
        _, _, stop = extract_snapshot_features(t, budget=8)
        assert 0 <= stop <= 7

    def test_no_nans_in_output(self):
        t = _stub_payload([0.7, 0.3], [0, 1, 0, 1, 0])
        X, y, _ = extract_snapshot_features(t, budget=5)
        assert not np.isnan(X).any(), "NaN in features"
        assert not np.isnan(y).any(), "NaN in labels"

    def test_single_child_line_tree(self):
        # Line tree: only one child. wdl_var = 0 at all steps, best_q constant.
        t = _stub_payload([0.6], [0, 0, 0, 0])
        X, y, stop = extract_snapshot_features(t, budget=4)
        assert X.shape == (4, 4)
        assert np.all(X[:, 1] == 0.0), "wdl_var should be 0 for single-child tree"
        assert np.all(X[:, 0] == pytest.approx(0.6)), "best_q should always be q_final[0]"

    def test_constant_halt_rewards_stop_immediately(self):
        # If best_q is constant across all steps, the oracle should halt at step 0
        # (no value in waiting since halt_reward never improves).
        t = _stub_payload([0.5, 0.5], [0] * 10)
        _, _, stop = extract_snapshot_features(t, budget=10)
        assert stop == 0, f"Constant rewards should yield stop=0, got {stop}"
