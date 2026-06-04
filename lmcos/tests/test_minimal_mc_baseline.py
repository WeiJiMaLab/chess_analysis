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
    extract_snapshot_features,
    compute_sign_accuracy,
)

# ---------------------------------------------------------------------------
# MinimalMC architecture
# ---------------------------------------------------------------------------

class TestMinimalMC:
    def test_output_shape(self):
        model = MinimalMC(input_dim=5, hidden_dim=64, hidden_layers=2)
        x = torch.randn(32, 5)
        out = model(x)
        assert out.shape == (32, 1), f"Expected (32,1), got {out.shape}"

    def test_parameter_count_small(self):
        """Minimal MC should have far fewer params than GNN+MC baseline (~millions)."""
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


# ---------------------------------------------------------------------------
# extract_snapshot_features
# ---------------------------------------------------------------------------

class TestExtractSnapshotFeatures:
    @pytest.fixture
    def example_tree(self):
        path = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined/filtered_shard_00000/000000_root_0.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_output_shape(self, example_tree):
        X, y = extract_snapshot_features(example_tree, budget=43)
        T = len(example_tree["oracle_root_q_trace"])
        assert X.shape[0] == min(T, 43), f"Expected {min(T,43)} rows, got {X.shape[0]}"
        assert X.shape[1] >= 3, "Expected at least 3 features per snapshot"
        assert y.shape == (X.shape[0],)

    def test_no_nans(self, example_tree):
        X, y = extract_snapshot_features(example_tree, budget=43)
        assert not np.isnan(X).any(), "NaN in features"
        assert not np.isnan(y).any(), "NaN in labels"

    def test_labels_binary(self, example_tree):
        _, y = extract_snapshot_features(example_tree, budget=43)
        assert set(y).issubset({-1.0, 1.0}), f"Labels not binary: {set(y)}"

    def test_best_q_in_unit_interval(self, example_tree):
        X, _ = extract_snapshot_features(example_tree, budget=43)
        best_q_col = X[:, 0]  # first feature = best_q at each step
        assert np.all((best_q_col >= 0) & (best_q_col <= 1 + 1e-6)), "best_q outside [0,1]"


# ---------------------------------------------------------------------------
# compute_sign_accuracy
# ---------------------------------------------------------------------------

class TestComputeSignAccuracy:
    def test_perfect(self):
        y = np.array([1.0, -1.0, 1.0, -1.0])
        preds = np.array([0.5, -0.5, 0.5, -0.5])
        assert compute_sign_accuracy(y, preds) == 1.0

    def test_all_wrong(self):
        y = np.array([1.0, 1.0, 1.0])
        preds = np.array([-1.0, -1.0, -1.0])
        assert compute_sign_accuracy(y, preds) == 0.0

    def test_chance(self):
        y = np.array([1.0, -1.0, 1.0, -1.0])
        preds = np.array([1.0, 1.0, -1.0, -1.0])
        assert compute_sign_accuracy(y, preds) == 0.5
