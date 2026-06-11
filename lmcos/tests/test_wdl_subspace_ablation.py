"""Unit tests for the pure-math core of the WDL-subspace ablation.

These cover the parts that don't need cluster data: the projection algebra
(complementarity, idempotence, full-rank identity), the decoder readout-basis
orthonormality, and the head-dim inference that reconstructs the controller.
The integration check (baseline regret reproducing the controller's recorded
number) runs on the cluster against the real cache.
"""

import torch

from cts.analysis.wdl_subspace_ablation import (
    _apply_projection,
    _default_k_values,
    _infer_head_dims,
    _subspace_projection_matrix,
    _wdl_readout_basis,
)
from cts.models.gnn import ChildWdlHead
from cts.models.mc import MetaController


def _orthonormal_basis(d_embed: int, seed: int = 0) -> torch.Tensor:
    """Right-singular vectors of a random matrix: an orthonormal [d_embed, d_embed] basis."""
    torch.manual_seed(seed)
    matrix = torch.randn(2 * d_embed, d_embed)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=True)
    return vh.transpose(0, 1).contiguous()


def test_subspace_projection_is_symmetric_and_idempotent():
    basis = _orthonormal_basis(8)
    for k in (1, 3, 8):
        projection = _subspace_projection_matrix(basis, k)
        assert torch.allclose(projection, projection.transpose(0, 1), atol=1e-5)
        assert torch.allclose(projection @ projection, projection, atol=1e-5)


def test_keep_plus_ablate_reconstructs_z():
    basis = _orthonormal_basis(8, seed=1)
    z = torch.randn(5, 8)
    for k in (1, 4, 8):
        projection = _subspace_projection_matrix(basis, k)
        keep = _apply_projection(z, projection, "keep")
        ablate = _apply_projection(z, projection, "ablate")
        assert torch.allclose(keep + ablate, z, atol=1e-5)


def test_full_rank_keep_is_identity_and_ablate_is_zero():
    basis = _orthonormal_basis(8, seed=2)
    z = torch.randn(4, 8)
    projection = _subspace_projection_matrix(basis, 8)
    assert torch.allclose(_apply_projection(z, projection, "keep"), z, atol=1e-5)
    assert torch.allclose(_apply_projection(z, projection, "ablate"), torch.zeros_like(z), atol=1e-5)


def test_apply_projection_rejects_unknown_mode():
    basis = _orthonormal_basis(4)
    projection = _subspace_projection_matrix(basis, 2)
    try:
        _apply_projection(torch.randn(3, 4), projection, "nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown projection mode")


def test_wdl_readout_basis_orthonormal_and_descending():
    decoder = ChildWdlHead(d_embed=8, hidden_dim=16, device="cpu")
    basis, singular_values = _wdl_readout_basis(decoder, 8)
    assert basis.shape == (8, 8)
    assert torch.allclose(basis.transpose(0, 1) @ basis, torch.eye(8), atol=1e-5)
    assert torch.all(singular_values[:-1] >= singular_values[1:] - 1e-6)


def test_infer_head_dims_single_head():
    state_dict = {
        "advantage_head.0.weight": torch.zeros(256, 129),
        "advantage_head.2.weight": torch.zeros(256, 256),
        "advantage_head.4.weight": torch.zeros(256, 256),
        "advantage_head.6.weight": torch.zeros(1, 256),
    }
    assert _infer_head_dims(state_dict, separate_sign_head=False) == (256, 3)


def test_infer_head_dims_separate_head():
    state_dict = {
        "advantage_backbone.0.weight": torch.zeros(256, 129),
        "advantage_backbone.2.weight": torch.zeros(256, 256),
        "advantage_backbone.4.weight": torch.zeros(256, 256),
        "advantage_proj.weight": torch.zeros(1, 256),
        "sign_head.weight": torch.zeros(1, 256),
    }
    assert _infer_head_dims(state_dict, separate_sign_head=True) == (256, 3)


def test_infer_head_dims_matches_real_metacontroller():
    """Round-trip against a real MetaController and confirm z_t leads the head input."""
    model = MetaController(
        k=1,
        node_feat=5,
        device="cpu",
        node_embed_hidden=128,
        d_embed=128,
        d_message=128,
        n_heads=4,
        d_att=32,
        hidden_dim=256,
        hidden_layers=3,
        controller_inputs=["z_t", "T_t"],
    )
    assert _infer_head_dims(model.state_dict(), separate_sign_head=False) == (256, 3)
    # [z_t (128), T_t (1)] -> first Linear reads 129 inputs, z_t the first 128.
    assert tuple(model.advantage_head[0].weight.shape) == (256, 129)


def test_default_k_values_spans_one_to_d_embed():
    k_values = _default_k_values(128)
    assert k_values[0] == 1
    assert k_values[-1] == 128
    assert k_values == sorted(set(k_values))
    assert all(1 <= k <= 128 for k in k_values)
