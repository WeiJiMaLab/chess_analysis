"""Tests ``resume_checkpoint`` and ``save_every_epoch`` in ``cts.train.controller_train``."""
from __future__ import annotations

from pathlib import Path

import torch

from cts.train.controller_train import ControllerTrainConfig, _build_model_and_optimizer, _feature_schema
from cts.train.gnn_pretrain import save_encoder_checkpoint


def _dummy_encoder_ckpt(tmp_path: Path, schema) -> str:
    path = tmp_path / "dummy_encoder.pt"
    if path.exists():
        return str(path)
    from cts.models.mc import MetaController

    m = MetaController(k=1, node_feat=len(schema.feature_names), device="cpu", node_embed_hidden=8,
                       d_embed=4, d_message=4, n_heads=1, d_att=4, hidden_dim=8, hidden_layers=1,
                       controller_inputs=["z_t", "T_t"])
    save_encoder_checkpoint(str(path), m.encoder)
    return str(path)


def test_resume_checkpoint_overwrites_fresh_head_weights(tmp_path):
    schema = _feature_schema()
    enc_ckpt = _dummy_encoder_ckpt(tmp_path, schema)

    # Train a "prior run": build a model, mutate its head weights away from init, save full state.
    config_a = ControllerTrainConfig(
        packed_train_data="unused", packed_validation_data="unused", encoder_checkpoint=enc_ckpt,
        k=1, node_embed_hidden=8, d_embed=4, d_message=4, n_heads=1, d_att=4,
        hidden_dim=8, hidden_layers=1, controller_inputs=["z_t", "T_t"], seed=1,
    )
    model_a, _, _ = _build_model_and_optimizer(config_a, schema)
    with torch.no_grad():
        for p in model_a.parameters():
            p.add_(1.0)  # guaranteed to differ from any fresh re-init
    resume_path = tmp_path / "resume_source.pt"
    torch.save({"model_state_dict": model_a.state_dict(), "metadata": {}}, resume_path)

    # Build a FRESH model with a different seed (so init would differ without resume), then resume.
    config_b = ControllerTrainConfig(
        packed_train_data="unused", packed_validation_data="unused", encoder_checkpoint=enc_ckpt,
        resume_checkpoint=str(resume_path),
        k=1, node_embed_hidden=8, d_embed=4, d_message=4, n_heads=1, d_att=4,
        hidden_dim=8, hidden_layers=1, controller_inputs=["z_t", "T_t"], seed=99,
    )
    model_b, _, _ = _build_model_and_optimizer(config_b, schema)

    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(p_a, p_b), f"resumed param {name_b} does not match the saved source"


def test_resume_checkpoint_respects_freeze_flag_after_loading(tmp_path):
    """resume_checkpoint loads BEFORE the freeze/unfreeze branch, so a frozen resume still ends
    up with requires_grad=False on the encoder (the load itself must not silently unfreeze it)."""
    schema = _feature_schema()
    enc_ckpt = _dummy_encoder_ckpt(tmp_path, schema)
    config_a = ControllerTrainConfig(
        packed_train_data="unused", packed_validation_data="unused", encoder_checkpoint=enc_ckpt,
        k=1, node_embed_hidden=8, d_embed=4, d_message=4, n_heads=1, d_att=4,
        hidden_dim=8, hidden_layers=1, controller_inputs=["z_t", "T_t"], unfreeze_encoder=True,
    )
    model_a, _, _ = _build_model_and_optimizer(config_a, schema)
    resume_path = tmp_path / "resume_source2.pt"
    torch.save({"model_state_dict": model_a.state_dict(), "metadata": {}}, resume_path)

    config_b = ControllerTrainConfig(
        packed_train_data="unused", packed_validation_data="unused", encoder_checkpoint=enc_ckpt,
        resume_checkpoint=str(resume_path),
        k=1, node_embed_hidden=8, d_embed=4, d_message=4, n_heads=1, d_att=4,
        hidden_dim=8, hidden_layers=1, controller_inputs=["z_t", "T_t"], unfreeze_encoder=False,
    )
    model_b, _, _ = _build_model_and_optimizer(config_b, schema)
    assert all(not p.requires_grad for p in model_b.encoder.parameters())
