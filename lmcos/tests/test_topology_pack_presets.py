"""Regression tests for topology pack presets + weighted topology loss."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from cts.core.providers.base import TreeExpansionProvider
from cts.core.schema import TREE_ENCODER_FEATURE_NAMES, tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeExample, collate_tensorized_examples_for_topology
from cts.core.tree import ExpansionChild
from cts.data.preprocess_gnn.pack import PackPretrainConfig, main as pack_main
from cts.data.preprocess_gnn.teacher_targets import TeacherSearchConfig, build_pretrain_example, save_pretrain_example
from cts.models.gnn import TopologyModel
from cts.train.gnn_pretrain import NodePretrainConfig, NodePretrainer


class _PackSmokeProvider(TreeExpansionProvider):
    def root_features(self, fen: str) -> dict[str, float]:
        return _wdl(0.35, 0.3, 0.35)

    def expand_node(self, fen: str, depth: int, max_children: int | None = None):
        children = {
            "root": [
                ExpansionChild("a", "a", _wdl(0.6, 0.2, 0.2, prior=0.7)),
                ExpansionChild("b", "b", _wdl(0.2, 0.4, 0.4, prior=0.3)),
            ],
            "a": [ExpansionChild("a1", "a1", _wdl(0.55, 0.2, 0.25, prior=1.0))],
            "b": [ExpansionChild("b1", "b1", _wdl(0.6, 0.1, 0.3, prior=1.0))],
        }.get(fen, [])
        return children if max_children is None else children[:max_children]

    def provider_metadata(self) -> dict[str, str]:
        return {"provider": "pack-smoke"}


def _wdl(win: float, draw: float, loss: float, prior: float = 1.0) -> dict[str, float]:
    total = win + draw + loss
    p_win, p_draw, p_loss = win / total, draw / total, loss / total
    value = p_win - p_loss
    return {
        "value": value,
        "prior": prior,
        "wdl_win": p_win,
        "wdl_draw": p_draw,
        "wdl_loss": p_loss,
        "wdl_var": (p_win + p_loss) - value * value,
    }


def _smoke_search_config() -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=2,
        search_budget=8,
        c_puct=1.0,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id="pack-smoke",
    )


def test_pack_config_rejects_wide_without_supervision_shard() -> None:
    with pytest.raises(ValueError):
        PackPretrainConfig(node_targets=True, topology_supervision_shard=False)


def test_pack_config_rejects_num_workers() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        PackPretrainConfig.model_validate({"num_workers": 8})


def test_pack_topology_smoke_reaches_second_train_shard() -> None:
    """Serial pack must finish shard 1 and write shard 2 (regression for per-shard pool hang)."""
    provider = _PackSmokeProvider()
    search_config = _smoke_search_config()
    train_count = 5
    shard_size = 2

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_dir = root / "raw"
        raw_dir.mkdir()
        split_root = root / "split"
        output_root = root / "packed"
        split_root.mkdir()

        train_paths: list[Path] = []
        for index in range(train_count):
            example = build_pretrain_example(
                "root",
                provider,
                search_config,
                root_position_id=f"smoke-{index}",
            )
            path = raw_dir / f"{index:06d}.pt"
            save_pretrain_example(str(path), example)
            train_paths.append(path)

        val_path = raw_dir / "val000.pt"
        save_pretrain_example(
            str(val_path),
            build_pretrain_example("root", provider, search_config, root_position_id="smoke-val"),
        )

        (split_root / "train_manifest.txt").write_text(
            "\n".join(str(p) for p in train_paths) + "\n",
            encoding="utf-8",
        )
        (split_root / "validation_manifest.txt").write_text(f"{val_path}\n", encoding="utf-8")

        pack_main(
            PackPretrainConfig(
                split_root=str(split_root),
                output_root=str(output_root),
                shard_size=shard_size,
                log_interval=1,
                clear=True,
                node_targets=True,
                topology_supervision_shard=True,
            )
        )

        train_dir = output_root / "train"
        assert (train_dir / "shard_00000.pt").is_file()
        assert (train_dir / "shard_00001.pt").is_file()  # regression: must pass shard 1 boundary
        assert (train_dir / "shard_00002.pt").is_file()

        manifest = output_root / "train_manifest.json"
        assert manifest.is_file()
        payload = manifest.read_text(encoding="utf-8")
        assert '"total_examples": 5' in payload
        assert (output_root / "validation" / "shard_00000.pt").is_file()


def test_pack_config_maps_include_topology_targets_yaml() -> None:
    cfg = PackPretrainConfig.model_validate({"include_topology_targets": True})
    assert cfg.topology_supervision_shard is True
    assert cfg.node_targets is False


def test_pack_config_maps_teacher_topology_yaml_to_full_template() -> None:
    cfg = PackPretrainConfig.model_validate({"include_teacher_topology": True})
    assert cfg.node_targets is True
    assert cfg.topology_supervision_shard is True


def test_collate_topology_derives_targets_from_wide_node_features() -> None:
    schema = tree_encoder_feature_schema(node_targets=True)
    tail = torch.tensor([[25.0, 0.0]], dtype=torch.float32)
    base = torch.zeros(1, len(TREE_ENCODER_FEATURE_NAMES), dtype=torch.float32)
    wide = torch.cat([base, tail], dim=-1)
    ex = TensorizedTreeExample(
        node_features=wide,
        parent_index=torch.tensor([-1], dtype=torch.long),
        edge_parent=torch.empty(0, dtype=torch.long),
        edge_child=torch.empty(0, dtype=torch.long),
        edge_slot=torch.empty(0, dtype=torch.long),
        depth=torch.zeros(1, dtype=torch.long),
        node_targets=torch.zeros(1, dtype=torch.float32),
        feature_names=tuple(schema.feature_names),
        edge_wdl_targets=None,
        topology_targets=None,
    )
    tree_batch, topology_targets = collate_tensorized_examples_for_topology([ex])
    assert tree_batch.node_features.shape == (1, len(TREE_ENCODER_FEATURE_NAMES))
    assert torch.allclose(topology_targets, tail)


@pytest.mark.parametrize("loss_type", ["huber", "mse"])
def test_topology_weighted_loss_masks_tail_dims(loss_type: str) -> None:
    model = TopologyModel(
        k=2,
        node_feat=5,
        device="cpu",
        node_embed_hidden=16,
        d_embed=32,
        d_message=32,
        n_heads=2,
        d_att=16,
        decoder_hidden=32,
        num_topology_targets=2,
    )
    cfg = NodePretrainConfig(loss_type=loss_type, topology_target_weights=(1.0, 1.0))
    trainer = NodePretrainer(model=model, device="cpu", train_examples=[], validation_examples=[], config=cfg)
    pred = torch.zeros(2, 2)
    targ_finite = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    targ_with_nan = torch.tensor([[3.0, 3.0], [float("nan"), float("nan")]])
    loss_finite = float(trainer._regression_loss(pred, targ_finite))
    loss_with_nan = float(trainer._regression_loss(pred, targ_with_nan))
    assert abs(loss_finite - loss_with_nan) < 1e-5

