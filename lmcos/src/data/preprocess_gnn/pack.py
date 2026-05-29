"""Flatten per-tree `.pt` pretrain examples into batched shards the GNN eats.

Given split manifests pointing at teacher trees (`cts_raw_pretrain_example_*`), this CLI
loads each example, aligns columns with the encoder schema (optionally widening with
scaled teacher topology), and writes chunky ``.pt`` shards plus JSON manifests — the
usual input to Child-WDL or topology Encoder/TopologyHead training — Huber on the four topology
vectors lines up encoder **node** states (root included) with scaled teacher summaries, while checkpoints
stay five columns wide at the embedding input.

Pipeline: manifest paths → tensors per node/edge → stack many trees → shard files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from cts.core.schema import (
    TEACHER_NODETARGETS_FEATURE_NAMES,
    nodetargets_target_feature_names,
    NodeFeatureSchema,
    tree_encoder_feature_schema,
)
from cts.core.tensorizer import TensorizedTreeExample
from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord


class PackPretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split"
    output_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed"
    shard_size: int = 2000
    log_interval: int = 1
    single_shard: bool = False
    clear: bool = False

    #: Append supervision targets directly to node features.
    node_targets: bool = False
    #: Flat ``nodetargets_targets`` in each shard + manifest flag ``nodetargets_targets: true``.
    node_supervision_shard: bool = False

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_topology_yaml(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "num_workers" in out:
            raise ValueError(
                "Pack configs no longer accept `num_workers`; packing runs serially in-process. "
                "Remove `num_workers` from your YAML."
            )
        # Migrate legacy topology_features key to node_targets
        if "topology_features" in out:
            out["node_targets"] = out.pop("topology_features")
        if "topology_supervision_shard" in out:
            out["node_supervision_shard"] = out.pop("topology_supervision_shard")

        lt_key = "include_topology_targets"
        tt_key = "include_teacher_topology"
        if lt_key in out or tt_key in out:
            inc_lt = bool(out.pop(lt_key, False))
            inc_tt = bool(out.pop(tt_key, False))
            if inc_lt and inc_tt:
                raise ValueError(
                    f"Deprecated mutually exclusive YAML keys `{lt_key}` and `{tt_key}` cannot both be true. "
                    "Use node_targets / node_supervision_shard; see LAB_NOTEBOOK."
                )
            if inc_lt:
                out.setdefault("node_supervision_shard", True)
                out.setdefault("node_targets", False)
            elif inc_tt:
                out.setdefault("node_targets", True)
                out.setdefault("node_supervision_shard", True)
        return out

    @model_validator(mode="after")
    def topology_wide_must_ship_supervision(self) -> PackPretrainConfig:
        if self.node_targets and not self.node_supervision_shard:
            raise ValueError(
                "node_targets=true requires node_supervision_shard=true so node pretrain "
                "has explicit targets and shards stay consistent; see LAB_NOTEBOOK presets."
            )
        return self


def read_manifest(path: Path) -> list[Path]:
    """Paths from ``train_manifest.txt`` / ``validation_manifest.txt`` (one path per line)."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        examples = [Path(line.strip()) for line in handle if line.strip()]
    if not examples:
        raise ValueError(f"No example paths found in manifest: {path}")
    return examples


def raise_if_example_cannot_pack(
    record: RawPretrainExampleRecord,
    schema: NodeFeatureSchema,
    *,
    example_path_label: str,
    need_teacher_topology_columns: bool,
) -> None:
    """Raise ``ValueError`` if this raw example cannot populate the chosen encoder columns."""
    feature_index = {name: index for index, name in enumerate(record.feature_names)}
    num_nodes = int(record.parent_index.shape[0])
    # Narrow shards still consume teacher targets tensors for the flat supervision row.
    if need_teacher_topology_columns:
        for name in TEACHER_NODETARGETS_FEATURE_NAMES:
            column = getattr(record, name)
            if int(column.shape[0]) != num_nodes:
                raise ValueError(
                    f"{example_path_label} has {name} length {column.shape[0]} != num_nodes={num_nodes}."
                )
    for required_feature in schema.feature_names:
        if required_feature in TEACHER_NODETARGETS_FEATURE_NAMES and need_teacher_topology_columns:
            continue  # lengths already enforced above
        if required_feature not in feature_index:
            raise ValueError(f"{example_path_label} is missing required tree encoder feature {required_feature!r}.")
        column = record.node_features[:, feature_index[required_feature]]
        if torch.isnan(column).any():
            bad_node = int(torch.nonzero(torch.isnan(column), as_tuple=False)[0].item())
            raise ValueError(
                f"{example_path_label} node_id={bad_node} is missing required tree encoder feature {required_feature!r}."
            )


def tensorize_example_for_pack(
    path: Path,
    schema: NodeFeatureSchema,
    *,
    node_targets: bool,
    node_supervision_shard: bool,
) -> TensorizedTreeExample:
    record = RawPretrainExampleRecord.load(path)
    need_topo = node_targets or node_supervision_shard
    raise_if_example_cannot_pack(
        record,
        schema,
        example_path_label=str(path),
        need_teacher_topology_columns=need_topo,
    )
    return record.to_tensorized_tree_example(
        schema,
        node_targets=node_targets,
        topology_supervision_shard=node_supervision_shard,
    )


def tensorize_paths_for_shard(
    shard_paths: list[Path],
    schema: NodeFeatureSchema,
    *,
    node_targets: bool,
    node_supervision_shard: bool,
    split_name: str,
    shard_index: int,
    total_shards: int,
    total_examples_before_shard: int,
    total_examples_in_split: int,
    start_time: float,
    log_interval: int,
) -> list[TensorizedTreeExample]:
    results: list[TensorizedTreeExample] = []
    for completed_in_shard, path in enumerate(shard_paths, start=1):
        results.append(
            tensorize_example_for_pack(
                path,
                schema,
                node_targets=node_targets,
                node_supervision_shard=node_supervision_shard,
            )
        )
        if completed_in_shard % log_interval == 0 or completed_in_shard == len(shard_paths):
            elapsed = time.time() - start_time
            completed_total = total_examples_before_shard + completed_in_shard
            print(
                f"split={split_name} shard={shard_index + 1}/{total_shards} "
                f"shard_examples={completed_in_shard}/{len(shard_paths)} "
                f"packed_examples={completed_total}/{total_examples_in_split} "
                f"elapsed_s={elapsed:.1f} base_examples_per_s={completed_total / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
    return results


def pack_train_or_val_split_to_shards(
    manifest_path: Path,
    output_root: Path,
    shard_size: int,
    log_interval: int,
    *,
    schema: NodeFeatureSchema,
    node_targets: bool,
    node_supervision_shard: bool,
) -> tuple[Path, int]:
    """Writes ``shard_*.pt`` under ``output_root/<split>/`` plus ``<split>_manifest.json``.

    Args: returns ``(manifest_path, example_count)``
    """
    # Split name is derived from the manifest filename ("train_manifest.txt"
    # → "train") so the packed manifest and subdirectory stay aligned.
    split_name = manifest_path.stem.replace("_manifest", "")
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    example_paths = read_manifest(manifest_path)
    start_time = time.time()
    entries = []
    total_examples = 0
    total_shards = (len(example_paths) + shard_size - 1) // shard_size

    for shard_index, start in enumerate(range(0, len(example_paths), shard_size)):
        shard_paths = example_paths[start : start + shard_size]
        tensorized_examples = tensorize_paths_for_shard(
            shard_paths,
            schema,
            node_targets=node_targets,
            node_supervision_shard=node_supervision_shard,
            split_name=split_name,
            shard_index=shard_index,
            total_shards=total_shards,
            total_examples_before_shard=total_examples,
            total_examples_in_split=len(example_paths),
            start_time=start_time,
            log_interval=log_interval,
        )
        shard_path = split_output_dir / f"shard_{shard_index:05d}.pt"
        # node_ptr/edge_ptr are CSR-style offsets: example i's nodes live in
        # the flat node_features tensor at rows [node_ptr[i], node_ptr[i+1]).
        # Same convention for edge_ptr against the flat edge tensors.
        node_ptr = [0]
        edge_ptr = [0]
        node_features_parts = []
        parent_index_parts = []
        edge_parent_parts = []
        edge_child_parts = []
        edge_slot_parts = []
        depth_parts = []
        target_parts = []
        edge_wdl_target_parts = []
        topology_targets_parts = []

        for tensorized in tensorized_examples:
            # Move everything to CPU before stacking — the resulting tensors
            # are saved to disk and the encoder loader handles device transfer.
            node_features_parts.append(tensorized.node_features.cpu())
            parent_index_parts.append(tensorized.parent_index.cpu())
            edge_parent_parts.append(tensorized.edge_parent.cpu())
            edge_child_parts.append(tensorized.edge_child.cpu())
            edge_slot_parts.append(tensorized.edge_slot.cpu())
            depth_parts.append(tensorized.depth.cpu())
            target_parts.append(tensorized.node_targets.cpu())
            if tensorized.edge_wdl_targets is not None:
                edge_wdl_target_parts.append(tensorized.edge_wdl_targets.cpu())
            if node_supervision_shard:
                if tensorized.nodetargets_targets is None:
                    raise ValueError("node_supervision_shard requires nodetargets_targets on every example.")
                topology_targets_parts.append(tensorized.nodetargets_targets.cpu())
            node_ptr.append(node_ptr[-1] + int(tensorized.node_features.shape[0]))
            edge_ptr.append(edge_ptr[-1] + int(tensorized.edge_parent.shape[0]))

        payload = {
            "format": "cts_tensorized_pretrain_shard_v1",
            "num_examples": len(tensorized_examples),
            "source_manifest": str(manifest_path),
            "source_paths": [str(path) for path in shard_paths],
            "feature_names": list(schema.feature_names),
            "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
            "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
            "node_features": torch.cat(node_features_parts, dim=0),
            "parent_index": torch.cat(parent_index_parts, dim=0),
            # Edge tensors may be empty for root-only examples; ``torch.cat``
            # rejects empty input lists, so substitute a zero-length tensor.
            "edge_parent": torch.cat(edge_parent_parts, dim=0) if edge_parent_parts else torch.empty(0, dtype=torch.long),
            "edge_child": torch.cat(edge_child_parts, dim=0) if edge_child_parts else torch.empty(0, dtype=torch.long),
            "edge_slot": torch.cat(edge_slot_parts, dim=0) if edge_slot_parts else torch.empty(0, dtype=torch.long),
            "depth": torch.cat(depth_parts, dim=0),
            "node_targets": torch.cat(target_parts, dim=0),
            # Only emit edge_wdl_targets when *every* example in the shard
            # provided it; mixing presence/absence across the shard would
            # break the flat indexing the loader assumes.
            "edge_wdl_targets": (
                torch.cat(edge_wdl_target_parts, dim=0)
                if edge_wdl_target_parts and len(edge_wdl_target_parts) == len(tensorized_examples)
                else None
            ),
        }
        if node_supervision_shard:
            if len(topology_targets_parts) != len(tensorized_examples):
                raise ValueError("nodetargets_targets missing for some examples in shard.")
            payload["topology_targets"] = torch.cat(topology_targets_parts, dim=0)
            payload["topology_target_feature_names"] = list(nodetargets_target_feature_names())
        torch.save(payload, shard_path)
        entries.append(
            {
                "path": str(shard_path),
                "num_examples": len(tensorized_examples),
                "shard_index": shard_index,
            }
        )
        total_examples += len(tensorized_examples)

    packed_manifest_path = output_root / f"{split_name}_manifest.json"
    with packed_manifest_path.open("w", encoding="utf-8") as handle:
        manifest_payload = {
            "format": "cts_tensorized_pretrain_manifest_v1",
            "split": split_name,
            "total_examples": total_examples,
            "entries": entries,
        }
        if node_supervision_shard:
            manifest_payload["topology_targets"] = True
        if node_targets:
            manifest_payload["node_targets"] = True
        json.dump(manifest_payload, handle, indent=2)

    return packed_manifest_path, total_examples


def main(config: PackPretrainConfig) -> None:
    """Pack ``train_manifest.txt`` and ``validation_manifest.txt`` beside ``split_root``."""
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive.")

    split_root = Path(config.split_root)
    output_root = Path(config.output_root)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"

    if config.clear and output_root.exists():
        # Remove leaf files first, then bottom-up the now-empty directories.
        # ``rglob`` is reverse-sorted so deeper paths get rmdir'd before
        # their parents.
        for child in output_root.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(output_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()

    output_root.mkdir(parents=True, exist_ok=True)

    train_paths = read_manifest(train_manifest)
    validation_paths = read_manifest(validation_manifest)
    # ``single_shard`` collapses each split to one file; pick a shard size
    # large enough to cover whichever split has more examples.
    shard_size = max(len(train_paths), len(validation_paths)) if config.single_shard else config.shard_size

    schema = tree_encoder_feature_schema(node_targets=config.node_targets)
    nt, ts = config.node_targets, config.node_supervision_shard
    train_packed_manifest, train_count = pack_train_or_val_split_to_shards(
        train_manifest,
        output_root,
        shard_size,
        config.log_interval,
        schema=schema,
        node_targets=nt,
        node_supervision_shard=ts,
    )
    validation_packed_manifest, validation_count = pack_train_or_val_split_to_shards(
        validation_manifest,
        output_root,
        shard_size,
        config.log_interval,
        schema=schema,
        node_targets=nt,
        node_supervision_shard=ts,
    )

    print(f"train_manifest={train_packed_manifest}")
    print(f"validation_manifest={validation_packed_manifest}")
    print(f"train_examples={train_count}")
    print(f"validation_examples={validation_count}")
    print(f"output_root={output_root}")
    print(f"single_shard={config.single_shard}")
    print(f"effective_shard_size={shard_size}")
    print(f"node_targets={nt}")
    print(f"node_supervision_shard={ts}")
    if config.node_supervision_shard:
        print(f"nodetargets_target_columns={list(nodetargets_target_feature_names())}")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(PackPretrainConfig, main)
