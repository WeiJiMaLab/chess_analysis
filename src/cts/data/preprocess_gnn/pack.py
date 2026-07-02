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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from cts.core.schema import (
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
    # Per-tree tensorization is embarrassingly parallel; >0 spreads it over a
    # ProcessPoolExecutor (shard assembly/save stays serial). 0 = serial.
    num_workers: int = 0


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
) -> None:
    """Raise ``ValueError`` if this raw example cannot populate the chosen encoder columns."""
    feature_index = {name: index for index, name in enumerate(record.feature_names)}
    for required_feature in schema.feature_names:
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
) -> TensorizedTreeExample:
    record = RawPretrainExampleRecord.load(path)
    raise_if_example_cannot_pack(record, schema, example_path_label=str(path))
    return record.to_tensorized_tree_example(schema)


def _log_shard_progress(completed_in_shard, shard_paths, split_name, shard_index, total_shards,
                        total_examples_before_shard, total_examples_in_split, start_time, log_interval):
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


def tensorize_paths_for_shard(
    shard_paths: list[Path],
    schema: NodeFeatureSchema,
    *,
    split_name: str,
    shard_index: int,
    total_shards: int,
    total_examples_before_shard: int,
    total_examples_in_split: int,
    start_time: float,
    log_interval: int,
    num_workers: int = 0,
) -> list[TensorizedTreeExample]:
    """Tensorize this shard's per-tree examples, in input order. ``num_workers`` > 0
    spreads the (independent) per-tree work over a process pool; results are written
    back into the input slot so the output order is preserved regardless of finish order."""
    args = dict(split_name=split_name, shard_index=shard_index, total_shards=total_shards,
                total_examples_before_shard=total_examples_before_shard,
                total_examples_in_split=total_examples_in_split, start_time=start_time,
                log_interval=log_interval)
    if num_workers <= 0:
        results: list[TensorizedTreeExample] = []
        for completed_in_shard, path in enumerate(shard_paths, start=1):
            results.append(tensorize_example_for_pack(path, schema))
            _log_shard_progress(completed_in_shard, shard_paths, **args)
        return results

    slots: list[TensorizedTreeExample | None] = [None] * len(shard_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(tensorize_example_for_pack, path, schema): i
                   for i, path in enumerate(shard_paths)}
        for completed_in_shard, future in enumerate(as_completed(futures), start=1):
            slots[futures[future]] = future.result()
            _log_shard_progress(completed_in_shard, shard_paths, **args)
    return slots


def pack_train_or_val_split_to_shards(
    manifest_path: Path,
    output_root: Path,
    shard_size: int,
    log_interval: int,
    *,
    schema: NodeFeatureSchema,
    num_workers: int = 0,
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
            split_name=split_name,
            shard_index=shard_index,
            total_shards=total_shards,
            total_examples_before_shard=total_examples,
            total_examples_in_split=len(example_paths),
            start_time=start_time,
            log_interval=log_interval,
            num_workers=num_workers,
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

    schema = tree_encoder_feature_schema()
    train_packed_manifest, train_count = pack_train_or_val_split_to_shards(
        train_manifest,
        output_root,
        shard_size,
        config.log_interval,
        schema=schema,
        num_workers=config.num_workers,
    )
    validation_packed_manifest, validation_count = pack_train_or_val_split_to_shards(
        validation_manifest,
        output_root,
        shard_size,
        config.log_interval,
        schema=schema,
        num_workers=config.num_workers,
    )

    print(f"train_manifest={train_packed_manifest}")
    print(f"validation_manifest={validation_packed_manifest}")
    print(f"train_examples={train_count}")
    print(f"validation_examples={validation_count}")
    print(f"output_root={output_root}")
    print(f"single_shard={config.single_shard}")
    print(f"effective_shard_size={shard_size}")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(PackPretrainConfig, main)
