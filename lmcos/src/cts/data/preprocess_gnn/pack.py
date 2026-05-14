"""Pack raw pretrain prefix examples into encoder-ready shard files.

Runs after ``prepare_pretrain_split.py`` and ``derive_pretrain_prefixes.py``:
loads each per-example ``RawPretrainExampleRecord`` listed in the split's
text manifest, projects its features onto the canonical encoder schema, and
concatenates many examples into a single ``.pt`` shard with flat ``node_ptr``
and ``edge_ptr`` offsets. The resulting per-split JSON manifest is what the
encoder pretraining slurm script feeds into its dataset loader.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict


from cts.core.schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord


class PackPretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split"
    output_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed"
    shard_size: int = 2000
    num_workers: int = 0
    log_interval: int = 1
    single_shard: bool = False
    clear: bool = False


def _read_manifest(path: Path) -> list[Path]:
    """Read a newline-delimited list of example paths from a split manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        examples = [Path(line.strip()) for line in handle if line.strip()]
    if not examples:
        raise ValueError(f"No example paths found in manifest: {path}")
    return examples


def _feature_schema() -> NodeFeatureSchema:
    """Return the canonical encoder schema used for every shard in this run."""
    return tree_encoder_feature_schema()


def _validate_tree_encoder_record_features(record: RawPretrainExampleRecord, *, context: str) -> None:
    """Fail loudly if a raw record is missing any required encoder feature.

    Older raw records may pre-date WDL features, which would otherwise show
    up as silent NaN columns inside packed shards. We catch both the
    missing-name case and any NaN entries up front so the offending file
    path appears in the error.

    Args:
        record: raw per-example record loaded from disk.
        context: short label (typically the file path) interpolated into
            the error message so the caller can locate the bad record.
    """
    feature_index = {name: index for index, name in enumerate(record.feature_names)}
    for required_feature in tree_encoder_feature_schema().feature_names:
        if required_feature not in feature_index:
            raise ValueError(f"{context} is missing required tree encoder feature {required_feature!r}.")
        column = record.node_features[:, feature_index[required_feature]]
        if torch.isnan(column).any():
            bad_node = int(torch.nonzero(torch.isnan(column), as_tuple=False)[0].item())
            raise ValueError(
                f"{context} node_id={bad_node} is missing required tree encoder feature {required_feature!r}."
            )


def _tensorize_example_path(task: tuple[str, tuple[str, ...]]) -> object:
    """Worker entry point: load one raw record and return its tensorized form.

    Accepts a ``(path, feature_names)`` tuple rather than a schema object so
    it can be pickled cleanly into a ``ProcessPoolExecutor`` worker.
    """
    path_str, feature_names = task
    schema = NodeFeatureSchema(feature_names)
    record = RawPretrainExampleRecord.load(path_str)
    _validate_tree_encoder_record_features(record, context=str(path_str))
    return record.to_tensorized_tree_example(schema)


def _load_tensorized_examples(
    shard_paths: list[Path],
    schema: NodeFeatureSchema,
    num_workers: int,
    *,
    split_name: str,
    shard_index: int,
    total_shards: int,
    total_examples_before_shard: int,
    total_examples_in_split: int,
    start_time: float,
    log_interval: int,
) -> list[object]:
    """Load and tensorize one shard's worth of examples, optionally in parallel.

    Logs progress every ``log_interval`` examples and at shard end so the
    slurm log shows a running examples/second rate. Falls back to serial
    execution if ``num_workers <= 1`` or if process-pool startup fails with
    ``PermissionError``/``OSError`` (seen on some cluster filesystems).
    """
    tasks = [(str(path), tuple(schema.feature_names)) for path in shard_paths]
    if num_workers <= 1:
        # Serial path: simpler, used when num_workers <= 1 or as a fallback.
        results = []
        for completed_in_shard, task in enumerate(tasks, start=1):
            results.append(_tensorize_example_path(task))
            if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                elapsed = time.time() - start_time
                completed_total = total_examples_before_shard + completed_in_shard
                print(
                    f"split={split_name} shard={shard_index + 1}/{total_shards} "
                    f"shard_examples={completed_in_shard}/{len(tasks)} "
                    f"packed_examples={completed_total}/{total_examples_in_split} "
                    f"elapsed_s={elapsed:.1f} base_examples_per_s={completed_total / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        return results
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Preserve input order by recording each future's original index
            # and writing into a pre-sized results list as futures complete.
            futures = {
                executor.submit(_tensorize_example_path, task): index
                for index, task in enumerate(tasks)
            }
            results = [None] * len(tasks)
            for completed_in_shard, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                    elapsed = time.time() - start_time
                    completed_total = total_examples_before_shard + completed_in_shard
                    print(
                        f"split={split_name} shard={shard_index + 1}/{total_shards} "
                        f"shard_examples={completed_in_shard}/{len(tasks)} "
                        f"packed_examples={completed_total}/{total_examples_in_split} "
                        f"elapsed_s={elapsed:.1f} base_examples_per_s={completed_total / max(elapsed, 1e-6):.2f}",
                        flush=True,
                    )
            return results
    except (PermissionError, OSError):
        # Some cluster filesystems forbid the semaphore/socket creation that
        # ProcessPoolExecutor needs. Fall back to the serial loop above.
        results = []
        for completed_in_shard, task in enumerate(tasks, start=1):
            results.append(_tensorize_example_path(task))
            if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                elapsed = time.time() - start_time
                completed_total = total_examples_before_shard + completed_in_shard
                print(
                    f"split={split_name} shard={shard_index + 1}/{total_shards} "
                    f"shard_examples={completed_in_shard}/{len(tasks)} "
                    f"packed_examples={completed_total}/{total_examples_in_split} "
                    f"elapsed_s={elapsed:.1f} base_examples_per_s={completed_total / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        return results


def _pack_split(manifest_path: Path, output_root: Path, shard_size: int, num_workers: int, log_interval: int) -> tuple[Path, int]:
    """Pack one split (train or validation) into shard files and a JSON manifest.

    Writes ``shard_NNNNN.pt`` files under ``output_root/<split>/`` and a
    ``<split>_manifest.json`` index at ``output_root``. Returns the manifest
    path and the total example count for the split.

    Args:
        manifest_path: text manifest listing per-example raw record paths.
        output_root: directory where the packed split subtree is created.
        shard_size: maximum number of examples per output shard file.
        num_workers: process-pool size used by ``_load_tensorized_examples``.
        log_interval: emit a progress line every ``log_interval`` examples.
    """
    # Split name is derived from the manifest filename ("train_manifest.txt"
    # → "train") so the packed manifest and subdirectory stay aligned.
    split_name = manifest_path.stem.replace("_manifest", "")
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    example_paths = _read_manifest(manifest_path)
    start_time = time.time()
    entries = []
    total_examples = 0
    total_shards = (len(example_paths) + shard_size - 1) // shard_size
    schema = _feature_schema()

    for shard_index, start in enumerate(range(0, len(example_paths), shard_size)):
        shard_paths = example_paths[start : start + shard_size]
        tensorized_examples = _load_tensorized_examples(
            shard_paths,
            schema,
            num_workers,
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
        json.dump(
            {
                "format": "cts_tensorized_pretrain_manifest_v1",
                "split": split_name,
                "total_examples": total_examples,
                "entries": entries,
            },
            handle,
            indent=2,
        )

    return packed_manifest_path, total_examples


def main(config: PackPretrainConfig) -> None:
    """CLI entry point: pack the train and validation splits at ``split_root``."""
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
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

    train_paths = _read_manifest(train_manifest)
    validation_paths = _read_manifest(validation_manifest)
    # ``single_shard`` collapses each split to one file; pick a shard size
    # large enough to cover whichever split has more examples.
    shard_size = max(len(train_paths), len(validation_paths)) if config.single_shard else config.shard_size

    train_packed_manifest, train_count = _pack_split(
        train_manifest,
        output_root,
        shard_size,
        config.num_workers,
        config.log_interval,
    )
    validation_packed_manifest, validation_count = _pack_split(
        validation_manifest,
        output_root,
        shard_size,
        config.num_workers,
        config.log_interval,
    )

    print(f"train_manifest={train_packed_manifest}")
    print(f"validation_manifest={validation_packed_manifest}")
    print(f"train_examples={train_count}")
    print(f"validation_examples={validation_count}")
    print(f"output_root={output_root}")
    print(f"num_workers={config.num_workers}")
    print(f"single_shard={config.single_shard}")
    print(f"effective_shard_size={shard_size}")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(PackPretrainConfig, main)
