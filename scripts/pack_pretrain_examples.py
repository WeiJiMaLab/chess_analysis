from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schema import NodeFeatureSchema, require_tree_encoder_scalar_features, tree_encoder_feature_schema
from tensorizer import tensorize_tree_with_targets


def _read_manifest(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        examples = [Path(line.strip()) for line in handle if line.strip()]
    if not examples:
        raise ValueError(f"No example paths found in manifest: {path}")
    return examples


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


def _validate_tree_encoder_example_features(example, *, context: str) -> None:
    for node in example.tree.iter_nodes():
        require_tree_encoder_scalar_features(
            node.scalar_features,
            context=f"{context} node_id={node.node_id} fen={node.fen!r}",
        )


def _pack_split(
    manifest_path: Path,
    output_root: Path,
    shard_size: int,
) -> tuple[Path, int]:
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
        tensorized_examples = []
        for path in shard_paths:
            example = torch.load(path, weights_only=False)
            _validate_tree_encoder_example_features(example, context=str(path))
            tensorized_examples.append(
                tensorize_tree_with_targets(
                    example.tree,
                    example.node_target_values,
                    schema=schema,
                    device="cpu",
                )
            )
        shard_path = split_output_dir / f"shard_{shard_index:05d}.pt"
        node_ptr = [0]
        edge_ptr = [0]
        node_features_parts = []
        parent_index_parts = []
        edge_parent_parts = []
        edge_child_parts = []
        edge_slot_parts = []
        depth_parts = []
        target_parts = []

        for tensorized in tensorized_examples:
            node_features_parts.append(tensorized.node_features.cpu())
            parent_index_parts.append(tensorized.parent_index.cpu())
            edge_parent_parts.append(tensorized.edge_parent.cpu())
            edge_child_parts.append(tensorized.edge_child.cpu())
            edge_slot_parts.append(tensorized.edge_slot.cpu())
            depth_parts.append(tensorized.depth.cpu())
            target_parts.append(tensorized.node_targets.cpu())
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
            "edge_parent": torch.cat(edge_parent_parts, dim=0) if edge_parent_parts else torch.empty(0, dtype=torch.long),
            "edge_child": torch.cat(edge_child_parts, dim=0) if edge_child_parts else torch.empty(0, dtype=torch.long),
            "edge_slot": torch.cat(edge_slot_parts, dim=0) if edge_slot_parts else torch.empty(0, dtype=torch.long),
            "depth": torch.cat(depth_parts, dim=0),
            "node_targets": torch.cat(target_parts, dim=0),
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
        elapsed = time.time() - start_time
        print(
            f"split={split_name} shard={shard_index + 1}/{total_shards} "
            f"packed_examples={total_examples}/{len(example_paths)} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack pretrain example manifests into larger shard files.")
    parser.add_argument(
        "--split-root",
        default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split",
    )
    parser.add_argument(
        "--output-root",
        default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed",
    )
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--single-shard", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"

    if args.clear and output_root.exists():
        for child in output_root.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(output_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()

    output_root.mkdir(parents=True, exist_ok=True)

    train_paths = _read_manifest(train_manifest)
    validation_paths = _read_manifest(validation_manifest)
    shard_size = max(len(train_paths), len(validation_paths)) if args.single_shard else args.shard_size

    train_packed_manifest, train_count = _pack_split(train_manifest, output_root, shard_size)
    validation_packed_manifest, validation_count = _pack_split(validation_manifest, output_root, shard_size)

    print(f"train_manifest={train_packed_manifest}")
    print(f"validation_manifest={validation_packed_manifest}")
    print(f"train_examples={train_count}")
    print(f"validation_examples={validation_count}")
    print(f"output_root={output_root}")
    print(f"single_shard={args.single_shard}")
    print(f"effective_shard_size={shard_size}")


if __name__ == "__main__":
    main()
