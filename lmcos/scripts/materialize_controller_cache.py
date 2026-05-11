"""Standalone parallel materialization of frozen-encoder controller cache.

Usage:
  # Three parallel workers, each on a 1-hour GPU job:
  python3 scripts/materialize_controller_cache.py materialize \
    --packed-data .../train_manifest.json \
    --encoder-checkpoint .../tree_encoder.pt \
    --output-dir .../cache_parts \
    --worker-index 0 --num-workers 3

  # After all workers finish, merge:
  python3 scripts/materialize_controller_cache.py merge \
    --output-dir .../cache_parts \
    --num-workers 3 \
    --final-cache .../train_manifest.materialized_train_tree_encoder.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from cts_pretrain import load_encoder_checkpoint
from schema import tree_encoder_feature_schema
from scripts.train_fitted_q_controller import (
    ComputeAdvantageTreeSearchModel,
    PackedControllerCollator,
    PackedControllerEpisodeDataset,
    _materialized_cache_shard_dir,
)


def materialize_worker(args: argparse.Namespace) -> None:
    schema = tree_encoder_feature_schema()
    device = torch.device(args.device)

    model = ComputeAdvantageTreeSearchModel(
        k=args.k,
        node_feat=len(schema.feature_names),
        device=args.device,
        node_embed_hidden=args.node_embed_hidden,
        d_embed=args.d_embed,
        d_message=args.d_message,
        n_heads=args.n_heads,
        d_att=args.d_att,
        q_hidden=args.q_hidden,
        q_hidden_layers=args.q_hidden_layers,
    )
    load_encoder_checkpoint(args.encoder_checkpoint, model.encoder)
    model.freeze_encoder()
    model.eval()

    full_dataset = PackedControllerEpisodeDataset(args.packed_data)
    n = len(full_dataset)
    chunk = (n + args.num_workers - 1) // args.num_workers
    start = args.worker_index * chunk
    end = min(start + chunk, n)
    indices = list(range(start, end))
    subset = Subset(full_dataset, indices)

    print(
        f"[materialize] worker={args.worker_index}/{args.num_workers} "
        f"episodes={start}..{end} ({len(indices)} of {n})",
        flush=True,
    )

    loader = DataLoader(
        subset,
        batch_size=args.episode_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=PackedControllerCollator(),
    )

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / f"worker_{args.worker_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: List[str] = []
    shard_sizes: List[int] = []
    shard_features: List[torch.Tensor] = []
    shard_targets: List[torch.Tensor] = []
    shard_oracle_steps: List[torch.Tensor] = []
    shard_snapshots = 0
    total_snapshots = 0
    total_episodes = 0
    shard_index = 0
    started = time.time()

    def flush_shard() -> None:
        nonlocal shard_features, shard_targets, shard_oracle_steps, shard_snapshots, shard_index
        if not shard_features:
            return
        features = torch.cat(shard_features, dim=0)
        targets = torch.cat(shard_targets, dim=0)
        oracle = torch.cat(shard_oracle_steps, dim=0)
        shard_path = shard_dir / f"shard_{shard_index:05d}.pt"
        torch.save(
            {
                "format": "cts_materialized_advantage_cache_shard_v2",
                "features": features,
                "target_advantages": targets,
                "oracle_stop_steps": oracle,
            },
            shard_path,
        )
        shard_paths.append(str(shard_path))
        shard_sizes.append(int(features.shape[0]))
        shard_features = []
        shard_targets = []
        shard_oracle_steps = []
        shard_snapshots = 0
        shard_index += 1

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if batch is None:
                continue
            features = model.encode_with_state_features(
                batch.tree_batch, batch.tree_sizes, batch.time_budgets,
            ).detach().cpu()
            targets = batch.target_advantages.detach().cpu()
            oracle_steps = torch.tensor(batch.oracle_stop_steps, dtype=torch.int32)
            path_lengths = torch.tensor(batch.path_lengths, dtype=torch.int32)
            oracle_per_snapshot = oracle_steps.repeat_interleave(path_lengths)
            shard_features.append(features)
            shard_targets.append(targets)
            shard_oracle_steps.append(oracle_per_snapshot)
            batch_snapshots = int(features.shape[0])
            shard_snapshots += batch_snapshots
            total_snapshots += batch_snapshots
            total_episodes += len(batch.paths)
            if shard_snapshots >= args.max_snapshots_per_shard:
                flush_shard()
            if batch_index % args.log_interval == 0 or batch_index == len(loader):
                elapsed = time.time() - started
                print(
                    f"worker={args.worker_index} batch={batch_index}/{len(loader)} "
                    f"episodes={total_episodes} snapshots={total_snapshots} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    flush_shard()

    manifest_path = shard_dir / "manifest.pt"
    torch.save(
        {
            "shard_paths": shard_paths,
            "shard_sizes": shard_sizes,
            "total_snapshots": total_snapshots,
            "total_episodes": total_episodes,
            "worker_index": args.worker_index,
        },
        manifest_path,
    )
    elapsed = time.time() - started
    print(
        f"[materialize] worker={args.worker_index} done "
        f"episodes={total_episodes} snapshots={total_snapshots} "
        f"shards={len(shard_paths)} elapsed_s={elapsed:.1f}",
        flush=True,
    )


def merge_workers(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    all_shard_paths: List[str] = []
    all_shard_sizes: List[int] = []
    total_snapshots = 0

    for worker_index in range(args.num_workers):
        manifest_path = output_dir / f"worker_{worker_index:02d}" / "manifest.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing worker manifest: {manifest_path}")
        worker_manifest = torch.load(manifest_path, weights_only=False)
        all_shard_paths.extend(worker_manifest["shard_paths"])
        all_shard_sizes.extend(worker_manifest["shard_sizes"])
        total_snapshots += worker_manifest["total_snapshots"]
        print(
            f"worker={worker_index} snapshots={worker_manifest['total_snapshots']} "
            f"shards={len(worker_manifest['shard_paths'])}",
            flush=True,
        )

    final_cache_path = Path(args.final_cache)
    final_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Also write a shard directory symlink/copy structure that matches what the
    # training script expects: {cache_path}.d/ containing all shards.
    expected_shard_dir = _materialized_cache_shard_dir(args.final_cache)
    expected_shard_dir.mkdir(parents=True, exist_ok=True)
    # Symlink each worker shard into the expected directory with globally unique names.
    remapped_paths: List[str] = []
    for i, src in enumerate(all_shard_paths):
        dst = expected_shard_dir / f"shard_{i:05d}.pt"
        src_path = Path(src)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_path.resolve())
        remapped_paths.append(str(dst))

    payload = {
        "format": "cts_materialized_advantage_cache_v2",
        "metadata": {
            "manifest_path": args.manifest_path,
            "encoder_checkpoint": args.encoder_checkpoint,
        },
        "shards": [
            {"path": p, "examples": s}
            for p, s in zip(remapped_paths, all_shard_sizes)
        ],
        "examples": total_snapshots,
    }
    torch.save(payload, final_cache_path)
    print(
        f"[merge] wrote {final_cache_path} "
        f"total_snapshots={total_snapshots} total_shards={len(remapped_paths)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- materialize subcommand --
    mat = subparsers.add_parser("materialize")
    mat.add_argument("--packed-data", required=True)
    mat.add_argument("--encoder-checkpoint", required=True)
    mat.add_argument("--output-dir", required=True)
    mat.add_argument("--worker-index", type=int, required=True)
    mat.add_argument("--num-workers", type=int, required=True)
    mat.add_argument("--device", default="cuda")
    mat.add_argument("--episode-batch-size", type=int, default=8)
    mat.add_argument("--max-snapshots-per-shard", type=int, default=250000)
    mat.add_argument("--log-interval", type=int, default=100)
    # Architecture (must match encoder checkpoint).
    mat.add_argument("--k", type=int, default=1)
    mat.add_argument("--node-embed-hidden", type=int, default=128)
    mat.add_argument("--d-embed", type=int, default=128)
    mat.add_argument("--d-message", type=int, default=128)
    mat.add_argument("--n-heads", type=int, default=4)
    mat.add_argument("--d-att", type=int, default=32)
    mat.add_argument("--q-hidden", type=int, default=256)
    mat.add_argument("--q-hidden-layers", type=int, default=3)

    # -- merge subcommand --
    mrg = subparsers.add_parser("merge")
    mrg.add_argument("--output-dir", required=True, help="Same as materialize --output-dir")
    mrg.add_argument("--num-workers", type=int, required=True)
    mrg.add_argument("--final-cache", required=True, help="Path the training script expects")
    mrg.add_argument("--manifest-path", required=True, help="Packed manifest path (for cache metadata)")
    mrg.add_argument("--encoder-checkpoint", required=True, help="Encoder checkpoint (for cache metadata)")

    args = parser.parse_args()
    if args.command == "materialize":
        materialize_worker(args)
    elif args.command == "merge":
        merge_workers(args)


if __name__ == "__main__":
    main()
