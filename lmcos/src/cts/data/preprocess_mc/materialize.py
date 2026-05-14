"""Pre-compute frozen-encoder activations for fitted-Q controller training.

Runs the encoder forward on every snapshot in a packed-episodes manifest and
writes a flat tensor cache (``z_root`` per snapshot, target advantage, oracle
stop step) which ``train_fitted_q_controller.py`` mmaps at training time. This
makes controller training I/O-light and GPU-bound on the small MLP head
instead of repeatedly re-encoding the same trees.

Sits between ``filter_packed_episodes.py`` and ``train_fitted_q_controller.py``.
Two subcommands: ``materialize`` writes one worker's slice as numbered shards
(safe to run in parallel SLURM jobs); ``merge`` collects the per-worker
manifests into the single cache file the training script expects.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Subset

from cts.core.schema import tree_encoder_feature_schema
from cts.models.mc import MetaController
from cts.train.controller_train import (
    ControllerEpisodeDataset,
    _materialized_cache_shard_dir,
    collate_controller_episodes,
)
from cts.train.gnn_pretrain import load_encoder_architecture, load_encoder_checkpoint


class MaterializeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["materialize", "merge"]

    # Shared by both subcommands.
    output_dir: str
    num_workers: int

    # materialize-only.
    packed_data: Optional[str] = None
    encoder_checkpoint: Optional[str] = None
    worker_index: Optional[int] = None
    device: str = "cuda"
    episode_batch_size: int = 8
    max_snapshots_per_shard: int = 250000
    log_interval: int = 100
    # Advantage-head architecture. The encoder's architecture is read out
    # of the checkpoint metadata; only the head shape is configurable here
    # (the head is reconstructed with random weights since we only use the
    # encoder for materialization).
    hidden_dim: int = 256
    hidden_layers: int = 3

    # merge-only.
    final_cache: Optional[str] = None
    manifest_path: Optional[str] = None


def materialize_worker(config: MaterializeConfig) -> None:
    """Encode this worker's slice of the dataset and write numbered shards.

    Splits the dataset into ``num_workers`` contiguous chunks by episode
    index and processes chunk ``worker_index``. Each worker writes its own
    ``worker_XX/`` subdirectory with shards plus a manifest; the ``merge``
    subcommand stitches them into the final cache.
    """
    schema = tree_encoder_feature_schema()
    device = torch.device(config.device)

    # Encoder architecture comes from the checkpoint's embedded metadata so
    # it always matches the saved weights. The head is configurable here
    # (we throw away its random weights after this run — only the encoder
    # outputs are consumed).
    architecture = load_encoder_architecture(config.encoder_checkpoint)
    model = MetaController(
        k=architecture["k"],
        node_feat=architecture.get("node_feat", len(schema.feature_names)),
        device=config.device,
        node_embed_hidden=architecture["node_embed_hidden"],
        d_embed=architecture["d_embed"],
        d_message=architecture["d_message"],
        n_heads=architecture["n_heads"],
        d_att=architecture["d_att"],
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    )
    load_encoder_checkpoint(config.encoder_checkpoint, model.encoder)
    model.freeze_encoder()
    model.eval()

    # Carve the dataset into equal-sized contiguous slices by episode index.
    # Ceiling division ensures the last worker gets the remainder rather
    # than dropping it; ``end`` is clamped so the final slice doesn't overrun.
    full_dataset = ControllerEpisodeDataset(config.packed_data)
    n = len(full_dataset)
    chunk = (n + config.num_workers - 1) // config.num_workers
    start = config.worker_index * chunk
    end = min(start + chunk, n)
    indices = list(range(start, end))
    subset = Subset(full_dataset, indices)

    print(
        f"[materialize] worker={config.worker_index}/{config.num_workers} "
        f"episodes={start}..{end} ({len(indices)} of {n})",
        flush=True,
    )

    # num_workers=0 keeps loading in the main process: the encoder forward
    # dominates wall time, so spawning DataLoader workers adds overhead
    # without speedup.
    loader = DataLoader(
        subset,
        batch_size=config.episode_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_controller_episodes,
    )

    output_dir = Path(config.output_dir)
    shard_dir = output_dir / f"worker_{config.worker_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Per-shard buffers accumulate until we hit ``max_snapshots_per_shard``,
    # at which point ``flush_shard`` writes them to disk and resets.
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
        """Concatenate buffered batches and write one shard file."""
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
            # Encode every snapshot in the batch and move to CPU immediately
            # so the GPU isn't held up waiting on disk writes.
            features = model.encode_with_state_features(
                batch.tree_batch, batch.tree_sizes, batch.time_budgets,
            ).detach().cpu()
            targets = batch.target_advantages.detach().cpu()
            oracle_steps = torch.tensor(batch.oracle_stop_steps, dtype=torch.int32)
            path_lengths = torch.tensor(batch.path_lengths, dtype=torch.int32)
            # Expand per-episode oracle stop step into per-snapshot rows so
            # the training script can broadcast it without re-deriving the
            # episode→snapshot mapping at train time.
            oracle_per_snapshot = oracle_steps.repeat_interleave(path_lengths)
            shard_features.append(features)
            shard_targets.append(targets)
            shard_oracle_steps.append(oracle_per_snapshot)
            batch_snapshots = int(features.shape[0])
            shard_snapshots += batch_snapshots
            total_snapshots += batch_snapshots
            total_episodes += len(batch.paths)
            if shard_snapshots >= config.max_snapshots_per_shard:
                flush_shard()
            if batch_index % config.log_interval == 0 or batch_index == len(loader):
                elapsed = time.time() - started
                print(
                    f"worker={config.worker_index} batch={batch_index}/{len(loader)} "
                    f"episodes={total_episodes} snapshots={total_snapshots} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    # Final partial shard (if any rows remain after the last full flush).
    flush_shard()

    # Per-worker manifest; ``merge`` reads these to build the global file list.
    manifest_path = shard_dir / "manifest.pt"
    torch.save(
        {
            "shard_paths": shard_paths,
            "shard_sizes": shard_sizes,
            "total_snapshots": total_snapshots,
            "total_episodes": total_episodes,
            "worker_index": config.worker_index,
        },
        manifest_path,
    )
    elapsed = time.time() - started
    print(
        f"[materialize] worker={config.worker_index} done "
        f"episodes={total_episodes} snapshots={total_snapshots} "
        f"shards={len(shard_paths)} elapsed_s={elapsed:.1f}",
        flush=True,
    )


def merge_workers(config: MaterializeConfig) -> None:
    """Stitch per-worker manifests into the single cache file training expects.

    Reads each ``worker_XX/manifest.pt``, renumbers shard files globally,
    symlinks them into the shard directory the training script looks for,
    and writes the top-level cache pointing at the symlinks.
    """
    output_dir = Path(config.output_dir)
    all_shard_paths: List[str] = []
    all_shard_sizes: List[int] = []
    total_snapshots = 0

    for worker_index in range(config.num_workers):
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

    final_cache_path = Path(config.final_cache)
    final_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # The training script looks for shards in ``{cache_path}.d/`` next to the
    # cache file. Rather than copy gigabytes, symlink each worker's shard
    # into that directory under a globally-unique name.
    expected_shard_dir = _materialized_cache_shard_dir(config.final_cache)
    expected_shard_dir.mkdir(parents=True, exist_ok=True)
    remapped_paths: List[str] = []
    for i, src in enumerate(all_shard_paths):
        dst = expected_shard_dir / f"shard_{i:05d}.pt"
        src_path = Path(src)
        # Remove any stale symlink/file so reruns are idempotent.
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_path.resolve())
        remapped_paths.append(str(dst))

    payload = {
        "format": "cts_materialized_advantage_cache_v2",
        "metadata": {
            "manifest_path": config.manifest_path,
            "encoder_checkpoint": config.encoder_checkpoint,
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


def main(config: MaterializeConfig) -> None:
    """Dispatch to the requested subcommand based on ``config.command``."""
    if config.command == "materialize":
        materialize_worker(config)
    elif config.command == "merge":
        merge_workers(config)


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(MaterializeConfig, main)
