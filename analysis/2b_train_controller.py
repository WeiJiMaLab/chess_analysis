#!/usr/bin/env python3
"""Stage 2b: train the metacontroller on ysagiv materialized caches (cheap proxy run).

Mirrors ``cts.train.controller_train``'s frozen-encoder path for ``max_batches``
optimizer steps, then plots per-batch loss. Use this as a stable stand-in for a
full Slurm run when iterating on analysis or configs.

Loss (same as production, ``sign_loss_weight`` default 0.1)::

    total_loss = advantage_mse + sign_loss_weight * sign_bce

- **advantage_mse** — regression: match the oracle advantage ``A = Q_continue − Q_halt``
  (how much better continuing search is vs stopping now). Measures magnitude errors.
- **sign_bce** — classification: predict ``sign(A)`` (continue vs halt). Anchors the
  halt/continue decision even when ``A`` is near zero. Often larger in raw value than MSE.

Usage::

    python3 analysis/2b_train_controller.py --max-batches 1000
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_ANALYSIS_ROOT = Path(__file__).resolve().parent
_LMCOS_ROOT = _ANALYSIS_ROOT.parent / "lmcos"
for _p in (_ANALYSIS_ROOT, _LMCOS_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cts._config import load_config
from cts.core.schema import tree_encoder_feature_schema
from cts.train.controller_train import (
    ControllerTrainConfig,
    MaterializedCache,
    _advantage_loss_components,
    _build_model_and_optimizer,
    _load_materialized_cache,
    _sign_auxiliary_loss,
)

from config import (
    ANALYSIS_ROOT,
    CACHE_SUBTREE_WEIGHTED_TRAIN,
    CONTROLLER_TRAIN_CONFIG_BEST,
    PACKED_TRAIN_MANIFEST,
)


@dataclass
class BatchRecord:
    batch: int
    elapsed_s: float
    total_loss: float
    advantage_mse: float
    sign_bce: float
    snapshots: int


def _print_scale(cache: MaterializedCache, batch_size: int) -> int:
    batches_per_epoch = math.ceil(cache.examples / batch_size)
    print("Train materialized cache scale")
    print(f"  total snapshots:     {cache.examples:,}")
    print(f"  batch_size:          {batch_size}")
    print(f"  batches / epoch:     {batches_per_epoch:,}")
    print(f"  train episodes:      ~576,830  (many snapshots per episode)")
    return batches_per_epoch


def train_max_batches(
    *,
    model,
    cache: MaterializedCache,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
    max_batches: int,
    seed: int,
) -> list[BatchRecord]:
    """Same loop as ``_train_materialized_cache_epoch``, capped at ``max_batches``."""
    model.train()
    records: list[BatchRecord] = []
    batch_counter = 0
    started = time.perf_counter()

    shard_order = list(range(len(cache.shard_paths)))
    random.Random(seed).shuffle(shard_order)

    for shard_index in shard_order:
        if batch_counter >= max_batches:
            break
        payload = torch.load(cache.shard_paths[shard_index], weights_only=False)
        features = payload["features"]
        target_advantages = payload["target_advantages"]
        order = torch.randperm(features.shape[0])
        for start in range(0, int(features.shape[0]), batch_size):
            if batch_counter >= max_batches:
                break
            batch_counter += 1
            batch_index = order[start : start + batch_size]
            batch_features = features[batch_index].to(device, non_blocking=True)
            batch_targets = target_advantages[batch_index].to(device, non_blocking=True)

            predicted, sign_logits = model.predict_from_features(batch_features)
            advantage_mse, _, _ = _advantage_loss_components(predicted, batch_targets)
            sign_loss = _sign_auxiliary_loss(sign_logits, batch_targets)
            total_loss = advantage_mse + sign_loss_weight * sign_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            records.append(
                BatchRecord(
                    batch=batch_counter,
                    elapsed_s=time.perf_counter() - started,
                    total_loss=float(total_loss.item()),
                    advantage_mse=float(advantage_mse.item()),
                    sign_bce=float(sign_loss.item()),
                    snapshots=int(batch_targets.shape[0]),
                )
            )

    if not records:
        raise RuntimeError("No batches ran — is the train cache present?")
    return records


def plot_loss_vs_batch(records: list[BatchRecord], output: Path, sign_loss_weight: float) -> None:
    batches = [r.batch for r in records]
    total_loss = [r.total_loss for r in records]
    mse = [r.advantage_mse for r in records]
    bce = [r.sign_bce for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(batches, total_loss, linewidth=1.2, color="#4c72b0")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Loss")
    ax.set_title("Total loss vs batch index")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(batches, mse, label="advantage_mse (regression on A)", linewidth=1.2)
    ax.plot(batches, bce, label="sign_bce (halt vs continue)", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Loss component")
    ax.set_title("Loss components vs batch index")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Stage 2b controller ({len(records)} batches, "
        f"total = mse + {sign_loss_weight}×bce)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS_ROOT / "outputs" / "2b_train_controller_loss.png",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONTROLLER_TRAIN_CONFIG_BEST,
    )
    args = parser.parse_args()

    if not CACHE_SUBTREE_WEIGHTED_TRAIN.is_file():
        raise SystemExit(f"Missing train cache: {CACHE_SUBTREE_WEIGHTED_TRAIN}")

    train_config = load_config(ControllerTrainConfig, str(args.config))
    device = torch.device(args.device)
    schema = tree_encoder_feature_schema()

    cache = _load_materialized_cache(
        str(CACHE_SUBTREE_WEIGHTED_TRAIN),
        manifest_path=str(PACKED_TRAIN_MANIFEST),
        encoder_checkpoint=train_config.encoder_checkpoint,
    )
    batches_per_epoch = _print_scale(cache, train_config.batch_size)

    model, optimizer, _ = _build_model_and_optimizer(train_config, schema)
    model.to(device)

    print(f"Running {args.max_batches} training batches on {device} ...", flush=True)
    t0 = time.perf_counter()
    records = train_max_batches(
        model=model,
        cache=cache,
        optimizer=optimizer,
        device=device,
        batch_size=train_config.batch_size,
        sign_loss_weight=train_config.sign_loss_weight,
        max_batches=args.max_batches,
        seed=train_config.seed,
    )
    wall = time.perf_counter() - t0
    per_batch = wall / len(records)
    print()
    print(f"Completed {len(records)} batches in {wall:.2f} s ({per_batch*1000:.1f} ms/batch)")
    print(f"Extrapolated train-only 1 epoch: {per_batch * batches_per_epoch / 60:.1f} min")
    print(
        f"Last batch: total={records[-1].total_loss:.4f} "
        f"mse={records[-1].advantage_mse:.4f} bce={records[-1].sign_bce:.4f}"
    )

    plot_loss_vs_batch(records, args.output, train_config.sign_loss_weight)


if __name__ == "__main__":
    main()
