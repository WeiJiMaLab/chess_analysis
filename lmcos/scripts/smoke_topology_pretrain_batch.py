#!/usr/bin/env python3
"""Smoke: load a packed node targets shard and run one Huber training step.

Avoids ``PackedTensorizedShardDataset`` full-corpus preload — one ``torch.load``
per shard, then only ``--num-examples`` trees are sliced out for a single batch.

Example (production shard, CPU):
  python3 scripts/smoke_topology_pretrain_batch.py \
    --shard /scratch/gpfs/GRIFFITHS/hl4291/chess/CTS/data/pretrain_packed_50k_topology_full/train/shard_00000.pt \
    --num-examples 4 --batch-size 2 --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cts.core.schema import tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeExample
from cts.data.build_tree import BuildTreeConfig, _build_nodetargets_model
from cts.train.gnn_pretrain import NodePretrainConfig, NodePretrainer


def _examples_from_shard(shard_path: Path, num_examples: int) -> list[TensorizedTreeExample]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "cts_tensorized_pretrain_shard_v1":
        raise ValueError(f"Unexpected shard format: {shard_path}")
    # Read either old key or new key to support legacy and new shards transparently
    shard_topology = payload.get("topology_targets")
    if shard_topology is None:
        raise ValueError(f"Shard missing topology/node targets: {shard_path}")

    node_ptr = payload["node_ptr"]
    edge_ptr = payload["edge_ptr"]
    feature_names = tuple(payload["feature_names"])
    edge_wdl_targets = payload.get("edge_wdl_targets")
    total = int(payload["num_examples"])
    take = min(num_examples, total)

    examples: list[TensorizedTreeExample] = []
    for i in range(take):
        ns = int(node_ptr[i].item())
        ne = int(node_ptr[i + 1].item())
        es = int(edge_ptr[i].item())
        ee = int(edge_ptr[i + 1].item())
        edge_slot = payload["edge_slot"][es:ee].clone()
        node_feat_block = payload["node_features"][ns:ne].clone()
        examples.append(
            TensorizedTreeExample(
                node_features=node_feat_block,
                parent_index=payload["parent_index"][ns:ne].clone(),
                edge_parent=payload["edge_parent"][es:ee].clone(),
                edge_child=payload["edge_child"][es:ee].clone(),
                edge_slot=edge_slot,
                depth=payload["depth"][ns:ne].clone(),
                node_targets=payload["node_targets"][ns:ne].clone(),
                feature_names=feature_names,
                edge_wdl_targets=edge_wdl_targets[es:ee].clone() if edge_wdl_targets is not None else None,
                nodetargets_targets=shard_topology[ns:ne].clone(),
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="One-batch node targets pretrain smoke.")
    parser.add_argument("--shard", type=Path, required=True, help="Path to shard_XXXXX.pt")
    parser.add_argument("--num-examples", type=int, default=4, help="Trees to slice from shard")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--k", type=int, default=1, help="Encoder message-passing rounds")
    args = parser.parse_args()

    if not args.shard.is_file():
        raise FileNotFoundError(args.shard)

    t0 = time.perf_counter()
    examples = _examples_from_shard(args.shard, args.num_examples)
    load_s = time.perf_counter() - t0
    print(f"loaded shard={args.shard} examples={len(examples)} elapsed_s={load_s:.2f}")
    if examples:
        ex0 = examples[0]
        print(
            f"  example[0] nodes={ex0.node_features.shape[0]} "
            f"node_feat_cols={ex0.node_features.shape[1]} "
            f"nodetargets_targets={tuple(ex0.nodetargets_targets.shape)}"
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda not available; falling back to cpu")
        device = "cpu"

    schema = tree_encoder_feature_schema(node_targets=False)
    build_cfg = BuildTreeConfig(
        command="pretrain-nodetargets-encoder",
        device=device,
        k=args.k,
        d_embed=64,
        d_message=64,
        n_heads=2,
        d_att=16,
        node_embed_hidden=64,
        decoder_hidden=64,
    )
    model = _build_nodetargets_model(build_cfg, schema)
    trainer = NodePretrainer(
        model=model,
        device=device,
        train_examples=examples,
        validation_examples=examples[:1],
        config=NodePretrainConfig(
            batch_size=min(args.batch_size, len(examples)),
            learning_rate=1e-3,
            epochs=1,
            loss_type="huber",
            huber_delta=1.0,
            shuffle=False,
            num_workers=0,
        ),
    )

    t1 = time.perf_counter()
    metrics = trainer.train_epoch(
        1,
        batch_progress_callback=lambda epoch, phase, bi, tb, seen_ex, seen_nodes, loss: print(
            f"  batch {bi}/{tb} phase={phase} trees={seen_ex} nodes={seen_nodes} huber_loss={loss:.6f}"
        ),
    )
    step_s = time.perf_counter() - t1
    print(
        f"train_epoch done total_loss={metrics.total_loss:.6f} "
        f"supervised_nodes={metrics.num_supervised_nodes} elapsed_s={step_s:.2f}"
    )
    print("smoke ok")


if __name__ == "__main__":
    main()
