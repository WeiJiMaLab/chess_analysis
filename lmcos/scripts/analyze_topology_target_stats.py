#!/usr/bin/env python3
"""Coarse stats for packed topology_targets columns (sample shards or full manifest)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from cts.core.schema import topology_target_feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-shards", type=int, default=0, help="0 = all shards in manifest")
    parser.add_argument("--max-nodes", type=int, default=2_000_000, help="Stop after this many nodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)
    entries = list(manifest["entries"])
    if args.max_shards > 0:
        rng = random.Random(args.seed)
        rng.shuffle(entries)
        entries = entries[: args.max_shards]

    names = topology_target_feature_names()
    n_dims = len(names)
    count = 0
    sum_ = torch.zeros(n_dims, dtype=torch.float64)
    sum_sq = torch.zeros(n_dims, dtype=torch.float64)
    vmin = torch.full((n_dims,), float("inf"))
    vmax = torch.full((n_dims,), float("-inf"))
    # Welford-style extrema on quantile samples
    reservoir: list[torch.Tensor] = []
    reservoir_cap = 500_000

    for entry in entries:
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        topo = payload.get("topology_targets")
        if topo is None:
            raise ValueError(f"No topology_targets in {entry['path']}")
        n_take = int(entry.get("num_examples", payload["num_examples"]))
        # slice only examples in this shard entry
        node_ptr = payload["node_ptr"]
        ns = int(node_ptr[0].item())
        ne = int(node_ptr[min(n_take, int(node_ptr.shape[0]) - 1)].item())
        block = topo[ns:ne].to(torch.float64)
        if block.numel() == 0:
            continue
        count += block.shape[0]
        sum_ += block.sum(dim=0)
        sum_sq += (block * block).sum(dim=0)
        vmin = torch.minimum(vmin, block.min(dim=0).values)
        vmax = torch.maximum(vmax, block.max(dim=0).values)
        if len(reservoir) < reservoir_cap:
            need = reservoir_cap - len(reservoir)
            if block.shape[0] <= need:
                reservoir.append(block)
            else:
                idx = torch.randperm(block.shape[0])[:need]
                reservoir.append(block[idx])
        if count >= args.max_nodes:
            break
        del payload

    if count == 0:
        raise SystemExit("No nodes collected.")

    mean = sum_ / count
    var = (sum_sq / count) - mean * mean
    std = torch.sqrt(var.clamp_min(0.0))
    sample = torch.cat(reservoir, dim=0)
    qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=sample.dtype)
    quantiles = torch.quantile(sample, qs, dim=0)

    print(f"manifest={args.manifest}")
    print(f"nodes={count:,} shards_scanned={len(entries)}")
    print()
    header = f"{'dim':<24} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        print(
            f"{name:<24} {mean[i]:10.4f} {std[i]:10.4f} "
            f"{vmin[i]:10.4f} {vmax[i]:10.4f}"
        )
    print()
    print("Quantiles (reservoir sample):")
    for q_idx, q in enumerate(qs):
        parts = " ".join(f"{quantiles[q_idx, j]:8.4f}" for j in range(n_dims))
        print(f"  p{int(q*100):02d}: {parts}")
    print()
    print("Column labels:", ", ".join(names))
    # z-score reference: pred and target would both use (x - mean) / std
    print("\nSuggested z-score scales (train mean/std):")
    for i, name in enumerate(names):
        s = float(std[i])
        print(f"  {name}: mean={mean[i]:.6f} std={s:.6f}" + ("  (std~0!)" if s < 1e-6 else ""))


if __name__ == "__main__":
    main()
