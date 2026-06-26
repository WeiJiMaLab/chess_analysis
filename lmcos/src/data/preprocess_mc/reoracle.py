"""Re-run the budgeted oracle DP with a new cost config on an existing packed dataset.

Reads halt_rewards and tree_sizes from the existing packed mc shards, re-runs
compute_budgeted_oracle with a new BudgetedOracleConfig, and writes a patched
materialized cache (same z_t features, new target_advantages / oracle_stop_steps).
Also writes a patched packed manifest so controller_train oracle-config validation passes.

Usage:
    python -m cts.data.preprocess_mc.reoracle --config reoracle.yaml
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from pydantic import BaseModel

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    BudgetBucket,
    budgeted_oracle_metadata,
    compute_budgeted_oracle,
)


class ReOracleConfig(BaseModel):
    # Source: existing packed mc manifest (provides halt_rewards, tree_sizes)
    packed_train_manifest: str
    packed_validation_manifest: str
    # Source: existing materialized cache (provides z_t features)
    materialized_train_cache: str
    materialized_validation_cache: str
    # Output directory
    output_dir: str
    # New oracle parameters
    time_lambda: float = 18.537
    time_p: float = 2.8
    time_tau: float = 2.5
    time_delta: int = 1
    timeout_value: float = -1.0
    time_mode: str = "power_law"
    maintenance_scale: float = 0.0
    maintenance_ref_nodes: float = 30.0
    maintenance_exponent: float = 1.1
    samples_per_bucket: int = 2
    seed: int = 0


def _oracle_config(cfg: ReOracleConfig) -> BudgetedOracleConfig:
    return BudgetedOracleConfig(
        maintenance_scale=cfg.maintenance_scale,
        maintenance_ref_nodes=cfg.maintenance_ref_nodes,
        maintenance_exponent=cfg.maintenance_exponent,
        time_lambda=cfg.time_lambda,
        time_p=cfg.time_p,
        time_tau=cfg.time_tau,
        time_delta=cfg.time_delta,
        timeout_value=cfg.timeout_value,
        time_mode=cfg.time_mode,
    )


def _iter_episodes(manifest_path: str):
    """Yield (halt_rewards, tree_sizes, starting_budget, oracle_value_old) per episode in manifest order."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest["entries"]:
        shard = torch.load(entry["path"], map_location="cpu", weights_only=False)
        esp = shard["episode_step_ptr"].numpy()
        hr = shard["trajectory_halt_rewards"].numpy()
        scf = shard["step_node_cutoffs"].numpy()  # tree sizes per step
        budgets = shard["starting_budgets"].numpy()
        for i in range(len(esp) - 1):
            ep_hr = hr[esp[i] : esp[i + 1]].tolist()
            ep_sz = scf[esp[i] : esp[i + 1]].tolist()
            yield ep_hr, ep_sz, int(budgets[i])


def _patch_materialized_cache(
    cache_path: str,
    new_advantages: List[float],
    new_stop_steps: List[int],
    output_cache_path: str,
) -> None:
    """Write a new materialized cache with same features but patched oracle targets."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    shard_dir_src = Path(str(cache_path) + ".d")

    out_path = Path(output_cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_shard_dir = Path(str(output_cache_path) + ".d")
    out_shard_dir.mkdir(parents=True, exist_ok=True)

    adv_tensor = torch.tensor(new_advantages, dtype=torch.float32)
    stop_tensor = torch.tensor(new_stop_steps, dtype=torch.int32)

    offset = 0
    new_shards = []
    for shard_meta in cache["shards"]:
        src_shard = torch.load(shard_meta["path"], map_location="cpu", weights_only=False)
        n = src_shard["features"].shape[0]
        patched = {
            "format": src_shard["format"],
            "features": src_shard["features"],
            "target_advantages": adv_tensor[offset : offset + n],
            "oracle_stop_steps": stop_tensor[offset : offset + n],
        }
        idx = len(new_shards)
        dst = out_shard_dir / f"shard_{idx:05d}.pt"
        torch.save(patched, dst)
        new_shards.append({"path": str(dst), "examples": n})
        offset += n
        print(f"  shard {idx}: {n} snapshots (offset {offset})", flush=True)

    assert offset == len(new_advantages), f"Mismatch: {offset} != {len(new_advantages)}"

    out_manifest = {k: v for k, v in cache.items() if k != "shards"}
    out_manifest["shards"] = new_shards
    torch.save(out_manifest, out_path)
    print(f"Wrote {out_path} ({offset} total snapshots)", flush=True)


def _patch_packed_manifest(
    manifest_path: str,
    oracle_config: BudgetedOracleConfig,
    output_manifest_path: str,
) -> None:
    """Write a copy of the packed manifest with updated oracle metadata fields."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    meta = budgeted_oracle_metadata(oracle_config)
    manifest.update({k: v for k, v in meta.items() if k != "oracle_type"})
    manifest["oracle_type"] = meta["oracle_type"]
    manifest["time_mode"] = oracle_config.time_mode
    Path(output_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Wrote manifest {output_manifest_path}", flush=True)


def reoracle(cfg: ReOracleConfig) -> None:
    oracle = _oracle_config(cfg)
    out = Path(cfg.output_dir)

    for split, packed_manifest, mat_cache in [
        ("train", cfg.packed_train_manifest, cfg.materialized_train_cache),
        ("validation", cfg.packed_validation_manifest, cfg.materialized_validation_cache),
    ]:
        print(f"\n=== {split} ===", flush=True)

        # Re-run oracle on all episodes in manifest order
        new_advantages: List[float] = []
        new_stop_steps: List[int] = []
        n_episodes = 0

        for halt_rewards, tree_sizes, starting_budget in _iter_episodes(packed_manifest):
            policy = compute_budgeted_oracle(halt_rewards, tree_sizes, starting_budget, oracle)
            new_advantages.extend(policy.target_advantages)
            # oracle_stop_steps in materialized cache is per-snapshot (same value repeated for each step)
            # Each step's oracle_stop_steps entry = optimal stop from that step onward
            new_stop_steps.extend(policy.stop_steps)
            n_episodes += 1
            if n_episodes % 10000 == 0:
                print(f"  episodes={n_episodes} snapshots={len(new_advantages)}", flush=True)

        print(f"  total episodes={n_episodes} snapshots={len(new_advantages)}", flush=True)

        # Patch materialized cache
        out_cache = str(out / f"{split}_cache.pt")
        _patch_materialized_cache(mat_cache, new_advantages, new_stop_steps, out_cache)

        # Patch packed manifest (for oracle config validation in controller_train)
        out_manifest = str(out / f"{split}_manifest.json")
        _patch_packed_manifest(packed_manifest, oracle, out_manifest)


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(ReOracleConfig, reoracle)
