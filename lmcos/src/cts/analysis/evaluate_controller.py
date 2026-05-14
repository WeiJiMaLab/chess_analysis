"""Closed-loop evaluation of a trained controller checkpoint.

Instantiates the encoder + controller + budgeted-oracle provider from a saved
checkpoint, replays each packed validation episode under the controller's
greedy policy, and reports actual halt regret against the oracle. Used as a
post-training sanity check that the diagnostics-measured loss actually
translates into real performance. Run manually; no standard slurm script.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ConfigDict

from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_mc.oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    budgeted_oracle_metadata,
)
from cts.models.mc import MetaController
from cts.train.controller_train import (
    MaterializedCache,
    ControllerEpisodeDataset,
    _build_packed_loader,
    _default_materialized_cache_path,
    _load_materialized_cache,
    _save_materialized_cache_shards,
    _validate_packed_manifest_oracle,
    _write_diagnostics,
    evaluate_batched_greedy_policy,
)


class EvaluateControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str
    packed_validation_data: str
    encoder_checkpoint: str
    output_diagnostics: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    # Architecture args: these must match the checkpoint exactly. The training
    # script writes them into checkpoint["metadata"] for reference, but the
    # loader doesn't auto-read them, so the caller passes them explicitly.
    k: int = 1
    node_embed_hidden: int = 128
    d_embed: int = 128
    d_message: int = 128
    n_heads: int = 4
    d_att: int = 32
    hidden_dim: int = 256
    hidden_layers: int = 3
    separate_sign_head: bool = False
    episode_batch_size: int = 8
    num_workers: int = 0
    log_interval: int = 10000
    materialized_cache_shard_snapshots: int = 250000
    # Oracle config (defaults match training script). Changing any of these
    # invalidates a previously-materialized cache for the same manifest.
    maintenance_scale: float = 0.0
    maintenance_ref_nodes: float = 30.0
    maintenance_exponent: float = 1.1
    time_lambda: float = 18.537
    time_p: float = 2.8
    time_tau: float = 2.5
    time_delta: int = 1
    timeout_value: float = -1.0
    samples_per_bucket: int = 2
    scramble_min_time: int = 1
    scramble_max_time: int = 3
    medium_small_min_time: int = 4
    medium_small_max_time: int = 10
    medium_large_min_time: int = 11
    medium_large_max_time: int = 25
    large_min_time: int = 26
    large_max_time: int = 60
    very_large_min_time: int = 61
    very_large_max_time: int = 120


def _oracle_config(config: EvaluateControllerConfig) -> BudgetedOracleConfig:
    """Assemble the oracle config from the loaded config. Defaults match the training script."""
    return BudgetedOracleConfig(
        maintenance_scale=config.maintenance_scale,
        maintenance_ref_nodes=config.maintenance_ref_nodes,
        maintenance_exponent=config.maintenance_exponent,
        time_lambda=config.time_lambda,
        time_p=config.time_p,
        time_tau=config.time_tau,
        time_delta=config.time_delta,
        timeout_value=config.timeout_value,
        samples_per_bucket=config.samples_per_bucket,
        # Five canonical compute-budget buckets; min/max times are in node-expansions.
        budget_buckets=[
            BudgetBucket("scramble", config.scramble_min_time, config.scramble_max_time),
            BudgetBucket("medium-small", config.medium_small_min_time, config.medium_small_max_time),
            BudgetBucket("medium-large", config.medium_large_min_time, config.medium_large_max_time),
            BudgetBucket("large", config.large_min_time, config.large_max_time),
            BudgetBucket("very-large", config.very_large_min_time, config.very_large_max_time),
        ],
    )


def main(config: EvaluateControllerConfig) -> None:
    device = torch.device(config.device)
    oracle_config = _oracle_config(config)
    # Guard: refuse to evaluate against a manifest whose recorded oracle
    # config doesn't match the one we just assembled. Avoids silently
    # comparing apples-to-oranges across runs.
    _validate_packed_manifest_oracle(config.packed_validation_data, oracle_config)
    schema = tree_encoder_feature_schema()

    # Build model and load checkpoint. Architecture must match the saved
    # state dict exactly; load_state_dict will raise on mismatched keys.
    model = MetaController(
        k=config.k,
        node_feat=len(schema.feature_names),
        device=config.device,
        node_embed_hidden=config.node_embed_hidden,
        d_embed=config.d_embed,
        d_message=config.d_message,
        n_heads=config.n_heads,
        d_att=config.d_att,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        separate_sign_head=config.separate_sign_head,
    )
    # Trust the encoder embedded in the controller checkpoint. --encoder-checkpoint
    # is kept only as the cache key so multiple controllers trained against the same
    # encoder share a materialized cache.
    checkpoint = torch.load(config.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.freeze_encoder()
    print(f"[evaluate_controller] checkpoint={config.checkpoint}", flush=True)
    if "metadata" in checkpoint:
        print(f"[evaluate_controller] checkpoint_metadata={json.dumps(checkpoint['metadata'])}", flush=True)

    # Load or materialize validation cache. The cache holds the encoder's
    # per-snapshot tree activations so each evaluation re-uses them instead
    # of re-running the (frozen) encoder on every episode replay.
    cache_path = _default_materialized_cache_path(
        config.packed_validation_data, config.encoder_checkpoint, "validation",
    )
    validation_loader = _build_packed_loader(
        config.packed_validation_data,
        batch_size=config.episode_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    if Path(cache_path).exists():
        print(f"[evaluate_controller] stage=load_cache path={cache_path}", flush=True)
        cache = _load_materialized_cache(
            cache_path,
            manifest_path=config.packed_validation_data,
            encoder_checkpoint=config.encoder_checkpoint,
        )
    else:
        # First-time path: walk the loader once to materialize and shard the
        # cache to disk. Subsequent runs against the same manifest reuse it.
        print("[evaluate_controller] stage=materialize_cache", flush=True)
        cache = _save_materialized_cache_shards(
            cache_path,
            model,
            validation_loader,
            manifest_path=config.packed_validation_data,
            encoder_checkpoint=config.encoder_checkpoint,
            log_interval=config.log_interval,
            split_name="validation",
            max_snapshots_per_shard=config.materialized_cache_shard_snapshots,
        )

    # Run greedy evaluation: replay every packed episode with the controller
    # picking actions argmax-style and compare halt-step / regret against
    # the oracle's optimal stop.
    validation_dataset = ControllerEpisodeDataset(config.packed_validation_data)
    # ``diagnostics`` is None when no output file was requested; the eval
    # function checks for None and skips per-step record-keeping in that case.
    diagnostics: List[Dict[str, Any]] = [] if config.output_diagnostics else None

    print("[evaluate_controller] stage=greedy_eval", flush=True)
    greedy_metrics = evaluate_batched_greedy_policy(
        model,
        validation_dataset,
        cache,
        oracle_config,
        log_interval=config.log_interval,
        diagnostics_out=diagnostics,
    )
    print(
        f"exact_stop_step_accuracy={greedy_metrics.exact_stop_step_accuracy:.3f} "
        f"first_action_accuracy={greedy_metrics.first_action_accuracy:.3f} "
        f"average_return={greedy_metrics.average_return:.3f} "
        f"average_oracle_value={greedy_metrics.average_oracle_value:.3f} "
        f"average_regret={greedy_metrics.average_regret:.3f} "
        f"average_expansions={greedy_metrics.average_expansions:.3f} "
        f"evaluated_episodes={greedy_metrics.evaluated_episodes}",
        flush=True,
    )
    if diagnostics is not None:
        _write_diagnostics(diagnostics, config.output_diagnostics)
        print(f"[evaluate_controller] diagnostics_written={config.output_diagnostics}", flush=True)

    print("[evaluate_controller] stage=done", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(EvaluateControllerConfig, main)
