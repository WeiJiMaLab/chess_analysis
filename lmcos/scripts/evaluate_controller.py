"""Evaluate a trained controller checkpoint on an arbitrary packed validation set."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from budgeted_controller_oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    budgeted_oracle_metadata,
)
from schema import tree_encoder_feature_schema
from train_fitted_q_controller import (
    ComputeAdvantageTreeSearchModel,
    MaterializedCache,
    PackedControllerEpisodeDataset,
    _build_packed_loader,
    _default_materialized_cache_path,
    _load_materialized_cache,
    _save_materialized_cache_shards,
    _validate_packed_manifest_oracle,
    _write_diagnostics,
    evaluate_batched_greedy_policy,
)


def _oracle_config(args: argparse.Namespace) -> BudgetedOracleConfig:
    return BudgetedOracleConfig(
        maintenance_scale=args.maintenance_scale,
        maintenance_ref_nodes=args.maintenance_ref_nodes,
        maintenance_exponent=args.maintenance_exponent,
        time_lambda=args.time_lambda,
        time_p=args.time_p,
        time_tau=args.time_tau,
        time_delta=args.time_delta,
        timeout_value=args.timeout_value,
        samples_per_bucket=args.samples_per_bucket,
        budget_buckets=[
            BudgetBucket("scramble", args.scramble_min_time, args.scramble_max_time),
            BudgetBucket("medium-small", args.medium_small_min_time, args.medium_small_max_time),
            BudgetBucket("medium-large", args.medium_large_min_time, args.medium_large_max_time),
            BudgetBucket("large", args.large_min_time, args.large_max_time),
            BudgetBucket("very-large", args.very_large_min_time, args.very_large_max_time),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a controller checkpoint on a packed validation set.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained controller checkpoint.")
    parser.add_argument("--packed-validation-data", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output-diagnostics", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    # Architecture args (must match checkpoint).
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    parser.add_argument("--q-hidden", type=int, default=256)
    parser.add_argument("--q-hidden-layers", type=int, default=3)
    parser.add_argument("--separate-sign-head", action="store_true")
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--materialized-cache-shard-snapshots", type=int, default=250000)
    # Oracle config (defaults match training script).
    parser.add_argument("--maintenance-scale", type=float, default=0.0)
    parser.add_argument("--maintenance-ref-nodes", type=float, default=30.0)
    parser.add_argument("--maintenance-exponent", type=float, default=1.1)
    parser.add_argument("--time-lambda", type=float, default=18.537)
    parser.add_argument("--time-p", type=float, default=2.8)
    parser.add_argument("--time-tau", type=float, default=2.5)
    parser.add_argument("--time-delta", type=int, default=1)
    parser.add_argument("--timeout-value", type=float, default=-1.0)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--scramble-min-time", type=int, default=1)
    parser.add_argument("--scramble-max-time", type=int, default=3)
    parser.add_argument("--medium-small-min-time", type=int, default=4)
    parser.add_argument("--medium-small-max-time", type=int, default=10)
    parser.add_argument("--medium-large-min-time", type=int, default=11)
    parser.add_argument("--medium-large-max-time", type=int, default=25)
    parser.add_argument("--large-min-time", type=int, default=26)
    parser.add_argument("--large-max-time", type=int, default=60)
    parser.add_argument("--very-large-min-time", type=int, default=61)
    parser.add_argument("--very-large-max-time", type=int, default=120)
    args = parser.parse_args()

    device = torch.device(args.device)
    oracle_config = _oracle_config(args)
    _validate_packed_manifest_oracle(args.packed_validation_data, oracle_config)
    schema = tree_encoder_feature_schema()

    # Build model and load checkpoint.
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
        separate_sign_head=args.separate_sign_head,
    )
    # Trust the encoder embedded in the controller checkpoint. --encoder-checkpoint
    # is kept only as the cache key so multiple controllers trained against the same
    # encoder share a materialized cache.
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.freeze_encoder()
    print(f"[evaluate_controller] checkpoint={args.checkpoint}", flush=True)
    if "metadata" in checkpoint:
        print(f"[evaluate_controller] checkpoint_metadata={json.dumps(checkpoint['metadata'])}", flush=True)

    # Load or materialize validation cache.
    cache_path = _default_materialized_cache_path(
        args.packed_validation_data, args.encoder_checkpoint, "validation",
    )
    validation_loader = _build_packed_loader(
        args.packed_validation_data,
        batch_size=args.episode_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if Path(cache_path).exists():
        print(f"[evaluate_controller] stage=load_cache path={cache_path}", flush=True)
        cache = _load_materialized_cache(
            cache_path,
            manifest_path=args.packed_validation_data,
            encoder_checkpoint=args.encoder_checkpoint,
        )
    else:
        print("[evaluate_controller] stage=materialize_cache", flush=True)
        cache = _save_materialized_cache_shards(
            cache_path,
            model,
            validation_loader,
            manifest_path=args.packed_validation_data,
            encoder_checkpoint=args.encoder_checkpoint,
            log_interval=args.log_interval,
            split_name="validation",
            max_snapshots_per_shard=args.materialized_cache_shard_snapshots,
        )

    # Run greedy evaluation.
    validation_dataset = PackedControllerEpisodeDataset(args.packed_validation_data)
    diagnostics: List[Dict[str, Any]] = [] if args.output_diagnostics else None

    print("[evaluate_controller] stage=greedy_eval", flush=True)
    greedy_metrics = evaluate_batched_greedy_policy(
        model,
        validation_dataset,
        cache,
        oracle_config,
        log_interval=args.log_interval,
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
        _write_diagnostics(diagnostics, args.output_diagnostics)
        print(f"[evaluate_controller] diagnostics_written={args.output_diagnostics}", flush=True)

    print("[evaluate_controller] stage=done", flush=True)


if __name__ == "__main__":
    main()
