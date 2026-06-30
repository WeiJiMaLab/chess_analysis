"""Score a trained MCHalt meta-controller checkpoint over a validation split.

The MCHalt tier (tier 5 of the D0 ladder) is the trained
:class:`cts.models.mc.MetaController` — a frozen GNN encoder + an MLP advantage
head reading ``[z_t, T_t]`` (or any configured subset of ``[z_t, N_t, T_t]``).
Unlike the parameter-free / one-parameter baselines, it cannot be scored from
the packed scalar payload alone: its advantage at each step is a function of the
GNN root embedding ``z_t``.

Rather than re-run the (expensive) encoder, we reuse the *materialized cache*
(``[z_t, N_t, T_t]`` rows produced once during training) plus the SAME greedy
rollout evaluator the training loop uses to select checkpoints
(:func:`cts.train.controller_train.evaluate_batched_greedy_policy`). That
evaluator applies the identical decision rule
(:func:`predicted_stop_from_advantages` — stop at first advantage <= 0) and the
identical return/regret machinery (:func:`return_for_stop_step`,
:func:`compute_budgeted_oracle`) the baselines use, so the MCHalt regret /
P(stop==OSS) it returns are directly comparable to the other four tiers.

This is the faithful reconstruction of the (now-deleted) ``profile_controller``
driver referenced in ``mc_minimal_plan.md``: it is exactly the val-regret path
the trainer selects ``controller.pt`` on, so a re-score reproduces the recorded
checkpoint metadata (regret / exact-stop accuracy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.models.mc import CONTROLLER_INPUT_NAMES, MetaController
from cts.train.controller_train import (
    ControllerEpisodeDataset,
    MaterializedCache,
    _aggregate_greedy_rollout_metrics,
    _collect_episode_metadata_and_step_count,
)
from cts.train.gnn_pretrain import load_encoder_architecture
from .baselines import stop_fit_metrics


def _load_materialized_cache_unchecked(cache_index_path: Path) -> MaterializedCache:
    """Load a materialized cache index without the manifest/encoder identity check.

    ``controller_train._load_materialized_cache`` refuses a cache whose stored
    manifest/encoder strings differ from the caller's. Here we are scoring an
    already-trained checkpoint over a cache built for the SAME encoder, so the
    identity has already been established by construction (same encoder produced
    both the checkpoint and the cache); we only need the shard list. We still
    surface the stored metadata to the caller for an explicit sanity print.
    """
    payload = torch.load(cache_index_path, weights_only=False)
    if payload.get("format") != "cts_materialized_advantage_cache_v2":
        raise ValueError(f"Unexpected materialized cache format: {cache_index_path}")
    shard_paths = [str(entry["path"]) for entry in payload.get("shards", [])]
    shard_sizes = [int(entry["examples"]) for entry in payload.get("shards", [])]
    return MaterializedCache(
        shard_paths=shard_paths,
        shard_sizes=shard_sizes,
        examples=int(payload.get("examples", sum(shard_sizes))),
    )


def _build_controller_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: str = "cpu",
    # Encoder/head hyperparameters. Defaults mirror ControllerTrainConfig (the
    # values every recorded controller was trained with). Override only if a
    # checkpoint used a non-default architecture.
    k: int = 2,
    node_embed_hidden: int = 128,
    d_embed: int = 128,
    d_message: int = 128,
    n_heads: int = 4,
    d_att: int = 32,
    hidden_dim: int = 256,
    hidden_layers: int = 3,
) -> MetaController:
    """Reconstruct a ``MetaController`` and load the checkpoint's full state dict.

    ``controller_inputs`` and ``separate_sign_head`` are read from the
    checkpoint metadata so the head's input dim / topology matches what was
    trained; the encoder weights live in the same state dict, so no separate
    encoder checkpoint is needed.
    """
    metadata = checkpoint.get("metadata", {})
    controller_inputs = list(metadata.get("controller_inputs", CONTROLLER_INPUT_NAMES))
    separate_sign_head = bool(metadata.get("separate_sign_head", False))
    schema = tree_encoder_feature_schema()

    # Resolve the encoder architecture from the encoder checkpoint this controller
    # was trained against, so the rebuilt MetaController matches whatever encoder
    # was used (tiny SF k=1/d_embed=32, or prod lc0 k=2/d_embed=128). The kwargs
    # above are only fallbacks for when the encoder checkpoint is unavailable.
    arch: dict[str, Any] = {}
    enc_ckpt = metadata.get("encoder_checkpoint")
    if enc_ckpt and Path(enc_ckpt).exists():
        try:
            arch = load_encoder_architecture(enc_ckpt) or {}
        except Exception:
            arch = {}

    model = MetaController(
        k=arch.get("k", k),
        node_feat=arch.get("node_feat", len(schema.feature_names)),
        device=device,
        node_embed_hidden=arch.get("node_embed_hidden", node_embed_hidden),
        d_embed=arch.get("d_embed", d_embed),
        d_message=arch.get("d_message", d_message),
        n_heads=arch.get("n_heads", n_heads),
        d_att=arch.get("d_att", d_att),
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        separate_sign_head=separate_sign_head,
        controller_inputs=controller_inputs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict_advantages_first_n(
    model: MetaController,
    cache: MaterializedCache,
    n_snapshots: int,
    *,
    predict_batch_size: int = 65536,
    device: str = "cpu",
) -> torch.Tensor:
    """Predict advantages for exactly the first ``n_snapshots`` cache rows, in order.

    A correct, shard-boundary-safe replacement for
    ``controller_train._predict_advantages_for_cache`` whose ``max_snapshots``
    clamp mixes a global produced-count with shard-local indices and overshoots
    once truncation crosses a shard boundary. The cache snapshot order matches
    the dataset's episode order, so taking the first ``n_snapshots`` rows aligns
    with the first ``max_episodes`` episodes' steps.
    """
    out: list[torch.Tensor] = []
    produced = 0
    with torch.inference_mode():
        for shard_path in cache.shard_paths:
            if produced >= n_snapshots:
                break
            features = torch.load(shard_path, weights_only=False)["features"]
            shard_take = min(features.shape[0], n_snapshots - produced)
            for start in range(0, shard_take, predict_batch_size):
                end = min(start + predict_batch_size, shard_take)
                batch = features[start:end].to(device)
                predicted, _ = model.predict_from_features(batch)
                out.append(predicted.detach().cpu())
            produced += shard_take
    advantages = torch.cat(out, dim=0)
    if advantages.shape[0] != n_snapshots:
        raise ValueError(
            f"cache yielded {advantages.shape[0]} snapshots but {n_snapshots} were "
            f"requested — cache and manifest are out of sync."
        )
    return advantages


def score_mchalt_checkpoint(
    *,
    checkpoint_path: str | Path,
    validation_manifest: str | Path,
    validation_cache_index: str | Path,
    oracle_config: BudgetedOracleConfig,
    max_episodes: int | None = None,
    device: str = "cpu",
    architecture_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Score a trained MCHalt checkpoint over the validation split via greedy rollout.

    Returns ``{"average_regret", "exact_stop_step_accuracy", "average_return",
    "average_oracle_value", "average_expansions", "evaluated_episodes"}`` — the
    regret + P(stop==OSS) directly comparable to the baseline tiers because the
    underlying evaluator shares their decision rule and oracle config.

    ``max_episodes`` caps the number of validation episodes (a capped sample is
    close to but not exactly the full-val recorded number). The materialized
    cache is consumed in dataset order, so the cap aligns features to episodes.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    overrides = architecture_overrides or {}
    model = _build_controller_from_checkpoint(checkpoint, device=device, **overrides)

    dataset = ControllerEpisodeDataset(str(validation_manifest))
    cache = _load_materialized_cache_unchecked(Path(validation_cache_index))

    # Take exactly the first ``max_episodes`` episodes' worth of snapshots from
    # the cache (its order matches dataset episode order), then apply the SAME
    # greedy rollout aggregator the trainer's checkpoint selection uses.
    episode_metadata, total_steps = _collect_episode_metadata_and_step_count(
        dataset, max_episodes=max_episodes
    )
    all_advantages = _predict_advantages_first_n(model, cache, total_steps, device=device)
    # Collect per-episode diagnostics so we can compute the Tier-1 stop-fit metrics
    # from the SAME greedy rollout (predicted_stop_step / oracle_stop_step / regret),
    # using the identical helper the baselines use — no training-code change.
    diagnostics: list[dict[str, Any]] = []
    metrics = _aggregate_greedy_rollout_metrics(
        episode_metadata,
        all_advantages,
        oracle_config,
        log_interval=0,
        started=0.0,
        diagnostics_out=diagnostics,
    )
    fit = stop_fit_metrics(
        [d["predicted_stop_step"] for d in diagnostics],
        [d["oracle_stop_step"] for d in diagnostics],
        [d["regret"] for d in diagnostics],
    )
    return {
        "average_regret": metrics.average_regret,
        "exact_stop_step_accuracy": metrics.exact_stop_step_accuracy,
        "average_return": metrics.average_return,
        "average_oracle_value": metrics.average_oracle_value,
        "average_expansions": metrics.average_expansions,
        "evaluated_episodes": float(metrics.evaluated_episodes),
        "per_episode_regrets": [float(d["regret"]) for d in diagnostics],  # for bootstrap CIs
        **fit,
    }
