#!/usr/bin/env python3
"""Stage 2b: train the metacontroller on ysagiv materialized caches (cheap proxy run).

Named variants (see ``config.STAGE2B_VARIANTS``) match the Yotam May 2026 comparison.
**Best model:** ``subtree_weight_root+budget`` (subtree-weighted encoder, z_t + T_t).

Usage::

    python3 analysis/2b_train_controller.py --variant subtree_weight_root+budget --max-batches 1000 --save
    python3 analysis/2b_train_controller.py --all --max-batches 1000 --save
    python3 analysis/2b_train_controller.py --all --max-batches 1000 --save --eval-interval 200
    python3 analysis/2b_plot_loss.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

_ANALYSIS_ROOT = Path(__file__).resolve().parent
_LMCOS_ROOT = _ANALYSIS_ROOT.parent / "lmcos"
for _p in (_ANALYSIS_ROOT, _LMCOS_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cts._config import load_config
from cts.core.schema import tree_encoder_feature_schema
from cts.models.mc import MetaController
from cts.train.controller_train import (
    BudgetedOracleConfig,
    ControllerEpisodeDataset,
    ControllerTrainConfig,
    EpisodeMetadata,
    MaterializedCache,
    _advantage_loss_components,
    _build_model_and_optimizer,
    _extract_all_episode_metadata,
    _load_materialized_cache,
    _oracle_config,
    _predict_advantages_for_cache,
    _save_checkpoint,
    _sign_auxiliary_loss,
    evaluate_materialized_cache_predictions,
)

from stage2b_eval_policy import evaluate_expected_regret_batched

from config import (
    HL4291_2B_DIR,
    STAGE2B_DEFAULT_VARIANT,
    STAGE2B_VARIANT_ORDER,
    STAGE2B_VARIANTS,
    Stage2bVariant,
)


@dataclass
class EvalContext:
    """Validation cache + episode metadata for periodic MSE / expected-regret eval."""

    validation_cache: MaterializedCache
    episode_metadata: list[EpisodeMetadata]
    oracle_config: BudgetedOracleConfig
    eval_subset_steps: int
    eval_episodes: int
    total_validation_episodes: int
    validation_snapshots: int
    stop_temperature: float


@dataclass
class TrainMetrics:
    """Per-batch training losses and periodic validation snapshots."""

    batch: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    advantage_mse: list[float] = field(default_factory=list)
    sign_bce: list[float] = field(default_factory=list)
    sign_loss_weight: float = 0.1
    n_batches: int = 0
    wall_s: float = 0.0
    batches_per_epoch: int | None = None
    # Periodic eval (aligned to training batch index; batch 0 = before any steps).
    eval_batch: list[int] = field(default_factory=list)
    eval_validation_mse: list[float] = field(default_factory=list)
    eval_validation_total_loss: list[float] = field(default_factory=list)
    eval_expected_regret: list[float] = field(default_factory=list)
    eval_expected_expansions: list[float] = field(default_factory=list)
    eval_evaluated_episodes: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.batch)


def build_eval_context(
    train_config: ControllerTrainConfig,
    *,
    eval_max_episodes: int,
    stop_temperature: float,
) -> EvalContext:
    """Load validation cache and episode metadata for periodic MSE + expected regret."""
    val_cache_path = train_config.materialized_validation_cache
    if not val_cache_path or not Path(val_cache_path).is_file():
        raise SystemExit(f"Missing validation cache: {val_cache_path}")
    validation_cache = _load_materialized_cache(
        val_cache_path,
        manifest_path=str(train_config.packed_validation_data),
        encoder_checkpoint=train_config.encoder_checkpoint,
    )
    full_metadata = _extract_all_episode_metadata(
        ControllerEpisodeDataset(str(train_config.packed_validation_data))
    )
    if eval_max_episodes > 0 and eval_max_episodes < len(full_metadata):
        episode_metadata = full_metadata[:eval_max_episodes]
    else:
        episode_metadata = full_metadata
    eval_subset_steps = sum(meta.num_steps for meta in episode_metadata)
    return EvalContext(
        validation_cache=validation_cache,
        episode_metadata=episode_metadata,
        oracle_config=_oracle_config(train_config),
        eval_subset_steps=eval_subset_steps,
        eval_episodes=len(episode_metadata),
        total_validation_episodes=len(full_metadata),
        validation_snapshots=validation_cache.examples,
        stop_temperature=stop_temperature,
    )


def run_periodic_eval(
    model: MetaController,
    batch_index: int,
    metrics: TrainMetrics,
    eval_ctx: EvalContext,
    *,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
    predict_batch_size: int,
) -> None:
    """Record validation MSE and expected regret at ``batch_index`` (model left in train mode after)."""
    val_metrics = evaluate_materialized_cache_predictions(
        model,
        eval_ctx.validation_cache,
        device=device,
        batch_size=batch_size,
        sign_loss_weight=sign_loss_weight,
    )
    all_advantages = _predict_advantages_for_cache(
        model, eval_ctx.validation_cache, predict_batch_size
    )
    if eval_ctx.eval_subset_steps < all_advantages.shape[0]:
        all_advantages = all_advantages[: eval_ctx.eval_subset_steps]
    expected_metrics = evaluate_expected_regret_batched(
        eval_ctx.episode_metadata,
        all_advantages,
        eval_ctx.oracle_config,
        temperature=eval_ctx.stop_temperature,
    )
    metrics.eval_batch.append(batch_index)
    metrics.eval_validation_mse.append(float(val_metrics.advantage_mse))
    metrics.eval_validation_total_loss.append(float(val_metrics.total_loss))
    metrics.eval_expected_regret.append(float(expected_metrics.average_regret))
    metrics.eval_expected_expansions.append(float(expected_metrics.average_expansions))
    metrics.eval_evaluated_episodes.append(int(expected_metrics.evaluated_episodes))
    print(
        f"eval@batch={batch_index:>5d}  "
        f"val_mse={val_metrics.advantage_mse:.4f}  "
        f"val_total={val_metrics.total_loss:.4f}  "
        f"expected_regret={expected_metrics.average_regret:.4f}  "
        f"E[expansions]={expected_metrics.average_expansions:.3f}  "
        f"tau={eval_ctx.stop_temperature:g}  "
        f"episodes={expected_metrics.evaluated_episodes}",
        flush=True,
    )
    model.train()


def _should_eval(batch: int, n_batches: int, eval_interval: int) -> bool:
    if eval_interval <= 0:
        return False
    if batch == 0 or batch == n_batches:
        return True
    return batch % eval_interval == 0


def train(
    model: MetaController,
    n_batches: int,
    *,
    cache: MaterializedCache,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
    seed: int = 0,
    log_interval: int = 10,
    eval_ctx: EvalContext | None = None,
    eval_interval: int = 0,
    eval_predict_batch_size: int = 65536,
) -> tuple[MetaController, TrainMetrics]:
    """Run ``n_batches`` steps on the materialized cache; return model and all batch metrics."""
    metrics = TrainMetrics(sign_loss_weight=sign_loss_weight)
    model.train()
    batch_counter = 0
    started = time.perf_counter()

    if eval_ctx is not None and _should_eval(0, n_batches, eval_interval):
        run_periodic_eval(
            model,
            0,
            metrics,
            eval_ctx,
            device=device,
            batch_size=batch_size,
            sign_loss_weight=sign_loss_weight,
            predict_batch_size=eval_predict_batch_size,
        )

    shard_order = list(range(len(cache.shard_paths)))
    random.Random(seed).shuffle(shard_order)

    pbar = tqdm(total=n_batches, desc="train", unit="batch")
    try:
        for shard_index in shard_order:
            if batch_counter >= n_batches:
                break
            payload = torch.load(cache.shard_paths[shard_index], weights_only=False)
            features = payload["features"]
            target_advantages = payload["target_advantages"]
            order = torch.randperm(features.shape[0])
            for start in range(0, int(features.shape[0]), batch_size):
                if batch_counter >= n_batches:
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

                pbar.update(1)
                total_f = float(total_loss.item())
                mse_f = float(advantage_mse.item())
                bce_f = float(sign_loss.item())
                elapsed = time.perf_counter() - started

                metrics.batch.append(batch_counter)
                metrics.total_loss.append(total_f)
                metrics.advantage_mse.append(mse_f)
                metrics.sign_bce.append(bce_f)

                if _should_log(batch_counter, n_batches, log_interval):
                    _print_batch(batch_counter, total_f, mse_f, bce_f, elapsed)

                if eval_ctx is not None and _should_eval(batch_counter, n_batches, eval_interval):
                    if metrics.eval_batch and metrics.eval_batch[-1] == batch_counter:
                        pass
                    else:
                        run_periodic_eval(
                            model,
                            batch_counter,
                            metrics,
                            eval_ctx,
                            device=device,
                            batch_size=batch_size,
                            sign_loss_weight=sign_loss_weight,
                            predict_batch_size=eval_predict_batch_size,
                        )
    finally:
        pbar.close()

    if (
        eval_ctx is not None
        and eval_interval > 0
        and batch_counter > 0
        and (not metrics.eval_batch or metrics.eval_batch[-1] != batch_counter)
    ):
        run_periodic_eval(
            model,
            batch_counter,
            metrics,
            eval_ctx,
            device=device,
            batch_size=batch_size,
            sign_loss_weight=sign_loss_weight,
            predict_batch_size=eval_predict_batch_size,
        )

    if batch_counter == 0:
        raise RuntimeError("No batches ran — is the train cache present?")

    metrics.n_batches = batch_counter
    metrics.wall_s = time.perf_counter() - started
    return model, metrics


def save_controller(
    model: MetaController,
    path: Path,
    *,
    variant: Stage2bVariant,
    train_config: ControllerTrainConfig,
    metrics: TrainMetrics,
) -> None:
    metadata: dict[str, Any] = {
        "stage": "analysis_2b_proxy",
        "variant": variant.name,
        "variant_label": variant.plot_label,
        "n_batches": metrics.n_batches,
        "wall_s": metrics.wall_s,
        "batches_per_epoch": metrics.batches_per_epoch,
        "encoder_checkpoint": train_config.encoder_checkpoint,
        "unfreeze_encoder": train_config.unfreeze_encoder,
        "controller_inputs": list(train_config.controller_inputs),
        "sign_loss_weight": train_config.sign_loss_weight,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "lmcos_config_yaml": str(variant.lmcos_config),
        "final_total_loss": metrics.total_loss[-1] if metrics.total_loss else None,
        "final_advantage_mse": metrics.advantage_mse[-1] if metrics.advantage_mse else None,
        "final_sign_bce": metrics.sign_bce[-1] if metrics.sign_bce else None,
        "final_eval_validation_mse": metrics.eval_validation_mse[-1] if metrics.eval_validation_mse else None,
        "final_eval_expected_regret": (
            metrics.eval_expected_regret[-1] if metrics.eval_expected_regret else None
        ),
    }
    _save_checkpoint(str(path), model, metadata)
    print(f"wrote {path}")


def save_metrics(
    metrics: TrainMetrics,
    path: Path,
    *,
    variant: Stage2bVariant,
    extra: dict[str, Any] | None = None,
) -> None:
    run_info: dict[str, Any] = {
        "variant": variant.name,
        "variant_label": variant.plot_label,
        "description": variant.description,
    }
    if extra:
        run_info.update(extra)
    payload: dict[str, Any] = {
        "summary": {
            "n_batches": metrics.n_batches,
            "wall_s": metrics.wall_s,
            "batches_per_epoch": metrics.batches_per_epoch,
            "sign_loss_weight": metrics.sign_loss_weight,
            "final_total_loss": metrics.total_loss[-1] if metrics.total_loss else None,
            "final_advantage_mse": metrics.advantage_mse[-1] if metrics.advantage_mse else None,
            "final_sign_bce": metrics.sign_bce[-1] if metrics.sign_bce else None,
            "final_eval_validation_mse": (
                metrics.eval_validation_mse[-1] if metrics.eval_validation_mse else None
            ),
            "final_eval_expected_regret": (
                metrics.eval_expected_regret[-1] if metrics.eval_expected_regret else None
            ),
        },
        "run": run_info,
        "curves": asdict(metrics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def build_model_and_cache(
    train_config: ControllerTrainConfig,
    device: torch.device,
) -> tuple[MetaController, torch.optim.Optimizer, MaterializedCache, int]:
    schema = tree_encoder_feature_schema()
    train_cache = train_config.materialized_train_cache
    if not train_cache or not Path(train_cache).is_file():
        raise SystemExit(f"Missing train cache: {train_cache}")
    cache = _load_materialized_cache(
        train_cache,
        manifest_path=str(train_config.packed_train_data),
        encoder_checkpoint=train_config.encoder_checkpoint,
    )
    batches_per_epoch = _print_scale(cache, train_config.batch_size)
    model, optimizer, _ = _build_model_and_optimizer(train_config, schema)
    model.to(device)
    return model, optimizer, cache, batches_per_epoch


def run_variant(
    variant: Stage2bVariant,
    *,
    max_batches: int,
    log_interval: int,
    eval_interval: int,
    eval_max_episodes: int,
    eval_predict_batch_size: int,
    eval_stop_temperature: float,
    device: torch.device,
    save: bool,
) -> TrainMetrics:
    print(f"\n=== variant: {variant.name} ===")
    print(f"    {variant.description}")
    print(f"    config: {variant.lmcos_config}")
    train_config = load_config(ControllerTrainConfig, str(variant.lmcos_config))
    model, optimizer, cache, batches_per_epoch = build_model_and_cache(train_config, device)

    eval_ctx: EvalContext | None = None
    if eval_interval > 0:
        eval_ctx = build_eval_context(
            train_config,
            eval_max_episodes=eval_max_episodes,
            stop_temperature=eval_stop_temperature,
        )
        print(
            "  validation snapshots: {:,}".format(eval_ctx.validation_snapshots),
            flush=True,
        )
        print(
            "  policy eval episodes:  {:,} / {:,}".format(
                eval_ctx.eval_episodes,
                eval_ctx.total_validation_episodes,
            ),
            flush=True,
        )
        print(
            f"  stop rule: P(stop|a) = sigmoid(-a / {eval_ctx.stop_temperature:g}); "
            "remaining mass at final step",
            flush=True,
        )

    model, metrics = train(
        model,
        max_batches,
        cache=cache,
        optimizer=optimizer,
        device=device,
        batch_size=train_config.batch_size,
        sign_loss_weight=train_config.sign_loss_weight,
        seed=train_config.seed,
        log_interval=log_interval,
        eval_ctx=eval_ctx,
        eval_interval=eval_interval,
        eval_predict_batch_size=eval_predict_batch_size,
    )
    metrics.batches_per_epoch = batches_per_epoch

    per_batch = metrics.wall_s / metrics.n_batches
    print(
        f"Done [{variant.name}]: {metrics.n_batches} batches in {metrics.wall_s:.2f} s "
        f"({per_batch*1000:.1f} ms/batch)"
    )
    if batches_per_epoch:
        print(f"Extrapolated train-only 1 epoch: {per_batch * batches_per_epoch / 60:.1f} min")

    if save:
        save_controller(
            model,
            variant.controller_path(),
            variant=variant,
            train_config=train_config,
            metrics=metrics,
        )
        save_metrics(
            metrics,
            variant.metrics_path(),
            variant=variant,
            extra={
                "max_batches_requested": max_batches,
                "log_interval": log_interval,
                "eval_interval": eval_interval,
                "eval_max_episodes": eval_max_episodes,
                "eval_predict_batch_size": eval_predict_batch_size,
                "eval_stop_temperature": eval_stop_temperature,
                "eval_stop_rule": "sigmoid(-advantage / temperature); absorb at last step",
                "device": str(device),
            },
        )
    return metrics


def _should_log(batch: int, n_batches: int, log_interval: int) -> bool:
    if batch == 1 or batch == n_batches:
        return True
    return log_interval > 0 and batch % log_interval == 0


def _print_batch(batch: int, total: float, mse: float, bce: float, elapsed_s: float) -> None:
    print(
        f"batch={batch:>5d}  elapsed={elapsed_s:6.1f}s  "
        f"total={total:.4f}  mse={mse:.4f}  sign_bce={bce:.4f}",
        flush=True,
    )


def _print_scale(cache: MaterializedCache, batch_size: int) -> int:
    batches_per_epoch = math.ceil(cache.examples / batch_size)
    print("  total snapshots:     {:,}".format(cache.examples))
    print("  batch_size:          {}".format(batch_size))
    print("  batches / epoch:     {:,}".format(batches_per_epoch))
    return batches_per_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Run validation MSE + expected regret every N batches (0 disables). "
        "Always runs at batch 0 and at the final batch.",
    )
    parser.add_argument(
        "--eval-max-episodes",
        type=int,
        default=3000,
        help="Expected-regret subsample size from validation manifest (0 = all ~30k). "
        "Validation MSE always uses the full validation cache.",
    )
    parser.add_argument(
        "--eval-stop-temperature",
        type=float,
        default=1.0,
        help="Temperature for probabilistic stopping: P(stop)=sigmoid(-advantage/tau). "
        "Smaller tau approaches hard stop-at-advantage<=0.",
    )
    parser.add_argument(
        "--eval-predict-batch-size",
        type=int,
        default=65536,
        help="Forward batch size for advantage prediction over the validation cache.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--variant",
        choices=list(STAGE2B_VARIANTS),
        default=None,
        help=f"Named run (default: {STAGE2B_DEFAULT_VARIANT.name}).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all variants in comparison order.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Write {{name}}_controller.pt and {{name}}_metrics.json under {HL4291_2B_DIR}",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    HL4291_2B_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        names = STAGE2B_VARIANT_ORDER
    else:
        name = args.variant or STAGE2B_DEFAULT_VARIANT.name
        names = [name]

    for name in names:
        run_variant(
            STAGE2B_VARIANTS[name],
            max_batches=args.max_batches,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            eval_max_episodes=args.eval_max_episodes,
            eval_predict_batch_size=args.eval_predict_batch_size,
            eval_stop_temperature=args.eval_stop_temperature,
            device=device,
            save=args.save,
        )


if __name__ == "__main__":
    main()
