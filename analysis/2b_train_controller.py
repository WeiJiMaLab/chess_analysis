#!/usr/bin/env python3
"""Stage 2b: train the metacontroller on ysagiv materialized caches (cheap proxy run).

Named variants (see ``config.STAGE2B_VARIANTS``) match the Yotam May 2026 comparison.
**Best model:** ``subtree_weight_root+budget`` (subtree-weighted encoder, z_t + T_t).

Usage::

    python3 analysis/2b_train_controller.py --variant subtree_weight_root+budget --max-batches 1000 --save
    python3 analysis/2b_train_controller.py --all --max-batches 1000 --save
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
    ControllerTrainConfig,
    MaterializedCache,
    _advantage_loss_components,
    _build_model_and_optimizer,
    _load_materialized_cache,
    _save_checkpoint,
    _sign_auxiliary_loss,
)

from config import (
    HL4291_2B_DIR,
    STAGE2B_DEFAULT_VARIANT,
    STAGE2B_VARIANT_ORDER,
    STAGE2B_VARIANTS,
    Stage2bVariant,
)


@dataclass
class TrainMetrics:
    """Per-batch training losses (one entry per optimizer step)."""

    batch: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    advantage_mse: list[float] = field(default_factory=list)
    sign_bce: list[float] = field(default_factory=list)
    sign_loss_weight: float = 0.1
    n_batches: int = 0
    wall_s: float = 0.0
    batches_per_epoch: int | None = None

    def __len__(self) -> int:
        return len(self.batch)


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
) -> tuple[MetaController, TrainMetrics]:
    """Run ``n_batches`` steps on the materialized cache; return model and all batch metrics."""
    metrics = TrainMetrics(sign_loss_weight=sign_loss_weight)
    model.train()
    batch_counter = 0
    started = time.perf_counter()

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
    finally:
        pbar.close()

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
    device: torch.device,
    save: bool,
) -> TrainMetrics:
    print(f"\n=== variant: {variant.name} ===")
    print(f"    {variant.description}")
    print(f"    config: {variant.lmcos_config}")
    train_config = load_config(ControllerTrainConfig, str(variant.lmcos_config))
    model, optimizer, cache, batches_per_epoch = build_model_and_cache(train_config, device)

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
            device=device,
            save=args.save,
        )


if __name__ == "__main__":
    main()
