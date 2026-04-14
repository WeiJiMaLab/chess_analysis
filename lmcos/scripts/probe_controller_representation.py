from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from controller_oracle import compute_oracle_policy
from GNN import PolicyValueTreeSearchModel
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from supervised_branch import (
    TeacherSearchConfig,
    build_trimmed_decision_episode,
    load_encoder_checkpoint,
    load_raw_pretrain_example_paths,
)
from tensorizer import TreeTensorizer
from tree import SearchTree


@dataclass(frozen=True)
class ProbeSample:
    tree: SearchTree
    target_action: int
    target_advantage: float
    phase_index: int
    oracle_stop_step: int


@dataclass(frozen=True)
class ProbeTensorDataset:
    embeddings: torch.Tensor
    labels: torch.Tensor
    advantages: torch.Tensor
    phase_indices: torch.Tensor
    oracle_stop_steps: torch.Tensor
    metadata: Dict[str, Any]
    processed_examples: int
    total_examples: int
    complete: bool

    def as_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(
            self.embeddings.to(torch.float32),
            self.labels.to(torch.int64),
            self.advantages.to(torch.float32),
            self.phase_indices.to(torch.int64),
            self.oracle_stop_steps.to(torch.int64),
        )


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


def _teacher_config(args: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="controller_probe_v1",
        search_config_id="controller_probe",
    )

def _select_paths(
    data: str,
    sample_size: int,
    seed: int,
) -> List[str]:
    paths = load_raw_pretrain_example_paths(data)
    rng = random.Random(seed)
    if sample_size > 0 and sample_size < len(paths):
        paths = rng.sample(paths, sample_size)
    return paths


def _load_samples_from_paths(
    paths: Sequence[str],
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    log_interval: int,
    start_example_index: int,
    total_examples: int,
) -> List[ProbeSample]:
    samples: List[ProbeSample] = []
    for index, path in enumerate(paths, start=1):
        example = torch.load(path, weights_only=False)
        try:
            episode = build_trimmed_decision_episode(example, quality_config)
        except ValueError:
            continue
        halt_rewards = [episode.final_root_q_values[move] for move in episode.best_moves]
        policy = compute_oracle_policy(halt_rewards, continue_cost)
        actions = policy.actions
        stop_step = policy.optimal_stop_step
        for phase_index, (tree, action, halt_reward) in enumerate(zip(episode.snapshots, actions, halt_rewards)):
            if phase_index + 1 < len(halt_rewards):
                continue_value = -continue_cost + policy.values[phase_index + 1]
            else:
                continue_value = -continue_cost + float(halt_reward)
            samples.append(
                ProbeSample(
                    tree=tree,
                    target_action=int(action),
                    target_advantage=float(continue_value - float(halt_reward)),
                    phase_index=phase_index,
                    oracle_stop_step=stop_step,
                )
            )
        absolute_index = start_example_index + index
        if index % log_interval == 0 or index == len(paths):
            print(
                f"[probe] loaded_examples={absolute_index}/{total_examples} accumulated_samples={len(samples)}",
                flush=True,
            )
    if not samples:
        return []
    return samples


def _extract_root_embeddings(
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
    samples: Sequence[ProbeSample],
    batch_size: int,
    log_interval: int,
    initial_completed: int = 0,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch = tensorizer.tensorize_forest([sample.tree for sample in batch_samples])
            encoded = model.encoder(batch)
            features.append(encoded.root_states.detach().cpu())
            completed = initial_completed + min(start + batch_size, len(samples))
            if completed % log_interval == 0 or completed == initial_completed + len(samples):
                print(
                    f"[probe] embedded_samples={completed}",
                    flush=True,
                )
    return torch.cat(features, dim=0)


def _cache_metadata(
    split_name: str,
    data: str,
    encoder_checkpoint: str,
    sample_size: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "split_name": split_name,
        "data": data,
        "encoder_checkpoint": encoder_checkpoint,
        "sample_size": sample_size,
        "seed": seed,
        "continue_cost": args.continue_cost,
        "search_budget": args.search_budget,
        "max_depth": args.max_depth,
        "c_puct": args.c_puct,
        "k": args.k,
        "node_embed_hidden": args.node_embed_hidden,
        "d_embed": args.d_embed,
        "d_message": args.d_message,
        "n_heads": args.n_heads,
        "d_att": args.d_att,
        "controller_hidden": args.controller_hidden,
        "value_hidden": args.value_hidden,
        "target_normalization_version": "controller_probe_v1",
    }


def _load_cached_dataset(cache_path: Path, expected_metadata: Dict[str, Any]) -> ProbeTensorDataset | None:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata != expected_metadata:
        print(
            f"[probe] cache_mismatch path={cache_path} expected={json.dumps(expected_metadata, sort_keys=True)} "
            f"found={json.dumps(metadata, sort_keys=True)}",
            flush=True,
        )
        return None
    print(f"[probe] cache_hit path={cache_path}", flush=True)
    advantages = payload.get("advantages")
    if advantages is None:
        advantages = torch.full_like(payload["labels"], float("nan"), dtype=torch.float32)
    return ProbeTensorDataset(
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        advantages=advantages,
        phase_indices=payload["phase_indices"],
        oracle_stop_steps=payload["oracle_stop_steps"],
        metadata=metadata,
        processed_examples=int(payload.get("processed_examples", 0)),
        total_examples=int(payload.get("total_examples", 0)),
        complete=bool(payload.get("complete", False)),
    )


def _save_cached_dataset(dataset: ProbeTensorDataset, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": dataset.embeddings,
            "labels": dataset.labels,
            "advantages": dataset.advantages,
            "phase_indices": dataset.phase_indices,
            "oracle_stop_steps": dataset.oracle_stop_steps,
            "metadata": dataset.metadata,
            "processed_examples": dataset.processed_examples,
            "total_examples": dataset.total_examples,
            "complete": dataset.complete,
        },
        cache_path,
    )
    print(
        f"[probe] cache_saved path={cache_path} processed_examples={dataset.processed_examples}/{dataset.total_examples} "
        f"complete={dataset.complete}",
        flush=True,
    )


def _build_or_load_dataset(
    split_name: str,
    data: str,
    sample_size: int,
    seed: int,
    cache_path: Path | None,
    args: argparse.Namespace,
    quality_config: TeacherSearchConfig,
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
) -> ProbeTensorDataset:
    selected_paths = _select_paths(data, sample_size, seed)
    metadata = _cache_metadata(split_name, data, args.encoder_checkpoint, sample_size, seed, args)
    cached: ProbeTensorDataset | None = None
    if cache_path is not None:
        cached = _load_cached_dataset(cache_path, metadata)
        if cached is not None:
            if cached.complete and cached.processed_examples == len(selected_paths):
                return cached
            print(
                f"[probe] cache_resume path={cache_path} processed_examples={cached.processed_examples}/{len(selected_paths)}",
                flush=True,
            )

    completed_examples = cached.processed_examples if cached is not None else 0
    embeddings_parts: List[torch.Tensor] = []
    labels_parts: List[torch.Tensor] = []
    advantage_parts: List[torch.Tensor] = []
    phase_parts: List[torch.Tensor] = []
    oracle_parts: List[torch.Tensor] = []
    if cached is not None:
        if cached.embeddings.numel() > 0:
            embeddings_parts.append(cached.embeddings)
        if cached.labels.numel() > 0:
            labels_parts.append(cached.labels)
            if cached.advantages.numel() > 0:
                advantage_parts.append(cached.advantages)
            phase_parts.append(cached.phase_indices)
            oracle_parts.append(cached.oracle_stop_steps)

    total_embedded_samples = int(sum(part.shape[0] for part in embeddings_parts))
    chunk_size = max(1, args.cache_save_interval_examples)

    for chunk_start in range(completed_examples, len(selected_paths), chunk_size):
        chunk_paths = selected_paths[chunk_start : chunk_start + chunk_size]
        print(
            f"[probe] stage=load_samples split={split_name} examples={chunk_start + 1}-{chunk_start + len(chunk_paths)}/{len(selected_paths)}",
            flush=True,
        )
        samples = _load_samples_from_paths(
            chunk_paths,
            quality_config,
            continue_cost=args.continue_cost,
            log_interval=args.load_log_interval,
            start_example_index=chunk_start,
            total_examples=len(selected_paths),
        )
        if samples:
            print(
                f"[probe] stage=extract_embeddings split={split_name} chunk_samples={len(samples)}",
                flush=True,
            )
            embeddings = _extract_root_embeddings(
                model,
                tensorizer,
                samples,
                args.embedding_batch_size,
                log_interval=args.embedding_log_interval,
                initial_completed=total_embedded_samples,
            )
            embeddings_parts.append(embeddings)
            labels_parts.append(torch.tensor([sample.target_action for sample in samples], dtype=torch.int64))
            advantage_parts.append(torch.tensor([sample.target_advantage for sample in samples], dtype=torch.float32))
            phase_parts.append(torch.tensor([sample.phase_index for sample in samples], dtype=torch.int64))
            oracle_parts.append(torch.tensor([sample.oracle_stop_step for sample in samples], dtype=torch.int64))
            total_embedded_samples += int(embeddings.shape[0])

        completed_examples = chunk_start + len(chunk_paths)
        partial_dataset = ProbeTensorDataset(
            embeddings=torch.cat(embeddings_parts, dim=0) if embeddings_parts else torch.empty((0, model.encoder.d_embed)),
            labels=torch.cat(labels_parts, dim=0) if labels_parts else torch.empty((0,), dtype=torch.int64),
            advantages=torch.cat(advantage_parts, dim=0) if advantage_parts else torch.empty((0,), dtype=torch.float32),
            phase_indices=torch.cat(phase_parts, dim=0) if phase_parts else torch.empty((0,), dtype=torch.int64),
            oracle_stop_steps=torch.cat(oracle_parts, dim=0) if oracle_parts else torch.empty((0,), dtype=torch.int64),
            metadata=metadata,
            processed_examples=completed_examples,
            total_examples=len(selected_paths),
            complete=False,
        )
        if cache_path is not None:
            _save_cached_dataset(partial_dataset, cache_path)

    dataset = ProbeTensorDataset(
        embeddings=torch.cat(embeddings_parts, dim=0) if embeddings_parts else torch.empty((0, model.encoder.d_embed)),
        labels=torch.cat(labels_parts, dim=0) if labels_parts else torch.empty((0,), dtype=torch.int64),
        advantages=torch.cat(advantage_parts, dim=0) if advantage_parts else torch.empty((0,), dtype=torch.float32),
        phase_indices=torch.cat(phase_parts, dim=0) if phase_parts else torch.empty((0,), dtype=torch.int64),
        oracle_stop_steps=torch.cat(oracle_parts, dim=0) if oracle_parts else torch.empty((0,), dtype=torch.int64),
        metadata=metadata,
        processed_examples=len(selected_paths),
        total_examples=len(selected_paths),
        complete=True,
    )
    if cache_path is not None:
        _save_cached_dataset(dataset, cache_path)
    return dataset


def _backfill_advantages_from_paths(
    dataset: ProbeTensorDataset,
    split_name: str,
    data: str,
    sample_size: int,
    seed: int,
    cache_path: Path | None,
    args: argparse.Namespace,
    quality_config: TeacherSearchConfig,
) -> ProbeTensorDataset:
    has_complete_advantages = (
        dataset.advantages.shape[0] == dataset.labels.shape[0]
        and not torch.isnan(dataset.advantages).any().item()
    )
    if has_complete_advantages:
        return dataset

    print(
        f"[probe] stage=backfill_advantages split={split_name} existing_advantages={dataset.advantages.shape[0]} "
        f"labels={dataset.labels.shape[0]}",
        flush=True,
    )
    selected_paths = _select_paths(data, sample_size, seed)
    samples = _load_samples_from_paths(
        selected_paths,
        quality_config,
        continue_cost=args.continue_cost,
        log_interval=args.load_log_interval,
        start_example_index=0,
        total_examples=len(selected_paths),
    )
    advantages = torch.tensor([sample.target_advantage for sample in samples], dtype=torch.float32)
    labels = torch.tensor([sample.target_action for sample in samples], dtype=torch.int64)
    phase_indices = torch.tensor([sample.phase_index for sample in samples], dtype=torch.int64)
    oracle_stop_steps = torch.tensor([sample.oracle_stop_step for sample in samples], dtype=torch.int64)
    if advantages.shape[0] != dataset.embeddings.shape[0]:
        raise ValueError(
            f"Cannot backfill advantages for {split_name}: recomputed {advantages.shape[0]} samples "
            f"but cache has {dataset.embeddings.shape[0]} embeddings."
        )
    if labels.shape == dataset.labels.shape and not torch.equal(labels, dataset.labels):
        raise ValueError(f"Cannot backfill advantages for {split_name}: recomputed action labels do not match cache.")

    updated = ProbeTensorDataset(
        embeddings=dataset.embeddings,
        labels=labels,
        advantages=advantages,
        phase_indices=phase_indices,
        oracle_stop_steps=oracle_stop_steps,
        metadata=dataset.metadata,
        processed_examples=dataset.processed_examples,
        total_examples=dataset.total_examples,
        complete=dataset.complete,
    )
    if cache_path is not None:
        _save_cached_dataset(updated, cache_path)
    return updated


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, device: torch.device) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _build_probe(probe_type: str, input_dim: int, hidden_dim: int, device: torch.device) -> nn.Module:
    if probe_type == "mlp":
        return MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
    if probe_type == "linear":
        return LinearProbe(input_dim=input_dim, device=device)
    raise ValueError(f"Unsupported probe_type: {probe_type}")


def _action_histogram(labels: Sequence[int]) -> Dict[int, int]:
    histogram: Dict[int, int] = {}
    for label in labels:
        histogram[label] = histogram.get(label, 0) + 1
    return histogram


def _tensor_action_histogram(labels: torch.Tensor) -> Dict[int, int]:
    histogram: Dict[int, int] = {}
    for label, count in zip(*torch.unique(labels, return_counts=True)):
        histogram[int(label.item())] = int(count.item())
    return histogram


def _format_metric(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else "nan"


def _round_metric_dict(values: Dict[int, float]) -> Dict[int, float]:
    return {key: round(value, 3) if math.isfinite(value) else float("nan") for key, value in values.items()}


def _make_loader(
    dataset: ProbeTensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset.as_tensor_dataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _evaluate_probe(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    probe.eval()
    total = 0
    correct = 0
    positive_total = 0
    positive_correct = 0
    negative_total = 0
    negative_correct = 0
    first_total = 0
    first_correct = 0
    phase_total: Dict[int, int] = {}
    phase_correct: Dict[int, int] = {}

    with torch.inference_mode():
        for batch_x, batch_y, _batch_advantage, batch_phase, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_phase = batch_phase.to(device, non_blocking=True)

            logits = probe(batch_x)
            preds = (logits >= 0.0).to(torch.int64)
            matches = preds == batch_y

            total += int(batch_y.numel())
            correct += int(matches.sum().item())
            positive_mask = batch_y == 1
            negative_mask = batch_y == 0
            positive_total += int(positive_mask.sum().item())
            negative_total += int(negative_mask.sum().item())
            if int(positive_mask.sum().item()) > 0:
                positive_correct += int(matches[positive_mask].sum().item())
            if int(negative_mask.sum().item()) > 0:
                negative_correct += int(matches[negative_mask].sum().item())

            first_mask = batch_phase == 0
            first_total += int(first_mask.sum().item())
            if int(first_mask.sum().item()) > 0:
                first_correct += int(matches[first_mask].sum().item())

            for phase in torch.unique(batch_phase).tolist():
                phase = int(phase)
                mask = batch_phase == phase
                phase_total[phase] = phase_total.get(phase, 0) + int(mask.sum().item())
                phase_correct[phase] = phase_correct.get(phase, 0) + int(matches[mask].sum().item())

    phase_accuracy = {
        phase: phase_correct[phase] / phase_total[phase]
        for phase in sorted(phase_total)
    }
    positive_accuracy = positive_correct / positive_total if positive_total > 0 else float("nan")
    negative_accuracy = negative_correct / negative_total if negative_total > 0 else float("nan")
    return {
        "accuracy": correct / total if total > 0 else float("nan"),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "first_action_accuracy": first_correct / first_total if first_total > 0 else float("nan"),
        "phase_accuracy": phase_accuracy,
    }


def _evaluate_advantage_probe(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    probe.eval()
    total = 0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    sign_correct = 0
    first_total = 0
    first_sign_correct = 0
    phase_total: Dict[int, int] = {}
    phase_sign_correct: Dict[int, int] = {}

    with torch.inference_mode():
        for batch_x, _batch_y, batch_advantage, batch_phase, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_advantage = batch_advantage.to(device, non_blocking=True)
            batch_phase = batch_phase.to(device, non_blocking=True)
            predictions = probe(batch_x)
            errors = predictions - batch_advantage
            matches = (predictions > 0.0) == (batch_advantage > 0.0)

            total += int(batch_advantage.numel())
            total_squared_error += float(torch.sum(errors.square()).item())
            total_absolute_error += float(torch.sum(torch.abs(errors)).item())
            sign_correct += int(matches.sum().item())

            first_mask = batch_phase == 0
            first_total += int(first_mask.sum().item())
            if int(first_mask.sum().item()) > 0:
                first_sign_correct += int(matches[first_mask].sum().item())

            for phase in torch.unique(batch_phase).tolist():
                phase = int(phase)
                mask = batch_phase == phase
                phase_total[phase] = phase_total.get(phase, 0) + int(mask.sum().item())
                phase_sign_correct[phase] = phase_sign_correct.get(phase, 0) + int(matches[mask].sum().item())

    phase_sign_accuracy = {
        phase: phase_sign_correct[phase] / phase_total[phase]
        for phase in sorted(phase_total)
    }
    return {
        "mse": total_squared_error / total if total > 0 else float("nan"),
        "mae": total_absolute_error / total if total > 0 else float("nan"),
        "sign_accuracy": sign_correct / total if total > 0 else float("nan"),
        "first_sign_accuracy": first_sign_correct / first_total if first_total > 0 else float("nan"),
        "phase_sign_accuracy": phase_sign_accuracy,
    }


def _majority_label(labels: torch.Tensor) -> int:
    values, counts = torch.unique(labels, return_counts=True)
    return int(values[counts.argmax()].item())


def _evaluate_constant_baseline(
    labels: torch.Tensor,
    phase_indices: torch.Tensor,
    constant_label: int,
) -> Dict[str, Any]:
    preds = torch.full_like(labels, constant_label)
    matches = preds == labels
    accuracy = float(matches.float().mean().item())

    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_accuracy = float(matches[positive_mask].float().mean().item()) if int(positive_mask.sum().item()) > 0 else float("nan")
    negative_accuracy = float(matches[negative_mask].float().mean().item()) if int(negative_mask.sum().item()) > 0 else float("nan")

    first_mask = phase_indices == 0
    first_action_accuracy = (
        float(matches[first_mask].float().mean().item()) if int(first_mask.sum().item()) > 0 else float("nan")
    )

    phase_accuracy: Dict[int, float] = {}
    for phase in sorted(set(int(value) for value in phase_indices.tolist())):
        mask = phase_indices == phase
        phase_accuracy[phase] = float(matches[mask].float().mean().item())

    return {
        "constant_label": constant_label,
        "accuracy": accuracy,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "first_action_accuracy": first_action_accuracy,
        "phase_accuracy": phase_accuracy,
    }


def _evaluate_constant_advantage_baseline(advantages: torch.Tensor, constant_value: float = 0.0) -> Dict[str, float]:
    preds = torch.full_like(advantages, float(constant_value), dtype=torch.float32)
    errors = preds - advantages.to(torch.float32)
    return {
        "constant_value": float(constant_value),
        "mse": float(torch.mean(errors.square()).item()),
        "mae": float(torch.mean(torch.abs(errors)).item()),
        "sign_accuracy": float(((preds > 0.0) == (advantages > 0.0)).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small MLP probe on frozen encoder root states to predict offline oracle halt/continue actions."
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    parser.add_argument("--controller-hidden", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--train-sample-size", type=int, default=0)
    parser.add_argument("--validation-sample-size", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-save-interval-examples", type=int, default=1000)
    parser.add_argument("--load-log-interval", type=int, default=250)
    parser.add_argument("--embedding-log-interval", type=int, default=5000)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-target", choices=["action", "advantage"], default="action")
    parser.add_argument("--probe-type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    schema = _feature_schema()
    tensorizer = TreeTensorizer(schema, device=args.device)
    quality_config = _teacher_config(args)

    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else None
    train_cache_path = cache_dir / "train_probe_cache.pt" if cache_dir is not None else None
    validation_cache_path = cache_dir / "validation_probe_cache.pt" if cache_dir is not None else None

    model: PolicyValueTreeSearchModel | None = None
    tensorizer_for_encoder: TreeTensorizer | None = None

    def _get_encoder_stack() -> tuple[PolicyValueTreeSearchModel, TreeTensorizer]:
        nonlocal model, tensorizer_for_encoder
        if model is None:
            print("[probe] stage=build_encoder", flush=True)
            model = PolicyValueTreeSearchModel(
                k=args.k,
                node_feat=len(schema.feature_names),
                device=args.device,
                node_embed_hidden=args.node_embed_hidden,
                d_embed=args.d_embed,
                d_message=args.d_message,
                n_heads=args.n_heads,
                d_att=args.d_att,
                controller_hidden=args.controller_hidden,
                value_hidden=args.value_hidden,
            )
            load_encoder_checkpoint(args.encoder_checkpoint, model.encoder)
            model.freeze_encoder()
            model.eval()
            tensorizer_for_encoder = tensorizer
        assert tensorizer_for_encoder is not None
        return model, tensorizer_for_encoder

    expected_train_metadata = _cache_metadata("train", args.train_data, args.encoder_checkpoint, args.train_sample_size, args.seed, args)
    expected_val_metadata = _cache_metadata(
        "validation",
        args.validation_data,
        args.encoder_checkpoint,
        args.validation_sample_size,
        args.seed + 1,
        args,
    )
    train_dataset = _load_cached_dataset(train_cache_path, expected_train_metadata) if train_cache_path is not None else None
    validation_dataset = (
        _load_cached_dataset(validation_cache_path, expected_val_metadata) if validation_cache_path is not None else None
    )

    if train_dataset is None or validation_dataset is None:
        encoder_model, encoder_tensorizer = _get_encoder_stack()
        if train_dataset is None:
            train_dataset = _build_or_load_dataset(
                split_name="train",
                data=args.train_data,
                sample_size=args.train_sample_size,
                seed=args.seed,
                cache_path=train_cache_path,
                args=args,
                quality_config=quality_config,
                model=encoder_model,
                tensorizer=encoder_tensorizer,
            )
        if validation_dataset is None:
            validation_dataset = _build_or_load_dataset(
                split_name="validation",
                data=args.validation_data,
                sample_size=args.validation_sample_size,
                seed=args.seed + 1,
                cache_path=validation_cache_path,
                args=args,
                quality_config=quality_config,
                model=encoder_model,
                tensorizer=encoder_tensorizer,
            )

    if args.probe_target == "advantage":
        train_dataset = _backfill_advantages_from_paths(
            train_dataset,
            split_name="train",
            data=args.train_data,
            sample_size=args.train_sample_size,
            seed=args.seed,
            cache_path=train_cache_path,
            args=args,
            quality_config=quality_config,
        )
        validation_dataset = _backfill_advantages_from_paths(
            validation_dataset,
            split_name="validation",
            data=args.validation_data,
            sample_size=args.validation_sample_size,
            seed=args.seed + 1,
            cache_path=validation_cache_path,
            args=args,
            quality_config=quality_config,
        )

    print(
        f"[probe] cached_train_samples={len(train_dataset.labels)} cached_validation_samples={len(validation_dataset.labels)} "
        f"train_action_histogram={_tensor_action_histogram(train_dataset.labels)} "
        f"validation_action_histogram={_tensor_action_histogram(validation_dataset.labels)}",
        flush=True,
    )

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = _make_loader(
        train_dataset,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    train_majority = _majority_label(train_dataset.labels)
    train_majority_metrics = _evaluate_constant_baseline(
        train_dataset.labels,
        train_dataset.phase_indices,
        train_majority,
    )
    validation_majority_metrics = _evaluate_constant_baseline(
        validation_dataset.labels,
        validation_dataset.phase_indices,
        train_majority,
    )
    print(
        f"[probe] majority_baseline label={train_majority} "
        f"train_acc={_format_metric(train_majority_metrics['accuracy'])} "
        f"train_bal_acc={_format_metric(train_majority_metrics['balanced_accuracy'])} "
        f"val_acc={_format_metric(validation_majority_metrics['accuracy'])} "
        f"val_bal_acc={_format_metric(validation_majority_metrics['balanced_accuracy'])} "
        f"val_first_action_acc={_format_metric(validation_majority_metrics['first_action_accuracy'])}",
        flush=True,
    )
    if args.probe_target == "advantage":
        train_advantage_baseline = _evaluate_constant_advantage_baseline(train_dataset.advantages, constant_value=0.0)
        validation_advantage_baseline = _evaluate_constant_advantage_baseline(validation_dataset.advantages, constant_value=0.0)
        print(
            f"[probe] advantage_zero_baseline "
            f"train_mse={_format_metric(train_advantage_baseline['mse'])} "
            f"train_mae={_format_metric(train_advantage_baseline['mae'])} "
            f"train_sign_acc={_format_metric(train_advantage_baseline['sign_accuracy'])} "
            f"val_mse={_format_metric(validation_advantage_baseline['mse'])} "
            f"val_mae={_format_metric(validation_advantage_baseline['mae'])} "
            f"val_sign_acc={_format_metric(validation_advantage_baseline['sign_accuracy'])}",
            flush=True,
        )

    probe = _build_probe(
        probe_type=args.probe_type,
        input_dim=int(train_dataset.embeddings.shape[1]),
        hidden_dim=args.probe_hidden_dim,
        device=device,
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.probe_learning_rate)

    print("[probe] stage=train_probe", flush=True)
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y, batch_advantage, _, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            if args.probe_target == "action":
                batch_target = batch_y.to(device, non_blocking=pin_memory).to(torch.float32)
                predictions = probe(batch_x)
                loss = F.binary_cross_entropy_with_logits(predictions, batch_target)
            else:
                batch_target = batch_advantage.to(device, non_blocking=pin_memory).to(torch.float32)
                predictions = probe(batch_x)
                loss = F.mse_loss(predictions, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_target.shape[0])
            total_examples += int(batch_target.shape[0])

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.probe_epochs:
            if args.probe_target == "action":
                train_metrics = _evaluate_probe(probe, train_eval_loader, device)
                val_metrics = _evaluate_probe(probe, validation_loader, device)
                print(
                    f"epoch={epoch}/{args.probe_epochs} "
                    f"loss={_format_metric(total_loss / max(total_examples, 1))} "
                    f"train_acc={_format_metric(train_metrics['accuracy'])} "
                    f"train_bal_acc={_format_metric(train_metrics['balanced_accuracy'])} "
                    f"val_acc={_format_metric(val_metrics['accuracy'])} "
                    f"val_bal_acc={_format_metric(val_metrics['balanced_accuracy'])} "
                    f"train_first_action_acc={_format_metric(train_metrics['first_action_accuracy'])} "
                    f"val_first_action_acc={_format_metric(val_metrics['first_action_accuracy'])}",
                    flush=True,
                )
            else:
                train_metrics = _evaluate_advantage_probe(probe, train_eval_loader, device)
                val_metrics = _evaluate_advantage_probe(probe, validation_loader, device)
                print(
                    f"epoch={epoch}/{args.probe_epochs} "
                    f"loss={_format_metric(total_loss / max(total_examples, 1))} "
                    f"train_mse={_format_metric(train_metrics['mse'])} "
                    f"train_mae={_format_metric(train_metrics['mae'])} "
                    f"train_sign_acc={_format_metric(train_metrics['sign_accuracy'])} "
                    f"val_mse={_format_metric(val_metrics['mse'])} "
                    f"val_mae={_format_metric(val_metrics['mae'])} "
                    f"val_sign_acc={_format_metric(val_metrics['sign_accuracy'])} "
                    f"train_first_sign_acc={_format_metric(train_metrics['first_sign_accuracy'])} "
                    f"val_first_sign_acc={_format_metric(val_metrics['first_sign_accuracy'])}",
                    flush=True,
                )

    train_metrics = _evaluate_probe(probe, train_eval_loader, device) if args.probe_target == "action" else _evaluate_advantage_probe(probe, train_eval_loader, device)
    val_metrics = _evaluate_probe(probe, validation_loader, device) if args.probe_target == "action" else _evaluate_advantage_probe(probe, validation_loader, device)

    print("[probe] stage=final_metrics", flush=True)
    if args.probe_target == "advantage":
        print(f"train_mse={_format_metric(train_metrics['mse'])}")
        print(f"train_mae={_format_metric(train_metrics['mae'])}")
        print(f"train_sign_accuracy={_format_metric(train_metrics['sign_accuracy'])}")
        print(f"train_first_sign_accuracy={_format_metric(train_metrics['first_sign_accuracy'])}")
        print(f"validation_mse={_format_metric(val_metrics['mse'])}")
        print(f"validation_mae={_format_metric(val_metrics['mae'])}")
        print(f"validation_sign_accuracy={_format_metric(val_metrics['sign_accuracy'])}")
        print(f"validation_first_sign_accuracy={_format_metric(val_metrics['first_sign_accuracy'])}")
        print(f"train_phase_sign_accuracy={_round_metric_dict(train_metrics['phase_sign_accuracy'])}")
        print(f"validation_phase_sign_accuracy={_round_metric_dict(val_metrics['phase_sign_accuracy'])}")
        return
    print(f"train_accuracy={_format_metric(train_metrics['accuracy'])}")
    print(f"train_balanced_accuracy={_format_metric(train_metrics['balanced_accuracy'])}")
    print(f"train_positive_accuracy={_format_metric(train_metrics['positive_accuracy'])}")
    print(f"train_negative_accuracy={_format_metric(train_metrics['negative_accuracy'])}")
    print(f"validation_accuracy={_format_metric(val_metrics['accuracy'])}")
    print(f"validation_balanced_accuracy={_format_metric(val_metrics['balanced_accuracy'])}")
    print(f"validation_positive_accuracy={_format_metric(val_metrics['positive_accuracy'])}")
    print(f"validation_negative_accuracy={_format_metric(val_metrics['negative_accuracy'])}")
    print(f"train_first_action_accuracy={_format_metric(train_metrics['first_action_accuracy'])}")
    print(f"validation_first_action_accuracy={_format_metric(val_metrics['first_action_accuracy'])}")
    print(f"train_phase_accuracy={_round_metric_dict(train_metrics['phase_accuracy'])}")
    print(f"validation_phase_accuracy={_round_metric_dict(val_metrics['phase_accuracy'])}")


if __name__ == "__main__":
    main()
