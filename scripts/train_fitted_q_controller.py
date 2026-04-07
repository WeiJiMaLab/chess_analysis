from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from controller_oracle import compute_oracle_policy
from cts_pretrain import load_encoder_checkpoint
from GNN import TreeEncoderOutput, TreeNN
from schema import NodeFeatureSchema
from supervised_branch import (
    TeacherSearchConfig,
    build_trimmed_decision_episode_with_halt_rewards,
    load_raw_pretrain_example_paths,
)
from tensorizer import TreeBatch, TreeTensorizer
from tree import SearchTree


@dataclass(frozen=True)
class FittedQEpisode:
    path: str
    snapshots: List[SearchTree]
    halt_rewards: List[float]
    q_targets: torch.Tensor
    oracle_stop_step: int
    oracle_value: float


@dataclass(frozen=True)
class FittedQBatch:
    tree_batch: TreeBatch
    q_targets: torch.Tensor
    paths: List[str]
    skipped: int


@dataclass(frozen=True)
class AdvantageMetrics:
    advantage_mse: float
    mean_abs_advantage_error: float
    sign_accuracy: float
    examples: int


@dataclass(frozen=True)
class GreedyPolicyMetrics:
    exact_stop_step_accuracy: float
    first_action_accuracy: float
    average_return: float
    average_oracle_value: float
    average_regret: float
    average_expansions: float
    evaluated_episodes: int
    skipped_episodes: int


class ComputeAdvantageTreeSearchModel(nn.Module):
    def __init__(
        self,
        *,
        k: int,
        node_feat: int,
        device: str,
        node_embed_hidden: int,
        d_embed: int,
        d_message: int,
        n_heads: int,
        d_att: int,
        q_hidden: int,
    ) -> None:
        super().__init__()
        self.encoder = TreeNN(
            k=k,
            node_feat=node_feat,
            device=device,
            node_embed_hidden=node_embed_hidden,
            d_embed=d_embed,
            d_message=d_message,
            n_heads=n_heads,
            d_att=d_att,
        )
        resolved_device = self.encoder.device
        self.advantage_head = nn.Sequential(
            nn.Linear(self.encoder.d_embed, q_hidden, device=resolved_device),
            nn.ReLU(),
            nn.Linear(q_hidden, 1, device=resolved_device),
        )

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, tree_batch: TreeBatch) -> torch.Tensor:
        encoded: TreeEncoderOutput = self.encoder(tree_batch)
        return self.advantage_head(encoded.root_states).squeeze(-1)


class FittedQEpisodeDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        quality_config: TeacherSearchConfig,
        continue_cost: float,
        reward_scale: float,
    ) -> None:
        self.paths = list(paths)
        self.quality_config = quality_config
        self.continue_cost = float(continue_cost)
        self.reward_scale = float(reward_scale)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> FittedQEpisode | None:
        path = self.paths[index]
        try:
            example = torch.load(path, weights_only=False)
            episode, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(
                example,
                self.quality_config,
            )
        except ValueError:
            return None

        scaled_rewards = [self.reward_scale * reward for reward in halt_rewards]
        q_targets, oracle_stop_step, oracle_value = bellman_q_targets(
            scaled_rewards,
            self.continue_cost,
        )
        return FittedQEpisode(
            path=path,
            snapshots=list(episode.snapshots),
            halt_rewards=scaled_rewards,
            q_targets=q_targets,
            oracle_stop_step=oracle_stop_step,
            oracle_value=oracle_value,
        )


class FittedQCollator:
    def __init__(self, tensorizer: TreeTensorizer) -> None:
        self.tensorizer = tensorizer

    def __call__(self, episodes: Sequence[FittedQEpisode | None]) -> FittedQBatch | None:
        valid = [episode for episode in episodes if episode is not None]
        if not valid:
            return None

        trees: List[SearchTree] = []
        q_targets = []
        paths = []
        for episode in valid:
            trees.extend(episode.snapshots)
            q_targets.append(episode.q_targets)
            paths.append(episode.path)

        return FittedQBatch(
            tree_batch=self.tensorizer.tensorize_forest(trees, validate=False),
            q_targets=torch.cat(q_targets, dim=0),
            paths=paths,
            skipped=len(episodes) - len(valid),
        )


def bellman_q_targets(
    halt_rewards: Sequence[float],
    continue_cost: float,
) -> tuple[torch.Tensor, int, float]:
    policy = compute_oracle_policy(halt_rewards, continue_cost)
    targets: List[List[float]] = []
    for index, halt_reward in enumerate(halt_rewards):
        if index + 1 < len(halt_rewards):
            continue_value = -continue_cost + policy.values[index + 1]
        else:
            continue_value = -continue_cost + float(halt_reward)
        targets.append([continue_value, float(halt_reward)])
    return (
        torch.tensor(targets, dtype=torch.float32),
        policy.optimal_stop_step,
        float(policy.values[0]),
    )


def _feature_schema() -> NodeFeatureSchema:
    return NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})


def _quality_config(args: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def _sample_paths(paths: Sequence[str], max_examples: int, seed: int) -> List[str]:
    selected = list(paths)
    if max_examples > 0 and max_examples < len(selected):
        rng = random.Random(seed)
        selected = rng.sample(selected, max_examples)
    return selected


def _build_loader(
    paths: Sequence[str],
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    reward_scale: float,
    tensorizer: TreeTensorizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = FittedQEpisodeDataset(
        paths=paths,
        quality_config=quality_config,
        continue_cost=continue_cost,
        reward_scale=reward_scale,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=FittedQCollator(tensorizer),
        generator=generator if shuffle else None,
    )


def _target_advantages(targets: torch.Tensor) -> torch.Tensor:
    return targets[:, 0] - targets[:, 1]


def _advantage_loss_components(
    predicted_advantages: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_advantages = _target_advantages(targets)
    advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
    mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))
    sign_accuracy = ((predicted_advantages > 0) == (target_advantages > 0)).float().mean()
    return advantage_mse, mean_abs_advantage_error, sign_accuracy


def _advantage_sign_correct_count(predicted_advantages: torch.Tensor, targets: torch.Tensor) -> int:
    target_advantages = _target_advantages(targets)
    return int(((predicted_advantages > 0) == (target_advantages > 0)).sum().item())


def _train_epoch(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
) -> AdvantageMetrics:
    model.train()
    total_advantage_mse = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    skipped = 0
    started = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        targets = batch.q_targets.to(device, non_blocking=True)
        predicted_advantages = model(batch.tree_batch)
        advantage_mse, mean_abs_advantage_error, _sign_accuracy = _advantage_loss_components(
            predicted_advantages,
            targets,
        )

        optimizer.zero_grad()
        advantage_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(targets.shape[0])
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += _advantage_sign_correct_count(predicted_advantages.detach(), targets)
        skipped += int(batch.skipped)

        if log_interval > 0 and batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"snapshots={total_examples} skipped_episodes={skipped} "
                f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if total_examples == 0:
        raise ValueError("Training loader produced no valid compute-advantage snapshots.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_advantage_predictions(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> AdvantageMetrics:
    model.eval()
    total_advantage_mse = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            targets = batch.q_targets.to(device, non_blocking=True)
            predicted_advantages = model(batch.tree_batch)
            advantage_mse, mean_abs_advantage_error, _sign_accuracy = _advantage_loss_components(
                predicted_advantages,
                targets,
            )
            examples = int(targets.shape[0])
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += _advantage_sign_correct_count(predicted_advantages, targets)
    if total_examples == 0:
        raise ValueError("Evaluation loader produced no valid compute-advantage snapshots.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _predict_stop_step(
    model: ComputeAdvantageTreeSearchModel,
    tensorizer: TreeTensorizer,
    episode: FittedQEpisode,
) -> int:
    model.eval()
    with torch.inference_mode():
        tree_batch = tensorizer.tensorize_forest(episode.snapshots, validate=False)
        predicted_advantages = model(tree_batch).detach().cpu().tolist()
    for step_index, advantage in enumerate(predicted_advantages):
        if float(advantage) <= 0:
            return step_index
    return len(predicted_advantages) - 1


def evaluate_greedy_policy(
    model: ComputeAdvantageTreeSearchModel,
    paths: Sequence[str],
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    reward_scale: float,
    tensorizer: TreeTensorizer,
    *,
    log_interval: int,
) -> GreedyPolicyMetrics:
    dataset = FittedQEpisodeDataset(paths, quality_config, continue_cost, reward_scale)
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    evaluated = 0
    skipped = 0
    started = time.time()

    for index in range(len(dataset)):
        episode = dataset[index]
        if episode is None:
            skipped += 1
            continue
        predicted_stop = _predict_stop_step(model, tensorizer, episode)
        predicted_return = -continue_cost * predicted_stop + episode.halt_rewards[predicted_stop]

        exact += int(predicted_stop == episode.oracle_stop_step)
        first_action += int((predicted_stop == 0) == (episode.oracle_stop_step == 0))
        total_return += predicted_return
        total_oracle_value += episode.oracle_value
        total_expansions += predicted_stop
        evaluated += 1

        if log_interval > 0 and (index + 1) % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"greedy_eval_progress={index + 1}/{len(dataset)} "
                f"evaluated={evaluated} skipped={skipped} "
                f"exact_stop_step_accuracy={exact / max(evaluated, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no valid fitted-Q episodes.")
    return GreedyPolicyMetrics(
        exact_stop_step_accuracy=exact / evaluated,
        first_action_accuracy=first_action / evaluated,
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle_value / evaluated,
        average_regret=(total_oracle_value - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
        skipped_episodes=skipped,
    )


def _save_checkpoint(path: str, model: ComputeAdvantageTreeSearchModel, metadata: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an offline counterfactual compute-advantage controller on generated tree snapshot trajectories."
        )
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    parser.add_argument("--q-hidden", type=int, default=128, help="Hidden width for the compute-advantage head.")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of full episodes per DataLoader batch.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--greedy-eval-interval", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[compute_advantage] stage=load_paths", flush=True)
    train_paths = _sample_paths(load_raw_pretrain_example_paths(args.train_data), args.max_train_examples, args.seed)
    validation_paths = _sample_paths(
        load_raw_pretrain_example_paths(args.validation_data),
        args.max_validation_examples,
        args.seed + 1,
    )
    print(
        f"[compute_advantage] train_examples={len(train_paths)} validation_examples={len(validation_paths)}",
        flush=True,
    )

    schema = _feature_schema()
    tensorizer = TreeTensorizer(schema, device="cpu")
    quality_config = _quality_config(args)
    device = torch.device(args.device)

    print("[compute_advantage] stage=build_model", flush=True)
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
    )
    load_encoder_checkpoint(args.encoder_checkpoint, model.encoder)
    if not args.unfreeze_encoder:
        model.freeze_encoder()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = _build_loader(
        train_paths,
        quality_config,
        args.continue_cost,
        args.reward_scale,
        tensorizer,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validation_loader = _build_loader(
        validation_paths,
        quality_config,
        args.continue_cost,
        args.reward_scale,
        tensorizer,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    best_validation_advantage_mse = float("inf")
    best_metadata: dict | None = None
    print("[compute_advantage] stage=train_start", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_advantage_mse={train_metrics.advantage_mse:.3f} "
            f"train_mean_abs_advantage_error={train_metrics.mean_abs_advantage_error:.3f} "
            f"train_sign_accuracy={train_metrics.sign_accuracy:.3f} "
            f"train_snapshots={train_metrics.examples}",
            flush=True,
        )

        validation_metrics = None
        if args.validation_interval > 0 and (epoch % args.validation_interval == 0 or epoch == args.epochs):
            validation_metrics = evaluate_advantage_predictions(model, validation_loader, device=device)
            print(
                f"validation_epoch={epoch}/{args.epochs} "
                f"validation_advantage_mse={validation_metrics.advantage_mse:.3f} "
                f"validation_mean_abs_advantage_error={validation_metrics.mean_abs_advantage_error:.3f} "
                f"validation_sign_accuracy={validation_metrics.sign_accuracy:.3f} "
                f"validation_snapshots={validation_metrics.examples}",
                flush=True,
            )
            if validation_metrics.advantage_mse < best_validation_advantage_mse:
                best_validation_advantage_mse = validation_metrics.advantage_mse
                best_metadata = {
                    "stage": "compute_advantage_controller",
                    "epoch": epoch,
                    "validation_advantage_mse": validation_metrics.advantage_mse,
                    "validation_sign_accuracy": validation_metrics.sign_accuracy,
                    "continue_cost": args.continue_cost,
                    "reward_scale": args.reward_scale,
                    "encoder_checkpoint": args.encoder_checkpoint,
                    "unfreeze_encoder": args.unfreeze_encoder,
                }
                if args.output_checkpoint:
                    print(f"[compute_advantage] stage=save_best path={args.output_checkpoint}", flush=True)
                    _save_checkpoint(args.output_checkpoint, model, best_metadata)

        if args.greedy_eval_interval > 0 and (epoch % args.greedy_eval_interval == 0 or epoch == args.epochs):
            greedy_metrics = evaluate_greedy_policy(
                model,
                validation_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                tensorizer,
                log_interval=args.log_interval,
            )
            print(
                f"greedy_epoch={epoch}/{args.epochs} "
                f"exact_stop_step_accuracy={greedy_metrics.exact_stop_step_accuracy:.3f} "
                f"first_action_accuracy={greedy_metrics.first_action_accuracy:.3f} "
                f"average_return={greedy_metrics.average_return:.3f} "
                f"average_oracle_value={greedy_metrics.average_oracle_value:.3f} "
                f"average_regret={greedy_metrics.average_regret:.3f} "
                f"average_expansions={greedy_metrics.average_expansions:.3f} "
                f"evaluated_episodes={greedy_metrics.evaluated_episodes} "
                f"skipped_episodes={greedy_metrics.skipped_episodes}",
                flush=True,
            )

    if args.output_checkpoint and best_metadata is None:
        _save_checkpoint(
            args.output_checkpoint,
            model,
            {
                "stage": "compute_advantage_controller",
                "epoch": args.epochs,
                "continue_cost": args.continue_cost,
                "reward_scale": args.reward_scale,
                "encoder_checkpoint": args.encoder_checkpoint,
                "unfreeze_encoder": args.unfreeze_encoder,
            },
        )
    print("[compute_advantage] stage=done", flush=True)


if __name__ == "__main__":
    main()
