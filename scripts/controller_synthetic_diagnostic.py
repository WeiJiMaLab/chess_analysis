from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from controller_oracle import optimal_stop_step as _optimal_stop_step
from GNN import PolicyValueTreeSearchModel, TreeEncoderOutput
from schema import NodeFeatureSchema
from supervised_branch import (
    ControllerOnlyEnv,
    FrozenEncoderControllerTrainer,
    PPOConfig,
    StepResult,
    evaluate_controller,
)
from tensorizer import TreeTensorizer
from tree import ExpansionChild, SearchTree


def _make_schema(num_steps: int) -> NodeFeatureSchema:
    feature_names = [f"phase_{index}" for index in range(num_steps)]
    return NodeFeatureSchema.from_ordered_features(feature_names, defaults={name: 0.0 for name in feature_names})


def _make_snapshot(step_index: int, num_steps: int, best_value: float) -> SearchTree:
    root_features = {f"phase_{index}": 1.0 if index == step_index else 0.0 for index in range(num_steps)}
    tree = SearchTree()
    root_id = tree.create_root("synthetic-root", root_features)
    tree.add_children(
        root_id,
        [
            ExpansionChild("best", f"best-{step_index}", root_features),
            ExpansionChild("other", f"other-{step_index}", root_features),
        ],
    )
    return tree


class FixedFeatureEncoder(nn.Module):
    def __init__(self, node_feat: int, device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.d_embed = node_feat

    def forward(self, tree_batch):
        node_states = tree_batch.node_features.to(self.device)
        root_states = node_states[tree_batch.root_index.to(self.device)]
        return TreeEncoderOutput(node_states=node_states, root_states=root_states)


class SyntheticHaltEnv(ControllerOnlyEnv):
    def __init__(self, halt_rewards: Sequence[float], continue_cost: float) -> None:
        if not halt_rewards:
            raise ValueError("halt_rewards must be non-empty.")
        self.halt_rewards = [float(value) for value in halt_rewards]
        self.continue_cost = float(continue_cost)
        num_steps = len(halt_rewards)
        self.snapshots = [
            _make_snapshot(step_index, num_steps, value) for step_index, value in enumerate(halt_rewards)
        ]
        self._index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self) -> SearchTree:
        self._index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        return self.snapshots[self._index]

    def step(self, action: int) -> StepResult:
        if self._done:
            raise ValueError("Episode already finished. Call reset().")
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1.")

        self._episode_length += 1
        done = False
        reward = 0.0
        if action == 1:
            reward = self.halt_rewards[self._index]
            done = True
        else:
            reward = -self.continue_cost
            if self._index + 1 < len(self.snapshots):
                self._index += 1
            else:
                reward += self.halt_rewards[self._index]
                done = True

        self._episode_return += reward
        self._done = done
        info = {}
        if done:
            info = {
                "episode_return": self._episode_return,
                "episode_length": self._episode_length,
                "expansions": self._index,
                "terminal_quality": self.halt_rewards[self._index],
                "halted": action == 1,
            }
        return StepResult(next_tree=self.snapshots[self._index], reward=reward, done=done, info=info)


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    halt_rewards: List[float]
    continue_cost: float
    expected_stop_step: int


SCENARIOS = {
    "always-halt": SyntheticScenario(
        name="always-halt",
        halt_rewards=[0.50, 0.40, 0.30],
        continue_cost=0.05,
        expected_stop_step=0,
    ),
    "always-continue": SyntheticScenario(
        name="always-continue",
        halt_rewards=[0.10, 0.30, 0.60],
        continue_cost=0.05,
        expected_stop_step=2,
    ),
    "continue-once": SyntheticScenario(
        name="continue-once",
        halt_rewards=[0.10, 0.30, 0.31],
        continue_cost=0.05,
        expected_stop_step=1,
    ),
    "continue-twice": SyntheticScenario(
        name="continue-twice",
        halt_rewards=[0.05, 0.10, 0.35, 0.36],
        continue_cost=0.05,
        expected_stop_step=2,
    ),
    "halt-late-drop": SyntheticScenario(
        name="halt-late-drop",
        halt_rewards=[0.08, 0.26, 0.24, 0.23],
        continue_cost=0.05,
        expected_stop_step=1,
    ),
    "borderline-halt": SyntheticScenario(
        name="borderline-halt",
        halt_rewards=[0.20, 0.25, 0.29],
        continue_cost=0.05,
        expected_stop_step=0,
    ),
    "small-gains-always-continue": SyntheticScenario(
        name="small-gains-always-continue",
        halt_rewards=[0.00, 0.07, 0.14, 0.21],
        continue_cost=0.05,
        expected_stop_step=3,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic PPO sanity checks for the halt/continue controller.")
    parser.add_argument("--scenario", choices=["all", *sorted(SCENARIOS.keys())], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def _run_scenario(args: argparse.Namespace, scenario: SyntheticScenario) -> None:
    oracle_stop = _optimal_stop_step(scenario.halt_rewards, scenario.continue_cost)
    schema = _make_schema(len(scenario.halt_rewards))
    tensorizer = TreeTensorizer(schema, device=args.device)
    encoder = FixedFeatureEncoder(node_feat=len(schema.feature_names), device=args.device)
    model = PolicyValueTreeSearchModel(
        k=2,
        node_feat=len(schema.feature_names),
        device=args.device,
        node_embed_hidden=64,
        d_embed=len(schema.feature_names),
        d_message=len(schema.feature_names),
        n_heads=4,
        d_att=16,
        controller_hidden=64,
        value_hidden=64,
        encoder=encoder,
    )
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[SyntheticHaltEnv(scenario.halt_rewards, scenario.continue_cost) for _ in range(args.num_envs)],
        config=PPOConfig(
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        ),
    )

    print(f"scenario={scenario.name}")
    print(f"halt_rewards={scenario.halt_rewards}")
    print(f"continue_cost={scenario.continue_cost:.6f}")
    print(f"oracle_stop_step={oracle_stop}")
    print("state_encoding=one_hot_step")

    for update_idx in range(1, args.num_updates + 1):
        metrics = trainer.train_update()
        if update_idx % args.log_interval == 0 or update_idx == args.num_updates:
            print(
                f"update={update_idx}/{args.num_updates} "
                f"policy_loss={metrics.policy_loss:.6f} "
                f"value_loss={metrics.value_loss:.6f} "
                f"entropy={metrics.entropy:.6f} "
                f"mean_episode_return={metrics.mean_episode_return:.6f} "
                f"mean_episode_length={metrics.mean_episode_length:.3f} "
                f"halt_rate={metrics.halt_rate:.3f}",
                flush=True,
            )

    def _make_env() -> SyntheticHaltEnv:
        return SyntheticHaltEnv(scenario.halt_rewards, scenario.continue_cost)

    evaluation = evaluate_controller(model, tensorizer, _make_env, num_episodes=args.eval_episodes)
    print(
        f"average_return={evaluation.average_return:.6f} "
        f"average_expansions={evaluation.average_expansions:.6f} "
        f"average_terminal_quality={evaluation.average_terminal_quality:.6f}"
    )
    print("halt_step_histogram=")
    for step in sorted(evaluation.halt_step_histogram):
        print(f"  step_{step}={evaluation.halt_step_histogram[step]}")
    print("")


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    scenarios = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]
    for scenario in scenarios:
        _run_scenario(args, scenario)


if __name__ == "__main__":
    main()
