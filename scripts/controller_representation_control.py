from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from controller_oracle import compute_oracle_policy as _compute_oracle_policy
from controller_oracle import optimal_stop_step as _optimal_stop_step
from GNN import PolicyValueTreeSearchModel, TreeEncoderOutput
from schema import NodeFeatureSchema
from supervised_branch import (
    ControllerOnlyEnv,
    FrozenEncoderControllerTrainer,
    PPOConfig,
    StepResult,
    TeacherSearchConfig,
    build_trimmed_decision_episode_with_halt_rewards,
    evaluate_controller,
    load_raw_pretrain_example_paths,
)
from tensorizer import TreeTensorizer
from tree import SearchTree


@dataclass(frozen=True)
class ControlEpisode:
    example_path: str
    halt_rewards: List[float]
    oracle_stop_step: int
    label_index: int
    oracle_actions: List[int]


@dataclass(frozen=True)
class OracleAgreementMetrics:
    exact_stop_step_accuracy: float
    first_action_accuracy: float
    confusion_matrix: Dict[int, Dict[int, int]]


@dataclass(frozen=True)
class RolloutCoverageMetrics:
    transitions_by_oracle_step: Dict[int, int]
    transitions_by_phase: Dict[int, int]
    actions_by_phase: Dict[int, Dict[int, int]]
    phase_by_oracle_step: Dict[int, Dict[int, int]]


def _episode_summary_line(episode: ControlEpisode) -> str:
    halt_rewards = ",".join(f"{reward:.6f}" for reward in episode.halt_rewards)
    oracle_actions = ",".join(str(action) for action in episode.oracle_actions)
    return (
        f"path={episode.example_path} "
        f"oracle_stop_step={episode.oracle_stop_step} "
        f"label_index={episode.label_index} "
        f"halt_rewards=[{halt_rewards}] "
        f"oracle_actions=[{oracle_actions}]"
    )


def _episode_summary_dict(episode: ControlEpisode) -> Dict[str, object]:
    return {
        "example_path": episode.example_path,
        "oracle_stop_step": episode.oracle_stop_step,
        "label_index": episode.label_index,
        "halt_rewards": list(episode.halt_rewards),
        "oracle_actions": list(episode.oracle_actions),
    }


def _make_trace_writer(debug_dir: Path) -> Callable[[str, Dict[str, object]], None]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    trace_path = debug_dir / "trace.jsonl"

    def _write(event: str, payload: Dict[str, object]) -> None:
        record = {"event": event, **payload}
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return _write


class FixedFeatureEncoder(nn.Module):
    def __init__(self, node_feat: int, device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.d_embed = node_feat

    def forward(self, tree_batch):
        node_states = tree_batch.node_features.to(self.device)
        root_states = node_states[tree_batch.root_index.to(self.device)]
        return TreeEncoderOutput(node_states=node_states, root_states=root_states)

def _teacher_config(args: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="representation_control_v1",
        search_config_id="representation_control",
    )


def _load_control_episodes(
    data: str,
    representation: str,
    continue_cost: float,
    reward_scale: float,
    quality_config: TeacherSearchConfig,
    max_examples: int,
    episodes_per_stop_step: int,
    seed: int,
) -> List[ControlEpisode]:
    paths = load_raw_pretrain_example_paths(data)
    rng = random.Random(seed)
    if max_examples > 0 and max_examples < len(paths):
        paths = rng.sample(paths, max_examples)

    episodes: List[ControlEpisode] = []
    for example_index, path in enumerate(paths):
        example = torch.load(path, weights_only=False)
        try:
            _, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(
                example,
                quality_config,
            )
        except ValueError:
            continue
        halt_rewards = [reward_scale * reward for reward in halt_rewards]
        oracle_policy = _compute_oracle_policy(halt_rewards, continue_cost)
        oracle_stop = oracle_policy.optimal_stop_step
        if representation == "oracle-stop-step":
            label_index = oracle_stop
        elif representation == "oracle-action-now":
            label_index = 0
        elif representation == "example-id":
            label_index = len(episodes)
        else:
            raise ValueError(f"Unsupported representation: {representation}")
        episodes.append(
            ControlEpisode(
                example_path=path,
                halt_rewards=halt_rewards,
                oracle_stop_step=oracle_stop,
                label_index=label_index,
                oracle_actions=list(oracle_policy.actions),
            )
        )
    if not episodes:
        raise ValueError("No usable control episodes were produced.")
    if episodes_per_stop_step > 0:
        grouped: Dict[int, List[ControlEpisode]] = {}
        for episode in episodes:
            grouped.setdefault(episode.oracle_stop_step, []).append(episode)
        balanced: List[ControlEpisode] = []
        for stop_step in sorted(grouped):
            bucket = list(grouped[stop_step])
            rng.shuffle(bucket)
            balanced.extend(bucket[:episodes_per_stop_step])
        if not balanced:
            raise ValueError("episodes_per_stop_step filtering removed all episodes.")
        episodes = balanced
    return episodes


def _feature_names(representation: str, max_steps: int, num_labels: int) -> List[str]:
    if representation == "oracle-action-now":
        return ["action_0", "action_1"]
    return [f"phase_{idx}" for idx in range(max_steps)] + [f"label_{idx}" for idx in range(num_labels)]


def _build_schema(
    max_steps: int,
    num_labels: int,
    representation: str = "oracle-stop-step",
) -> NodeFeatureSchema:
    feature_names = _feature_names(representation, max_steps, num_labels)
    return NodeFeatureSchema.from_ordered_features(feature_names, defaults={name: 0.0 for name in feature_names})


def _make_observation_tree(
    step_index: int,
    label_index: int,
    max_steps: int,
    num_labels: int,
    representation: str = "oracle-stop-step",
    oracle_action: int | None = None,
) -> SearchTree:
    features: Dict[str, float] = {}
    if representation == "oracle-action-now":
        if oracle_action not in (0, 1):
            raise ValueError("oracle_action_now representation requires oracle_action in {0,1}.")
        features["action_0"] = 1.0 if oracle_action == 0 else 0.0
        features["action_1"] = 1.0 if oracle_action == 1 else 0.0
    else:
        for idx in range(max_steps):
            features[f"phase_{idx}"] = 1.0 if idx == step_index else 0.0
        for idx in range(num_labels):
            features[f"label_{idx}"] = 1.0 if idx == label_index else 0.0
    tree = SearchTree()
    tree.create_root("control-root", features)
    return tree


def _decode_observation_tree(
    tree: SearchTree,
    max_steps: int,
    num_labels: int,
    representation: str = "oracle-stop-step",
) -> Tuple[int, int]:
    if representation == "oracle-action-now":
        raise ValueError("oracle-action-now observations do not encode phase/label indices.")
    root = tree.get_node(tree.root_id)
    phase_index = 0
    label_index = 0
    for idx in range(max_steps):
        if float(root.scalar_features.get(f"phase_{idx}", 0.0)) > 0.5:
            phase_index = idx
            break
    for idx in range(num_labels):
        if float(root.scalar_features.get(f"label_{idx}", 0.0)) > 0.5:
            label_index = idx
            break
    return phase_index, label_index


class RepresentationControlEnv(ControllerOnlyEnv):
    def __init__(
        self,
        episodes: Sequence[ControlEpisode],
        continue_cost: float,
        max_steps: int,
        num_labels: int,
        representation: str = "oracle-stop-step",
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if not episodes:
            raise ValueError("episodes must be non-empty.")
        self.episodes = list(episodes)
        self.continue_cost = float(continue_cost)
        self.max_steps = max_steps
        self.num_labels = num_labels
        self.representation = representation
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._episode_index = -1
        self._step_index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        self._current: ControlEpisode | None = None

    def _current_tree(self) -> SearchTree:
        assert self._current is not None
        return _make_observation_tree(
            step_index=self._step_index,
            label_index=self._current.label_index,
            max_steps=self.max_steps,
            num_labels=self.num_labels,
            representation=self.representation,
            oracle_action=self._current.oracle_actions[self._step_index] if self.representation == "oracle-action-now" else None,
        )

    def reset(self) -> SearchTree:
        if self.shuffle:
            self._episode_index = self._rng.randrange(len(self.episodes))
        else:
            self._episode_index = (self._episode_index + 1) % len(self.episodes)
        self._current = self.episodes[self._episode_index]
        self._step_index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        return self._current_tree()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise ValueError("Episode already finished. Call reset().")
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1.")
        assert self._current is not None

        self._episode_length += 1
        done = False
        reward = 0.0
        if action == 1:
            reward = self._current.halt_rewards[self._step_index]
            done = True
        else:
            reward = -self.continue_cost
            if self._step_index + 1 < len(self._current.halt_rewards):
                self._step_index += 1
            else:
                reward += self._current.halt_rewards[self._step_index]
                done = True

        self._episode_return += reward
        self._done = done
        info = {}
        if done:
            info = {
                "episode_return": self._episode_return,
                "episode_length": self._episode_length,
                "expansions": self._step_index,
                "terminal_quality": self._current.halt_rewards[self._step_index],
                "halted": action == 1,
                "oracle_stop_step": self._current.oracle_stop_step,
                "example_path": self._current.example_path,
            }
        return StepResult(next_tree=self._current_tree(), reward=reward, done=done, info=info)


def _predict_stop_step(
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
    env: RepresentationControlEnv,
) -> int:
    tree = env.reset()
    done = False
    stop_step = 0
    while not done:
        tree_batch = tensorizer.tensorize_tree(tree)
        output = model(tree_batch)
        halt_logit = float(output.halt_logits.squeeze(0).item())
        action = 1 if halt_logit >= 0.0 else 0
        step_result = env.step(action)
        tree = step_result.next_tree
        done = step_result.done
        if done:
            stop_step = int(step_result.info["expansions"])
    return stop_step


def evaluate_oracle_agreement(
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
    episodes: Sequence[ControlEpisode],
    continue_cost: float,
    max_steps: int,
    num_labels: int,
) -> OracleAgreementMetrics:
    model.eval()
    exact_matches = 0
    first_action_matches = 0
    confusion_matrix: Dict[int, Dict[int, int]] = {}

    with torch.no_grad():
        for index, episode in enumerate(episodes):
            env = RepresentationControlEnv(
                [episode],
                continue_cost=continue_cost,
                max_steps=max_steps,
                num_labels=num_labels,
                seed=index,
                shuffle=False,
            )
            predicted_stop_step = _predict_stop_step(model, tensorizer, env)
            oracle_stop_step = episode.oracle_stop_step
            if predicted_stop_step == oracle_stop_step:
                exact_matches += 1
            if (predicted_stop_step == 0) == (oracle_stop_step == 0):
                first_action_matches += 1
            row = confusion_matrix.setdefault(oracle_stop_step, {})
            row[predicted_stop_step] = row.get(predicted_stop_step, 0) + 1

    total = len(episodes)
    return OracleAgreementMetrics(
        exact_stop_step_accuracy=exact_matches / total,
        first_action_accuracy=first_action_matches / total,
        confusion_matrix=confusion_matrix,
    )


def _collect_rollout_coverage(
    trainer: FrozenEncoderControllerTrainer,
    episodes: Sequence[ControlEpisode],
    max_steps: int,
    num_labels: int,
    representation: str,
) -> RolloutCoverageMetrics:
    if representation == "oracle-action-now":
        raise ValueError("Rollout coverage is not available for oracle-action-now because phase and episode labels are intentionally hidden.")
    label_to_oracle_step = {episode.label_index: episode.oracle_stop_step for episode in episodes}
    transitions_by_oracle_step: Dict[int, int] = {}
    transitions_by_phase: Dict[int, int] = {}
    actions_by_phase: Dict[int, Dict[int, int]] = {}
    phase_by_oracle_step: Dict[int, Dict[int, int]] = {}

    rollout, _ = trainer._collect_rollout()
    for transition in rollout:
        phase_index, label_index = _decode_observation_tree(
            transition.observation,
            max_steps=max_steps,
            num_labels=num_labels,
            representation=representation,
        )
        oracle_stop_step = label_to_oracle_step[label_index]
        transitions_by_oracle_step[oracle_stop_step] = transitions_by_oracle_step.get(oracle_stop_step, 0) + 1
        transitions_by_phase[phase_index] = transitions_by_phase.get(phase_index, 0) + 1
        phase_counts = phase_by_oracle_step.setdefault(oracle_stop_step, {})
        phase_counts[phase_index] = phase_counts.get(phase_index, 0) + 1
        action_counts = actions_by_phase.setdefault(phase_index, {})
        action_counts[transition.action] = action_counts.get(transition.action, 0) + 1

    return RolloutCoverageMetrics(
        transitions_by_oracle_step=transitions_by_oracle_step,
        transitions_by_phase=transitions_by_phase,
        actions_by_phase=actions_by_phase,
        phase_by_oracle_step=phase_by_oracle_step,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control diagnostic: train the production PPO path on intentionally trivial representations."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--representation", choices=["oracle-stop-step", "oracle-action-now", "example-id"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--episodes-per-stop-step", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--debug-dir")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    quality_config = _teacher_config(args)
    episodes = _load_control_episodes(
        data=args.data,
        representation=args.representation,
        continue_cost=args.continue_cost,
        reward_scale=args.reward_scale,
        quality_config=quality_config,
        max_examples=args.max_examples,
        episodes_per_stop_step=args.episodes_per_stop_step,
        seed=args.seed,
    )
    max_steps = max(len(episode.halt_rewards) for episode in episodes)
    num_labels = max(episode.label_index for episode in episodes) + 1
    trace_callback = None
    trace_state = {"update_index": 0}
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        base_trace_callback = _make_trace_writer(debug_dir)
        config_path = debug_dir / "run_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "data": args.data,
                    "representation": args.representation,
                    "continue_cost": args.continue_cost,
                    "reward_scale": args.reward_scale,
                    "max_examples": args.max_examples,
                    "episodes_per_stop_step": args.episodes_per_stop_step,
                    "seed": args.seed,
                    "num_envs": args.num_envs,
                    "rollout_steps": args.rollout_steps,
                    "num_updates": args.num_updates,
                    "ppo_epochs": args.ppo_epochs,
                    "minibatch_size": args.minibatch_size,
                    "learning_rate": args.learning_rate,
                    "eval_episodes": args.eval_episodes,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        selected_path = debug_dir / "selected_episodes.jsonl"
        with selected_path.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(_episode_summary_dict(episode), sort_keys=True) + "\n")

        def _trace(event: str, payload: Dict[str, object]) -> None:
            base_trace_callback(
                event,
                {
                    "update_index": trace_state["update_index"],
                    **payload,
                },
            )

        trace_callback = _trace

    if args.representation == "oracle-action-now":
        num_labels = 2
        max_steps = 1
    schema = _build_schema(max_steps=max_steps, num_labels=num_labels, representation=args.representation)
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
    envs = [
        RepresentationControlEnv(
            episodes,
            args.continue_cost,
            max_steps=max_steps,
            num_labels=num_labels,
            representation=args.representation,
            seed=args.seed + env_index,
            shuffle=True,
        )
        for env_index in range(args.num_envs)
    ]
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=envs,
        config=PPOConfig(
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        ),
        trace_callback=trace_callback,
    )

    oracle_histogram: Dict[int, int] = {}
    for episode in episodes:
        oracle_histogram[episode.oracle_stop_step] = oracle_histogram.get(episode.oracle_stop_step, 0) + 1

    print(f"representation={args.representation}")
    print(f"episodes={len(episodes)}")
    print(f"continue_cost={args.continue_cost:.6f}")
    print(f"reward_scale={args.reward_scale:.6f}")
    print(f"max_steps={max_steps}")
    print(f"num_labels={num_labels}")
    print(f"oracle_stop_histogram={oracle_histogram}")
    if args.debug_dir:
        print(f"debug_dir={args.debug_dir}")
    print("selected_episodes=")
    for oracle_step in sorted(oracle_histogram):
        print(f"  oracle_{oracle_step}:")
        for episode in episodes:
            if episode.oracle_stop_step == oracle_step:
                print(f"    {_episode_summary_line(episode)}")

    if args.inspect_only:
        return

    if args.representation != "oracle-action-now":
        coverage = _collect_rollout_coverage(
            trainer,
            episodes,
            max_steps=max_steps,
            num_labels=num_labels,
            representation=args.representation,
        )
        print(f"rollout_transitions_by_oracle_step={coverage.transitions_by_oracle_step}")
        print(f"rollout_transitions_by_phase={coverage.transitions_by_phase}")
        print(f"rollout_actions_by_phase={coverage.actions_by_phase}")
        print(f"rollout_phase_by_oracle_step={coverage.phase_by_oracle_step}")
        if trace_callback is not None:
            trace_callback(
                "rollout_coverage",
                {
                    "transitions_by_oracle_step": coverage.transitions_by_oracle_step,
                    "transitions_by_phase": coverage.transitions_by_phase,
                    "actions_by_phase": coverage.actions_by_phase,
                    "phase_by_oracle_step": coverage.phase_by_oracle_step,
                },
            )
    else:
        print("rollout_coverage=unavailable_for_oracle_action_now")

    for update_idx in range(1, args.num_updates + 1):
        trace_state["update_index"] = update_idx
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

    def _make_env() -> RepresentationControlEnv:
        return RepresentationControlEnv(
            episodes,
            args.continue_cost,
            max_steps=max_steps,
            num_labels=num_labels,
            representation=args.representation,
            seed=args.seed,
            shuffle=False,
        )

    evaluation = evaluate_controller(
        model,
        tensorizer,
        _make_env,
        num_episodes=args.eval_episodes,
        trace_callback=trace_callback,
    )
    print(
        f"average_return={evaluation.average_return:.6f} "
        f"average_expansions={evaluation.average_expansions:.6f} "
        f"average_terminal_quality={evaluation.average_terminal_quality:.6f}"
    )
    print("halt_step_histogram=")
    for step in sorted(evaluation.halt_step_histogram):
        print(f"  step_{step}={evaluation.halt_step_histogram[step]}")

    oracle_agreement = evaluate_oracle_agreement(
        model,
        tensorizer,
        episodes,
        continue_cost=args.continue_cost,
        max_steps=max_steps,
        num_labels=num_labels,
    )
    print(f"exact_stop_step_accuracy={oracle_agreement.exact_stop_step_accuracy:.6f}")
    print(f"first_action_accuracy={oracle_agreement.first_action_accuracy:.6f}")
    print("oracle_confusion_matrix=")
    for oracle_step in sorted(oracle_agreement.confusion_matrix):
        row = oracle_agreement.confusion_matrix[oracle_step]
        formatted = " ".join(
            f"pred_{predicted_step}={row[predicted_step]}"
            for predicted_step in sorted(row)
        )
        print(f"  oracle_{oracle_step}: {formatted}")


if __name__ == "__main__":
    main()
