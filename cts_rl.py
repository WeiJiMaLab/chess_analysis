from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from GNN import PolicyValueTreeSearchModel
from cts_episode_envs import ControllerOnlyEnv
from cts_pretrain import load_encoder_checkpoint
from tensorizer import TreeTensorizer, TensorizedTreeObservation, collate_tensorized_observations
from tree import SearchTree


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 8
    max_grad_norm: float = 1.0


@dataclass
class TreeObservationSnapshot:
    tree: SearchTree
    tensorized: Optional[TensorizedTreeObservation] = None

    @classmethod
    def from_tree(
        cls,
        tree: SearchTree,
        tensorizer: Optional[TreeTensorizer] = None,
    ) -> "TreeObservationSnapshot":
        frozen_tree = tree.clone()
        tensorized = None
        if tensorizer is not None:
            tensorized = tensorizer.tensorize_tree_observation(frozen_tree, validate=False)
        return cls(tree=frozen_tree, tensorized=tensorized)


RolloutObservation = Union[SearchTree, TreeObservationSnapshot]


@dataclass
class RolloutTransition:
    observation: RolloutObservation
    env_id: int
    action: int
    reward: float
    done: bool
    old_log_prob: float
    value: float
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class PPOTrainMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    explained_variance: float
    mean_episode_return: float
    mean_episode_length: float
    halt_rate: float
    num_episodes: int


@dataclass
class EvaluationMetrics:
    average_return: float
    average_expansions: float
    average_terminal_quality: float
    halt_step_histogram: Dict[int, int]
    quality_by_expansions: Dict[int, float]


@dataclass(frozen=True)
class ReinforceConfig:
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    batch_episodes: int = 128
    entropy_coef: float = 0.0
    use_return_normalization: bool = True


@dataclass
class ReinforceEpisodeTransition:
    observation: RolloutObservation
    action: int
    reward: float
    done: bool


@dataclass
class ReinforceTrainMetrics:
    policy_loss: float
    entropy: float
    mean_episode_return: float
    mean_episode_length: float
    halt_rate: float
    num_episodes: int


def _trainable_parameters(*modules: torch.nn.Module) -> Tuple[torch.nn.Parameter, ...]:
    parameters: List[torch.nn.Parameter] = []
    for module in modules:
        parameters.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    return tuple(parameters)


def _snapshot_observation(tree: SearchTree, tensorizer: Optional[TreeTensorizer] = None) -> TreeObservationSnapshot:
    return TreeObservationSnapshot.from_tree(tree, tensorizer=tensorizer)


def _observation_tree(observation: RolloutObservation) -> SearchTree:
    if isinstance(observation, TreeObservationSnapshot):
        return observation.tree
    return observation


def _observation_tensorized(
    observation: RolloutObservation,
    tensorizer: TreeTensorizer,
) -> TensorizedTreeObservation:
    if isinstance(observation, TreeObservationSnapshot) and observation.tensorized is not None:
        return observation.tensorized
    return tensorizer.tensorize_tree_observation(_observation_tree(observation), validate=False)


def _tensorized_observation_batch(
    observations: Sequence[RolloutObservation],
    tensorizer: TreeTensorizer,
):
    tensorized = [_observation_tensorized(observation, tensorizer) for observation in observations]
    return collate_tensorized_observations(tensorized)


class FrozenEncoderControllerTrainer:
    def __init__(
        self,
        model: PolicyValueTreeSearchModel,
        tensorizer: TreeTensorizer,
        envs: Sequence[ControllerOnlyEnv],
        config: PPOConfig,
        encoder_checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        trace_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        if not envs:
            raise ValueError("At least one environment is required.")

        self.model = model
        self.tensorizer = tensorizer
        self.envs = list(envs)
        self.config = config
        self.trace_callback = trace_callback

        if encoder_checkpoint_path is not None:
            load_encoder_checkpoint(encoder_checkpoint_path, self.model.encoder)
        if freeze_encoder:
            self.model.freeze_encoder()

        self._trainable_parameters = _trainable_parameters(
            self.model.encoder,
            self.model.halt_controller,
            self.model.value_head,
        )
        if not self._trainable_parameters:
            raise ValueError("FrozenEncoderControllerTrainer requires at least one trainable parameter.")
        self.optimizer = torch.optim.Adam(self._trainable_parameters, lr=config.learning_rate)
        self.current_trees = [env.reset() for env in self.envs]

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.trace_callback is not None:
            self.trace_callback(event, payload)

    def _inference_output(self, tree: SearchTree):
        self.model.eval()
        with torch.inference_mode():
            tree_batch = self.tensorizer.tensorize_tree(tree, validate=False)
            return self.model(tree_batch)

    def _inference_output_from_observation(self, observation: RolloutObservation):
        self.model.eval()
        with torch.inference_mode():
            tree_batch = _tensorized_observation_batch([observation], self.tensorizer)
            return self.model(tree_batch)

    def _batch_inference_outputs_from_observations(
        self,
        observations: Sequence[RolloutObservation],
    ):
        self.model.eval()
        with torch.inference_mode():
            tree_batch = _tensorized_observation_batch(observations, self.tensorizer)
            return self.model(tree_batch)

    def _predict_single(self, tree: SearchTree) -> Tuple[Bernoulli, float]:
        output = self._inference_output(tree)
        distribution = Bernoulli(logits=output.halt_logits.squeeze(0))
        value = float(output.state_value.squeeze(0).item())
        return distribution, value

    def _predict_observation(self, observation: RolloutObservation) -> Tuple[Bernoulli, float]:
        output = self._inference_output_from_observation(observation)
        distribution = Bernoulli(logits=output.halt_logits.squeeze(0))
        value = float(output.state_value.squeeze(0).item())
        return distribution, value

    def _bootstrap_value(self, tree: SearchTree) -> float:
        output = self._inference_output(tree)
        return float(output.state_value.squeeze(0).item())

    def _batch_bootstrap_values(self, trees: Sequence[SearchTree]) -> List[float]:
        if not trees:
            return []
        self.model.eval()
        with torch.inference_mode():
            tree_batch = self.tensorizer.tensorize_forest(trees, validate=False)
            output = self.model(tree_batch)
        return output.state_value.detach().cpu().tolist()

    def _collect_rollout(self) -> Tuple[List[RolloutTransition], List[Dict[str, Any]]]:
        rollout: List[RolloutTransition] = []
        completed_episodes: List[Dict[str, Any]] = []

        for step_idx in range(self.config.rollout_steps):
            observations = [
                _snapshot_observation(self.current_trees[env_id], tensorizer=self.tensorizer)
                for env_id in range(len(self.envs))
            ]
            output = self._batch_inference_outputs_from_observations(observations)
            distributions = Bernoulli(logits=output.halt_logits)
            sampled_actions = distributions.sample()
            log_probs = distributions.log_prob(sampled_actions).detach().cpu().tolist()
            actions = sampled_actions.to(dtype=torch.int64).detach().cpu().tolist()
            values = output.state_value.detach().cpu().tolist()

            for env_id, env in enumerate(self.envs):
                rollout_index = len(rollout)
                observation = observations[env_id]
                action = int(actions[env_id])
                log_prob = float(log_probs[env_id])
                value = float(values[env_id])

                step_result = env.step(action)
                self._trace(
                    "rollout_transition",
                    {
                        "phase": "pre_gae",
                        "step_index": rollout_index,
                        "rollout_step": step_idx,
                        "env_id": env_id,
                        "observation_root_features": dict(
                            observation.tree.get_node(observation.tree.root_id).scalar_features
                        ),
                        "action": action,
                        "reward": float(step_result.reward),
                        "done": bool(step_result.done),
                        "old_log_prob": log_prob,
                        "value": value,
                        "info": dict(step_result.info),
                    },
                )
                rollout.append(
                    RolloutTransition(
                        observation=observation,
                        env_id=env_id,
                        action=action,
                        reward=float(step_result.reward),
                        done=step_result.done,
                        old_log_prob=log_prob,
                        value=value,
                    )
                )

                if step_result.done:
                    completed_episodes.append(dict(step_result.info))
                    self._trace(
                        "completed_episode",
                        {
                            "step_index": rollout_index,
                            "rollout_step": step_idx,
                            "env_id": env_id,
                            "info": dict(step_result.info),
                        },
                    )
                    self.current_trees[env_id] = env.reset()
                else:
                    self.current_trees[env_id] = step_result.next_tree

        transitions_by_env: Dict[int, List[int]] = {}
        for index, transition in enumerate(rollout):
            transitions_by_env.setdefault(transition.env_id, []).append(index)

        bootstrap_env_ids = [
            env_id
            for env_id, indices in transitions_by_env.items()
            if not rollout[indices[-1]].done
        ]
        bootstrap_values = self._batch_bootstrap_values([self.current_trees[env_id] for env_id in bootstrap_env_ids])
        next_value_by_env = {
            env_id: bootstrap_value
            for env_id, bootstrap_value in zip(bootstrap_env_ids, bootstrap_values)
        }

        for env_id, indices in transitions_by_env.items():
            next_value = 0.0 if rollout[indices[-1]].done else next_value_by_env[env_id]
            gae = 0.0
            for index in reversed(indices):
                transition = rollout[index]
                nonterminal = 0.0 if transition.done else 1.0
                delta = transition.reward + self.config.gamma * next_value * nonterminal - transition.value
                gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
                transition.advantage = gae
                transition.return_value = transition.value + gae
                self._trace(
                    "rollout_transition",
                    {
                        "phase": "post_gae",
                        "rollout_index": index,
                        "env_id": transition.env_id,
                        "action": transition.action,
                        "reward": transition.reward,
                        "done": transition.done,
                        "value": transition.value,
                        "advantage": transition.advantage,
                        "return_value": transition.return_value,
                    },
                )
                next_value = transition.value

        return rollout, completed_episodes

    def _update_policy(self, rollout: Sequence[RolloutTransition]) -> Tuple[float, float, float, float, float, float]:
        if not rollout:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        advantages = torch.tensor(
            [transition.advantage for transition in rollout],
            dtype=torch.float32,
            device=self.model.encoder.device,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = torch.tensor(
            [transition.return_value for transition in rollout],
            dtype=torch.float32,
            device=self.model.encoder.device,
        )
        old_log_probs = torch.tensor(
            [transition.old_log_prob for transition in rollout],
            dtype=torch.float32,
            device=self.model.encoder.device,
        )
        actions = torch.tensor(
            [transition.action for transition in rollout],
            dtype=torch.float32,
            device=self.model.encoder.device,
        )

        indices = list(range(len(rollout)))
        device = self.model.encoder.device
        total_policy_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_value_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_entropy = torch.zeros((), dtype=torch.float32, device=device)
        total_approx_kl = torch.zeros((), dtype=torch.float32, device=device)
        total_clipfrac = torch.zeros((), dtype=torch.float32, device=device)
        update_steps = 0

        value_predictions = torch.tensor(
            [transition.value for transition in rollout],
            dtype=torch.float32,
            device=self.model.encoder.device,
        )
        returns_var = torch.var(returns, unbiased=False)
        if float(returns_var.item()) <= 1e-8:
            explained_variance = 0.0
        else:
            explained_variance = float(
                (1.0 - torch.var(returns - value_predictions, unbiased=False) / returns_var).item()
            )

        self.model.train()
        for epoch_index in range(self.config.ppo_epochs):
            random.shuffle(indices)
            for batch_start in range(0, len(indices), self.config.minibatch_size):
                batch_indices = indices[batch_start : batch_start + self.config.minibatch_size]
                observations = [_observation_tensorized(rollout[index].observation, self.tensorizer) for index in batch_indices]
                tree_batch = collate_tensorized_observations(observations)
                output = self.model(tree_batch)

                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                distribution = Bernoulli(logits=output.halt_logits)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(output.state_value, batch_returns)
                log_ratio = new_log_probs - batch_old_log_probs
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()

                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                if self.trace_callback is not None:
                    self._trace(
                        "ppo_minibatch",
                        {
                            "epoch_index": epoch_index,
                            "batch_start": batch_start,
                            "batch_indices": list(batch_indices),
                            "actions": batch_actions.detach().cpu().tolist(),
                            "old_log_probs": batch_old_log_probs.detach().cpu().tolist(),
                            "new_log_probs": new_log_probs.detach().cpu().tolist(),
                            "advantages": batch_advantages.detach().cpu().tolist(),
                            "returns": batch_returns.detach().cpu().tolist(),
                            "state_values": output.state_value.detach().cpu().tolist(),
                            "halt_logits": output.halt_logits.detach().cpu().tolist(),
                            "ratios": ratio.detach().cpu().tolist(),
                            "policy_loss": float(policy_loss.item()),
                            "value_loss": float(value_loss.item()),
                            "entropy": float(entropy.item()),
                            "approx_kl": float(approx_kl.item()),
                            "clipfrac": float(clipfrac.item()),
                        },
                    )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters, self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += entropy.detach()
                total_approx_kl += approx_kl.detach()
                total_clipfrac += clipfrac.detach()
                update_steps += 1

        if update_steps == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, explained_variance
        return (
            float((total_policy_loss / update_steps).item()),
            float((total_value_loss / update_steps).item()),
            float((total_entropy / update_steps).item()),
            float((total_approx_kl / update_steps).item()),
            float((total_clipfrac / update_steps).item()),
            explained_variance,
        )

    def train_update(self) -> PPOTrainMetrics:
        rollout, completed_episodes = self._collect_rollout()
        policy_loss, value_loss, entropy, approx_kl, clipfrac, explained_variance = self._update_policy(rollout)

        if completed_episodes:
            mean_episode_return = sum(item["episode_return"] for item in completed_episodes) / len(completed_episodes)
            mean_episode_length = sum(item["episode_length"] for item in completed_episodes) / len(completed_episodes)
            halt_rate = sum(1.0 for item in completed_episodes if item["halted"]) / len(completed_episodes)
        else:
            mean_episode_return = 0.0
            mean_episode_length = 0.0
            halt_rate = 0.0

        metrics = PPOTrainMetrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            approx_kl=approx_kl,
            clipfrac=clipfrac,
            explained_variance=explained_variance,
            mean_episode_return=mean_episode_return,
            mean_episode_length=mean_episode_length,
            halt_rate=halt_rate,
            num_episodes=len(completed_episodes),
        )
        self._trace(
            "train_update_metrics",
            {
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "approx_kl": metrics.approx_kl,
                "clipfrac": metrics.clipfrac,
                "explained_variance": metrics.explained_variance,
                "mean_episode_return": metrics.mean_episode_return,
                "mean_episode_length": metrics.mean_episode_length,
                "halt_rate": metrics.halt_rate,
                "num_episodes": metrics.num_episodes,
            },
        )
        return metrics

    def train(self, num_updates: int) -> List[PPOTrainMetrics]:
        return [self.train_update() for _ in range(num_updates)]


class ReinforceControllerTrainer:
    def __init__(
        self,
        model: PolicyValueTreeSearchModel,
        tensorizer: TreeTensorizer,
        envs: Sequence[ControllerOnlyEnv],
        config: ReinforceConfig,
        encoder_checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ) -> None:
        if not envs:
            raise ValueError("At least one environment is required.")

        self.model = model
        self.tensorizer = tensorizer
        self.envs = list(envs)
        self.config = config

        if encoder_checkpoint_path is not None:
            load_encoder_checkpoint(encoder_checkpoint_path, self.model.encoder)
        if freeze_encoder:
            self.model.freeze_encoder()

        self._trainable_parameters = _trainable_parameters(
            self.model.encoder,
            self.model.halt_controller,
        )
        if not self._trainable_parameters:
            raise ValueError("ReinforceControllerTrainer requires at least one trainable parameter.")
        self.optimizer = torch.optim.Adam(self._trainable_parameters, lr=config.learning_rate)
        self.current_trees = [env.reset() for env in self.envs]

    def _inference_output(self, tree: SearchTree):
        self.model.eval()
        with torch.inference_mode():
            tree_batch = self.tensorizer.tensorize_tree(tree, validate=False)
            return self.model(tree_batch)

    def _inference_output_from_observation(self, observation: RolloutObservation):
        self.model.eval()
        with torch.inference_mode():
            tree_batch = _tensorized_observation_batch([observation], self.tensorizer)
            return self.model(tree_batch)

    def _batch_inference_outputs_from_observations(
        self,
        observations: Sequence[RolloutObservation],
    ):
        self.model.eval()
        with torch.inference_mode():
            tree_batch = _tensorized_observation_batch(observations, self.tensorizer)
            return self.model(tree_batch)

    def _predict_single(self, tree: SearchTree) -> Bernoulli:
        output = self._inference_output(tree)
        return Bernoulli(logits=output.halt_logits.squeeze(0))

    def _predict_observation(self, observation: RolloutObservation) -> Bernoulli:
        output = self._inference_output_from_observation(observation)
        return Bernoulli(logits=output.halt_logits.squeeze(0))

    def _collect_batch(self) -> Tuple[List[List[ReinforceEpisodeTransition]], List[Dict[str, Any]]]:
        completed_episodes: List[List[ReinforceEpisodeTransition]] = []
        completed_infos: List[Dict[str, Any]] = []
        active_episodes: Dict[int, List[ReinforceEpisodeTransition]] = {env_id: [] for env_id in range(len(self.envs))}

        while len(completed_episodes) < self.config.batch_episodes:
            observations = [
                _snapshot_observation(self.current_trees[env_id], tensorizer=self.tensorizer)
                for env_id in range(len(self.envs))
            ]
            output = self._batch_inference_outputs_from_observations(observations)
            distributions = Bernoulli(logits=output.halt_logits)
            actions = distributions.sample().to(dtype=torch.int64).detach().cpu().tolist()

            for env_id, env in enumerate(self.envs):
                observation = observations[env_id]
                action = int(actions[env_id])

                step_result = env.step(action)
                active_episodes[env_id].append(
                    ReinforceEpisodeTransition(
                        observation=observation,
                        action=action,
                        reward=float(step_result.reward),
                        done=bool(step_result.done),
                    )
                )

                if step_result.done:
                    completed_episodes.append(active_episodes[env_id])
                    completed_infos.append(dict(step_result.info))
                    active_episodes[env_id] = []
                    self.current_trees[env_id] = env.reset()
                    if len(completed_episodes) >= self.config.batch_episodes:
                        break
                else:
                    self.current_trees[env_id] = step_result.next_tree

        return completed_episodes, completed_infos

    def _update_policy(
        self,
        episodes: Sequence[Sequence[ReinforceEpisodeTransition]],
    ) -> Tuple[float, float]:
        observations: List[RolloutObservation] = []
        actions: List[int] = []
        returns: List[float] = []

        for episode in episodes:
            episode_return = 0.0
            for transition in reversed(episode):
                episode_return += transition.reward
                observations.append(transition.observation)
                actions.append(transition.action)
                returns.append(episode_return)

        if not observations:
            return 0.0, 0.0

        returns_tensor = torch.tensor(
            list(reversed(returns)),
            dtype=torch.float32,
            device=self.model.encoder.device,
        )
        if self.config.use_return_normalization:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
        actions_tensor = torch.tensor(
            list(reversed(actions)),
            dtype=torch.float32,
            device=self.model.encoder.device,
        )
        tensorized_observations = [
            _observation_tensorized(observation, self.tensorizer) for observation in reversed(observations)
        ]
        tree_batch = collate_tensorized_observations(tensorized_observations)
        output = self.model(tree_batch)
        distribution = Bernoulli(logits=output.halt_logits)
        log_probs = distribution.log_prob(actions_tensor)
        entropy = distribution.entropy().mean()

        policy_loss = -(log_probs * returns_tensor).mean()
        loss = policy_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_parameters, self.config.max_grad_norm)
        self.optimizer.step()

        return float(policy_loss.item()), float(entropy.item())

    def train_update(self) -> ReinforceTrainMetrics:
        episodes, episode_infos = self._collect_batch()
        policy_loss, entropy = self._update_policy(episodes)

        mean_episode_return = sum(item["episode_return"] for item in episode_infos) / len(episode_infos)
        mean_episode_length = sum(item["episode_length"] for item in episode_infos) / len(episode_infos)
        halt_rate = sum(1.0 for item in episode_infos if item["halted"]) / len(episode_infos)

        return ReinforceTrainMetrics(
            policy_loss=policy_loss,
            entropy=entropy,
            mean_episode_return=mean_episode_return,
            mean_episode_length=mean_episode_length,
            halt_rate=halt_rate,
            num_episodes=len(episode_infos),
        )


def evaluate_controller(
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
    env_factory: Callable[[], ControllerOnlyEnv],
    num_episodes: int,
    trace_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> EvaluationMetrics:
    model.eval()

    episode_returns: List[float] = []
    episode_expansions: List[int] = []
    episode_qualities: List[float] = []
    halt_step_histogram: Dict[int, int] = {}
    quality_totals_by_expansion: Dict[int, float] = {}
    quality_counts_by_expansion: Dict[int, int] = {}

    with torch.inference_mode():
        env = env_factory()
        for _ in range(num_episodes):
            tree = env.reset()
            done = False

            while not done:
                tree_batch = tensorizer.tensorize_tree(tree, validate=False)
                output = model(tree_batch)
                halt_logit = float(output.halt_logits.squeeze(0).item())
                action = 1 if halt_logit >= 0.0 else 0
                step_result = env.step(action)
                if trace_callback is not None:
                    trace_callback(
                        "eval_step",
                        {
                            "episode_index": len(episode_returns),
                            "observation_root_features": dict(tree.get_node(tree.root_id).scalar_features),
                            "halt_logit": halt_logit,
                            "action": action,
                            "reward": float(step_result.reward),
                            "done": bool(step_result.done),
                            "info": dict(step_result.info),
                        },
                    )
                tree = step_result.next_tree.clone()
                done = step_result.done
                if done:
                    info = step_result.info
                    episode_returns.append(float(info["episode_return"]))
                    episode_expansions.append(int(info["expansions"]))
                    episode_qualities.append(float(info["terminal_quality"]))
                    halt_step = int(info["expansions"])
                    halt_step_histogram[halt_step] = halt_step_histogram.get(halt_step, 0) + 1
                    quality_totals_by_expansion[halt_step] = (
                        quality_totals_by_expansion.get(halt_step, 0.0) + float(info["terminal_quality"])
                    )
                    quality_counts_by_expansion[halt_step] = quality_counts_by_expansion.get(halt_step, 0) + 1

    quality_by_expansions = {}
    for expansions, total_quality in quality_totals_by_expansion.items():
        quality_by_expansions[expansions] = total_quality / quality_counts_by_expansion[expansions]

    return EvaluationMetrics(
        average_return=sum(episode_returns) / len(episode_returns),
        average_expansions=sum(episode_expansions) / len(episode_expansions),
        average_terminal_quality=sum(episode_qualities) / len(episode_qualities),
        halt_step_histogram=halt_step_histogram,
        quality_by_expansions=quality_by_expansions,
    )
