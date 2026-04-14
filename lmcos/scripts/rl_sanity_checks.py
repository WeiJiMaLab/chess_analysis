from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GNN import PolicyValueTreeSearchModel
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from supervised_branch import (
    FrozenEncoderControllerTrainer,
    PPOConfig,
    ReinforceConfig,
    ReinforceControllerTrainer,
    ReinforceEpisodeTransition,
    RolloutTransition,
    ToyHaltEnv,
    load_encoder_checkpoint,
)
from tensorizer import TreeTensorizer
from tree import ExpansionChild, SearchTree


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


def _make_snapshot(best_value: float) -> SearchTree:
    tree = SearchTree()
    root_id = tree.create_root("toy-root", {"value": 0.0, "prior": 1.0})
    tree.add_children(
        root_id,
        [
            ExpansionChild("best", f"best-{best_value}", {"value": best_value, "prior": 0.7}),
            ExpansionChild("other", f"other-{best_value}", {"value": 0.1, "prior": 0.3}),
        ],
    )
    return tree


def _make_halt_favored_env(continue_cost: float) -> ToyHaltEnv:
    return ToyHaltEnv(
        [
            _make_snapshot(0.8),
            _make_snapshot(0.8),
            _make_snapshot(0.8),
        ],
        continue_cost=continue_cost,
    )


def _make_continue_favored_env(continue_cost: float) -> ToyHaltEnv:
    return ToyHaltEnv(
        [
            _make_snapshot(0.0),
            _make_snapshot(0.0),
            _make_snapshot(1.0),
        ],
        continue_cost=continue_cost,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(
    device: str,
    k: int,
    node_embed_hidden: int,
    d_embed: int,
    d_message: int,
    n_heads: int,
    d_att: int,
    controller_hidden: int,
    value_hidden: int,
) -> PolicyValueTreeSearchModel:
    schema = _feature_schema()
    return PolicyValueTreeSearchModel(
        k=k,
        node_feat=len(schema.feature_names),
        device=device,
        node_embed_hidden=node_embed_hidden,
        d_embed=d_embed,
        d_message=d_message,
        n_heads=n_heads,
        d_att=d_att,
        controller_hidden=controller_hidden,
        value_hidden=value_hidden,
    )


def _tree_tensorizer(device: str) -> TreeTensorizer:
    return TreeTensorizer(_feature_schema(), device=device)


def _load_optional_encoder(model: PolicyValueTreeSearchModel, checkpoint_path: str | None) -> None:
    if checkpoint_path is not None:
        load_encoder_checkpoint(checkpoint_path, model.encoder)


def _halt_logit(model: PolicyValueTreeSearchModel, tensorizer: TreeTensorizer, tree: SearchTree) -> float:
    model.eval()
    with torch.no_grad():
        output = model(tensorizer.tensorize_tree(tree))
    return float(output.halt_logits.squeeze(0).item())


def _action_log_prob(
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
    tree: SearchTree,
    action: int,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(tensorizer.tensorize_tree(tree))
        distribution = Bernoulli(logits=output.halt_logits.squeeze(0))
        action_tensor = torch.tensor(float(action), device=output.halt_logits.device)
        return float(distribution.log_prob(action_tensor).item())


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _clone_module_params(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _module_delta_norm(module: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for name, parameter in module.named_parameters():
        total += float((parameter.detach() - before[name]).pow(2).sum().item())
    return total ** 0.5


def _ppo_sign_case(
    device: str,
    checkpoint_path: str | None,
    positive_for_halt: bool,
    learning_rate: float,
    model_dims: Mapping[str, int],
) -> Dict[str, Any]:
    tensorizer = _tree_tensorizer(device)
    model = _build_model(
        device=device,
        k=model_dims["k"],
        node_embed_hidden=model_dims["node_embed_hidden"],
        d_embed=model_dims["d_embed"],
        d_message=model_dims["d_message"],
        n_heads=model_dims["n_heads"],
        d_att=model_dims["d_att"],
        controller_hidden=model_dims["controller_hidden"],
        value_hidden=model_dims["value_hidden"],
    )
    _load_optional_encoder(model, checkpoint_path)
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[_make_halt_favored_env(continue_cost=0.05)],
        config=PPOConfig(
            rollout_steps=1,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.0,
            ppo_epochs=1,
            minibatch_size=2,
            max_grad_norm=10.0,
        ),
        encoder_checkpoint_path=None,
        freeze_encoder=False,
    )

    observation = _make_halt_favored_env(continue_cost=0.05).reset()
    action_for_selected = 1
    opposing_action = 0
    selected_advantage = 1.0 if positive_for_halt else -1.0
    opposing_advantage = -selected_advantage

    before_halt_logit = _halt_logit(model, tensorizer, observation)
    before_selected_log_prob = _action_log_prob(model, tensorizer, observation, action_for_selected)

    positive_transition = RolloutTransition(
        observation=copy.deepcopy(observation),
        env_id=0,
        action=action_for_selected,
        reward=0.0,
        done=True,
        old_log_prob=_action_log_prob(model, tensorizer, observation, action_for_selected),
        value=0.0,
        advantage=selected_advantage,
        return_value=0.0,
    )
    opposing_transition = RolloutTransition(
        observation=copy.deepcopy(observation),
        env_id=0,
        action=opposing_action,
        reward=0.0,
        done=True,
        old_log_prob=_action_log_prob(model, tensorizer, observation, opposing_action),
        value=0.0,
        advantage=opposing_advantage,
        return_value=0.0,
    )

    trainer._update_policy([positive_transition, opposing_transition])

    after_halt_logit = _halt_logit(model, tensorizer, observation)
    after_selected_log_prob = _action_log_prob(model, tensorizer, observation, action_for_selected)
    delta = after_selected_log_prob - before_selected_log_prob

    return {
        "expected_direction": "increase" if positive_for_halt else "decrease",
        "passed": delta > 0.0 if positive_for_halt else delta < 0.0,
        "before_halt_logit": before_halt_logit,
        "after_halt_logit": after_halt_logit,
        "before_selected_log_prob": before_selected_log_prob,
        "after_selected_log_prob": after_selected_log_prob,
        "selected_log_prob_delta": delta,
    }


def _reinforce_sign_case(
    device: str,
    checkpoint_path: str | None,
    positive_for_halt: bool,
    learning_rate: float,
    model_dims: Mapping[str, int],
) -> Dict[str, Any]:
    tensorizer = _tree_tensorizer(device)
    model = _build_model(
        device=device,
        k=model_dims["k"],
        node_embed_hidden=model_dims["node_embed_hidden"],
        d_embed=model_dims["d_embed"],
        d_message=model_dims["d_message"],
        n_heads=model_dims["n_heads"],
        d_att=model_dims["d_att"],
        controller_hidden=model_dims["controller_hidden"],
        value_hidden=model_dims["value_hidden"],
    )
    _load_optional_encoder(model, checkpoint_path)
    trainer = ReinforceControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[_make_halt_favored_env(continue_cost=0.05)],
        config=ReinforceConfig(
            learning_rate=learning_rate,
            max_grad_norm=10.0,
            batch_episodes=1,
            entropy_coef=0.0,
            use_return_normalization=False,
        ),
        encoder_checkpoint_path=None,
        freeze_encoder=False,
    )

    observation = _make_halt_favored_env(continue_cost=0.05).reset()
    before_halt_logit = _halt_logit(model, tensorizer, observation)
    before_halt_log_prob = _action_log_prob(model, tensorizer, observation, 1)

    reward = 1.0 if positive_for_halt else -1.0
    episode = [[ReinforceEpisodeTransition(observation=copy.deepcopy(observation), action=1, reward=reward, done=True)]]
    trainer._update_policy(episode)

    after_halt_logit = _halt_logit(model, tensorizer, observation)
    after_halt_log_prob = _action_log_prob(model, tensorizer, observation, 1)
    delta = after_halt_log_prob - before_halt_log_prob

    return {
        "expected_direction": "increase" if positive_for_halt else "decrease",
        "passed": delta > 0.0 if positive_for_halt else delta < 0.0,
        "before_halt_logit": before_halt_logit,
        "after_halt_logit": after_halt_logit,
        "before_selected_log_prob": before_halt_log_prob,
        "after_selected_log_prob": after_halt_log_prob,
        "selected_log_prob_delta": delta,
    }


def _inspect_ppo_gradients(
    device: str,
    checkpoint_path: str | None,
    learning_rate: float,
    continue_cost: float,
    rollout_steps: int,
    num_envs: int,
    model_dims: Mapping[str, int],
) -> Dict[str, Any]:
    tensorizer = _tree_tensorizer(device)
    model = _build_model(
        device=device,
        k=model_dims["k"],
        node_embed_hidden=model_dims["node_embed_hidden"],
        d_embed=model_dims["d_embed"],
        d_message=model_dims["d_message"],
        n_heads=model_dims["n_heads"],
        d_att=model_dims["d_att"],
        controller_hidden=model_dims["controller_hidden"],
        value_hidden=model_dims["value_hidden"],
    )
    _load_optional_encoder(model, checkpoint_path)
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[_make_halt_favored_env(continue_cost=continue_cost) for _ in range(num_envs)],
        config=PPOConfig(
            rollout_steps=rollout_steps,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            ppo_epochs=1,
            minibatch_size=min(8, rollout_steps * num_envs),
            max_grad_norm=1.0,
        ),
        encoder_checkpoint_path=None,
        freeze_encoder=False,
    )
    rollout, completed_episodes = trainer._collect_rollout()
    advantages = torch.tensor(
        [transition.advantage for transition in rollout],
        dtype=torch.float32,
        device=model.encoder.device,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    returns = torch.tensor(
        [transition.return_value for transition in rollout],
        dtype=torch.float32,
        device=model.encoder.device,
    )
    old_log_probs = torch.tensor(
        [transition.old_log_prob for transition in rollout],
        dtype=torch.float32,
        device=model.encoder.device,
    )
    actions = torch.tensor(
        [transition.action for transition in rollout],
        dtype=torch.float32,
        device=model.encoder.device,
    )

    batch_indices = list(range(min(len(rollout), trainer.config.minibatch_size)))
    trees = [rollout[index].observation for index in batch_indices]
    tree_batch = tensorizer.tensorize_forest(trees)
    output = model(tree_batch)
    distribution = Bernoulli(logits=output.halt_logits)
    batch_actions = actions[batch_indices]
    batch_old_log_probs = old_log_probs[batch_indices]
    batch_advantages = advantages[batch_indices]
    batch_returns = returns[batch_indices]
    new_log_probs = distribution.log_prob(batch_actions)
    entropy = distribution.entropy().mean()
    ratio = torch.exp(new_log_probs - batch_old_log_probs)
    unclipped = ratio * batch_advantages
    clipped = torch.clamp(ratio, 1.0 - trainer.config.clip_epsilon, 1.0 + trainer.config.clip_epsilon) * batch_advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(output.state_value, batch_returns)
    loss = policy_loss + trainer.config.value_loss_coef * value_loss

    before_encoder = _clone_module_params(model.encoder)
    before_halt = _clone_module_params(model.halt_controller)
    before_value = _clone_module_params(model.value_head)

    trainer.optimizer.zero_grad()
    loss.backward()
    grad_stats = {
        "encoder": _module_grad_norm(model.encoder),
        "halt_controller": _module_grad_norm(model.halt_controller),
        "value_head": _module_grad_norm(model.value_head),
    }
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        trainer.config.max_grad_norm,
    )
    trainer.optimizer.step()

    delta_stats = {
        "encoder": _module_delta_norm(model.encoder, before_encoder),
        "halt_controller": _module_delta_norm(model.halt_controller, before_halt),
        "value_head": _module_delta_norm(model.value_head, before_value),
    }

    return {
        "rollout_size": len(rollout),
        "num_completed_episodes": len(completed_episodes),
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "mean_advantage": float(batch_advantages.mean().item()),
        "mean_return": float(batch_returns.mean().item()),
        "grad_norms": grad_stats,
        "parameter_delta_norms": delta_stats,
    }


def _inspect_reinforce_gradients(
    device: str,
    checkpoint_path: str | None,
    learning_rate: float,
    continue_cost: float,
    batch_episodes: int,
    num_envs: int,
    model_dims: Mapping[str, int],
) -> Dict[str, Any]:
    tensorizer = _tree_tensorizer(device)
    model = _build_model(
        device=device,
        k=model_dims["k"],
        node_embed_hidden=model_dims["node_embed_hidden"],
        d_embed=model_dims["d_embed"],
        d_message=model_dims["d_message"],
        n_heads=model_dims["n_heads"],
        d_att=model_dims["d_att"],
        controller_hidden=model_dims["controller_hidden"],
        value_hidden=model_dims["value_hidden"],
    )
    _load_optional_encoder(model, checkpoint_path)
    trainer = ReinforceControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[_make_halt_favored_env(continue_cost=continue_cost) for _ in range(num_envs)],
        config=ReinforceConfig(
            learning_rate=learning_rate,
            max_grad_norm=1.0,
            batch_episodes=batch_episodes,
            entropy_coef=0.0,
            use_return_normalization=True,
        ),
        encoder_checkpoint_path=None,
        freeze_encoder=False,
    )

    episodes, episode_infos = trainer._collect_batch()
    observations: List[SearchTree] = []
    actions: List[int] = []
    returns: List[float] = []
    for episode in episodes:
        episode_return = 0.0
        for transition in reversed(episode):
            episode_return += transition.reward
            observations.append(transition.observation)
            actions.append(transition.action)
            returns.append(episode_return)

    returns_tensor = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=model.encoder.device)
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
    actions_tensor = torch.tensor(list(reversed(actions)), dtype=torch.float32, device=model.encoder.device)
    trees = list(reversed(observations))
    tree_batch = tensorizer.tensorize_forest(trees)
    output = model(tree_batch)
    distribution = Bernoulli(logits=output.halt_logits)
    log_probs = distribution.log_prob(actions_tensor)
    entropy = distribution.entropy().mean()
    policy_loss = -(log_probs * returns_tensor).mean()
    loss = policy_loss

    before_encoder = _clone_module_params(model.encoder)
    before_halt = _clone_module_params(model.halt_controller)
    before_value = _clone_module_params(model.value_head)

    trainer.optimizer.zero_grad()
    loss.backward()
    grad_stats = {
        "encoder": _module_grad_norm(model.encoder),
        "halt_controller": _module_grad_norm(model.halt_controller),
        "value_head": _module_grad_norm(model.value_head),
    }
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        trainer.config.max_grad_norm,
    )
    trainer.optimizer.step()

    delta_stats = {
        "encoder": _module_delta_norm(model.encoder, before_encoder),
        "halt_controller": _module_delta_norm(model.halt_controller, before_halt),
        "value_head": _module_delta_norm(model.value_head, before_value),
    }

    return {
        "num_episodes": len(episode_infos),
        "mean_episode_return": sum(item["episode_return"] for item in episode_infos) / len(episode_infos),
        "mean_episode_length": sum(item["episode_length"] for item in episode_infos) / len(episode_infos),
        "policy_loss": float(policy_loss.item()),
        "entropy": float(entropy.item()),
        "grad_norms": grad_stats,
        "parameter_delta_norms": delta_stats,
    }


def _collect_drift(
    trainer: FrozenEncoderControllerTrainer | ReinforceControllerTrainer,
    model: PolicyValueTreeSearchModel,
    tensorizer: TreeTensorizer,
    fixed_tree: SearchTree,
    num_updates: int,
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for update_index in range(1, num_updates + 1):
        before = _halt_logit(model, tensorizer, fixed_tree)
        metrics = trainer.train_update()
        after = _halt_logit(model, tensorizer, fixed_tree)
        record = {
            "update": update_index,
            "halt_logit_before": before,
            "halt_logit_after": after,
            "halt_logit_delta": after - before,
        }
        if hasattr(metrics, "__dict__"):
            record["train_metrics"] = asdict(metrics)
        history.append(record)
    return history


def _ppo_drift(
    device: str,
    checkpoint_path: str | None,
    learning_rate: float,
    continue_cost: float,
    rollout_steps: int,
    num_envs: int,
    drift_updates: int,
    env_kind: str,
    model_dims: Mapping[str, int],
) -> List[Dict[str, Any]]:
    tensorizer = _tree_tensorizer(device)
    model = _build_model(
        device=device,
        k=model_dims["k"],
        node_embed_hidden=model_dims["node_embed_hidden"],
        d_embed=model_dims["d_embed"],
        d_message=model_dims["d_message"],
        n_heads=model_dims["n_heads"],
        d_att=model_dims["d_att"],
        controller_hidden=model_dims["controller_hidden"],
        value_hidden=model_dims["value_hidden"],
    )
    _load_optional_encoder(model, checkpoint_path)
    env_factory = _make_halt_favored_env if env_kind == "halt_favored" else _make_continue_favored_env
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[env_factory(continue_cost=continue_cost) for _ in range(num_envs)],
        config=PPOConfig(
            rollout_steps=rollout_steps,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            ppo_epochs=1,
            minibatch_size=min(8, rollout_steps * num_envs),
            max_grad_norm=1.0,
        ),
        encoder_checkpoint_path=None,
        freeze_encoder=False,
    )
    fixed_tree = env_factory(continue_cost=continue_cost).reset()
    return _collect_drift(trainer, model, tensorizer, fixed_tree, drift_updates)


def _reinforce_drift(
    device: str,
    checkpoint_path: str | None,
    learning_rate: float,
    continue_cost: float,
    batch_episodes: int,
    num_envs: int,
    drift_updates: int,
    env_kind: str,
    model_dims: Mapping[str, int],
) -> List[Dict[str, Any]]:
    tensorizer = _tree_tensorizer(device)
    model = _build_model(
        device=device,
        k=model_dims["k"],
        node_embed_hidden=model_dims["node_embed_hidden"],
        d_embed=model_dims["d_embed"],
        d_message=model_dims["d_message"],
        n_heads=model_dims["n_heads"],
        d_att=model_dims["d_att"],
        controller_hidden=model_dims["controller_hidden"],
        value_hidden=model_dims["value_hidden"],
    )
    _load_optional_encoder(model, checkpoint_path)
    env_factory = _make_halt_favored_env if env_kind == "halt_favored" else _make_continue_favored_env
    trainer = ReinforceControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[env_factory(continue_cost=continue_cost) for _ in range(num_envs)],
        config=ReinforceConfig(
            learning_rate=learning_rate,
            max_grad_norm=1.0,
            batch_episodes=batch_episodes,
            entropy_coef=0.0,
            use_return_normalization=True,
        ),
        encoder_checkpoint_path=None,
        freeze_encoder=False,
    )
    fixed_tree = env_factory(continue_cost=continue_cost).reset()
    return _collect_drift(trainer, model, tensorizer, fixed_tree, drift_updates)


def _run_ppo(args: argparse.Namespace) -> Dict[str, Any]:
    model_dims = {
        "k": args.k,
        "node_embed_hidden": args.node_embed_hidden,
        "d_embed": args.d_embed,
        "d_message": args.d_message,
        "n_heads": args.n_heads,
        "d_att": args.d_att,
        "controller_hidden": args.controller_hidden,
        "value_hidden": args.value_hidden,
    }
    return {
        "manual_sign": {
            "positive_for_halt": _ppo_sign_case(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                positive_for_halt=True,
                learning_rate=args.learning_rate,
                model_dims=model_dims,
            ),
            "negative_for_halt": _ppo_sign_case(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                positive_for_halt=False,
                learning_rate=args.learning_rate,
                model_dims=model_dims,
            ),
        },
        "gradient_and_parameter_norms": _inspect_ppo_gradients(
            device=args.device,
            checkpoint_path=args.encoder_checkpoint,
            learning_rate=args.learning_rate,
            continue_cost=args.continue_cost,
            rollout_steps=args.ppo_rollout_steps,
            num_envs=args.num_envs,
            model_dims=model_dims,
        ),
        "fixed_state_logit_drift": {
            "halt_favored": _ppo_drift(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                learning_rate=args.learning_rate,
                continue_cost=args.continue_cost,
                rollout_steps=args.ppo_rollout_steps,
                num_envs=args.num_envs,
                drift_updates=args.drift_updates,
                env_kind="halt_favored",
                model_dims=model_dims,
            ),
            "continue_favored": _ppo_drift(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                learning_rate=args.learning_rate,
                continue_cost=args.continue_cost,
                rollout_steps=args.ppo_rollout_steps,
                num_envs=args.num_envs,
                drift_updates=args.drift_updates,
                env_kind="continue_favored",
                model_dims=model_dims,
            ),
        },
    }


def _run_reinforce(args: argparse.Namespace) -> Dict[str, Any]:
    model_dims = {
        "k": args.k,
        "node_embed_hidden": args.node_embed_hidden,
        "d_embed": args.d_embed,
        "d_message": args.d_message,
        "n_heads": args.n_heads,
        "d_att": args.d_att,
        "controller_hidden": args.controller_hidden,
        "value_hidden": args.value_hidden,
    }
    return {
        "manual_sign": {
            "positive_for_halt": _reinforce_sign_case(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                positive_for_halt=True,
                learning_rate=args.learning_rate,
                model_dims=model_dims,
            ),
            "negative_for_halt": _reinforce_sign_case(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                positive_for_halt=False,
                learning_rate=args.learning_rate,
                model_dims=model_dims,
            ),
        },
        "gradient_and_parameter_norms": _inspect_reinforce_gradients(
            device=args.device,
            checkpoint_path=args.encoder_checkpoint,
            learning_rate=args.learning_rate,
            continue_cost=args.continue_cost,
            batch_episodes=args.reinforce_batch_episodes,
            num_envs=args.num_envs,
            model_dims=model_dims,
        ),
        "fixed_state_logit_drift": {
            "halt_favored": _reinforce_drift(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                learning_rate=args.learning_rate,
                continue_cost=args.continue_cost,
                batch_episodes=args.reinforce_batch_episodes,
                num_envs=args.num_envs,
                drift_updates=args.drift_updates,
                env_kind="halt_favored",
                model_dims=model_dims,
            ),
            "continue_favored": _reinforce_drift(
                device=args.device,
                checkpoint_path=args.encoder_checkpoint,
                learning_rate=args.learning_rate,
                continue_cost=args.continue_cost,
                batch_episodes=args.reinforce_batch_episodes,
                num_envs=args.num_envs,
                drift_updates=args.drift_updates,
                env_kind="continue_favored",
                model_dims=model_dims,
            ),
        },
    }


def _print_summary(report: Dict[str, Any]) -> None:
    for algorithm_name, section in report["results"].items():
        print(f"[{algorithm_name}] stage=manual_sign")
        for case_name, case in section["manual_sign"].items():
            print(
                f"{case_name} passed={case['passed']} expected={case['expected_direction']} "
                f"log_prob_delta={case['selected_log_prob_delta']:.6f} "
                f"halt_logit_before={case['before_halt_logit']:.6f} "
                f"halt_logit_after={case['after_halt_logit']:.6f}"
            )

        norms = section["gradient_and_parameter_norms"]
        print(f"[{algorithm_name}] stage=gradient_and_parameter_norms")
        print(
            f"encoder_grad_norm={norms['grad_norms']['encoder']:.6f} "
            f"halt_controller_grad_norm={norms['grad_norms']['halt_controller']:.6f} "
            f"value_head_grad_norm={norms['grad_norms']['value_head']:.6f}"
        )
        print(
            f"encoder_delta_norm={norms['parameter_delta_norms']['encoder']:.6f} "
            f"halt_controller_delta_norm={norms['parameter_delta_norms']['halt_controller']:.6f} "
            f"value_head_delta_norm={norms['parameter_delta_norms']['value_head']:.6f}"
        )

        print(f"[{algorithm_name}] stage=fixed_state_logit_drift")
        for env_kind, history in section["fixed_state_logit_drift"].items():
            final_record = history[-1]
            total_delta = sum(item["halt_logit_delta"] for item in history)
            print(
                f"{env_kind} updates={len(history)} "
                f"final_halt_logit={final_record['halt_logit_after']:.6f} "
                f"total_logit_delta={total_delta:.6f}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RL sanity diagnostics for PPO and REINFORCE.")
    parser.add_argument("--algorithm", choices=["ppo", "reinforce", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--output-json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-cost", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    parser.add_argument("--controller-hidden", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--drift-updates", type=int, default=5)
    parser.add_argument("--ppo-rollout-steps", type=int, default=8)
    parser.add_argument("--reinforce-batch-episodes", type=int, default=32)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _set_seed(args.seed)

    report: Dict[str, Any] = {
        "config": {
            "algorithm": args.algorithm,
            "device": args.device,
            "encoder_checkpoint": args.encoder_checkpoint,
            "seed": args.seed,
            "continue_cost": args.continue_cost,
            "learning_rate": args.learning_rate,
            "num_envs": args.num_envs,
            "drift_updates": args.drift_updates,
            "ppo_rollout_steps": args.ppo_rollout_steps,
            "reinforce_batch_episodes": args.reinforce_batch_episodes,
        },
        "results": {},
    }

    if args.algorithm in {"ppo", "both"}:
        report["results"]["ppo"] = _run_ppo(args)
    if args.algorithm in {"reinforce", "both"}:
        report["results"]["reinforce"] = _run_reinforce(args)

    _print_summary(report)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[sanity] wrote_json={output_path}")


if __name__ == "__main__":
    main()
