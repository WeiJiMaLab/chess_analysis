from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class BudgetBucket:
    name: str
    min_time: int
    max_time: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BudgetBucket.name must be non-empty.")
        if self.min_time <= 0:
            raise ValueError("BudgetBucket.min_time must be positive.")
        if self.max_time < self.min_time:
            raise ValueError("BudgetBucket.max_time must be >= min_time.")


DEFAULT_BUDGET_BUCKETS: tuple[BudgetBucket, ...] = (
    BudgetBucket("scramble", 1, 3),
    BudgetBucket("medium-small", 4, 10),
    BudgetBucket("medium-large", 11, 25),
    BudgetBucket("large", 26, 60),
    BudgetBucket("very-large", 61, 120),
)


@dataclass(frozen=True)
class BudgetedOracleConfig:
    maintenance_scale: float = 0.0025
    maintenance_ref_nodes: float = 30.0
    maintenance_exponent: float = 1.1
    time_lambda: float = 18.537
    time_p: float = 2.8
    time_tau: float = 2.5
    time_delta: int = 1
    timeout_value: float = -1.0
    budget_buckets: tuple[BudgetBucket, ...] = DEFAULT_BUDGET_BUCKETS
    samples_per_bucket: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.maintenance_scale < 0.0:
            raise ValueError("maintenance_scale must be non-negative.")
        if self.maintenance_ref_nodes <= 0.0:
            raise ValueError("maintenance_ref_nodes must be positive.")
        if self.maintenance_exponent <= 0.0:
            raise ValueError("maintenance_exponent must be positive.")
        if self.time_lambda < 0.0:
            raise ValueError("time_lambda must be non-negative.")
        if self.time_p <= 1.0:
            raise ValueError("time_p must be > 1.")
        if self.time_tau <= 0.0:
            raise ValueError("time_tau must be positive.")
        if self.time_delta <= 0:
            raise ValueError("time_delta must be positive.")
        if self.samples_per_bucket <= 0:
            raise ValueError("samples_per_bucket must be positive.")
        if not self.budget_buckets:
            raise ValueError("At least one budget bucket is required.")


@dataclass(frozen=True)
class SampledBudget:
    bucket_index: int
    bucket_name: str
    starting_budget: int


@dataclass(frozen=True)
class BudgetedOraclePolicy:
    halt_rewards: List[float]
    tree_sizes: List[int]
    time_budgets: List[int]
    continue_values: List[float]
    values: List[float]
    actions: List[int]
    stop_steps: List[int]
    target_advantages: List[float]
    starting_budget: int

    @property
    def optimal_stop_step(self) -> int:
        return self.stop_steps[0]

    @property
    def oracle_value(self) -> float:
        return self.values[0]


def maintenance_cost(num_nodes: int, config: BudgetedOracleConfig) -> float:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    return float(config.maintenance_scale * (float(num_nodes) / config.maintenance_ref_nodes) ** config.maintenance_exponent)


def time_cost(remaining_budget: int, config: BudgetedOracleConfig) -> float:
    if remaining_budget <= 0:
        raise ValueError("remaining_budget must be positive.")
    left = (remaining_budget - config.time_delta + config.time_tau) ** (-(config.time_p - 1.0))
    right = (remaining_budget + config.time_tau) ** (-(config.time_p - 1.0))
    return float(config.time_lambda * (left - right))


def continue_cost(num_nodes: int, remaining_budget: int, config: BudgetedOracleConfig) -> float:
    return maintenance_cost(num_nodes, config) + time_cost(remaining_budget, config)


def deterministic_starting_budgets(
    source_path: str,
    config: BudgetedOracleConfig,
) -> List[SampledBudget]:
    sampled: List[SampledBudget] = []
    for bucket_index, bucket in enumerate(config.budget_buckets):
        for draw_index in range(config.samples_per_bucket):
            digest = hashlib.sha256(
                f"{source_path}|{bucket_index}|{draw_index}|{config.seed}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
            rng = random.Random(seed)
            sampled.append(
                SampledBudget(
                    bucket_index=bucket_index,
                    bucket_name=bucket.name,
                    starting_budget=rng.randint(bucket.min_time, bucket.max_time),
                )
            )
    return sampled


def compute_budgeted_oracle(
    halt_rewards: Sequence[float],
    tree_sizes: Sequence[int],
    starting_budget: int,
    config: BudgetedOracleConfig,
) -> BudgetedOraclePolicy:
    if not halt_rewards:
        raise ValueError("halt_rewards must be non-empty.")
    if len(halt_rewards) != len(tree_sizes):
        raise ValueError("halt_rewards and tree_sizes must have the same length.")
    if starting_budget <= 0:
        raise ValueError("starting_budget must be positive.")

    max_steps = min(len(halt_rewards), starting_budget)
    truncated_rewards = [float(value) for value in halt_rewards[:max_steps]]
    truncated_sizes = [int(value) for value in tree_sizes[:max_steps]]
    time_budgets = [starting_budget - index for index in range(max_steps)]

    values = [0.0] * max_steps
    actions = [1] * max_steps  # 1 = halt, 0 = continue
    stop_steps = [0] * max_steps
    continue_values = [0.0] * max_steps
    target_advantages = [0.0] * max_steps

    for idx in range(max_steps - 1, -1, -1):
        halt_value = truncated_rewards[idx]
        remaining_budget = time_budgets[idx]

        if remaining_budget <= config.time_delta:
            next_value = float(config.timeout_value)
        elif idx + 1 < max_steps:
            next_value = values[idx + 1]
        else:
            next_value = truncated_rewards[idx]

        step_continue_value = -continue_cost(truncated_sizes[idx], remaining_budget, config) + next_value
        continue_values[idx] = step_continue_value
        target_advantages[idx] = step_continue_value - halt_value

        if halt_value >= step_continue_value:
            values[idx] = halt_value
            actions[idx] = 1
            stop_steps[idx] = idx
        else:
            values[idx] = step_continue_value
            actions[idx] = 0
            stop_steps[idx] = stop_steps[idx + 1] if idx + 1 < max_steps else idx

    return BudgetedOraclePolicy(
        halt_rewards=truncated_rewards,
        tree_sizes=truncated_sizes,
        time_budgets=time_budgets,
        continue_values=continue_values,
        values=values,
        actions=actions,
        stop_steps=stop_steps,
        target_advantages=target_advantages,
        starting_budget=starting_budget,
    )


def has_strong_budgeted_margins(
    halt_rewards: Sequence[float],
    tree_sizes: Sequence[int],
    starting_budget: int,
    config: BudgetedOracleConfig,
    min_decision_margin: float,
) -> bool:
    if min_decision_margin <= 0.0:
        return True
    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, starting_budget, config)
    for idx, halt_value in enumerate(policy.halt_rewards):
        margin = (
            float(halt_value) - policy.continue_values[idx]
            if policy.actions[idx] == 1
            else policy.continue_values[idx] - float(halt_value)
        )
        if margin < min_decision_margin:
            return False
        if policy.actions[idx] == 1:
            break
    return True


def return_for_stop_step(
    halt_rewards: Sequence[float],
    tree_sizes: Sequence[int],
    time_budgets: Sequence[int],
    stop_step: int,
    config: BudgetedOracleConfig,
) -> float:
    if len(halt_rewards) != len(tree_sizes) or len(halt_rewards) != len(time_budgets):
        raise ValueError("halt_rewards, tree_sizes, and time_budgets must have the same length.")
    if stop_step < 0 or stop_step >= len(halt_rewards):
        raise IndexError(stop_step)
    total = float(halt_rewards[stop_step])
    for idx in range(stop_step):
        total -= continue_cost(int(tree_sizes[idx]), int(time_budgets[idx]), config)
    return total


def budgeted_oracle_metadata(config: BudgetedOracleConfig) -> Dict[str, Any]:
    return {
        "oracle_type": "budgeted_controller_v1",
        "maintenance_scale": config.maintenance_scale,
        "maintenance_ref_nodes": config.maintenance_ref_nodes,
        "maintenance_exponent": config.maintenance_exponent,
        "time_lambda": config.time_lambda,
        "time_p": config.time_p,
        "time_tau": config.time_tau,
        "time_delta": config.time_delta,
        "timeout_value": config.timeout_value,
        "samples_per_bucket": config.samples_per_bucket,
        "budget_seed": config.seed,
        "budget_buckets": [
            {"name": bucket.name, "min_time": bucket.min_time, "max_time": bucket.max_time}
            for bucket in config.budget_buckets
        ],
    }


def budgeted_oracle_config_from_metadata(metadata: Dict[str, Any]) -> BudgetedOracleConfig | None:
    if metadata.get("oracle_type") != "budgeted_controller_v1":
        return None
    buckets = tuple(
        BudgetBucket(
            name=str(bucket["name"]),
            min_time=int(bucket["min_time"]),
            max_time=int(bucket["max_time"]),
        )
        for bucket in metadata["budget_buckets"]
    )
    return BudgetedOracleConfig(
        maintenance_scale=float(metadata["maintenance_scale"]),
        maintenance_ref_nodes=float(metadata["maintenance_ref_nodes"]),
        maintenance_exponent=float(metadata["maintenance_exponent"]),
        time_lambda=float(metadata["time_lambda"]),
        time_p=float(metadata["time_p"]),
        time_tau=float(metadata["time_tau"]),
        time_delta=int(metadata["time_delta"]),
        timeout_value=float(metadata["timeout_value"]),
        budget_buckets=buckets,
        samples_per_bucket=int(metadata["samples_per_bucket"]),
        seed=int(metadata.get("budget_seed", 0)),
    )
