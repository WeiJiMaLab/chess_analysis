from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from planning_cost import PlanningCostConfig, incremental_planning_cost, planning_cost_config


@dataclass(frozen=True)
class OraclePolicy:
    values: List[float]
    actions: List[int]
    stop_steps: List[int]

    @property
    def optimal_stop_step(self) -> int:
        return self.stop_steps[0]


def _resolve_cost_config(
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> PlanningCostConfig:
    if cost_config is not None:
        return cost_config
    return planning_cost_config(
        continue_cost,
        kind=planning_cost_kind,
        exponent=planning_cost_exponent,
    )


def compute_oracle_policy(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> OraclePolicy:
    if not halt_rewards:
        raise ValueError("halt_rewards must be non-empty.")
    resolved_cost = _resolve_cost_config(
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )

    values = [0.0] * len(halt_rewards)
    actions = [1] * len(halt_rewards)  # 1 = halt, 0 = continue
    stop_steps = [0] * len(halt_rewards)

    values[-1] = float(halt_rewards[-1])
    actions[-1] = 1
    stop_steps[-1] = len(halt_rewards) - 1

    for idx in range(len(halt_rewards) - 2, -1, -1):
        halt_now = float(halt_rewards[idx])
        continue_then = -incremental_planning_cost(idx, resolved_cost) + values[idx + 1]
        if halt_now >= continue_then:
            values[idx] = halt_now
            actions[idx] = 1
            stop_steps[idx] = idx
        else:
            values[idx] = continue_then
            actions[idx] = 0
            stop_steps[idx] = stop_steps[idx + 1]

    return OraclePolicy(values=values, actions=actions, stop_steps=stop_steps)


def optimal_stop_step(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> int:
    return compute_oracle_policy(
        halt_rewards,
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    ).optimal_stop_step


def optimal_values_and_actions(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> tuple[list[float], list[int]]:
    policy = compute_oracle_policy(
        halt_rewards,
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )
    return list(policy.values), list(policy.actions)


def has_strong_optimal_margins(
    halt_rewards: Sequence[float],
    continue_cost: float,
    min_decision_margin: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> bool:
    if min_decision_margin <= 0.0:
        return True

    resolved_cost = _resolve_cost_config(
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )
    policy = compute_oracle_policy(halt_rewards, continue_cost, cost_config=resolved_cost)
    for idx, halt_now in enumerate(halt_rewards):
        if idx == len(halt_rewards) - 1:
            continue_then = float("-inf")
        else:
            continue_then = -incremental_planning_cost(idx, resolved_cost) + policy.values[idx + 1]

        if policy.actions[idx] == 1:
            margin = float(halt_now) - continue_then
            if margin < min_decision_margin:
                return False
            break

        margin = continue_then - float(halt_now)
        if margin < min_decision_margin:
            return False

    return True
