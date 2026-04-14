from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PlanningCostConfig:
    kind: str = "linear"
    base_cost: float = 0.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.base_cost < 0.0:
            raise ValueError("base_cost must be non-negative.")
        if self.kind not in {"linear", "power"}:
            raise ValueError(f"Unsupported planning cost kind: {self.kind!r}")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive.")
        if self.kind == "linear" and abs(self.exponent - 1.0) > 1e-12:
            raise ValueError("Linear planning cost must use exponent=1.0.")


def planning_cost_config(
    continue_cost: float,
    *,
    kind: str = "linear",
    exponent: float = 1.0,
) -> PlanningCostConfig:
    resolved_exponent = 1.0 if kind == "linear" else exponent
    return PlanningCostConfig(kind=kind, base_cost=float(continue_cost), exponent=float(resolved_exponent))


def planning_cost_config_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    continue_cost_key: str = "continue_cost",
    kind_key: str = "planning_cost_kind",
    exponent_key: str = "planning_cost_exponent",
) -> PlanningCostConfig | None:
    if metadata is None:
        return None
    if continue_cost_key not in metadata:
        return None
    kind = str(metadata.get(kind_key, "linear"))
    exponent = float(metadata.get(exponent_key, 1.0))
    return planning_cost_config(float(metadata[continue_cost_key]), kind=kind, exponent=exponent)


def planning_cost_metadata(config: PlanningCostConfig) -> dict[str, float | str]:
    return {
        "continue_cost": float(config.base_cost),
        "planning_cost_kind": config.kind,
        "planning_cost_exponent": float(config.exponent),
    }


def cumulative_planning_cost(num_continues: int, config: PlanningCostConfig) -> float:
    if num_continues < 0:
        raise ValueError("num_continues must be non-negative.")
    if config.kind == "linear":
        return config.base_cost * num_continues
    return config.base_cost * (num_continues ** config.exponent)


def incremental_planning_cost(num_continues_so_far: int, config: PlanningCostConfig) -> float:
    if num_continues_so_far < 0:
        raise ValueError("num_continues_so_far must be non-negative.")
    return cumulative_planning_cost(num_continues_so_far + 1, config) - cumulative_planning_cost(
        num_continues_so_far,
        config,
    )


def return_for_stop_step(halt_rewards: list[float] | tuple[float, ...], stop_step: int, config: PlanningCostConfig) -> float:
    if stop_step < 0 or stop_step >= len(halt_rewards):
        raise IndexError(stop_step)
    return float(halt_rewards[stop_step]) - cumulative_planning_cost(stop_step, config)
