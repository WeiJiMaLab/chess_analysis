"""Budgeted oracle that defines the controller's reward function.

For each snapshot of a search tree, the oracle decides whether the controller
should ``halt`` (consume the current tree's quality) or ``expand`` (pay a
maintenance + time cost and continue). Buckets partition expansion-count
space so training data covers tight (1-3 expansions) through generous
(61-120 expansions) regimes. The dynamic-programming pass in
``compute_budgeted_oracle`` produces the per-step (halt, continue) values
that the fitted-Q controller learns from. Invoked at packing time from
``scripts/pack_controller_episodes.py`` and at train time from
``train_fitted_q_controller.py`` for config sanity checks.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class BudgetBucket:
    """Inclusive expansion-count range used to stratify training samples.

    Each bucket draws ``samples_per_bucket`` starting budgets uniformly from
    ``[min_time, max_time]`` so the controller sees representative episodes
    across the full search-budget range.
    """

    name: str  # human-readable bucket label (e.g. "scramble", "very-large")
    min_time: int  # inclusive lower bound on starting expansion budget
    max_time: int  # inclusive upper bound on starting expansion budget

    def __post_init__(self) -> None:
        # Validate at construction so a malformed config fails immediately,
        # not deep inside data packing.
        if not self.name:
            raise ValueError("BudgetBucket.name must be non-empty.")
        if self.min_time <= 0:
            raise ValueError("BudgetBucket.min_time must be positive.")
        if self.max_time < self.min_time:
            raise ValueError("BudgetBucket.max_time must be >= min_time.")


# Canonical bucket partition spanning 1-120 expansions. Changing these
# boundaries reshuffles which snapshots appear in the training set and is
# a dataset-format-breaking change.
DEFAULT_BUDGET_BUCKETS: tuple[BudgetBucket, ...] = (
    BudgetBucket("scramble", 1, 3),
    BudgetBucket("medium-small", 4, 10),
    BudgetBucket("medium-large", 11, 25),
    BudgetBucket("large", 26, 60),
    BudgetBucket("very-large", 61, 120),
)


@dataclass(frozen=True)
class BudgetedOracleConfig:
    """Hyperparameters of the budgeted oracle reward function.

    Two cost terms apply each time the controller chooses to expand:
    a maintenance term that scales with current tree size, and a time
    term that grows convexly as the remaining budget shrinks toward zero.
    Together they define the trade-off the controller must learn.
    """

    maintenance_scale: float = 0.0  # multiplier on the size-dependent cost; 0 disables maintenance
    maintenance_ref_nodes: float = 30.0  # reference tree size used to normalize maintenance cost
    maintenance_exponent: float = 1.1  # superlinear growth exponent for maintenance cost vs. node count
    time_lambda: float = 18.537  # overall scale of the time-cost term
    time_p: float = 2.8  # convexity exponent for the time-cost term; must be > 1
    time_tau: float = 2.5  # offset that softens the singularity as budget → 0
    time_delta: int = 1  # minimum remaining budget required to still consider continuing
    timeout_value: float = -1.0  # terminal value assigned when the budget is exhausted
    budget_buckets: tuple[BudgetBucket, ...] = DEFAULT_BUDGET_BUCKETS  # bucket partition over expansion counts
    samples_per_bucket: int = 2  # number of starting budgets drawn from each bucket per source tree
    seed: int = 0  # base seed mixed into the deterministic bucket sampler

    def __post_init__(self) -> None:
        # All scalar knobs are validated up front; downstream code assumes
        # these invariants without re-checking.
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
    """One starting-budget draw associated with a specific bucket."""

    bucket_index: int  # position of the parent bucket in config.budget_buckets
    bucket_name: str  # convenience copy of the bucket's name
    starting_budget: int  # sampled expansion budget within [bucket.min_time, bucket.max_time]


@dataclass(frozen=True)
class BudgetedOraclePolicy:
    """Per-step DP outputs for a single (tree, starting_budget) episode.

    All sequences are aligned and indexed by snapshot step (0 = before any
    expansion, 1 = after the first expansion, etc.). ``actions[i]`` is
    1 for halt and 0 for continue. ``stop_steps[i]`` is the optimal step
    at which to halt assuming optimal play from step ``i`` onward.
    """

    halt_rewards: List[float]  # per-step reward if the controller halts at that step
    tree_sizes: List[int]  # per-step node count, used by the maintenance cost
    time_budgets: List[int]  # remaining expansion budget at each step
    continue_values: List[float]  # per-step value of choosing "expand" (after DP)
    values: List[float]  # per-step optimal value V*(s)
    actions: List[int]  # per-step optimal action; 1 = halt, 0 = continue
    stop_steps: List[int]  # earliest optimal halt step from each step onward
    target_advantages: List[float]  # continue_value - halt_value; the controller's regression target
    starting_budget: int  # initial budget that parameterized this episode

    @property
    def optimal_stop_step(self) -> int:
        """Optimal halt step when starting from snapshot 0."""
        return self.stop_steps[0]

    @property
    def oracle_value(self) -> float:
        """Optimal value at snapshot 0; the return the controller should achieve."""
        return self.values[0]


def predicted_stop_from_advantages(advantages: Sequence[float]) -> int:
    """Greedy halt step from a per-step advantage trace (controller evaluation rule).

    Returns the first step where ``advantage <= 0``, or the last step if the trace
    never crosses zero. Matches ``controller_train.py`` episode evaluation.
    """
    predicted_stop = len(advantages) - 1
    for step_index, value in enumerate(advantages):
        if float(value) <= 0.0:
            predicted_stop = step_index
            break
    return predicted_stop


def maintenance_cost(num_nodes: int, config: BudgetedOracleConfig) -> float:
    """Size-dependent per-step cost: ``scale * (n / ref) ** exponent``."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    return float(config.maintenance_scale * (float(num_nodes) / config.maintenance_ref_nodes) ** config.maintenance_exponent)


def time_cost(remaining_budget: int, config: BudgetedOracleConfig) -> float:
    """Convex time cost that grows sharply as the remaining budget approaches zero.

    Implemented as the finite difference of an offset power-law potential,
    so summing per-step costs telescopes into a clean total cost-of-time
    expression with closed-form properties.
    """
    if remaining_budget <= 0:
        raise ValueError("remaining_budget must be positive.")
    # Two power-law evaluations one step apart; the subtraction is the
    # per-step time cost when integrated as the difference of a potential.
    left = (remaining_budget - config.time_delta + config.time_tau) ** (-(config.time_p - 1.0))
    right = (remaining_budget + config.time_tau) ** (-(config.time_p - 1.0))
    return float(config.time_lambda * (left - right))


def continue_cost(num_nodes: int, remaining_budget: int, config: BudgetedOracleConfig) -> float:
    """Total per-step cost of choosing to expand: maintenance + time."""
    return maintenance_cost(num_nodes, config) + time_cost(remaining_budget, config)


def deterministic_starting_budgets(
    source_path: str,
    config: BudgetedOracleConfig,
) -> List[SampledBudget]:
    """Generate reproducible starting-budget samples for a single source tree.

    Each (source_path, bucket, draw) triple hashes to a unique RNG seed, so
    the same tree always yields the same budgets across runs. This keeps the
    packed training set deterministic and exactly reproducible from the raw
    tree shards plus the oracle config.

    Args:
        source_path: path to the source tree file; mixed into the hash so
            different trees draw independent budgets.
        config: oracle config supplying buckets, sample count, and seed.
    """
    sampled: List[SampledBudget] = []
    for bucket_index, bucket in enumerate(config.budget_buckets):
        for draw_index in range(config.samples_per_bucket):
            # Hash source path + bucket + draw + seed into a 64-bit RNG seed.
            # SHA-256 is overkill cryptographically but trivially fast here
            # and guarantees uniform distribution across the keyspace.
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
    """Solve the per-step halt/continue MDP via backward DP.

    The controller faces a stopping problem: at each snapshot step it
    observes ``halt_rewards[i]`` and may either lock that value in (halt)
    or pay ``continue_cost`` and move to step ``i+1``. This function
    computes the optimal value function and action sequence in O(T).

    Args:
        halt_rewards: per-step reward for halting at that step.
        tree_sizes: per-step node count, aligned with ``halt_rewards``.
        starting_budget: initial expansion budget for the episode.
        config: oracle hyperparameters defining the cost terms.
    """
    if not halt_rewards:
        raise ValueError("halt_rewards must be non-empty.")
    if len(halt_rewards) != len(tree_sizes):
        raise ValueError("halt_rewards and tree_sizes must have the same length.")
    if starting_budget <= 0:
        raise ValueError("starting_budget must be positive.")

    # Episode is truncated either by data length or by the starting budget,
    # whichever is shorter. After truncation, step i has budget B - i left.
    max_steps = min(len(halt_rewards), starting_budget)
    truncated_rewards = [float(value) for value in halt_rewards[:max_steps]]
    truncated_sizes = [int(value) for value in tree_sizes[:max_steps]]
    time_budgets = [starting_budget - index for index in range(max_steps)]

    values = [0.0] * max_steps
    actions = [1] * max_steps  # 1 = halt, 0 = continue
    stop_steps = [0] * max_steps
    continue_values = [0.0] * max_steps
    target_advantages = [0.0] * max_steps

    # Standard backward induction: V*(T-1) is the trivial halt; for earlier
    # steps compare halt_value against (continue cost + V* of the next step).
    for idx in range(max_steps - 1, -1, -1):
        halt_value = truncated_rewards[idx]
        remaining_budget = time_budgets[idx]

        # Determine the value of being in the "next" state. If the budget
        # is about to run out, the controller is forced to time out.
        # Otherwise, use the next step's V* (or fall back to the current
        # halt value at the data-truncation boundary).
        if remaining_budget <= config.time_delta:
            next_value = float(config.timeout_value)
        elif idx + 1 < max_steps:
            next_value = values[idx + 1]
        else:
            next_value = truncated_rewards[idx]

        step_continue_value = -continue_cost(truncated_sizes[idx], remaining_budget, config) + next_value
        continue_values[idx] = step_continue_value
        target_advantages[idx] = step_continue_value - halt_value

        # Tie-break in favor of halting (>=) — matches the convention used
        # by the controller's argmax at inference time.
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
    """Return True iff every pre-halt decision has a margin >= ``min_decision_margin``.

    Used at packing time to filter out ambiguous episodes where the oracle's
    halt/continue choice is essentially a coin flip — those examples add
    noise to the controller without teaching it anything.

    Args:
        halt_rewards: per-step halt rewards (forwarded to the oracle).
        tree_sizes: per-step node counts (forwarded to the oracle).
        starting_budget: initial budget (forwarded to the oracle).
        config: oracle hyperparameters.
        min_decision_margin: minimum |halt - continue| value required at
            each step up to and including the optimal halt step.
    """
    if min_decision_margin <= 0.0:
        return True
    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, starting_budget, config)
    for idx, halt_value in enumerate(policy.halt_rewards):
        # Margin is signed by the chosen action so that "decisive" always
        # means the chosen action's value beats the alternative by margin.
        margin = (
            float(halt_value) - policy.continue_values[idx]
            if policy.actions[idx] == 1
            else policy.continue_values[idx] - float(halt_value)
        )
        if margin < min_decision_margin:
            return False
        # Once we've reached the halt step, future steps don't matter for
        # the realized trajectory — stop checking.
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
    """Total episode return if the controller halts at exactly ``stop_step``.

    The return is the halt reward at the stop step minus the accumulated
    continue costs paid for all expansions taken to reach it. Used for
    counterfactual evaluation: comparing a learned policy against the
    oracle's optimal stop step in the same episode.

    Args:
        halt_rewards: per-step halt rewards.
        tree_sizes: per-step node counts.
        time_budgets: per-step remaining budgets.
        stop_step: index at which the policy halts (0-based).
        config: oracle hyperparameters.
    """
    if len(halt_rewards) != len(tree_sizes) or len(halt_rewards) != len(time_budgets):
        raise ValueError("halt_rewards, tree_sizes, and time_budgets must have the same length.")
    if stop_step < 0 or stop_step >= len(halt_rewards):
        raise IndexError(stop_step)
    total = float(halt_rewards[stop_step])
    for idx in range(stop_step):
        total -= continue_cost(int(tree_sizes[idx]), int(time_budgets[idx]), config)
    return total


def budgeted_oracle_metadata(config: BudgetedOracleConfig) -> Dict[str, Any]:
    """Serialize a config into the JSON-compatible metadata stored alongside packed episodes.

    The ``oracle_type`` tag versions the schema so packed datasets can be
    safely round-tripped through ``budgeted_oracle_config_from_metadata``.
    """
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
    """Reverse of ``budgeted_oracle_metadata``; returns None for non-budgeted oracle types.

    Used by ``train_fitted_q_controller.py`` to assert that the training
    config matches the config under which the data was packed.
    """
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
