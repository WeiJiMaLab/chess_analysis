"""
Plot budgeted-oracle *planning return* vs search-tree size for one `PretrainExample` checkpoint.

**Requires** a ``torch.load`` that unpickles ``PretrainExample`` (same as ``pack_controller_episodes``).
Raw `cts_raw_pretrain_example_v1` dicts are not supported here (no teacher / halt sequence).

Y-axis: ``return_for_stop_step`` from `budgeted_controller_oracle` — halt reward at that expansion,
minus the cumulative **maintenance + time** cost of all earlier continues (for the chosen
``starting_budget``). X-axis: ``snapshot.num_nodes()`` at each expansion index (same as packed
``tree_sizes`` in the training pipeline).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_LMCOS = Path(__file__).resolve().parent.parent.parent / "lmcos"
if str(_LMCOS) not in sys.path:
    sys.path.insert(0, str(_LMCOS))

from budgeted_controller_oracle import (  # noqa: E402
    BudgetBucket,
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    return_for_stop_step,
)
from cts_episode_envs import build_trimmed_decision_episode_with_halt_rewards  # noqa: E402
from cts_pretrain import PretrainExample, TeacherSearchConfig  # noqa: E402


def _default_out() -> str:
    return str(Path(__file__).resolve().parent.parent / "figures" / "performance" / "budgeted_oracle_value.png")


def _default_pt() -> str:
    return os.path.join(
        "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data",
        "generated_trees_oracle96_trace_filtered",
        "shard_00000",
        "000000_root_0.pt",
    )


def _oracle_config_from_args(a: argparse.Namespace) -> BudgetedOracleConfig:
    return BudgetedOracleConfig(
        maintenance_scale=a.maintenance_scale,
        maintenance_ref_nodes=a.maintenance_ref_nodes,
        maintenance_exponent=a.maintenance_exponent,
        time_lambda=a.time_lambda,
        time_p=a.time_p,
        time_tau=a.time_tau,
        time_delta=a.time_delta,
        timeout_value=a.timeout_value,
        budget_buckets=(
            BudgetBucket("scramble", a.scramble_min_time, a.scramble_max_time),
            BudgetBucket("medium-small", a.medium_small_min_time, a.medium_small_max_time),
            BudgetBucket("medium-large", a.medium_large_min_time, a.medium_large_max_time),
            BudgetBucket("large", a.large_min_time, a.large_max_time),
            BudgetBucket("very-large", a.very_large_min_time, a.very_large_max_time),
        ),
        samples_per_bucket=a.samples_per_bucket,
        seed=a.seed,
    )


def _quality_config_from_args(a: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=a.max_depth,
        search_budget=a.search_budget,
        c_puct=a.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot budgeted return vs tree size for one PretrainExample (see pack_controller_episodes).",
    )
    p.add_argument("--pt-path", type=str, default=_default_pt(), help="Path to a PretrainExample .pt file")
    p.add_argument("--starting-budget", type=int, default=20, help="Planning time budget (steps); same as pack pipeline")
    p.add_argument("--reward-scale", type=float, default=1.0, help="Scale halt rewards before oracle (as in pack)")
    p.add_argument("--out", type=str, default=_default_out(), help="Output PNG path")
    p.add_argument("--search-budget", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--maintenance-scale", type=float, default=0.01)
    p.add_argument("--maintenance-ref-nodes", type=float, default=30.0)
    p.add_argument("--maintenance-exponent", type=float, default=1.1)
    p.add_argument("--time-lambda", type=float, default=18.537)
    p.add_argument("--time-p", type=float, default=2.8)
    p.add_argument("--time-tau", type=float, default=2.5)
    p.add_argument("--time-delta", type=int, default=1)
    p.add_argument("--timeout-value", type=float, default=-1.0)
    p.add_argument("--samples-per-bucket", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scramble-min-time", type=int, default=1)
    p.add_argument("--scramble-max-time", type=int, default=3)
    p.add_argument("--medium-small-min-time", type=int, default=4)
    p.add_argument("--medium-small-max-time", type=int, default=10)
    p.add_argument("--medium-large-min-time", type=int, default=11)
    p.add_argument("--medium-large-max-time", type=int, default=25)
    p.add_argument("--large-min-time", type=int, default=26)
    p.add_argument("--large-max-time", type=int, default=60)
    p.add_argument("--very-large-min-time", type=int, default=61)
    p.add_argument("--very-large-max-time", type=int, default=120)
    args = p.parse_args()

    qcfg = _quality_config_from_args(args)
    ocfg = _oracle_config_from_args(args)

    example = torch.load(args.pt_path, weights_only=False)
    if not isinstance(example, PretrainExample):
        raise SystemExit(
            f"Need a PretrainExample at {args.pt_path!r}; got {type(example).__name__}. "
            "Use a checkpoint produced in the pretrain / controller pack pipeline, not a raw v1 dict."
        )

    _episode, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(example, qcfg)
    tree_sizes = [s.num_nodes() for s in _episode.snapshots]
    if not tree_sizes or not halt_rewards:
        raise SystemExit("Empty episode after build_trimmed_decision_episode_with_halt_rewards.")

    scaled = [float(args.reward_scale) * r for r in halt_rewards]
    n = len(scaled)
    time_budgets = [args.starting_budget - i for i in range(n)]
    if time_budgets and time_budgets[0] != args.starting_budget:
        raise RuntimeError("internal: time budget mismatch")

    per_stop = []
    for k in range(n):
        per_stop.append(
            return_for_stop_step(scaled, tree_sizes, time_budgets, k, ocfg)
        )
    policy = compute_budgeted_oracle(scaled, tree_sizes, args.starting_budget, ocfg)
    k_star = policy.optimal_stop_step
    v_star = policy.oracle_value

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="tight")
    ax.plot(tree_sizes, per_stop, color="#4338ca", linewidth=2, label="return if you stop (net of cont. costs to here)")
    ax.axhline(v_star, color="#b45309", linestyle="--", linewidth=1.2, label=f"budgeted oracle value = {v_star:.4f}")
    if 0 <= k_star < n:
        ax.axvline(tree_sizes[k_star], color="#dc2626", linestyle=":", linewidth=1.5, label=f"optimal stop @ n={tree_sizes[k_star]} (step {k_star})")
    ax.set_xlabel("search tree size (nodes in snapshot)")
    ax.set_ylabel("net return (halt − Σ continue costs)")
    ax.set_title(
        f"Budgeted planning value vs tree size  ·  B={args.starting_budget}  ·  {Path(args.pt_path).name}\n"
        f"reward_scale={args.reward_scale}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print(f"  steps: {n}  tree_sizes: {tree_sizes[0]}..{tree_sizes[-1]}  optimal_stop_step={k_star}  oracle_value={v_star}")


if __name__ == "__main__":
    main()
