from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_oracle import compute_oracle_policy
from supervised_branch import GeneratedTreeHaltEnv, TeacherSearchConfig, load_raw_pretrain_example_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze offline optimal halt/continue targets under the current controller reward."
    )
    parser.add_argument("--data", required=True, help="Raw .pt directory or text manifest of raw pretrain examples.")
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    paths = load_raw_pretrain_example_paths(args.data)
    rng = random.Random(args.seed)
    if args.sample_size > 0 and args.sample_size < len(paths):
        paths = rng.sample(paths, args.sample_size)

    quality_config = TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="controller_analysis_v1",
        search_config_id="controller_analysis",
    )

    env = GeneratedTreeHaltEnv(
        example_paths=paths,
        quality_config=quality_config,
        continue_cost=args.continue_cost,
        seed=args.seed,
        shuffle=False,
        max_cache_size=1,
    )

    total_examples = 0
    should_continue_count = 0
    alignment_count = 0
    first_halt_reward_sum = 0.0
    optimal_value_sum = 0.0
    continue_advantage_sum = 0.0
    optimal_stop_step_histogram: dict[int, int] = {}

    for index, path in enumerate(paths, start=1):
        env.reset()
        if env._current_path != path:
            raise RuntimeError(f"Unexpected path order: expected {path}, got {env._current_path}")

        halt_rewards = [env._final_root_q_values[move] for move in env._best_moves]
        policy = compute_oracle_policy(halt_rewards, args.continue_cost)

        total_examples += 1
        first_halt_reward = halt_rewards[0]
        first_halt_reward_sum += first_halt_reward
        optimal_value_sum += policy.values[0]
        continue_advantage_sum += policy.values[0] - first_halt_reward

        if policy.optimal_stop_step > 0:
            should_continue_count += 1
        optimal_stop_step_histogram[policy.optimal_stop_step] = (
            optimal_stop_step_histogram.get(policy.optimal_stop_step, 0) + 1
        )

        if env._best_moves[0] == env._best_moves[-1]:
            alignment_count += 1

        if index % args.log_interval == 0 or index == len(paths):
            print(f"checked={index}/{len(paths)}", flush=True)

    def _safe_div(total: float, count: int) -> float:
        return total / count if count > 0 else float("nan")

    print(f"total_examples={total_examples}")
    print(f"continue_cost={args.continue_cost:.6f}")
    print(f"mean_first_halt_reward={_safe_div(first_halt_reward_sum, total_examples):.6f}")
    print(f"mean_optimal_value={_safe_div(optimal_value_sum, total_examples):.6f}")
    print(f"mean_continue_advantage={_safe_div(continue_advantage_sum, total_examples):.6f}")
    print(f"should_continue_count={should_continue_count}")
    print(f"should_continue_rate={_safe_div(should_continue_count, total_examples):.6f}")
    print(f"best_move_alignment_count={alignment_count}")
    print(f"best_move_alignment_rate={_safe_div(alignment_count, total_examples):.6f}")
    print("optimal_stop_step_histogram=")
    for step in sorted(optimal_stop_step_histogram):
        print(f"  step_{step}={optimal_stop_step_histogram[step]}")


if __name__ == "__main__":
    main()
