from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supervised_branch import GeneratedTreeHaltEnv, TeacherSearchConfig, load_raw_pretrain_example_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze how often additional search changes the best root move or materially improves root quality."
    )
    parser.add_argument("--data", required=True, help="Raw .pt directory or text manifest of raw pretrain examples.")
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--quality-threshold", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def _controller_episode_summary(
    env: GeneratedTreeHaltEnv,
) -> tuple[str, float, Optional[str], float]:
    first_snapshot = env._snapshots[0]
    first_quality = float(env._qualities[0])
    first_best_move = first_snapshot.get_node(first_snapshot.best_root_child()).incoming_move_uci

    final_snapshot = env._snapshots[-1]
    final_quality = float(env._qualities[-1])
    final_best_move = None
    if final_snapshot.root_children():
        final_best_move = final_snapshot.get_node(final_snapshot.best_root_child()).incoming_move_uci

    return first_best_move, first_quality, final_best_move, final_quality


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

    total_examples = 0
    examples_with_root_decision = 0
    best_move_changed = 0
    best_move_changed_nontrivial = 0
    quality_gain_nontrivial = 0
    first_decision_quality_sum = 0.0
    final_quality_sum = 0.0
    final_minus_first_decision_sum = 0.0

    env = GeneratedTreeHaltEnv(
        example_paths=paths,
        quality_config=quality_config,
        continue_cost=0.05,
        seed=args.seed,
        shuffle=False,
        max_cache_size=1,
    )

    for index, path in enumerate(paths, start=1):
        env.reset()
        if env._current_path != path:
            raise RuntimeError(f"Unexpected path order: expected {path}, got {env._current_path}")
        total_examples += 1
        first_move, first_q, final_move, final_q = _controller_episode_summary(env)
        examples_with_root_decision += 1
        first_decision_quality_sum += first_q
        final_quality_sum += final_q
        final_minus_first_decision_sum += final_q - first_q

        if first_move != final_move:
            best_move_changed += 1
            if final_q - first_q >= args.quality_threshold:
                best_move_changed_nontrivial += 1

        if final_q - first_q >= args.quality_threshold:
            quality_gain_nontrivial += 1

        if index % args.log_interval == 0 or index == len(paths):
            print(f"checked={index}/{len(paths)}", flush=True)

    def _safe_div(numerator: float, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    print(f"total_examples={total_examples}")
    print(f"examples_with_root_decision={examples_with_root_decision}")
    print(f"quality_threshold={args.quality_threshold:.6f}")
    print(
        f"mean_first_decision_quality={_safe_div(first_decision_quality_sum, examples_with_root_decision):.6f}"
    )
    print(f"mean_final_quality={_safe_div(final_quality_sum, examples_with_root_decision):.6f}")
    print(
        f"mean_final_minus_first_decision={_safe_div(final_minus_first_decision_sum, examples_with_root_decision):.6f}"
    )
    print(f"best_move_changed_count={best_move_changed}")
    print(
        f"best_move_changed_rate={_safe_div(best_move_changed, examples_with_root_decision):.6f}"
    )
    print(f"best_move_changed_nontrivial_count={best_move_changed_nontrivial}")
    print(
        f"best_move_changed_nontrivial_rate="
        f"{_safe_div(best_move_changed_nontrivial, examples_with_root_decision):.6f}"
    )
    print(f"quality_gain_nontrivial_count={quality_gain_nontrivial}")
    print(
        f"quality_gain_nontrivial_rate={_safe_div(quality_gain_nontrivial, examples_with_root_decision):.6f}"
    )


if __name__ == "__main__":
    main()
