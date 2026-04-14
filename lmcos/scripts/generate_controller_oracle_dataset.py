from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_oracle import (
    has_strong_optimal_margins as _has_strong_optimal_margins,
    optimal_stop_step as _optimal_stop_step,
    optimal_values_and_actions as _optimal_values_and_actions,
)
from supervised_branch import (
    PretrainExample,
    TeacherSearchConfig,
    build_trimmed_decision_episode_with_halt_rewards,
    compute_teacher_targets,
    save_pretrain_example_to_directory,
)
from tree import ExpansionChild, SearchTree


VALUE_GRID = [-1.0, -0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.0]
PRIOR_GRID = [0.2, 0.4, 0.6, 0.8]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate small synthetic raw-tree datasets with known oracle stop steps under the current controller objective."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-per-step", type=int, default=256)
    parser.add_argument("--validation-per-step", type=int, default=64)
    parser.add_argument("--oracle-steps", default="0,1,2")
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument(
        "--min-decision-margin",
        type=float,
        default=0.05,
        help="Require the optimal action at each visited decision point to beat the alternative by at least this margin.",
    )
    return parser


def _teacher_config(args: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="oracle_dataset_v1",
        search_config_id="oracle_dataset",
    )


def _validate_margin_request(target_steps: Sequence[int], continue_cost: float, min_decision_margin: float) -> None:
    if not target_steps:
        raise ValueError("oracle-steps must be non-empty.")
    if continue_cost < 0.0:
        raise ValueError("continue_cost must be non-negative.")
    if min_decision_margin < 0.0:
        raise ValueError("min_decision_margin must be non-negative.")

def _make_root_child(move: str, value: float, prior: float) -> ExpansionChild:
    return ExpansionChild(
        move_uci=move,
        fen=f"{move}_leaf",
        scalar_features={"value": value, "prior": prior},
    )


def _make_leaf_child(move: str, value: float, prior: float) -> ExpansionChild:
    return ExpansionChild(
        move_uci=move,
        fen=f"{move}_leaf",
        scalar_features={"value": value, "prior": prior},
        is_terminal=True,
    )


def _random_candidate_tree(rng: random.Random) -> SearchTree:
    tree = SearchTree()
    root_id = tree.create_root("oracle-root", {"value": 0.0, "prior": 1.0})
    root_children = [
        _make_root_child("a", rng.choice(VALUE_GRID), rng.choice(PRIOR_GRID)),
        _make_root_child("b", rng.choice(VALUE_GRID), rng.choice(PRIOR_GRID)),
    ]
    child_ids = tree.add_children(root_id, root_children)
    a_id, b_id = child_ids

    a_grandchildren = [
        _make_leaf_child("a1", rng.choice(VALUE_GRID), rng.choice(PRIOR_GRID)),
        _make_leaf_child("a2", rng.choice(VALUE_GRID), rng.choice(PRIOR_GRID)),
    ]
    b_grandchildren = [
        _make_leaf_child("b1", rng.choice(VALUE_GRID), rng.choice(PRIOR_GRID)),
        _make_leaf_child("b2", rng.choice(VALUE_GRID), rng.choice(PRIOR_GRID)),
    ]

    if rng.random() < 0.5:
        tree.add_children(a_id, a_grandchildren)
        tree.add_children(b_id, b_grandchildren)
    else:
        tree.add_children(b_id, b_grandchildren)
        tree.add_children(a_id, a_grandchildren)

    return tree


def _example_and_oracle(
    tree: SearchTree,
    config: TeacherSearchConfig,
    continue_cost: float,
) -> tuple[PretrainExample, int, list[float]]:
    teacher_result = compute_teacher_targets(tree, config)
    example = PretrainExample(
        tree=tree,
        node_target_values=teacher_result.node_target_values,
        metadata={},
    )
    _, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(example, config)
    oracle_stop = _optimal_stop_step(halt_rewards, continue_cost)
    return example, oracle_stop, halt_rewards


def _write_manifest(paths: Sequence[str], manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path}\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    target_steps = [int(token.strip()) for token in args.oracle_steps.split(",") if token.strip()]
    _validate_margin_request(target_steps, args.continue_cost, args.min_decision_margin)
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    validation_dir = output_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    config = _teacher_config(args)
    required_counts: Dict[str, Dict[int, int]] = {
        "train": {step: args.train_per_step for step in target_steps},
        "validation": {step: args.validation_per_step for step in target_steps},
    }
    saved_paths: Dict[str, List[str]] = {"train": [], "validation": []}
    saved_counts: Dict[str, Dict[int, int]] = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }

    split_cycle = ["train", "validation"]
    next_index = {"train": 0, "validation": 0}

    attempts = 0
    while attempts < args.max_attempts:
        attempts += 1
        tree = _random_candidate_tree(rng)
        example, oracle_stop, halt_rewards = _example_and_oracle(tree, config, args.continue_cost)
        if oracle_stop not in target_steps:
            if attempts % args.log_interval == 0:
                print(f"attempts={attempts} accepted={sum(len(paths) for paths in saved_paths.values())}", flush=True)
            continue
        if not _has_strong_optimal_margins(
            halt_rewards,
            continue_cost=args.continue_cost,
            min_decision_margin=args.min_decision_margin,
        ):
            if attempts % args.log_interval == 0:
                print(
                    f"attempts={attempts} accepted={sum(len(paths) for paths in saved_paths.values())} "
                    f"rejected_weak_margin=1",
                    flush=True,
                )
            continue

        split = None
        for candidate_split in split_cycle:
            if saved_counts[candidate_split][oracle_stop] < required_counts[candidate_split][oracle_stop]:
                split = candidate_split
                break
        if split is None:
            if all(
                saved_counts[candidate_split][step] >= required_counts[candidate_split][step]
                for candidate_split in ("train", "validation")
                for step in target_steps
            ):
                break
            continue

        split_dir = train_dir if split == "train" else validation_dir
        idx = next_index[split]
        next_index[split] += 1
        example.metadata = {
            "root_position_id": f"oracle_{split}_{idx}",
            "oracle_stop_step": oracle_stop,
            "halt_rewards": halt_rewards,
            "search_config_id": config.search_config_id,
        }
        path = save_pretrain_example_to_directory(str(split_dir), example, idx)
        saved_paths[split].append(path)
        saved_counts[split][oracle_stop] += 1

        if attempts % args.log_interval == 0:
            print(
                f"attempts={attempts} "
                f"train_counts={dict(saved_counts['train'])} "
                f"validation_counts={dict(saved_counts['validation'])}",
                flush=True,
            )

        if all(
            saved_counts[candidate_split][step] >= required_counts[candidate_split][step]
            for candidate_split in ("train", "validation")
            for step in target_steps
        ):
            break

    if not all(
        saved_counts[candidate_split][step] >= required_counts[candidate_split][step]
        for candidate_split in ("train", "validation")
        for step in target_steps
    ):
        raise RuntimeError(
            f"Failed to generate the requested oracle dataset within {args.max_attempts} attempts. "
            f"train_counts={dict(saved_counts['train'])} validation_counts={dict(saved_counts['validation'])}"
        )

    train_manifest = output_root / "train_manifest.txt"
    validation_manifest = output_root / "validation_manifest.txt"
    _write_manifest(saved_paths["train"], train_manifest)
    _write_manifest(saved_paths["validation"], validation_manifest)

    print(f"output_root={output_root}")
    print(f"train_examples={len(saved_paths['train'])}")
    print(f"validation_examples={len(saved_paths['validation'])}")
    print(f"train_counts={dict(saved_counts['train'])}")
    print(f"validation_counts={dict(saved_counts['validation'])}")
    print(f"train_manifest={train_manifest}")
    print(f"validation_manifest={validation_manifest}")


if __name__ == "__main__":
    main()
