from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import random
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cts_pretrain import (
    PretrainExample,
    TeacherSearchConfig,
    derive_prefix_pretrain_example,
    load_raw_pretrain_example_paths,
    save_pretrain_example_to_directory,
)
from schema import require_tree_encoder_scalar_features


def _validate_tree_encoder_example_features(example: PretrainExample, *, context: str) -> None:
    for node in example.tree.iter_nodes():
        require_tree_encoder_scalar_features(
            node.scalar_features,
            context=f"{context} node_id={node.node_id} fen={node.fen!r}",
        )


def _existing_output_for_index(output_dir: Path, index: int) -> Path | None:
    matches = sorted(output_dir.glob(f"{index:06d}_*.pt"))
    if not matches:
        return None
    return matches[0]


def _derive_prefix_task(task: tuple[str, int, TeacherSearchConfig, int, int, int]) -> PretrainExample:
    path_str, global_index, config, min_nodes, max_nodes, base_seed = task
    example = torch.load(path_str, weights_only=False)
    _validate_tree_encoder_example_features(example, context=str(path_str))
    example_rng = random.Random(f"{base_seed}:{global_index}")
    return derive_prefix_pretrain_example(
        example,
        config=config,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        rng=example_rng,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive variable-size root-prefix pretrain examples from existing raw trees."
    )
    parser.add_argument("--input-data", required=True, help="Raw .pt directory or text manifest of raw pretrain examples.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-nodes", type=int, required=True, help="Minimum expanded-node budget for the derived root prefix.")
    parser.add_argument("--max-nodes", type=int, required=True, help="Maximum expanded-node budget for the derived root prefix.")
    parser.add_argument("--search-budget", type=int, required=True)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.min_nodes <= 0:
        raise ValueError("min_nodes must be positive.")
    if args.max_nodes < args.min_nodes:
        raise ValueError("max_nodes must be >= min_nodes.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.start_index < 0:
        raise ValueError("start_index must be non-negative.")

    input_paths = load_raw_pretrain_example_paths(args.input_data)
    resolved_end = len(input_paths) if args.end_index is None else min(args.end_index, len(input_paths))
    if resolved_end < args.start_index:
        raise ValueError("end_index must be >= start_index.")
    selected_paths = input_paths[args.start_index:resolved_end]
    if not selected_paths:
        raise ValueError("No input examples selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id="prefix_pretrain_v1",
    )
    start_time = time.time()
    saved = 0
    skipped = 0
    work_items: list[tuple[str, int, str] | tuple[str, int, tuple[str, int, TeacherSearchConfig, int, int, int]]] = []
    process_tasks: list[tuple[str, int, TeacherSearchConfig, int, int, int]] = []
    for offset, path_str in enumerate(selected_paths):
        global_index = args.start_index + offset
        if args.resume and _existing_output_for_index(output_dir, global_index) is not None:
            work_items.append(("skip", global_index, path_str))
            continue
        task = (path_str, global_index, config, args.min_nodes, args.max_nodes, args.seed)
        work_items.append(("process", global_index, task))
        process_tasks.append(task)

    if args.num_workers <= 1:
        result_iter = iter(map(_derive_prefix_task, process_tasks))
        for offset, work_item in enumerate(work_items):
            mode = work_item[0]
            global_index = work_item[1]
            if mode == "skip":
                skipped += 1
            else:
                prefix_example = next(result_iter)
                save_pretrain_example_to_directory(str(output_dir), prefix_example, global_index)
                saved += 1

            completed = offset + 1
            if completed % args.log_interval == 0 or completed == len(selected_paths):
                elapsed = time.time() - start_time
                print(
                    f"progress={completed}/{len(selected_paths)} saved={saved} skipped={skipped} "
                    f"elapsed_s={elapsed:.1f} base_examples_per_s={completed / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        return

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        result_iter = iter(executor.map(_derive_prefix_task, process_tasks))
        for offset, work_item in enumerate(work_items):
            mode = work_item[0]
            global_index = work_item[1]
            if mode == "skip":
                skipped += 1
            else:
                prefix_example = next(result_iter)
                save_pretrain_example_to_directory(str(output_dir), prefix_example, global_index)
                saved += 1

            completed = offset + 1
            if completed % args.log_interval == 0 or completed == len(selected_paths):
                elapsed = time.time() - start_time
                print(
                    f"progress={completed}/{len(selected_paths)} saved={saved} skipped={skipped} "
                    f"elapsed_s={elapsed:.1f} base_examples_per_s={completed / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
