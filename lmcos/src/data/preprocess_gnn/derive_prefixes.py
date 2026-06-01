"""Derive variable-size root-prefix pretrain examples from full teacher trees.

For each full tree on disk, emits a single "prefix tree": the same tree
truncated to a randomly chosen expansion count in ``[min_nodes, max_nodes]``.
These prefixes are the supervision examples the encoder pretrains on — it
learns to predict each frontier child's WDL from the prefix tree state.
Runs after ``prepare_pretrain_split.py`` (which produces the raw full trees)
and before ``pack_pretrain_examples.py`` (which batches these prefixes for
training).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import random
import sys
import time
from pathlib import Path

import torch

from pydantic import BaseModel, ConfigDict

from cts.core.schema import require_tree_encoder_scalar_features
from cts.data.preprocess_gnn.teacher_targets import (
    PretrainExample,
    TeacherSearchConfig,
    derive_prefix_pretrain_example,
    load_pretrain_example,
    load_raw_pretrain_example_paths,
    save_pretrain_example_to_directory,
)


class DerivePrefixesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_data: str
    output_dir: str
    min_nodes: int
    max_nodes: int
    search_budget: int
    c_puct: float = 1.0
    max_depth: int = 10
    seed: int = 0
    start_index: int = 0
    end_index: int | None = None
    log_interval: int = 25
    num_workers: int = 0
    resume: bool = False


def _validate_tree_encoder_example_features(example: PretrainExample, *, context: str) -> None:
    """Fail loudly if any node is missing encoder features (e.g. older WDL-less data)."""
    for node in example.tree.iter_nodes():
        require_tree_encoder_scalar_features(
            node.scalar_features,
            context=f"{context} node_id={node.node_id} fen={node.fen!r}",
        )


def _existing_output_for_index(output_dir: Path, index: int) -> Path | None:
    """Return the existing output for ``index`` if resume-mode should skip it.

    Outputs are saved as ``{index:06d}_*.pt``; presence of any matching file
    is treated as a complete prior run for that index.
    """
    matches = sorted(output_dir.glob(f"{index:06d}_*.pt"))
    if not matches:
        return None
    return matches[0]


def _derive_prefix_task(task: tuple[str, int, TeacherSearchConfig, int, int, int]) -> PretrainExample:
    """Worker entry point: load one raw example, validate, and derive a prefix.

    Packed as a single positional tuple to be ``ProcessPoolExecutor.map``
    friendly. The per-example RNG is seeded by ``base_seed:global_index`` so
    runs are reproducible and parallel workers don't collide on the same
    random draw.
    """
    path_str, global_index, config, min_nodes, max_nodes, base_seed = task
    example = load_pretrain_example(path_str)
    _validate_tree_encoder_example_features(example, context=str(path_str))
    example_rng = random.Random(f"{base_seed}:{global_index}")
    return derive_prefix_pretrain_example(
        example,
        config=config,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        rng=example_rng,
    )


def main(config: DerivePrefixesConfig) -> None:
    if config.min_nodes <= 0:
        raise ValueError("min_nodes must be positive.")
    if config.max_nodes < config.min_nodes:
        raise ValueError("max_nodes must be >= min_nodes.")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if config.start_index < 0:
        raise ValueError("start_index must be non-negative.")

    # Slice the input list to the requested [start, end) window. This lets a
    # slurm array job partition the dataset across tasks without each task
    # reading the whole manifest twice.
    input_paths = load_raw_pretrain_example_paths(config.input_data)
    resolved_end = len(input_paths) if config.end_index is None else min(config.end_index, len(input_paths))
    if resolved_end < config.start_index:
        raise ValueError("end_index must be >= start_index.")
    selected_paths = input_paths[config.start_index:resolved_end]
    if not selected_paths:
        raise ValueError("No input examples selected.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Recorded into each saved example so downstream consumers can identify
    # which prefix-generation config produced the data.
    search_config = TeacherSearchConfig(
        max_depth=config.max_depth,
        search_budget=config.search_budget,
        c_puct=config.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id="prefix_pretrain_v1",
    )
    start_time = time.time()
    saved = 0
    skipped = 0
    # Two parallel lists: ``work_items`` preserves input order (including
    # skips) for progress reporting; ``process_tasks`` holds only the work
    # the workers actually need to run, in the same relative order so a
    # single ``result_iter`` matches up with the "process" entries.
    work_items: list[tuple[str, int, str] | tuple[str, int, tuple[str, int, TeacherSearchConfig, int, int, int]]] = []
    process_tasks: list[tuple[str, int, TeacherSearchConfig, int, int, int]] = []
    for offset, path_str in enumerate(selected_paths):
        global_index = config.start_index + offset
        if config.resume and _existing_output_for_index(output_dir, global_index) is not None:
            work_items.append(("skip", global_index, path_str))
            continue
        task = (path_str, global_index, search_config, config.min_nodes, config.max_nodes, config.seed)
        work_items.append(("process", global_index, task))
        process_tasks.append(task)

    # Single-process path: bypass the executor so debugging stack traces are
    # legible. ``map`` is lazy so derivation still overlaps with disk I/O.
    if config.num_workers <= 1:
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
            if completed % config.log_interval == 0 or completed == len(selected_paths):
                elapsed = time.time() - start_time
                print(
                    f"progress={completed}/{len(selected_paths)} saved={saved} skipped={skipped} "
                    f"elapsed_s={elapsed:.1f} base_examples_per_s={completed / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        return

    # Parallel path. Some shared filesystems (e.g. read-only mounts in
    # certain slurm setups) refuse the semaphore syscalls ProcessPoolExecutor
    # uses; on PermissionError/OSError we fall back to the sequential loop
    # rather than crashing the whole shard.
    try:
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
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
                if completed % config.log_interval == 0 or completed == len(selected_paths):
                    elapsed = time.time() - start_time
                    print(
                        f"progress={completed}/{len(selected_paths)} saved={saved} skipped={skipped} "
                        f"elapsed_s={elapsed:.1f} base_examples_per_s={completed / max(elapsed, 1e-6):.2f}",
                        flush=True,
                    )
    except (PermissionError, OSError):
        # Fallback: same loop body as the single-process path above.
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
            if completed % config.log_interval == 0 or completed == len(selected_paths):
                elapsed = time.time() - start_time
                print(
                    f"progress={completed}/{len(selected_paths)} saved={saved} skipped={skipped} "
                    f"elapsed_s={elapsed:.1f} base_examples_per_s={completed / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(DerivePrefixesConfig, main)
