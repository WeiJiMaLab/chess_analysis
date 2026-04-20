"""Analyze source-level root-action churn motifs directly from raw pretrain examples."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from cts_pretrain import PretrainExample, load_pretrain_example


def _iter_source_paths(source_root: Path) -> list[Path]:
    return sorted(source_root.rglob("*.pt"))


def _compressed_move_sequence(best_moves: list[str]) -> list[str]:
    compressed: list[str] = []
    for move in best_moves:
        if not compressed or compressed[-1] != move:
            compressed.append(move)
    return compressed


def _first_appearance_step(best_moves: list[str], target_move: str) -> int:
    return next(index for index, move in enumerate(best_moves) if move == target_move)


def _stabilization_step(best_moves: list[str], target_move: str) -> int:
    for index in range(len(best_moves)):
        if all(move == target_move for move in best_moves[index:]):
            return index
    raise ValueError("Target move never stabilizes.")


def _canonicalize_final_anchored_motif(compressed_moves: list[str]) -> str:
    final_move = compressed_moves[-1]
    mapping: dict[str, str] = {final_move: "A"}
    next_label = ord("B")
    labels: list[str] = []
    for move in compressed_moves:
        if move not in mapping:
            mapping[move] = chr(next_label)
            next_label += 1
        labels.append(mapping[move])
    return "".join(labels)


def _source_record(path: Path, example: PretrainExample) -> dict[str, Any] | None:
    best_moves = [str(move) for move in example.oracle_best_move_trace]
    if not best_moves:
        return None
    compressed = _compressed_move_sequence(best_moves)
    final_move = compressed[-1]
    first_appearance = _first_appearance_step(best_moves, final_move)
    stabilization = _stabilization_step(best_moves, final_move)
    top_level_category = "A" if len(compressed) == 1 else "X*A" if compressed.count(final_move) == 1 else "X*AB*A"
    halt_rewards = [float(example.oracle_final_root_q_values[move]) for move in best_moves]
    return {
        "source_path": str(path),
        "compressed_sequence": compressed,
        "canonical_motif": _canonicalize_final_anchored_motif(compressed),
        "top_level_category": top_level_category,
        "compressed_length": len(compressed),
        "num_unique_moves": len(set(compressed)),
        "final_move": final_move,
        "first_appearance_step": first_appearance,
        "stabilization_step": stabilization,
        "is_stable_from_start": first_appearance == 0 and stabilization == 0,
        "raw_length": len(best_moves),
        "min_post_first_A_halt_reward": min(halt_rewards[first_appearance:]),
        "halt_reward_at_first_A": halt_rewards[first_appearance],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _plot_top_level(records: list[dict[str, Any]], out_path: Path) -> dict[str, int]:
    counts = Counter(str(record["top_level_category"]) for record in records)
    categories = ["A", "X*A", "X*AB*A"]
    values = [counts.get(category, 0) for category in categories]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(categories, values, color=["tab:blue", "tab:orange", "tab:red"])
    ax.set_ylabel("Source paths")
    ax.set_title("Root-Preference Churn Motifs")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {category: counts.get(category, 0) for category in categories}


def _plot_lengths(records: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    categories = ["A", "X*A", "X*AB*A"]
    max_length = max((int(record["compressed_length"]) for record in records), default=1)
    lengths = list(range(1, max_length + 1))
    width = 0.25
    x = np.arange(len(lengths), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    summary: dict[str, Any] = {}
    for offset_index, category in enumerate(categories):
        counts = [
            sum(1 for record in records if str(record["top_level_category"]) == category and int(record["compressed_length"]) == length)
            for length in lengths
        ]
        offset = (offset_index - 1) * width
        ax.bar(x + offset, counts, width=width, label=category)
        summary[category] = {str(length): count for length, count in zip(lengths, counts) if count > 0}
    ax.set_xticks(x)
    ax.set_xticklabels([str(length) for length in lengths])
    ax.set_xlabel("Compressed sequence length")
    ax.set_ylabel("Source paths")
    ax.set_title("Compressed Length by Top-Level Motif")
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    top_level_counts = Counter(str(record["top_level_category"]) for record in records)
    exact_motif_counts = Counter(str(record["canonical_motif"]) for record in records)
    lengths_by_category: dict[str, Counter[int]] = defaultdict(Counter)
    unique_moves_by_category: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        category = str(record["top_level_category"])
        lengths_by_category[category][int(record["compressed_length"])] += 1
        unique_moves_by_category[category][int(record["num_unique_moves"])] += 1

    return {
        "num_source_paths": len(records),
        "top_level_counts": dict(top_level_counts),
        "exact_motif_counts": dict(exact_motif_counts),
        "length_distribution_by_category": {
            category: {str(length): count for length, count in sorted(counter.items())}
            for category, counter in sorted(lengths_by_category.items())
        },
        "unique_move_distribution_by_category": {
            category: {str(length): count for length, count in sorted(counter.items())}
            for category, counter in sorted(unique_moves_by_category.items())
        },
        "mean_raw_length": statistics.mean(int(record["raw_length"]) for record in records) if records else 0.0,
        "stable_from_start_fraction": (
            statistics.mean(1.0 if bool(record["is_stable_from_start"]) else 0.0 for record in records) if records else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze source-level root-action churn from raw oracle traces.")
    parser.add_argument("--source-root", required=True, help="Root directory containing raw pretrain examples.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary JSON, JSONL, and plots.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of source files to analyze.")
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = _iter_source_paths(source_root)
    if args.limit > 0:
        source_paths = source_paths[:args.limit]
    if not source_paths:
        raise ValueError(f"No raw examples found under {source_root}")

    records: list[dict[str, Any]] = []
    started = time.time()
    for index, path in enumerate(source_paths, start=1):
        example = load_pretrain_example(str(path))
        if not isinstance(example, PretrainExample):
            continue
        record = _source_record(path, example)
        if record is not None:
            records.append(record)
        if args.log_interval > 0 and (index % args.log_interval == 0 or index == len(source_paths)):
            elapsed = time.time() - started
            print(f"processed={index}/{len(source_paths)} kept={len(records)} elapsed_s={elapsed:.1f}", flush=True)

    _write_jsonl(output_dir / "root_action_churn_sources.jsonl", records)
    summary = _summarize(records)
    summary["top_level_plot"] = _plot_top_level(records, output_dir / "root_action_churn_top_level.png")
    summary["length_plot"] = _plot_lengths(records, output_dir / "root_action_churn_lengths.png")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "top_level_counts="
        + ", ".join(f"{key}={summary['top_level_counts'].get(key, 0)}" for key in ["A", "X*A", "X*AB*A"]),
        flush=True,
    )
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
