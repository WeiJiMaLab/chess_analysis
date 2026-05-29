"""Root-action churn motifs and lucky-early-halt detection.

Replays the teacher search on each source pretrain example to get the
per-step best-move trace, classifies the compressed sequence into the
A / X*A / X*AB*A motif categories, and aggregates per-episode "stop
phase" + "lucky early halt" stats used in the lab-notebook investigation
into how the oracle's halt timing lines up with root-move stabilization.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cts.analysis._budgeted._shared import (
    ANALYSIS_FIGURE_DPI,
    PretrainExample,
    build_trimmed_decision_episode,
    torch,
)
from cts.analysis._common import (
    TeacherSearchConfig,
    _canonicalize_final_anchored_motif,
    _compressed_move_sequence,
    _first_appearance_step,
    _stabilization_step,
)

# Pretrain-example loader is only needed inside ``_reconstruct_episode_move_trace``,
# but goes through the same torch-dependent import chain as the other CTS helpers.
try:
    from cts.data.preprocess_gnn.teacher_targets import load_pretrain_example
except ModuleNotFoundError:
    load_pretrain_example = None


# Tolerance used by the "lucky early halt" heuristic in _root_action_churn_records:
# the oracle's optimal stop step is flagged "lucky" only when a later step
# would have delivered a halt reward at least this much lower. 0.01 is large
# enough to skip noise (rewards round to ~3 sig figs) and small enough that
# any deliberately-lower halt reward still triggers the flag.
LUCKY_EARLY_HALT_EPS = 0.01


def _reconstruct_episode_move_trace(
    source_path: str,
    quality_config: TeacherSearchConfig,
) -> dict[str, Any]:
    """Like ``_reconstruct_episode_best_moves`` but also returns per-step halt rewards and final Q values."""
    if torch is None or build_trimmed_decision_episode is None or PretrainExample is None:
        raise RuntimeError("Torch/CTS episode helpers are unavailable; cannot reconstruct move traces.")
    example = load_pretrain_example(source_path)
    if not isinstance(example, PretrainExample):
        raise ValueError(f"Expected PretrainExample at {source_path}, got {type(example).__name__}.")
    episode = build_trimmed_decision_episode(example, quality_config)
    halt_rewards = [float(episode.final_root_q_values[move]) for move in episode.best_moves]
    return {
        "best_moves": list(episode.best_moves),
        "halt_rewards": halt_rewards,
        "final_root_q_values": dict(episode.final_root_q_values),
    }


def _move_trace_cache(
    diagnostics: list[dict[str, Any]],
    quality_config: TeacherSearchConfig,
) -> dict[str, dict[str, Any]]:
    """Cache the full move trace (moves + halt rewards) per source path."""
    cache: dict[str, dict[str, Any]] = {}
    for source_path in sorted({str(episode["source_path"]) for episode in diagnostics}):
        cache[source_path] = _reconstruct_episode_move_trace(source_path, quality_config)
    return cache


def _root_action_churn_records(
    diagnostics: list[dict[str, Any]],
    move_trace_cache: dict[str, dict[str, Any]],
    *,
    lucky_eps: float = LUCKY_EARLY_HALT_EPS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (per-source, per-episode) records describing root-move churn patterns.

    Used by the lab-notebook investigation into whether the oracle's optimal
    halt step lines up with when the root's best move stabilizes. Returns
    ``(source_records, episode_records)``: source records describe the
    move-trace motif (A / X*A / X*AB*A and friends); episode records add
    which phase the oracle's halt falls in and whether it was "lucky".
    """
    source_records: list[dict[str, Any]] = []
    source_lookup: dict[str, dict[str, Any]] = {}
    for source_path, trace in sorted(move_trace_cache.items()):
        best_moves = [str(move) for move in trace["best_moves"]]
        if not best_moves:
            continue
        compressed = _compressed_move_sequence(best_moves)
        final_move = compressed[-1]
        first_appearance = _first_appearance_step(best_moves, final_move)
        stabilization = _stabilization_step(best_moves, final_move)
        # Three high-level motif categories: the trace either never changed
        # (A), changed once before settling (X*A), or oscillated back and
        # forth before settling (X*AB*A).
        top_level_category = "A" if len(compressed) == 1 else "X*A" if compressed.count(final_move) == 1 else "X*AB*A"
        record = {
            "source_path": source_path,
            "compressed_sequence": compressed,
            "canonical_motif": _canonicalize_final_anchored_motif(compressed),
            "top_level_category": top_level_category,
            "compressed_length": len(compressed),
            "num_unique_moves": len(set(compressed)),
            "final_move": final_move,
            "first_appearance_step": first_appearance,
            "stabilization_step": stabilization,
            "is_stable_from_start": first_appearance == 0 and stabilization == 0,
        }
        source_records.append(record)
        source_lookup[source_path] = {
            **record,
            "best_moves": best_moves,
            "halt_rewards": [float(value) for value in trace["halt_rewards"]],
        }

    episode_records: list[dict[str, Any]] = []
    for episode in diagnostics:
        source_path = str(episode["source_path"])
        source_record = source_lookup.get(source_path)
        if source_record is None:
            continue
        oracle_stop = int(episode["oracle_stop_step"])
        best_moves = source_record["best_moves"]
        halt_rewards = source_record["halt_rewards"]
        final_move = str(source_record["final_move"])
        oracle_stop_move = str(best_moves[oracle_stop])
        first_appearance = int(source_record["first_appearance_step"])
        stabilization = int(source_record["stabilization_step"])
        # Three phases of the move trace; the halt either happens before the
        # final move has even appeared, after it appeared but before churn
        # stabilizes, or after it has fully stabilized.
        if oracle_stop < first_appearance:
            stop_phase = "before_first_A"
        elif oracle_stop < stabilization:
            stop_phase = "after_first_A_before_stabilization"
        else:
            stop_phase = "after_stabilization"
        # Lucky early halt: oracle halts on the final move before stabilization
        # AND a later halt would have gotten a meaningfully worse halt reward.
        lucky_early_halt = (
            oracle_stop_move == final_move
            and oracle_stop < stabilization
            and min(halt_rewards[oracle_stop:]) < halt_rewards[oracle_stop] - lucky_eps
        )
        episode_records.append(
            {
                "path": str(episode["path"]),
                "source_path": source_path,
                "budget_bucket_name": str(episode["budget_bucket_name"]),
                "starting_budget": int(episode["starting_budget"]),
                "oracle_stop_step": oracle_stop,
                "predicted_stop_step": int(episode["predicted_stop_step"]),
                "oracle_stop_move": oracle_stop_move,
                "final_move": final_move,
                "top_level_category": str(source_record["top_level_category"]),
                "canonical_motif": str(source_record["canonical_motif"]),
                "compressed_length": int(source_record["compressed_length"]),
                "num_unique_moves": int(source_record["num_unique_moves"]),
                "first_appearance_step": first_appearance,
                "stabilization_step": stabilization,
                "oracle_stop_phase": stop_phase,
                "oracle_stop_matches_final_move": oracle_stop_move == final_move,
                "lucky_early_halt": lucky_early_halt,
            }
        )
    return source_records, episode_records


def _summarize_root_action_churn(
    source_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate motif counts, length distributions, and lucky-halt rates from churn records."""
    top_level_counts = Counter(str(record["top_level_category"]) for record in source_records)
    exact_motif_counts = Counter(str(record["canonical_motif"]) for record in source_records)
    lengths_by_category: dict[str, Counter[int]] = defaultdict(Counter)
    unique_moves_by_category: dict[str, Counter[int]] = defaultdict(Counter)
    for record in source_records:
        category = str(record["top_level_category"])
        lengths_by_category[category][int(record["compressed_length"])] += 1
        unique_moves_by_category[category][int(record["num_unique_moves"])] += 1

    stop_phase_counts = Counter(str(record["oracle_stop_phase"]) for record in episode_records)
    lucky_early_halts = [record for record in episode_records if bool(record["lucky_early_halt"])]

    return {
        "num_source_paths": len(source_records),
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
        "oracle_stop_phase_counts": dict(stop_phase_counts),
        "lucky_early_halt_episodes": len(lucky_early_halts),
        "lucky_early_halt_fraction": len(lucky_early_halts) / max(len(episode_records), 1),
    }


def _plot_root_action_churn_top_level(source_records: list[dict[str, Any]], out_path: Path) -> dict[str, int]:
    """Bar plot of the three top-level motif categories (A / X*A / X*AB*A)."""
    counts = Counter(str(record["top_level_category"]) for record in source_records)
    categories = ["A", "X*A", "X*AB*A"]
    values = [counts.get(category, 0) for category in categories]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(categories, values, color=["tab:blue", "tab:orange", "tab:red"])
    ax.set_ylabel("Source paths")
    ax.set_title("Root-Preference Churn Motifs")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return {category: counts.get(category, 0) for category in categories}


def _plot_root_action_churn_lengths(source_records: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Grouped-bar plot of compressed-sequence length distribution per motif category."""
    categories = ["A", "X*A", "X*AB*A"]
    max_length = max((int(record["compressed_length"]) for record in source_records), default=1)
    lengths = list(range(1, max_length + 1))
    width = 0.25
    x = np.arange(len(lengths), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    summary: dict[str, Any] = {}
    for offset_index, category in enumerate(categories):
        counts = [
            sum(
                1
                for record in source_records
                if str(record["top_level_category"]) == category and int(record["compressed_length"]) == length
            )
            for length in lengths
        ]
        ax.bar(x + (offset_index - 1) * width, counts, width=width, label=category)
        summary[category] = {str(length): count for length, count in zip(lengths, counts) if count > 0}
    ax.set_xticks(x)
    ax.set_xticklabels([str(length) for length in lengths])
    ax.set_xlabel("Compressed sequence length")
    ax.set_ylabel("Source paths")
    ax.set_title("Motif Length Distribution")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary
