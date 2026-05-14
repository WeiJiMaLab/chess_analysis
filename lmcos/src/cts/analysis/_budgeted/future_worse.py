"""Future-worse halt analysis: same-move vs move-switch error split.

When the oracle halts because ``future_value_gain <= 0``, this module
asks whether the controller's mistake correlates with switching to a
different root move or merely halting at the wrong step on the same
move. Replays the teacher search to get per-episode best-move
sequences, then renders the two-panel fraction + regret comparison.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cts.analysis._budgeted._shared import (
    ANALYSIS_FIGURE_DPI,
    FIGSIZE_TWO_PANEL_WIDE,
    PretrainExample,
    build_trimmed_decision_episode,
    torch,
)
from cts.analysis._common import (
    TeacherSearchConfig,
    _episode_regret_decomposition,
)
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig

# Pretrain-example loader is only needed inside ``_reconstruct_episode_best_moves``,
# but goes through the same torch-dependent import chain as the other CTS helpers.
try:
    from cts.data.preprocess_gnn.teacher_targets import load_pretrain_example
except ModuleNotFoundError:
    load_pretrain_example = None


def _reconstruct_episode_best_moves(
    source_path: str,
    quality_config: TeacherSearchConfig,
) -> list[str]:
    """Replay the teacher search on the source pretrain example to get per-step best moves."""
    if torch is None or build_trimmed_decision_episode is None or PretrainExample is None:
        raise RuntimeError("Torch/CTS episode helpers are unavailable; cannot reconstruct best moves.")
    example = load_pretrain_example(source_path)
    if not isinstance(example, PretrainExample):
        raise ValueError(f"Expected PretrainExample at {source_path}, got {type(example).__name__}.")
    episode = build_trimmed_decision_episode(example, quality_config)
    return list(episode.best_moves)


def _best_move_sequences(
    diagnostics: list[dict[str, Any]],
    quality_config: TeacherSearchConfig,
) -> dict[str, list[str]]:
    """Cache best-move sequences keyed by source path; each source is replayed at most once."""
    cache: dict[str, list[str]] = {}
    for source_path in sorted({str(episode["source_path"]) for episode in diagnostics}):
        cache[source_path] = _reconstruct_episode_best_moves(source_path, quality_config)
    return cache


def _future_worse_move_switch_summary(
    diagnostics: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    best_move_cache: dict[str, list[str]],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, Any]:
    """Among ``future_already_worse``-halt episodes, split by whether the predicted move differs from the oracle's.

    Investigates the lab-notebook question: when the oracle halts because the
    future is worse, does the controller's mistake correlate with switching
    to a different root move, or does it merely halt at the wrong step on
    the same move?
    """
    stop_rows = {
        str(row["path"]): row
        for row in step_rows
        if bool(row["at_oracle_stop"]) and str(row["oracle_stop_driver"]) == "future_already_worse"
    }
    cases: list[dict[str, Any]] = []
    for episode in diagnostics:
        path = str(episode["path"])
        stop_row = stop_rows.get(path)
        if stop_row is None:
            continue
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        source_path = str(episode["source_path"])
        best_moves = best_move_cache[source_path]
        oracle_move = best_moves[oracle_stop]
        predicted_move = best_moves[predicted_stop]
        move_relation = "move_switch" if oracle_move != predicted_move else "same_move"
        decomposition = _episode_regret_decomposition(episode, oracle_config)
        cases.append(
            {
                "budget_bucket_name": str(episode["budget_bucket_name"]),
                "delta": predicted_stop - oracle_stop,
                "move_relation": move_relation,
                "oracle_move": oracle_move,
                "predicted_move": predicted_move,
                "halt_reward_term": float(decomposition["halt_reward_term"]),
                "maintenance_term": float(decomposition["maintenance_term"]),
                "time_term": float(decomposition["time_term"]),
                "regret": float(decomposition["saved_regret"]),
                "future_value_gain": float(stop_row["future_value_gain"]),
                "maintenance_cost": float(stop_row["maintenance_cost"]),
                "time_cost": float(stop_row["time_cost"]),
                "false_continue": bool(stop_row["false_continue"]),
            }
        )

    relation_counts = Counter(case["move_relation"] for case in cases)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_bucket[str(case["budget_bucket_name"])].append(case)
        by_relation[str(case["move_relation"])].append(case)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean-aggregate the key regret-decomposition fields plus false-continue rate."""
        return {
            "episodes": len(rows),
            "mean_regret": statistics.mean(float(row["regret"]) for row in rows),
            "mean_halt_reward_term": statistics.mean(float(row["halt_reward_term"]) for row in rows),
            "mean_maintenance_term": statistics.mean(float(row["maintenance_term"]) for row in rows),
            "mean_time_term": statistics.mean(float(row["time_term"]) for row in rows),
            "mean_delta": statistics.mean(float(row["delta"]) for row in rows),
            "false_continue_rate": statistics.mean(1.0 if row["false_continue"] else 0.0 for row in rows),
        }

    by_bucket_relation: dict[str, dict[str, Any]] = {}
    for bucket, rows in sorted(by_bucket.items()):
        relation_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            relation_map[str(row["move_relation"])].append(row)
        by_bucket_relation[bucket] = {relation: summarize(rel_rows) for relation, rel_rows in sorted(relation_map.items())}

    return {
        "overall_relation_counts": dict(relation_counts),
        "overall": summarize(cases) if cases else {},
        "by_relation": {relation: summarize(rows) for relation, rows in sorted(by_relation.items())},
        "by_bucket_relation": by_bucket_relation,
    }


def _plot_future_worse_move_switch(
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Two-panel plot of the same-move/move-switch split for future-worse halts."""
    bucket_summary = summary["by_bucket_relation"]
    buckets = list(bucket_summary)
    same_move = [bucket_summary[bucket].get("same_move", {}).get("episodes", 0) for bucket in buckets]
    move_switch = [bucket_summary[bucket].get("move_switch", {}).get("episodes", 0) for bucket in buckets]
    totals = np.maximum(np.array(same_move, dtype=np.float32) + np.array(move_switch, dtype=np.float32), 1.0)
    same_frac = np.array(same_move, dtype=np.float32) / totals
    switch_frac = np.array(move_switch, dtype=np.float32) / totals

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL_WIDE, constrained_layout=True)
    x = np.arange(len(buckets))
    axes[0].bar(x, same_frac, label="same_move", color="tab:blue")
    axes[0].bar(x, switch_frac, bottom=same_frac, label="move_switch", color="tab:red")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets, rotation=25)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Future-Worse Episodes: Same Move vs Move Switch")
    axes[0].legend()

    same_regret = [bucket_summary[bucket].get("same_move", {}).get("mean_regret", 0.0) for bucket in buckets]
    switch_regret = [bucket_summary[bucket].get("move_switch", {}).get("mean_regret", 0.0) for bucket in buckets]
    width = 0.35
    axes[1].bar(x - width / 2, same_regret, width=width, label="same_move", color="tab:blue")
    axes[1].bar(x + width / 2, switch_regret, width=width, label="move_switch", color="tab:red")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buckets, rotation=25)
    axes[1].set_ylabel("Mean regret")
    axes[1].set_title("Future-Worse Regret by Move Relation")
    axes[1].legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
