"""Cross-theme helpers and figure-styling constants for the budgeted-controller analyzer.

Holds the per-file constants promoted in the recent cleanup commit
(``ANALYSIS_FIGURE_DPI``, the ``FIGSIZE_*`` tuples, calibration / partial-
dependence bin edges) plus the small handful of utility helpers used by
two or more themed sub-modules (state-row builder, episode-error
classifier, source-path availability check, JSON / JSONL writers).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

# Optional CTS deps: only needed for analyses that reconstruct the underlying
# search-tree move trace from the source pretrain example. When unavailable
# (e.g. on a workstation without the full CTS env), those analyses are skipped
# instead of crashing the whole report.
try:
    import torch
    from cts.data.episode_envs import build_trimmed_decision_episode
    from cts.data.preprocess_gnn.teacher_targets import (
        PretrainExample,
        TeacherSearchConfig,
        load_pretrain_example,
    )
except ModuleNotFoundError:
    torch = None
    build_trimmed_decision_episode = None
    PretrainExample = None
    TeacherSearchConfig = None
    load_pretrain_example = None


# --- Figure styling constants ------------------------------------------------
# These mirror what cts.analysis._common will eventually own (a sibling agent
# is creating that file). Until it lands, the analysis-driver figures source
# their dpi + figsize values from here so individual plot helpers reference
# named constants instead of literal tuples.

# Raster DPI for every saved figure. 180 is the project standard: sharp enough
# for the lab notebook + paper figures without bloating the PNG payload.
ANALYSIS_FIGURE_DPI = 180
# Three-up loss/metric panel row (used by _plot_loss_curves).
FIGSIZE_LOSS_THREE_PANEL = (15, 4.5)
# Two-up calibration / partial-dependence row.
FIGSIZE_TWO_PANEL_WIDE = (12, 4.5)
# Single-axis figures default to one of these aspect ratios depending on
# whether they're a tall stacked-bar or a wide line plot.
FIGSIZE_SINGLE_WIDE = (9, 5)
FIGSIZE_SINGLE_TALL = (8, 5)

# --- Calibration / partial-dependence bin edges ------------------------------
# Edges for the |predicted advantage| calibration plot. Coarser at the high
# end because anything past 0.5 is "confidently correct" territory; the
# interesting calibration variance lives in [0, 0.5).
CALIBRATION_BIN_EDGES = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, math.inf]
# Edges for the partial-dependence heat-map on the time-budget (T_t) axis.
# Doubling roughly matches the long-tailed budget distribution.
PARTIAL_DEPENDENCE_TIME_BIN_EDGES = [1, 2, 3, 5, 10, 20, 40, 80, math.inf]


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    """Load the per-episode diagnostics JSONL into a list of dicts."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_state_rows(
    diagnostics: list[dict[str, Any]],
    *,
    max_rows: int | None = 500000,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Flatten episodes into per-step state rows, optionally reservoir-sampled.

    Many of the calibration / sign-accuracy / heat-map plots want one row per
    (episode, step), but with hundreds of thousands of episodes that blows
    memory. Reservoir-sample with a fixed seed so the same diagnostics file
    always yields the same plot. Returns ``(rows, total_rows_seen)`` so
    callers can report the population size alongside the sample.

    Args:
        diagnostics: per-episode dicts as loaded from the diagnostics JSONL.
        max_rows: cap on retained rows; ``None``/``<=0`` disables sampling.
        seed: RNG seed for the reservoir sampler.
    """
    rows: list[dict[str, Any]] = []
    total_rows = 0
    rng = random.Random(seed)
    for episode in diagnostics:
        for step_index, (tree_size, time_budget, pred_adv, target_adv) in enumerate(
            zip(
                episode["tree_sizes"],
                episode["time_budgets"],
                episode["predicted_advantages"],
                episode["target_advantages"],
            )
        ):
            pred_continue = float(pred_adv) > 0.0
            target_continue = float(target_adv) > 0.0
            row = {
                "source_path": episode["source_path"],
                "path": episode["path"],
                "budget_bucket_name": episode["budget_bucket_name"],
                "starting_budget": int(episode["starting_budget"]),
                "oracle_stop_step": int(episode["oracle_stop_step"]),
                "predicted_stop_step": int(episode["predicted_stop_step"]),
                "episode_regret": float(episode["regret"]),
                "step_index": step_index,
                "tree_size": int(tree_size),
                "time_budget": int(time_budget),
                "predicted_advantage": float(pred_adv),
                "target_advantage": float(target_adv),
                "pred_continue": pred_continue,
                "target_continue": target_continue,
                "sign_correct": pred_continue == target_continue,
                "state_error_type": (
                    "false_continue"
                    if pred_continue and not target_continue
                    else "false_halt"
                    if (not pred_continue) and target_continue
                    else "correct_continue"
                    if pred_continue
                    else "correct_halt"
                ),
            }
            total_rows += 1
            # Reservoir sampling: fill until full, then replace existing rows
            # with probability max_rows / total_rows_seen.
            if max_rows is None or max_rows <= 0 or len(rows) < max_rows:
                rows.append(row)
            else:
                replacement_index = rng.randrange(total_rows)
                if replacement_index < max_rows:
                    rows[replacement_index] = row
    return rows, total_rows


def _episode_error_type(episode: dict[str, Any]) -> str:
    """Classify the controller's stop step vs the oracle's: exact / over / under."""
    pred = int(episode["predicted_stop_step"])
    oracle = int(episode["oracle_stop_step"])
    if pred == oracle:
        return "exact"
    if pred > oracle:
        return "oversearch"
    return "undersearch"


def _source_paths_available(diagnostics: list[dict[str, Any]]) -> bool:
    """True iff every source-tree path referenced by ``diagnostics`` exists on disk."""
    source_paths = sorted({str(episode["source_path"]) for episode in diagnostics})
    return bool(source_paths) and all(Path(path).exists() for path in source_paths)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Pretty-print the full analysis payload as ``summary.json``."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append-style JSONL writer for the per-episode/per-step record dumps."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
