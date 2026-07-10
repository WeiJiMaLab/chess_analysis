"""Pack budget-augmented controller episodes into tensorized shards.

This is the controller data packing stage of the CTS pipeline. It reads a
pretrain split (already split into train/validation by root FEN), runs
``BudgetedControllerOracle`` on each source tree at multiple sampled starting
budgets, and writes packed episode shards (one trajectory per source tree,
many oracle episodes per trajectory) grouped under a per-split JSON manifest.

The shards are consumed downstream by ``materialize_controller_cache.py``
(which encodes nodes into the controller's input cache) and then by
``train_fitted_q_controller.py`` for training. Supports per-tree filtering
(``--min-halt-reward-range``, ``--exclude-xaba``), tree-level stratified
sampling, budget-sensitivity weighting, and ``ProcessPoolExecutor`` workers.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from cts.core.schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts.data.preprocess_gnn.teacher_targets import (
    EdgeStats,
    RawPretrainExampleRecord,
    TeacherSearchConfig,
    _backpropagate_path,
    _flip_wdl_target,
    _maybe_static_node_wdl,
    load_raw_pretrain_record,
)
from cts.data.preprocess_mc.oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    BudgetedOraclePolicy,
    budgeted_oracle_metadata,
    compute_budgeted_oracle,
    deterministic_starting_budgets,
    has_strong_budgeted_margins,
    return_for_stop_step,
)


class PackControllerEpisodesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split"
    output_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed"
    shard_size: int = 500
    num_workers: int = 0
    log_interval: int = 10
    reward_scale: float = 1.0
    min_halt_reward_range: float = 0.0
    min_decision_margin: float = 0.0
    exclude_xaba: bool = False
    sample_trees_by_budget_score: bool = False
    sample_trees_by_tree_strata: bool = False
    tree_stratification_mode: str = "dj"
    search_budget: int = 64
    max_depth: int = 10
    c_puct: float = 1.0
    maintenance_scale: float = 0.0
    maintenance_ref_nodes: float = 30.0
    maintenance_exponent: float = 1.1
    time_lambda: float = 18.537
    time_p: float = 2.8
    time_tau: float = 2.5
    time_delta: int = 1
    timeout_value: float = -1.0
    time_mode: str = "power_law"
    samples_per_bucket: int = 2
    fixed_budget: Optional[int] = None  # if set, every episode gets this one starting budget (1 per
                                         # tree), bypassing the 5-bucket stratification entirely
    seed: int = 0
    scramble_min_time: int = 1
    scramble_max_time: int = 3
    medium_small_min_time: int = 4
    medium_small_max_time: int = 10
    medium_large_min_time: int = 11
    medium_large_max_time: int = 25
    large_min_time: int = 26
    large_max_time: int = 60
    very_large_min_time: int = 61
    very_large_max_time: int = 120
    downsample_trivial: float = 1.0
    clear: bool = False

# Per-tree keep probability under budget-sensitivity sampling is linearly
# interpolated from this floor (at score=0) up to 1.0 (at score=max). The
# floor exists so the lowest-sensitivity trees still contribute ~25% of their
# expected mass: they're informative for calibration / negative examples even
# when the controller doesn't need to "decide" much on them, and dropping
# them entirely would make the packed split's budget-score distribution a
# top-heavy slice that no longer covers the easy-tree regime.
_BUDGET_SCORE_KEEP_FLOOR = 0.25

# Tree-level stratified sampling partitions the source-tree distribution into
# this many quantile bins per stratifying axis (stop-depth excess "d",
# budget-action variance "j"). 3 bins gives a low/mid/high split that's
# coarse enough to keep each bin well-populated for an inverse-frequency
# reweight, but fine enough that the rare top bin actually gets upweighted.
_TREE_STRATIFICATION_NUM_BINS = 3
# Mixture weight between the natural per-bin distribution (weight=0 → keep
# everything) and a fully uniform-over-bins distribution (weight=1 → keep
# prob inversely proportional to bin count). 0.5 halves the gap toward
# uniform — empirically the sweet spot that lifts rare strata without
# losing most of the dominant-stratum trees.
_TREE_STRATIFICATION_MIXTURE_WEIGHT = 0.5

# In "dj" mode, within each d-bin we multiply per-tree keep probabilities by
# these j-bin multipliers (low/mid/high j-bin). The multipliers favor the
# high-j tail (trees where budget actually changes the optimal action) and
# are then renormalized per d-bin so the bin's total expected mass is
# preserved — the multipliers redistribute *within* a d-bin, they don't
# shift mass across d-bins.
_TREE_STRATIFICATION_J_MULTIPLIERS = (1.0, 1.25, 1.5)


def _feature_schema() -> NodeFeatureSchema:
    """Return the canonical encoder feature schema; thin alias for clarity at call sites."""
    return tree_encoder_feature_schema()


def _quality_config(config: PackControllerEpisodesConfig) -> TeacherSearchConfig:
    """Build the teacher search config used to score halt rewards on each tree."""
    return TeacherSearchConfig(
        max_depth=config.max_depth,
        search_budget=config.search_budget,
        c_puct=config.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def _oracle_config(config: PackControllerEpisodesConfig) -> BudgetedOracleConfig:
    """Build the budgeted oracle config from CLI args.

    ``fixed_budget`` collapses the usual 5-bucket stratification to a single bucket
    (one starting budget for every episode, one episode per tree) — see
    outputs/reports/normative.md (R-EVALUATE): the per-episode budget is a hard cap (92% of
    packed episodes just run to exhaustion of their own sampled budget), which is the right
    design for the bucket-stratified regret analyses but conflates budget with episode length
    for a single-cost-regime assessment. Otherwise builds all 5 bucket ranges as before.
    """
    if config.fixed_budget is not None:
        return BudgetedOracleConfig(
            maintenance_scale=config.maintenance_scale,
            maintenance_ref_nodes=config.maintenance_ref_nodes,
            maintenance_exponent=config.maintenance_exponent,
            time_lambda=config.time_lambda,
            time_p=config.time_p,
            time_tau=config.time_tau,
            time_delta=config.time_delta,
            timeout_value=config.timeout_value,
            time_mode=config.time_mode,
            budget_buckets=(BudgetBucket("fixed", config.fixed_budget, config.fixed_budget),),
            samples_per_bucket=1,
            seed=config.seed,
        )
    return BudgetedOracleConfig(
        maintenance_scale=config.maintenance_scale,
        maintenance_ref_nodes=config.maintenance_ref_nodes,
        maintenance_exponent=config.maintenance_exponent,
        time_lambda=config.time_lambda,
        time_p=config.time_p,
        time_tau=config.time_tau,
        time_delta=config.time_delta,
        timeout_value=config.timeout_value,
        time_mode=config.time_mode,
        budget_buckets=(
            BudgetBucket("scramble", config.scramble_min_time, config.scramble_max_time),
            BudgetBucket("medium-small", config.medium_small_min_time, config.medium_small_max_time),
            BudgetBucket("medium-large", config.medium_large_min_time, config.medium_large_max_time),
            BudgetBucket("large", config.large_min_time, config.large_max_time),
            BudgetBucket("very-large", config.very_large_min_time, config.very_large_max_time),
        ),
        samples_per_bucket=config.samples_per_bucket,
        seed=config.seed,
    )


def _read_manifest(path: Path) -> List[Path]:
    """Read a newline-delimited manifest of example file paths. Raises on empty/missing."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        examples = [Path(line.strip()) for line in handle if line.strip()]
    if not examples:
        raise ValueError(f"No example paths found in manifest: {path}")
    return examples


def _halt_reward_range(halt_rewards: List[float]) -> float:
    """Span of the per-step halt rewards. Used as the trivial-tree filter signal."""
    if not halt_rewards:
        raise ValueError("halt_rewards must be non-empty.")
    return float(max(halt_rewards) - min(halt_rewards))


def _accepted_episode_count(result: Optional[dict[str, Any]]) -> int:
    """Count packed episodes in a per-tree result; 0 if the result was filtered out."""
    if not result:
        return 0
    return len(result.get("episodes", []))


def _compressed_move_sequence(best_moves: List[str]) -> List[str]:
    """Run-length collapse consecutive duplicates: ['A','A','B','A'] → ['A','B','A']."""
    compressed: List[str] = []
    for move in best_moves:
        if not compressed or compressed[-1] != move:
            compressed.append(move)
    return compressed


def _source_top_level_root_churn_category(record: Any) -> Optional[str]:
    """Classify a tree's root-best-move trace into one of three churn motifs.

    Returns "A" (no churn — best move stable), "X*A" (one switch ending on
    a settled move), or "X*AB*A" (churn ending on a move that was revisited).
    Used by ``--exclude-xaba`` to drop the noisiest motif. Returns None if
    the record carries no move sequence.
    """
    # Two record formats coexist: newer records carry an int index into
    # oracle_root_moves; older ones carry a string trace directly.
    if hasattr(record, "oracle_best_move_index"):
        best_moves = [
            str(record.oracle_root_moves[int(index)])
            for index in record.oracle_best_move_index.tolist()
        ]
    else:
        best_moves = [str(move) for move in record.oracle_best_move_trace]
    if not best_moves:
        return None
    compressed = _compressed_move_sequence(best_moves)
    final_move = compressed[-1]
    if len(compressed) == 1:
        return "A"
    if compressed.count(final_move) == 1:
        return "X*A"
    return "X*AB*A"


def _tree_stop_depth_excess(episodes: List[dict[str, Any]]) -> float:
    """Mean of ``max(oracle_stop_step - 1, 0)`` across episodes; the "d" stratification axis."""
    if not episodes:
        return 0.0
    return float(
        np.mean(
            np.asarray([max(int(episode["oracle_stop_step"]) - 1, 0) for episode in episodes], dtype=np.float32)
        )
    )


def _tree_budget_action_variance(episodes: List[dict[str, Any]]) -> float:
    """Mean per-step Bernoulli variance of the continue action across budgets; the "j" stratification axis.

    For each step index, computes p*(1-p) over the indicator "advantage > 0"
    aggregated across episodes that reach that step. Quantifies how much the
    optimal action varies with starting budget — high variance trees are the
    ones where budget actually matters.
    """
    if not episodes:
        return 0.0
    max_steps = max(len(episode["target_advantages"]) for episode in episodes)
    variances: List[float] = []
    for step_index in range(max_steps):
        continue_actions = [
            1.0 if float(episode["target_advantages"][step_index]) > 0.0 else 0.0
            for episode in episodes
            if step_index < len(episode["target_advantages"])
        ]
        # Need at least 2 samples for a meaningful Bernoulli variance estimate.
        if len(continue_actions) < 2:
            continue
        continue_probability = float(np.mean(np.asarray(continue_actions, dtype=np.float32)))
        variances.append(continue_probability * (1.0 - continue_probability))
    if not variances:
        return 0.0
    return float(np.mean(np.asarray(variances, dtype=np.float32)))


def _tree_stratification_stats(episodes: List[dict[str, Any]]) -> tuple[float, float]:
    """Bundle the (d, j) stratification axes for a single tree's episode set."""
    return _tree_stop_depth_excess(episodes), _tree_budget_action_variance(episodes)


def _tree_budget_sensitivity_score(
    episodes: List[dict[str, Any]],
    tree_sizes: List[int],
    oracle_config: BudgetedOracleConfig,
) -> float:
    """Score how much a tree benefits from a budget-aware policy vs. a fixed stop step.

    Compares the mean oracle value across budgets (budget-aware) against the
    best constant-stop-step value (budget-blind upper bound). Trees with
    large positive scores are the ones where the controller most needs to
    condition on remaining budget. Used by ``--sample-trees-by-budget-score``.
    """
    if not episodes or not tree_sizes:
        return 0.0
    budget_aware_values: List[float] = []
    max_shared_stop = -1
    for episode in episodes:
        halt_rewards = [float(value) for value in episode["halt_rewards"]]
        if not halt_rewards:
            continue
        budget_aware_values.append(float(episode["oracle_value"]))
        max_shared_stop = max(max_shared_stop, len(halt_rewards) - 1)
    if not budget_aware_values or max_shared_stop < 0:
        return 0.0
    budget_aware_value = float(np.mean(np.asarray(budget_aware_values, dtype=np.float32)))
    # Budget-blind baseline: for each candidate constant stop step, average
    # the realized return across episodes, then take the best stop step.
    budget_blind_value = max(
        float(
            np.mean(
                np.asarray(
                    [
                        return_for_stop_step(
                            halt_rewards,
                            tree_sizes[: len(halt_rewards)],
                            [int(episode["starting_budget"]) - idx for idx in range(len(halt_rewards))],
                            min(stop_step, len(halt_rewards) - 1),
                            oracle_config,
                        )
                        for episode in episodes
                        for halt_rewards in [[float(value) for value in episode["halt_rewards"]]]
                        if halt_rewards
                    ],
                    dtype=np.float32,
                )
            )
        )
        for stop_step in range(max_shared_stop + 1)
    )
    return budget_aware_value - budget_blind_value


def _tree_budget_sensitivity_keep_probability(score: float, max_score: float) -> float:
    """Linearly interpolate from ``_BUDGET_SCORE_KEEP_FLOOR`` (score=0) to 1.0 (score=max_score)."""
    if max_score <= 0.0:
        return 1.0
    clamped_score = min(max(float(score), 0.0), float(max_score))
    normalized = clamped_score / float(max_score)
    return _BUDGET_SCORE_KEEP_FLOOR + (1.0 - _BUDGET_SCORE_KEEP_FLOOR) * normalized


def _quantile_thresholds(values: List[float], num_bins: int) -> List[float]:
    """Return ``num_bins - 1`` cut points dividing ``values`` into equal-mass bins."""
    if not values or num_bins <= 1:
        return []
    quantiles = [bin_index / num_bins for bin_index in range(1, num_bins)]
    return [float(value) for value in np.quantile(np.asarray(values, dtype=np.float32), quantiles)]


def _quantile_bin(value: float, thresholds: List[float]) -> int:
    """Bucket a scalar by walking ascending thresholds; returns the bin index in ``[0, len(thresholds)]``."""
    bin_index = 0
    for threshold in thresholds:
        if value > threshold:
            bin_index += 1
    return bin_index


def _tree_stratum_key(
    stop_depth_excess: float,
    budget_action_variance: float,
    stop_depth_thresholds: List[float],
    budget_action_thresholds: List[float],
    stratification_mode: str,
) -> str:
    """Map a tree's (d, j) stats to a stratum string used as a stratification key."""
    d_bin = _quantile_bin(stop_depth_excess, stop_depth_thresholds)
    j_bin = _quantile_bin(budget_action_variance, budget_action_thresholds)
    if stratification_mode == "d":
        return f"d{d_bin}"
    if stratification_mode == "j":
        return f"j{j_bin}"
    return f"d{d_bin}_j{j_bin}"


def _tree_stratified_keep_probability(
    stratum_count: int,
    total_trees: int,
    num_nonempty_strata: int,
    max_relative_mass: float,
) -> float:
    """Convert a stratum's raw count into a per-tree keep probability via the mixture formula.

    The "relative mass" formula linearly interpolates between the natural
    distribution (mixture_weight=0 → keep all) and a fully uniform-over-strata
    distribution (mixture_weight=1 → keep prob inversely proportional to
    stratum count). Normalizing by the max keeps every probability in [0, 1].
    """
    if stratum_count <= 0 or total_trees <= 0 or num_nonempty_strata <= 0 or max_relative_mass <= 0.0:
        return 1.0
    relative_mass = (1.0 - _TREE_STRATIFICATION_MIXTURE_WEIGHT) + (
        _TREE_STRATIFICATION_MIXTURE_WEIGHT * float(total_trees) / (float(num_nonempty_strata) * float(stratum_count))
    )
    return min(relative_mass / max_relative_mass, 1.0)


def _compute_tree_stratified_keep_probabilities_from_rows(
    rows: List[dict[str, Any]],
    stratification_mode: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute per-tree keep probabilities for "d", "j", or "dj" stratification.

    Returns (path → keep_probability, metadata dict for the manifest). The
    "dj" mode is the most involved: first bin trees by stop-depth, then
    within each d-bin compute local j-bin quantile thresholds and reweight
    via the J_MULTIPLIERS, preserving each d-bin's total mass.

    Args:
        rows: per-tree stat rows produced by ``_tree_stats_one_task``;
            each row has ``source_path``, ``stop_depth_excess``, and
            ``budget_action_variance``.
        stratification_mode: one of "d", "j", "dj".
    """
    # Empty-input short circuit: nothing to stratify, return an empty mapping
    # and a minimal metadata block so the manifest still has the schema fields.
    if not rows:
        return {}, {
            "tree_stratification_mode": stratification_mode,
            "tree_stop_depth_thresholds": [],
            "tree_budget_action_variance_thresholds": [],
            "tree_strata_counts": {},
            "tree_stratification_num_bins": _TREE_STRATIFICATION_NUM_BINS,
            "tree_stratification_mixture_weight": _TREE_STRATIFICATION_MIXTURE_WEIGHT,
            "tree_stratification_j_multipliers": list(_TREE_STRATIFICATION_J_MULTIPLIERS),
        }

    # Global quantile thresholds for both axes; these define the bin
    # boundaries that every per-row classification uses.
    stop_depth_thresholds = _quantile_thresholds(
        [float(row["stop_depth_excess"]) for row in rows],
        _TREE_STRATIFICATION_NUM_BINS,
    )
    budget_action_thresholds = _quantile_thresholds(
        [float(row["budget_action_variance"]) for row in rows],
        _TREE_STRATIFICATION_NUM_BINS,
    )

    # ---- d-axis classification (used by all modes) ----------------------
    total_trees = len(rows)
    d_bin_by_path: dict[str, int] = {}
    d_counts: dict[int, int] = {}
    rows_by_d_bin: dict[int, List[dict[str, Any]]] = {}
    for row in rows:
        path_str = str(row["source_path"])
        d_bin = _quantile_bin(float(row["stop_depth_excess"]), stop_depth_thresholds)
        d_bin_by_path[path_str] = d_bin
        d_counts[d_bin] = d_counts.get(d_bin, 0) + 1
        rows_by_d_bin.setdefault(d_bin, []).append(row)

    # Per-d-bin keep probabilities (also reused as the base in "dj" mode).
    num_nonempty_d_bins = len(d_counts)
    d_relative_masses = {
        d_bin: (1.0 - _TREE_STRATIFICATION_MIXTURE_WEIGHT)
        + (_TREE_STRATIFICATION_MIXTURE_WEIGHT * float(total_trees) / (float(num_nonempty_d_bins) * float(count)))
        for d_bin, count in d_counts.items()
    }
    max_d_relative_mass = max(d_relative_masses.values()) if d_relative_masses else 1.0
    d_keep_probabilities = {
        d_bin: _tree_stratified_keep_probability(
            count,
            total_trees,
            num_nonempty_d_bins,
            max_d_relative_mass,
        )
        for d_bin, count in d_counts.items()
    }

    # ---- "d" mode: stop-depth only ---------------------------------------
    if stratification_mode == "d":
        keep_probabilities = {
            str(row["source_path"]): d_keep_probabilities[d_bin_by_path[str(row["source_path"])]]
            for row in rows
        }
        return keep_probabilities, {
            "tree_stratification_mode": stratification_mode,
            "tree_stop_depth_thresholds": stop_depth_thresholds,
            "tree_budget_action_variance_thresholds": budget_action_thresholds,
            "tree_strata_counts": {f"d{d_bin}": count for d_bin, count in sorted(d_counts.items())},
            "tree_stratification_num_bins": _TREE_STRATIFICATION_NUM_BINS,
            "tree_stratification_mixture_weight": _TREE_STRATIFICATION_MIXTURE_WEIGHT,
            "tree_stratification_j_multipliers": list(_TREE_STRATIFICATION_J_MULTIPLIERS),
        }

    # ---- "j" mode: budget-action variance only ---------------------------
    if stratification_mode == "j":
        j_bin_by_path: dict[str, int] = {}
        j_counts: dict[int, int] = {}
        for row in rows:
            path_str = str(row["source_path"])
            j_bin = _quantile_bin(float(row["budget_action_variance"]), budget_action_thresholds)
            j_bin_by_path[path_str] = j_bin
            j_counts[j_bin] = j_counts.get(j_bin, 0) + 1
        num_nonempty_j_bins = len(j_counts)
        j_relative_masses = {
            j_bin: (1.0 - _TREE_STRATIFICATION_MIXTURE_WEIGHT)
            + (_TREE_STRATIFICATION_MIXTURE_WEIGHT * float(total_trees) / (float(num_nonempty_j_bins) * float(count)))
            for j_bin, count in j_counts.items()
        }
        max_j_relative_mass = max(j_relative_masses.values()) if j_relative_masses else 1.0
        keep_probabilities = {
            path_str: _tree_stratified_keep_probability(
                j_counts[j_bin_by_path[path_str]],
                total_trees,
                num_nonempty_j_bins,
                max_j_relative_mass,
            )
            for path_str in j_bin_by_path
        }
        return keep_probabilities, {
            "tree_stratification_mode": stratification_mode,
            "tree_stop_depth_thresholds": stop_depth_thresholds,
            "tree_budget_action_variance_thresholds": budget_action_thresholds,
            "tree_strata_counts": {f"j{j_bin}": count for j_bin, count in sorted(j_counts.items())},
            "tree_stratification_num_bins": _TREE_STRATIFICATION_NUM_BINS,
            "tree_stratification_mixture_weight": _TREE_STRATIFICATION_MIXTURE_WEIGHT,
            "tree_stratification_j_multipliers": list(_TREE_STRATIFICATION_J_MULTIPLIERS),
        }

    # ---- "dj" mode: nested stratification --------------------------------
    # For each d-bin, compute local j-bin quantiles and reweight trees by
    # the J_MULTIPLIERS, normalized so each d-bin's total expected mass is
    # preserved (mean multiplier across the d-bin equals 1 after scaling).
    keep_probabilities: dict[str, float] = {}
    stratum_counts: dict[str, int] = {}
    j_thresholds_by_d_bin: dict[str, List[float]] = {}
    for d_bin, d_rows in rows_by_d_bin.items():
        base_keep_probability = d_keep_probabilities[d_bin]
        # If the d-bin already has prob 1.0, no further reweighting can help;
        # keep every tree and skip the j-bin quantile work.
        if base_keep_probability >= 1.0:
            for row in d_rows:
                path_str = str(row["source_path"])
                keep_probabilities[path_str] = 1.0
                stratum_key = f"d{d_bin}_j0"
                stratum_counts[stratum_key] = stratum_counts.get(stratum_key, 0) + 1
            j_thresholds_by_d_bin[f"d{d_bin}"] = []
            continue

        # Within this d-bin only, recompute j-bin quantile thresholds so
        # rare-j-bin behavior is preserved when one d-bin has a skewed j-distribution.
        d_budget_action_thresholds = _quantile_thresholds(
            [float(row["budget_action_variance"]) for row in d_rows],
            _TREE_STRATIFICATION_NUM_BINS,
        )
        j_thresholds_by_d_bin[f"d{d_bin}"] = d_budget_action_thresholds
        j_counts: dict[int, int] = {}
        j_bin_by_path: dict[str, int] = {}
        for row in d_rows:
            path_str = str(row["source_path"])
            j_bin = _quantile_bin(float(row["budget_action_variance"]), d_budget_action_thresholds)
            j_bin_by_path[path_str] = j_bin
            j_counts[j_bin] = j_counts.get(j_bin, 0) + 1
        # Mass-preserving normalizer: divide each per-tree multiplier by the
        # mean multiplier within the d-bin so the bin's total expected
        # accept count stays unchanged after j-bin reweighting.
        mean_multiplier = sum(
            float(j_counts[j_bin]) * _TREE_STRATIFICATION_J_MULTIPLIERS[j_bin]
            for j_bin in j_counts
        ) / float(len(d_rows))
        for row in d_rows:
            path_str = str(row["source_path"])
            j_bin = j_bin_by_path[path_str]
            keep_probabilities[path_str] = min(
                base_keep_probability * (_TREE_STRATIFICATION_J_MULTIPLIERS[j_bin] / mean_multiplier),
                1.0,
            )
            stratum_key = f"d{d_bin}_j{j_bin}"
            stratum_counts[stratum_key] = stratum_counts.get(stratum_key, 0) + 1

    return keep_probabilities, {
        "tree_stratification_mode": stratification_mode,
        "tree_stop_depth_thresholds": stop_depth_thresholds,
        "tree_budget_action_variance_thresholds": budget_action_thresholds,
        "tree_strata_counts": dict(sorted(stratum_counts.items())),
        "tree_stratification_num_bins": _TREE_STRATIFICATION_NUM_BINS,
        "tree_stratification_mixture_weight": _TREE_STRATIFICATION_MIXTURE_WEIGHT,
        "tree_stratification_j_multipliers": list(_TREE_STRATIFICATION_J_MULTIPLIERS),
        "tree_budget_action_variance_thresholds_by_d_bin": j_thresholds_by_d_bin,
    }


def _deterministic_unit_interval(identifier: str, seed: int) -> float:
    """Hash ``(seed, identifier)`` into a stable uniform draw in [0, 1).

    Used to make stratification and downsampling fully reproducible — the
    same identifier always produces the same accept/reject decision.
    """
    digest = hashlib.blake2b(f"{seed}:{identifier}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)


def _ordered_expansion_parent_ids(record: RawPretrainExampleRecord) -> List[int]:
    """Reconstruct expansion order from a raw record's CSR child arrays.

    Same logic as ``SearchTree.ordered_expansion_parent_ids``: because
    children are assigned contiguous ids at expansion, sorting by the
    first child's id reconstructs the original expansion sequence.
    """
    child_ptr = record.child_ptr.tolist()
    children_index = record.children_index.tolist()
    expansion_parents: List[tuple[int, int]] = []
    for node_id in range(int(record.parent_index.shape[0])):
        start = int(child_ptr[node_id])
        end = int(child_ptr[node_id + 1])
        if bool(record.is_expanded[node_id].item()) and end > start:
            expansion_parents.append((int(children_index[start]), node_id))
    expansion_parents.sort()
    return [node_id for _, node_id in expansion_parents]


# The static per-node value feature backed up at leaves during PUCT search
# (matches ``TeacherSearchConfig.value_feature``'s default in teacher_targets.py).
_VALUE_FEATURE_NAME = "value"


class _RecordFeatureNode:
    """Duck-typed stand-in for ``teacher_targets.SearchNode`` exposing only ``scalar_features``."""

    __slots__ = ("scalar_features",)

    def __init__(self, scalar_features: Dict[str, float]) -> None:
        self.scalar_features = scalar_features


class _RecordFeatureTree:
    """Duck-typed stand-in for ``teacher_targets.SearchTree`` exposing only ``get_node``.

    ``_maybe_static_node_wdl`` (imported from ``teacher_targets.py``) only ever calls
    ``tree.get_node(node_id).scalar_features`` -- this adapter supplies exactly that,
    read straight off the raw record's own dense feature matrix, without paying for a
    full ``RawPretrainExampleRecord.to_pretrain_example()`` rehydration (FEN replay via
    python-chess, metadata, one ``SearchNode`` per node) just to read four scalars per
    expansion event. Relies on ``RawPretrainExampleRecord._node_scalar_feature_dicts``
    (a "private" cross-module attribute access, same coupling-risk category as the
    ``teacher_targets._*`` imports at the top of this file -- see history.md's Deferred
    section).
    """

    def __init__(self, record: RawPretrainExampleRecord) -> None:
        self._nodes = [_RecordFeatureNode(features) for features in record._node_scalar_feature_dicts()]

    def get_node(self, node_id: int) -> _RecordFeatureNode:
        return self._nodes[node_id]


def _ancestor_edge_path(parent_index: List[int], node_id: int) -> List[Tuple[int, int]]:
    """Return the root-to-``node_id`` ancestor edge path, in root-to-leaf order.

    This is exactly the ``path`` that ``TreeSearch.generate``'s PUCT leaf selection
    takes when ``node_id`` gets picked as the leaf to expand: a leaf's root-to-leaf
    path is nothing more than its own ancestor chain, which ``parent_index`` already
    fixes uniquely regardless of *how* PUCT scoring chose it. That is precisely what
    makes retroactive replay of a *real expansion's* own backup possible without
    re-running leaf selection at all -- see ``_replay_backprop_history``'s docstring
    for the one category of backprop event this does NOT recover (repeat visits to an
    already-terminal node) and why that is an acceptable, small, well-characterized gap
    rather than something worth chasing with a full PUCT re-simulation.
    """
    path: List[Tuple[int, int]] = []
    current = node_id
    while parent_index[current] != -1:
        parent = parent_index[current]
        path.append((parent, current))
        current = parent
    path.reverse()
    return path


def _replay_backprop_history(
    record: RawPretrainExampleRecord,
    expansion_parent_ids: List[int],
    value_feature: str = _VALUE_FEATURE_NAME,
) -> Dict[str, np.ndarray]:
    """Retroactively replay a tree's own PUCT backprop history from data already on disk.

    This is the core of ``packhistory_trees``: the fix for the frozen-per-step-value bug
    described in history.md. No re-generation is needed because every *real* PUCT
    expansion event's backup is fully determined by information the final tree already
    carries:

      - *which* node got expanded, and in what order (``expansion_parent_ids``, from
        ``_ordered_expansion_parent_ids``);
      - the root-to-that-node ancestor path being backed up, which is fixed by
        ``parent_index`` alone (``_ancestor_edge_path``) -- PUCT's leaf-selection
        scoring only decided *which* leaf to expand next, never what path a given
        leaf's own backup takes, since a leaf's path back to root is unique in a tree;
      - the leaf's own static value/WDL (a per-node feature baked in at node creation
        and never mutated afterward -- see ``PUCTSearch.on_expand`` in
        teacher_targets.py -- read via ``_maybe_static_node_wdl``, imported not
        duplicated).

    Reuses ``EdgeStats``/``_backpropagate_path`` (imported from teacher_targets.py, not
    duplicated) to replay the exact same running-mean bookkeeping generation itself used.

    Root/creation order fix: newly created children are seeded into ``edge_stats`` in
    ASCENDING NODE ID order (via ``parent_index`` directly), not
    ``children_index[child_ptr[...]]`` CSR order -- CSR order is sorted by UCI move
    string in this raw format, not creation order, and node ids are assigned
    sequentially at creation time, so ascending id is the one true creation order.

    What this does NOT recover, and why that is the right tradeoff: PUCT leaf selection
    (``_select_leaf_by_puct`` in teacher_targets.py) can, and empirically does,
    repeatedly re-select an already-permanently-terminal node as the best leaf (e.g. a
    discovered forced mate) without ever expanding it again. Each such repeat visit
    still runs a full backprop up its ancestor path but creates no new node, so it
    leaves no trace in the final tree structure or in ``expansion_parent_ids`` -- this
    function does not attempt to recover those "wasted" backprop events. Doing so would
    require re-running the exact same deterministic PUCT leaf-selection procedure that
    produced them, using each node's *static* ``prior`` feature -- but that feature is
    only available post-hoc as float16 (this raw format's on-disk dtype, see
    ``RawPretrainExampleRecord.__post_init__``), a precision loss the live search itself
    never had. An earlier version of this function did attempt the full PUCT
    re-simulation and was found, via this module's own validation against real
    ``human_trees``, to occasionally (on ~1/40 sampled trees) diverge onto a *completely
    different* branch after a near-tied PUCT score got decided the other way by float16
    rounding -- a large, unbounded, hard-to-predict error, strictly worse than the
    small gap from just not chasing terminal-revisit backprop mass. Measured directly
    (40 sampled ``human_trees``, every step x every root child, ~94k comparisons against
    the tree's own stored ``oracle_root_q_trace``) for the simpler approach implemented
    here: median error 0.0, p95 = 1.5e-4, p99 = 2.2e-4, only 0.16% of comparisons above
    1e-3, concentrated in a small number of terminal-revisit-heavy trees (one outlier
    tree reached 0.056; every other sampled tree's worst step was <= 0.004). This
    matches, and gives fuller context for, the "~1e-4 residual" already flagged in
    history.md's Deferred section. See this stage's Progress Log entry in history.md
    for the full measurement and the schema-deviation note (this function ended up
    NOT needing a direct call to ``_flip_wdl_target``, since ``_backpropagate_path``
    already applies it internally).

    Returns a dict of parallel numpy arrays for the sparse update-log schema
    (``step_index``, ``node_id``, ``visit_count``, ``q_value``, ``wdl``), already sorted
    by ``(node_id, step_index)`` ascending (achieved for free by iterating node ids in
    order, since each node's own update history is already time-ordered by
    construction), plus the CSR ``node_update_ptr`` index and each node's *final*
    (as of the last replayed step) value/WDL for convenience.

    ``step_index`` convention -- read this before querying "value as of step t" from
    the log, it is a common off-by-one trap: ``step_index`` is the 0-indexed position
    of an expansion event within ``expansion_parent_ids`` (i.e. within
    ``record.oracle_trace_expansion_counts``, whose values are the 1-indexed
    ``num_expansions`` count). A log row with ``step_index == k`` reflects the state
    immediately after the ``(k + 1)``-th real expansion -- i.e. it lines up with
    ``record.oracle_root_q_trace[k]`` / ``record.oracle_trace_expansion_counts[k]``
    (== ``k + 1``) directly, at the *same* 0-indexed row ``k``, not ``k + 1``. A
    forward-fill "value as of step t" query into this log should therefore be called
    with ``t = k`` to reproduce oracle row ``k`` -- calling it with ``t = k + 1``
    (e.g. by mistakenly treating ``step_index`` as 1-indexed, matching
    ``oracle_trace_expansion_counts``' own values instead of its *index*) silently
    includes one extra expansion's worth of backprop and can produce spurious,
    sometimes large, mismatches against ``oracle_root_q_trace`` that look like a
    replay-accuracy bug but are actually a caller-side indexing bug. (Confirmed
    directly against one such apparent mismatch during this stage's own validation:
    error 0.0058 querying with the off-by-one convention at a step where the correctly
    -indexed query gives error 0.0 exactly -- see this stage's Progress Log entry in
    history.md.)
    """
    num_nodes = int(record.parent_index.shape[0])
    parent_index = record.parent_index.tolist()

    # Children in ascending node id order -- see the "Root/creation order fix" note
    # above. Iterating node_id ascending and appending yields exactly this order.
    children_by_parent: Dict[int, List[int]] = {}
    for node_id in range(1, num_nodes):
        children_by_parent.setdefault(parent_index[node_id], []).append(node_id)

    feature_tree = _RecordFeatureTree(record)
    edge_stats: Dict[Tuple[int, int], EdgeStats] = {}
    updates_by_node: Dict[int, List[Tuple[int, int, float, Tuple[float, float, float]]]] = {}

    for step_index, parent_id in enumerate(expansion_parent_ids):
        # Seed the newly created children's edges, mirroring PUCTSearch.on_expand.
        # These do not themselves get a log row: a freshly-born edge has
        # visit_count == 0 until some later expansion backs a value up through it.
        for child_id in children_by_parent.get(parent_id, []):
            edge_stats[(parent_id, child_id)] = EdgeStats()

        path = _ancestor_edge_path(parent_index, parent_id)
        if not path:
            # The root's own expansion event: no incoming edge, nothing to back up.
            continue

        leaf_value = float(feature_tree.get_node(parent_id).scalar_features[value_feature])
        leaf_wdl = _maybe_static_node_wdl(feature_tree, parent_id)
        _backpropagate_path(edge_stats, path, leaf_value, leaf_wdl)

        for ancestor_id, descendant_id in path:
            stats = edge_stats[(ancestor_id, descendant_id)]
            updates_by_node.setdefault(descendant_id, []).append(
                (step_index, stats.visit_count, stats.q_value, stats.mean_wdl)
            )

    step_index_col: List[int] = []
    node_id_col: List[int] = []
    visit_count_col: List[int] = []
    q_value_col: List[float] = []
    wdl_col: List[Tuple[float, float, float]] = []
    node_update_ptr = [0]
    final_value = np.zeros(num_nodes, dtype=np.float32)
    final_wdl = np.zeros((num_nodes, 3), dtype=np.float32)
    for node_id in range(num_nodes):
        rows = updates_by_node.get(node_id, [])
        for step, visit_count, q_value, wdl in rows:
            step_index_col.append(step)
            node_id_col.append(node_id)
            visit_count_col.append(visit_count)
            q_value_col.append(q_value)
            wdl_col.append(wdl)
        if rows:
            _, _, last_q_value, last_wdl = rows[-1]
            final_value[node_id] = last_q_value
            final_wdl[node_id] = last_wdl
        # Nodes with no update-log rows (never visited by any backprop event --
        # 88.8%/97.7% of nodes on the two trees measured during planning) keep the
        # EdgeStats()-zero default here, exactly matching what a live query against
        # generation's own edge_stats would have returned for an unvisited edge (see
        # teacher_targets._root_q_values_from_edge_stats's explicit 0.0 fallback).
        node_update_ptr.append(len(step_index_col))

    return {
        "update_log_step_index": np.asarray(step_index_col, dtype=np.int32),
        "update_log_node_id": np.asarray(node_id_col, dtype=np.int32),
        "update_log_visit_count": np.asarray(visit_count_col, dtype=np.int32),
        "update_log_q_value": np.asarray(q_value_col, dtype=np.float32),
        "update_log_wdl": (
            np.asarray(wdl_col, dtype=np.float32).reshape(-1, 3)
            if wdl_col
            else np.zeros((0, 3), dtype=np.float32)
        ),
        "node_update_ptr": np.asarray(node_update_ptr, dtype=np.int32),
        "final_value": final_value,
        "final_wdl": final_wdl,
    }


def build_compact_trajectory(
    record: RawPretrainExampleRecord,
    schema: NodeFeatureSchema | None = None,
    *,
    source_path: str = "",
) -> Optional[dict[str, Any]]:
    """Convert a raw pretrain record into the per-trajectory dict used at pack time.

    The returned ``halt_rewards`` and ``tree_sizes`` are the precomputed inputs passed
    to ``compute_budgeted_oracle`` when building controller episodes.
    """
    if schema is None:
        schema = _feature_schema()
    return _build_compact_trajectory(source_path, record, schema)


def _build_compact_trajectory(
    path_str: str,
    record: RawPretrainExampleRecord,
    schema: NodeFeatureSchema,
) -> Optional[dict[str, Any]]:
    """Convert a raw pretrain record into the per-trajectory dict packed into shards.

    Strips pre-root expansions: the controller's first decision happens
    when the search has reached the root, so we trim everything before the
    root's own expansion. Returns None if the record has no expansions or
    no root expansion.
    """
    expansion_parent_ids = _ordered_expansion_parent_ids(record)
    if not expansion_parent_ids:
        return None

    # The root expansion's position in the expansion order is the controller's
    # "step 0" — everything before it is teacher search overhead, not
    # something the controller has to decide about.
    try:
        root_rank = expansion_parent_ids.index(0)
    except ValueError:
        return None

    # Sanity-check that the record's oracle traces align with the expansion
    # sequence we just reconstructed. If they don't, the record is malformed.
    if int(record.oracle_best_move_index.shape[0]) != len(expansion_parent_ids):
        raise ValueError("oracle_best_move_index must align with the expansion sequence.")
    trace_counts = [int(value) for value in record.oracle_trace_expansion_counts.tolist()]
    if trace_counts != list(range(1, len(expansion_parent_ids) + 1)):
        raise ValueError("oracle_trace_expansion_counts must be contiguous positive counts.")

    # Build per-step node-count cutoffs by replaying expansions in order.
    # Each step adds a parent's children to the live node count.
    full_tree = record.to_tensorized_tree_example(schema)
    child_counts = (record.child_ptr[1:] - record.child_ptr[:-1]).to(dtype=torch.long)
    node_cutoffs: List[int] = []
    current_nodes = 1
    for parent_id in expansion_parent_ids:
        current_nodes += int(child_counts[parent_id].item())
        node_cutoffs.append(current_nodes)

    # Trim everything before the root's expansion: the controller only sees
    # snapshots from step ``root_rank`` onward.
    trimmed_node_cutoffs = node_cutoffs[root_rank:]
    trimmed_best_move_index = record.oracle_best_move_index.to(dtype=torch.long)[root_rank:]
    trimmed_halt_rewards = record.oracle_final_root_q_values.to(dtype=torch.float32)[trimmed_best_move_index]

    # Retroactively replay this tree's own backprop history (the actual bug fix --
    # see _replay_backprop_history's docstring). ``full_tree.node_features`` as
    # returned by ``to_tensorized_tree_example`` carries the STATIC, one-shot
    # per-node provider features (each node's own network eval, baked in at node
    # creation and never mutated -- see teacher_targets.value_features_from_wdl);
    # packing that directly, unchanged, for every step is exactly the bug this
    # stage fixes. The "value"/"wdl_win"/"wdl_draw"/"wdl_loss" columns are
    # overwritten below with each node's *final* replayed backed-up value/WDL
    # (its own incoming edge's EdgeStats, i.e. the same quantity
    # oracle_root_q_trace records for root children) so that even a consumer that
    # ignores the sparse update log below gets the converged, correct value
    # rather than the never-updated static eval. "wdl_var" is left untouched: the
    # update-log schema (per history.md) tracks only visit_count/q_value/wdl, not
    # a variance term, and wdl_var is a property of the node's own static WDL
    # distribution, not something backprop revises.
    #
    # Per-STEP (not just final) values are not densely materialized here -- doing
    # so would be an O(steps * nodes * features) blow-up per tree. Instead this
    # function packs (a) this static/final-state feature matrix and (b) the full
    # sparse update log (below), so downstream per-step consumption
    # (packhistory_MCmaterialize / packhistory_GNNpretrain) does an O(log M)
    # forward-fill lookup into the log rather than reading a pre-materialized
    # dense per-step tensor. See this stage's Progress Log entry in history.md for
    # this interpretation note.
    replay = _replay_backprop_history(record, expansion_parent_ids, value_feature=_VALUE_FEATURE_NAME)
    node_features = full_tree.node_features.clone()
    feature_columns = {name: index for index, name in enumerate(schema.feature_names)}
    if _VALUE_FEATURE_NAME in feature_columns:
        node_features[:, feature_columns[_VALUE_FEATURE_NAME]] = torch.from_numpy(replay["final_value"]).to(
            dtype=node_features.dtype
        )
    wdl_feature_names = ("wdl_win", "wdl_draw", "wdl_loss")
    if all(name in feature_columns for name in wdl_feature_names):
        for wdl_index, name in enumerate(wdl_feature_names):
            node_features[:, feature_columns[name]] = torch.from_numpy(replay["final_wdl"][:, wdl_index]).to(
                dtype=node_features.dtype
            )

    return {
        "num_steps": len(trimmed_node_cutoffs),
        "source_path": path_str,
        "node_features": node_features.numpy(),
        "parent_index": full_tree.parent_index.numpy(),
        "depth": full_tree.depth.numpy(),
        "child_ptr": record.child_ptr.numpy(),
        "edge_child": record.children_index.numpy(),
        "edge_slot": full_tree.edge_slot.numpy(),
        "expansion_parent_ids": np.asarray(expansion_parent_ids, dtype=np.int32),
        "first_decision_expansion_count": root_rank + 1,
        "step_node_cutoffs": np.asarray(trimmed_node_cutoffs, dtype=np.int32),
        "halt_rewards": trimmed_halt_rewards.numpy(),
        "tree_sizes": np.asarray(trimmed_node_cutoffs, dtype=np.int64),
        # Sparse per-edge backprop update log (packhistory_GNNpretrain's only
        # input, per history.md) -- full untrimmed history (0-indexed by position
        # in ``expansion_parent_ids``, NOT re-based to ``root_rank`` the way the
        # step_node_cutoffs/tree_sizes above are), since GNN pretraining samples
        # over the full 96-step search budget, not just the controller's
        # post-root-expansion decision window.
        "update_log_step_index": replay["update_log_step_index"],
        "update_log_node_id": replay["update_log_node_id"],
        "update_log_visit_count": replay["update_log_visit_count"],
        "update_log_q_value": replay["update_log_q_value"],
        "update_log_wdl": replay["update_log_wdl"],
        "node_update_ptr": replay["node_update_ptr"],
    }


def budgeted_oracle_from_trajectory(
    trajectory: Mapping[str, Any],
    starting_budget: int,
    config: BudgetedOracleConfig,
) -> BudgetedOraclePolicy:
    """Run ``compute_budgeted_oracle`` on a ``build_compact_trajectory`` result."""
    num_steps = min(starting_budget, len(trajectory["halt_rewards"]))
    halt_rewards = [float(value) for value in trajectory["halt_rewards"][:num_steps]]
    tree_sizes = [int(value) for value in trajectory["tree_sizes"][:num_steps]]
    return compute_budgeted_oracle(halt_rewards, tree_sizes, num_steps, config)


def build_compact_trajectory_from_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str = "",
) -> Optional[dict[str, Any]]:
    """Load a raw ``.pt`` payload and build the controller trajectory dict."""
    record = RawPretrainExampleRecord.from_payload(payload)
    return build_compact_trajectory(record, source_path=source_path)


def _build_packed_tree_result(
    path_str: str,
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    feature_names: Tuple[str, ...],
    oracle_config: BudgetedOracleConfig,
) -> Optional[dict[str, Any]]:
    """Load one source tree, apply all per-tree filters, and pack one episode per sampled budget.

    Returns None when the tree is dropped by any of:
      - failing to load as a ``RawPretrainExampleRecord``;
      - matching the X*AB*A churn motif (when ``exclude_xaba`` is set);
      - having no expansions or no root expansion;
      - having a halt-reward range below ``min_halt_reward_range``;
      - producing zero episodes after the per-budget margin filter.

    On success, returns a dict with the trajectory, packed episodes, and
    stratification stats needed downstream.
    """
    record = load_raw_pretrain_record(path_str)
    if not isinstance(record, RawPretrainExampleRecord):
        return None
    if exclude_xaba and _source_top_level_root_churn_category(record) == "X*AB*A":
        return None
    trajectory = build_compact_trajectory(record, NodeFeatureSchema(feature_names), source_path=path_str)
    if trajectory is None or trajectory["num_steps"] <= 0:
        return None

    scaled_rewards = [float(reward_scale * reward) for reward in trajectory["halt_rewards"].tolist()]
    tree_sizes = [int(value) for value in trajectory["tree_sizes"].tolist()]
    # Trivial-tree filter: trees where halt reward barely changes across
    # the trajectory teach the controller nothing about when to stop.
    if _halt_reward_range(scaled_rewards) < min_halt_reward_range:
        return None

    # Run the oracle once per sampled starting budget. The deterministic
    # sampler keeps this reproducible across runs.
    packed_episodes: List[dict[str, Any]] = []
    for sampled_budget in deterministic_starting_budgets(path_str, oracle_config):
        # Per-budget margin filter: drop episodes where the oracle's halt/continue
        # decision is too close to a coin flip to be a useful supervision signal.
        if not has_strong_budgeted_margins(
            scaled_rewards,
            tree_sizes,
            sampled_budget.starting_budget,
            oracle_config,
            min_decision_margin,
        ):
            continue

        policy = compute_budgeted_oracle(
            scaled_rewards,
            tree_sizes,
            sampled_budget.starting_budget,
            oracle_config,
        )
        max_steps = len(policy.halt_rewards)
        episode_key = f"{path_str}#bucket={sampled_budget.bucket_name}#budget={sampled_budget.starting_budget}"
        packed_episodes.append(
            {
                "num_steps": max_steps,
                "target_advantages": np.asarray(policy.target_advantages, dtype=np.float32),
                "oracle_stop_step": policy.optimal_stop_step,
                "oracle_value": policy.oracle_value,
                "starting_budget": policy.starting_budget,
                "bucket_index": sampled_budget.bucket_index,
                "bucket_name": sampled_budget.bucket_name,
                "episode_key": episode_key,
                "source_path": path_str,
                "halt_rewards": list(policy.halt_rewards),
            }
        )
    if not packed_episodes:
        return None
    stop_depth_excess, budget_action_variance = _tree_stratification_stats(packed_episodes)
    return {
        "trajectory": trajectory,
        "episodes": packed_episodes,
        "stop_depth_excess": stop_depth_excess,
        "budget_action_variance": budget_action_variance,
        "budget_score": _tree_budget_sensitivity_score(
            packed_episodes,
            tree_sizes,
            oracle_config,
        ),
    }


def _compute_max_budget_score(
    example_paths: List[Path],
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    feature_names: Tuple[str, ...],
    oracle_config: BudgetedOracleConfig,
    num_workers: int,
) -> float:
    """First pass over the split: find the global max budget-sensitivity score.

    Used as the normalizer when ``--sample-trees-by-budget-score`` is enabled
    so that the highest-sensitivity tree gets keep probability 1.0 and the
    lowest gets ``_BUDGET_SCORE_KEEP_FLOOR``.
    """
    tasks = [
        (
            str(path),
            reward_scale,
            min_halt_reward_range,
            min_decision_margin,
            exclude_xaba,
            feature_names,
            oracle_config,
        )
        for path in example_paths
    ]
    max_budget_score = 0.0
    # Serial path mirrors the parallel one so debugging stack traces stay simple.
    if num_workers <= 0:
        for task in tasks:
            result = _score_one_task(task)
            max_budget_score = max(max_budget_score, float(result.get("budget_score", 0.0)))
        return max_budget_score

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_score_one_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            max_budget_score = max(max_budget_score, float(result.get("budget_score", 0.0)))
    return max_budget_score


def _compute_tree_stratified_keep_probabilities(
    example_paths: List[Path],
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    feature_names: Tuple[str, ...],
    oracle_config: BudgetedOracleConfig,
    num_workers: int,
    stratification_mode: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """First pass over the split: compute per-tree (d, j) stats and turn them into keep probabilities.

    Used when ``--sample-trees-by-tree-strata`` is enabled. Mirrors the
    serial/parallel structure of ``_compute_max_budget_score``.
    """
    tasks = [
        (
            str(path),
            reward_scale,
            min_halt_reward_range,
            min_decision_margin,
            exclude_xaba,
            feature_names,
            oracle_config,
        )
        for path in example_paths
    ]
    rows: List[dict[str, Any]] = []
    if num_workers <= 0:
        for task in tasks:
            result = _tree_stats_one_task(task)
            if result is not None:
                rows.append(result)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_tree_stats_one_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rows.append(result)

    return _compute_tree_stratified_keep_probabilities_from_rows(rows, stratification_mode)


def _print_shard_progress(
    *,
    split_name: str,
    shard_index: int,
    total_shards: int,
    completed_in_shard: int,
    total_tasks: int,
    results: List[Optional[dict[str, Any]]],
    total_episodes: int,
    total_skipped: int,
    total_tree_strata_filtered: int,
    total_budget_score_filtered: int,
    start_time: float,
) -> None:
    """Emit one progress line summarizing how many results in the in-flight shard fall in each accept/skip/filter bucket."""
    elapsed = time.time() - start_time
    accepted_so_far = total_episodes + sum(_accepted_episode_count(r) for r in results)
    skipped_so_far = total_skipped + sum(1 for r in results if not r)
    tree_strata_filtered_so_far = total_tree_strata_filtered + sum(
        1 for r in results if r and r.get("status") == "filtered_tree_strata"
    )
    budget_score_filtered_so_far = total_budget_score_filtered + sum(
        1 for r in results if r and r.get("status") == "filtered_budget_score"
    )
    print(
        f"split={split_name} shard={shard_index + 1}/{total_shards} "
        f"shard_progress={completed_in_shard}/{total_tasks} "
        f"accepted={accepted_so_far} skipped={skipped_so_far} "
        f"tree_strata_filtered={tree_strata_filtered_so_far} "
        f"budget_score_filtered={budget_score_filtered_so_far} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )


def _process_shard_tasks(
    tasks: List[tuple],
    *,
    num_workers: int,
    log_interval: int,
    split_name: str,
    shard_index: int,
    total_shards: int,
    total_episodes: int,
    total_skipped: int,
    total_tree_strata_filtered: int,
    total_budget_score_filtered: int,
    start_time: float,
) -> List[Optional[dict[str, Any]]]:
    """Process the shard's tasks serially or via a worker pool, returning per-task results in input order.

    The parallel branch pre-allocates the output list and writes into the
    per-task slot so the output order matches the input task order regardless
    of which worker finishes first.
    """
    results: List[Optional[dict[str, Any]]]
    if num_workers <= 0:
        results = []
        for completed_in_shard, task in enumerate(tasks, start=1):
            results.append(_process_one_task(task))
            if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                _print_shard_progress(
                    split_name=split_name,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    completed_in_shard=completed_in_shard,
                    total_tasks=len(tasks),
                    results=results,
                    total_episodes=total_episodes,
                    total_skipped=total_skipped,
                    total_tree_strata_filtered=total_tree_strata_filtered,
                    total_budget_score_filtered=total_budget_score_filtered,
                    start_time=start_time,
                )
    else:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one_task, task): idx for idx, task in enumerate(tasks)}
            for completed_in_shard, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                    _print_shard_progress(
                        split_name=split_name,
                        shard_index=shard_index,
                        total_shards=total_shards,
                        completed_in_shard=completed_in_shard,
                        total_tasks=len(tasks),
                        results=results,
                        total_episodes=total_episodes,
                        total_skipped=total_skipped,
                        total_tree_strata_filtered=total_tree_strata_filtered,
                        total_budget_score_filtered=total_budget_score_filtered,
                        start_time=start_time,
                    )
    return results


def _accumulate_shard_buffers(
    results: List[Optional[dict[str, Any]]],
) -> tuple[dict[str, Any], int, int, int, int]:
    """Walk per-task results and concatenate trajectory/episode tensors plus CSR ptr arrays.

    Returns ``(buffers, shard_episodes, skipped, tree_strata_filtered,
    budget_score_filtered)`` where ``buffers`` carries every list/scalar that
    the shard serializer needs. CSR-style "ptr" arrays let downstream readers
    slice out a single trajectory or episode without scanning the whole shard.
    """
    trajectory_node_ptr = [0]
    trajectory_edge_ptr = [0]
    trajectory_child_ptr_ptr = [0]
    trajectory_expansion_parent_ptr = [0]
    trajectory_step_ptr = [0]
    # CSR offsets into the shard-level concatenated update-log arrays (mirrors
    # trajectory_edge_ptr's role for edge_child) and into the shard-level
    # concatenated node_update_ptr array (mirrors trajectory_child_ptr_ptr's role
    # for child_ptr, since node_update_ptr, like child_ptr, has length N_tree + 1
    # per tree and stores LOCAL per-tree offsets).
    trajectory_update_log_ptr = [0]
    trajectory_node_update_ptr_ptr = [0]
    all_node_features: List[torch.Tensor] = []
    all_parent_index: List[torch.Tensor] = []
    all_edge_child: List[torch.Tensor] = []
    all_edge_slot: List[torch.Tensor] = []
    all_depth: List[torch.Tensor] = []
    all_child_ptr: List[torch.Tensor] = []
    all_expansion_parent_ids: List[torch.Tensor] = []
    all_step_node_cutoffs: List[torch.Tensor] = []
    all_trajectory_halt_rewards: List[torch.Tensor] = []
    all_first_decision_expansion_counts: List[int] = []
    all_update_log_step_index: List[torch.Tensor] = []
    all_update_log_node_id: List[torch.Tensor] = []
    all_update_log_visit_count: List[torch.Tensor] = []
    all_update_log_q_value: List[torch.Tensor] = []
    all_update_log_wdl: List[torch.Tensor] = []
    all_node_update_ptr: List[torch.Tensor] = []
    trajectory_source_paths: List[str] = []
    episode_step_ptr = [0]
    episode_trajectory_index: List[int] = []
    all_target_advantages: List[torch.Tensor] = []
    all_oracle_stop_steps: List[int] = []
    all_oracle_values: List[float] = []
    all_starting_budgets: List[int] = []
    all_bucket_indices: List[int] = []
    all_bucket_names: List[str] = []
    all_episode_keys: List[str] = []
    shard_episodes = 0
    skipped = 0
    tree_strata_filtered = 0
    budget_score_filtered = 0

    for raw_result in results:
        # Categorize each task's outcome and update the appropriate counter.
        if not raw_result:
            skipped += 1
            continue
        if raw_result.get("status") == "filtered_tree_strata":
            tree_strata_filtered += 1
            skipped += 1
            continue
        if raw_result.get("status") == "filtered_budget_score":
            budget_score_filtered += 1
            skipped += 1
            continue

        # Append the trajectory's CSR slices and update the ptr arrays.
        trajectory_data = _numpy_to_torch(raw_result["trajectory"])
        trajectory_index = len(trajectory_source_paths)
        trajectory_source_paths.append(trajectory_data["source_path"])
        trajectory_node_ptr.append(trajectory_node_ptr[-1] + int(trajectory_data["node_features"].shape[0]))
        trajectory_edge_ptr.append(trajectory_edge_ptr[-1] + int(trajectory_data["edge_child"].shape[0]))
        trajectory_child_ptr_ptr.append(trajectory_child_ptr_ptr[-1] + int(trajectory_data["child_ptr"].shape[0]))
        trajectory_expansion_parent_ptr.append(
            trajectory_expansion_parent_ptr[-1] + int(trajectory_data["expansion_parent_ids"].shape[0])
        )
        trajectory_step_ptr.append(trajectory_step_ptr[-1] + int(trajectory_data["step_node_cutoffs"].shape[0]))
        trajectory_update_log_ptr.append(
            trajectory_update_log_ptr[-1] + int(trajectory_data["update_log_step_index"].shape[0])
        )
        trajectory_node_update_ptr_ptr.append(
            trajectory_node_update_ptr_ptr[-1] + int(trajectory_data["node_update_ptr"].shape[0])
        )

        all_node_features.append(trajectory_data["node_features"])
        all_parent_index.append(trajectory_data["parent_index"])
        all_edge_child.append(trajectory_data["edge_child"])
        all_edge_slot.append(trajectory_data["edge_slot"])
        all_depth.append(trajectory_data["depth"])
        all_child_ptr.append(trajectory_data["child_ptr"])
        all_expansion_parent_ids.append(trajectory_data["expansion_parent_ids"])
        all_step_node_cutoffs.append(trajectory_data["step_node_cutoffs"])
        all_trajectory_halt_rewards.append(trajectory_data["halt_rewards"])
        all_first_decision_expansion_counts.append(int(trajectory_data["first_decision_expansion_count"]))
        all_update_log_step_index.append(trajectory_data["update_log_step_index"])
        all_update_log_node_id.append(trajectory_data["update_log_node_id"])
        all_update_log_visit_count.append(trajectory_data["update_log_visit_count"])
        all_update_log_q_value.append(trajectory_data["update_log_q_value"])
        all_update_log_wdl.append(trajectory_data["update_log_wdl"])
        all_node_update_ptr.append(trajectory_data["node_update_ptr"])

        # Append each episode that survived filtering, recording which
        # trajectory it came from so the cache materializer can join them.
        for augmented_result in raw_result["episodes"]:
            episode_data = _numpy_to_torch(augmented_result)
            num_steps = episode_data["num_steps"]
            episode_step_ptr.append(episode_step_ptr[-1] + num_steps)
            episode_trajectory_index.append(trajectory_index)

            all_target_advantages.append(episode_data["target_advantages"])
            all_oracle_stop_steps.append(episode_data["oracle_stop_step"])
            all_oracle_values.append(episode_data["oracle_value"])
            all_starting_budgets.append(episode_data["starting_budget"])
            all_bucket_indices.append(episode_data["bucket_index"])
            all_bucket_names.append(episode_data["bucket_name"])
            all_episode_keys.append(episode_data["episode_key"])
            shard_episodes += 1

    buffers = {
        "trajectory_node_ptr": trajectory_node_ptr,
        "trajectory_edge_ptr": trajectory_edge_ptr,
        "trajectory_child_ptr_ptr": trajectory_child_ptr_ptr,
        "trajectory_expansion_parent_ptr": trajectory_expansion_parent_ptr,
        "trajectory_step_ptr": trajectory_step_ptr,
        "trajectory_update_log_ptr": trajectory_update_log_ptr,
        "trajectory_node_update_ptr_ptr": trajectory_node_update_ptr_ptr,
        "all_node_features": all_node_features,
        "all_parent_index": all_parent_index,
        "all_edge_child": all_edge_child,
        "all_edge_slot": all_edge_slot,
        "all_depth": all_depth,
        "all_child_ptr": all_child_ptr,
        "all_expansion_parent_ids": all_expansion_parent_ids,
        "all_step_node_cutoffs": all_step_node_cutoffs,
        "all_trajectory_halt_rewards": all_trajectory_halt_rewards,
        "all_first_decision_expansion_counts": all_first_decision_expansion_counts,
        "all_update_log_step_index": all_update_log_step_index,
        "all_update_log_node_id": all_update_log_node_id,
        "all_update_log_visit_count": all_update_log_visit_count,
        "all_update_log_q_value": all_update_log_q_value,
        "all_update_log_wdl": all_update_log_wdl,
        "all_node_update_ptr": all_node_update_ptr,
        "trajectory_source_paths": trajectory_source_paths,
        "episode_step_ptr": episode_step_ptr,
        "episode_trajectory_index": episode_trajectory_index,
        "all_target_advantages": all_target_advantages,
        "all_oracle_stop_steps": all_oracle_stop_steps,
        "all_oracle_values": all_oracle_values,
        "all_starting_budgets": all_starting_budgets,
        "all_bucket_indices": all_bucket_indices,
        "all_bucket_names": all_bucket_names,
        "all_episode_keys": all_episode_keys,
    }
    return buffers, shard_episodes, skipped, tree_strata_filtered, budget_score_filtered


def _serialize_shard_payload(
    shard_path: Path,
    buffers: dict[str, Any],
    *,
    shard_episodes: int,
    schema: NodeFeatureSchema,
    reward_scale: float,
    oracle_config: BudgetedOracleConfig,
) -> None:
    """Concat buffered tensors and write the shard's ``.pt`` payload to disk."""
    payload = {
        # New format for the packhistory_trees stage (rewrite of mc_pack): adds the
        # sparse per-edge backprop update log fields below. Bumped from
        # "cts_budgeted_controller_episode_shard_v4" (the old mc_pack format, still
        # used by pre-existing mc_packed/ shards, untouched by this stage) so
        # consumers can tell the two apart.
        "format": "cts_packhistory_trees_shard_v1",
        "num_trajectories": len(buffers["trajectory_source_paths"]),
        "num_episodes": shard_episodes,
        "feature_names": list(schema.feature_names),
        "reward_scale": reward_scale,
        "trajectory_node_ptr": torch.tensor(buffers["trajectory_node_ptr"], dtype=torch.long),
        "trajectory_edge_ptr": torch.tensor(buffers["trajectory_edge_ptr"], dtype=torch.long),
        "trajectory_child_ptr_ptr": torch.tensor(buffers["trajectory_child_ptr_ptr"], dtype=torch.long),
        "trajectory_expansion_parent_ptr": torch.tensor(buffers["trajectory_expansion_parent_ptr"], dtype=torch.long),
        "trajectory_step_ptr": torch.tensor(buffers["trajectory_step_ptr"], dtype=torch.long),
        # CSR offsets for the sparse update log -- see _accumulate_shard_buffers's
        # comment for how these relate to the (per-tree-local) node_update_ptr values.
        "trajectory_update_log_ptr": torch.tensor(buffers["trajectory_update_log_ptr"], dtype=torch.long),
        "trajectory_node_update_ptr_ptr": torch.tensor(buffers["trajectory_node_update_ptr_ptr"], dtype=torch.long),
        "episode_step_ptr": torch.tensor(buffers["episode_step_ptr"], dtype=torch.long),
        "episode_trajectory_index": torch.tensor(buffers["episode_trajectory_index"], dtype=torch.long),
        "node_features": torch.cat(buffers["all_node_features"], dim=0),
        "parent_index": torch.cat(buffers["all_parent_index"], dim=0),
        "edge_child": torch.cat(buffers["all_edge_child"], dim=0) if buffers["all_edge_child"] else torch.empty(0, dtype=torch.long),
        "edge_slot": torch.cat(buffers["all_edge_slot"], dim=0) if buffers["all_edge_slot"] else torch.empty(0, dtype=torch.long),
        "depth": torch.cat(buffers["all_depth"], dim=0),
        "child_ptr": torch.cat(buffers["all_child_ptr"], dim=0) if buffers["all_child_ptr"] else torch.empty(0, dtype=torch.long),
        "expansion_parent_ids": torch.cat(buffers["all_expansion_parent_ids"], dim=0) if buffers["all_expansion_parent_ids"] else torch.empty(0, dtype=torch.long),
        "step_node_cutoffs": torch.cat(buffers["all_step_node_cutoffs"], dim=0) if buffers["all_step_node_cutoffs"] else torch.empty(0, dtype=torch.long),
        "trajectory_halt_rewards": torch.cat(buffers["all_trajectory_halt_rewards"], dim=0)
        if buffers["all_trajectory_halt_rewards"]
        else torch.empty(0, dtype=torch.float32),
        "first_decision_expansion_counts": torch.tensor(buffers["all_first_decision_expansion_counts"], dtype=torch.long),
        # Sparse per-edge backprop update log, concatenated across every tree in
        # this shard. Sorted within each tree's own slice by (node_id, step_index)
        # ascending; node_update_ptr entries are LOCAL per-tree CSR offsets (mirrors
        # child_ptr's convention) -- use trajectory_update_log_ptr /
        # trajectory_node_update_ptr_ptr (above) to find a given tree's slice of
        # these concatenated arrays. See _replay_backprop_history's docstring for
        # the full schema and history.md's "Stage: packhistory_trees" section for
        # the source-of-truth spec.
        "update_log_step_index": torch.cat(buffers["all_update_log_step_index"], dim=0)
        if buffers["all_update_log_step_index"]
        else torch.empty(0, dtype=torch.int32),
        "update_log_node_id": torch.cat(buffers["all_update_log_node_id"], dim=0)
        if buffers["all_update_log_node_id"]
        else torch.empty(0, dtype=torch.int32),
        "update_log_visit_count": torch.cat(buffers["all_update_log_visit_count"], dim=0)
        if buffers["all_update_log_visit_count"]
        else torch.empty(0, dtype=torch.int32),
        "update_log_q_value": torch.cat(buffers["all_update_log_q_value"], dim=0)
        if buffers["all_update_log_q_value"]
        else torch.empty(0, dtype=torch.float32),
        "update_log_wdl": torch.cat(buffers["all_update_log_wdl"], dim=0)
        if buffers["all_update_log_wdl"]
        else torch.empty((0, 3), dtype=torch.float32),
        "node_update_ptr": torch.cat(buffers["all_node_update_ptr"], dim=0)
        if buffers["all_node_update_ptr"]
        else torch.empty(0, dtype=torch.int32),
        "target_advantages": torch.cat(buffers["all_target_advantages"], dim=0),
        "oracle_stop_steps": torch.tensor(buffers["all_oracle_stop_steps"], dtype=torch.long),
        "oracle_values": torch.tensor(buffers["all_oracle_values"], dtype=torch.float32),
        "starting_budgets": torch.tensor(buffers["all_starting_budgets"], dtype=torch.long),
        "budget_bucket_indices": torch.tensor(buffers["all_bucket_indices"], dtype=torch.long),
        "budget_bucket_names": buffers["all_bucket_names"],
        "episode_keys": buffers["all_episode_keys"],
        "trajectory_source_paths": buffers["trajectory_source_paths"],
        **budgeted_oracle_metadata(oracle_config),
    }
    torch.save(payload, shard_path)


def _write_split_manifest(
    packed_manifest_path: Path,
    *,
    split_name: str,
    total_episodes: int,
    total_skipped: int,
    total_tree_strata_filtered: int,
    total_budget_score_filtered: int,
    sample_trees_by_tree_strata: bool,
    tree_stratification_metadata: dict[str, Any],
    max_budget_score: float,
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    sample_trees_by_budget_score: bool,
    oracle_config: BudgetedOracleConfig,
    entries: List[dict],
) -> None:
    """Write the per-split JSON manifest. Carries all knobs and filter counts so a packed split is fully self-describing."""
    with packed_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "cts_budgeted_controller_episode_manifest_v4",
                "split": split_name,
                "total_episodes": total_episodes,
                "total_skipped": total_skipped,
                "total_tree_strata_filtered": total_tree_strata_filtered,
                "total_budget_score_filtered": total_budget_score_filtered,
                "sample_trees_by_tree_strata": sample_trees_by_tree_strata,
                **tree_stratification_metadata,
                "max_budget_score": max_budget_score,
                "budget_score_keep_floor": _BUDGET_SCORE_KEEP_FLOOR,
                "reward_scale": reward_scale,
                "min_halt_reward_range": min_halt_reward_range,
                "min_decision_margin": min_decision_margin,
                "sample_trees_by_budget_score": sample_trees_by_budget_score,
                **budgeted_oracle_metadata(oracle_config),
                "entries": entries,
            },
            handle,
            indent=2,
        )


def _compute_tree_strata_prepass(
    example_paths: List[Path],
    *,
    sample_trees_by_tree_strata: bool,
    tree_stratification_mode: str,
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    feature_names: Tuple[str, ...],
    oracle_config: BudgetedOracleConfig,
    num_workers: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run the optional tree-strata pre-pass; return (keep probs, metadata) or empty defaults when disabled."""
    if sample_trees_by_tree_strata:
        return _compute_tree_stratified_keep_probabilities(
            example_paths,
            reward_scale,
            min_halt_reward_range,
            min_decision_margin,
            exclude_xaba,
            feature_names,
            oracle_config,
            num_workers,
            tree_stratification_mode,
        )
    return {}, {
        "tree_stratification_mode": tree_stratification_mode,
        "tree_stop_depth_thresholds": [],
        "tree_budget_action_variance_thresholds": [],
        "tree_strata_counts": {},
        "tree_stratification_num_bins": _TREE_STRATIFICATION_NUM_BINS,
        "tree_stratification_mixture_weight": _TREE_STRATIFICATION_MIXTURE_WEIGHT,
    }


def _compute_budget_score_prepass(
    example_paths: List[Path],
    *,
    sample_trees_by_budget_score: bool,
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    feature_names: Tuple[str, ...],
    oracle_config: BudgetedOracleConfig,
    num_workers: int,
) -> float:
    """Run the optional budget-sensitivity pre-pass; return the global max score, or 0.0 when disabled."""
    if sample_trees_by_budget_score:
        return _compute_max_budget_score(
            example_paths,
            reward_scale,
            min_halt_reward_range,
            min_decision_margin,
            exclude_xaba,
            feature_names,
            oracle_config,
            num_workers,
        )
    return 0.0


def _pack_split(
    manifest_path: Path,
    output_root: Path,
    quality_config: TeacherSearchConfig,
    oracle_config: BudgetedOracleConfig,
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    sample_trees_by_tree_strata: bool,
    tree_stratification_mode: str,
    sample_trees_by_budget_score: bool,
    shard_size: int,
    num_workers: int,
    log_interval: int,
    downsample_trivial: float = 1.0,
) -> tuple[Path, int, int, int]:
    """Pack one split (train or validation) into sharded ``.pt`` files plus a JSON manifest.

    Performs up to three passes over the split:
      1. Optional: per-tree stat collection for stratified sampling.
      2. Optional: per-tree budget-score collection for sensitivity sampling.
      3. Main packing pass: loads each tree, runs the oracle, applies all
         filters, and accumulates packed shards.

    Returns (packed_manifest_path, total_episodes, total_skipped,
    total_tree_strata_filtered, total_budget_score_filtered).
    """
    split_name = manifest_path.stem.replace("_manifest", "")
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    example_paths = _read_manifest(manifest_path)
    schema = _feature_schema()
    feature_names = tuple(schema.feature_names)
    start_time = time.time()
    entries: List[dict] = []
    total_episodes = 0
    total_skipped = 0
    total_tree_strata_filtered = 0
    total_budget_score_filtered = 0
    total_shards = (len(example_paths) + shard_size - 1) // shard_size

    tree_keep_probabilities, tree_stratification_metadata = _compute_tree_strata_prepass(
        example_paths,
        sample_trees_by_tree_strata=sample_trees_by_tree_strata,
        tree_stratification_mode=tree_stratification_mode,
        reward_scale=reward_scale,
        min_halt_reward_range=min_halt_reward_range,
        min_decision_margin=min_decision_margin,
        exclude_xaba=exclude_xaba,
        feature_names=feature_names,
        oracle_config=oracle_config,
        num_workers=num_workers,
    )
    max_budget_score = _compute_budget_score_prepass(
        example_paths,
        sample_trees_by_budget_score=sample_trees_by_budget_score,
        reward_scale=reward_scale,
        min_halt_reward_range=min_halt_reward_range,
        min_decision_margin=min_decision_margin,
        exclude_xaba=exclude_xaba,
        feature_names=feature_names,
        oracle_config=oracle_config,
        num_workers=num_workers,
    )

    # Main packing loop, one shard at a time.
    for shard_index, shard_start in enumerate(range(0, len(example_paths), shard_size)):
        shard_paths = example_paths[shard_start: shard_start + shard_size]
        tasks = [
            (
                str(path),
                reward_scale,
                min_halt_reward_range,
                min_decision_margin,
                exclude_xaba,
                tree_keep_probabilities.get(str(path), 1.0),
                sample_trees_by_budget_score,
                max_budget_score,
                feature_names,
                oracle_config,
                downsample_trivial,
            )
            for path in shard_paths
        ]

        results = _process_shard_tasks(
            tasks,
            num_workers=num_workers,
            log_interval=log_interval,
            split_name=split_name,
            shard_index=shard_index,
            total_shards=total_shards,
            total_episodes=total_episodes,
            total_skipped=total_skipped,
            total_tree_strata_filtered=total_tree_strata_filtered,
            total_budget_score_filtered=total_budget_score_filtered,
            start_time=start_time,
        )

        buffers, shard_episodes, shard_skipped, shard_tree_strata, shard_budget_score = (
            _accumulate_shard_buffers(results)
        )
        total_skipped += shard_skipped
        total_tree_strata_filtered += shard_tree_strata
        total_budget_score_filtered += shard_budget_score

        # Skip empty shards entirely so the manifest doesn't reference
        # zero-sized files that downstream code would have to special-case.
        if shard_episodes == 0:
            continue

        shard_path = split_output_dir / f"shard_{shard_index:05d}.pt"
        _serialize_shard_payload(
            shard_path,
            buffers,
            shard_episodes=shard_episodes,
            schema=schema,
            reward_scale=reward_scale,
            oracle_config=oracle_config,
        )
        entries.append({"path": str(shard_path), "num_episodes": shard_episodes, "shard_index": shard_index})
        total_episodes += shard_episodes

    packed_manifest_path = output_root / f"{split_name}_manifest.json"
    _write_split_manifest(
        packed_manifest_path,
        split_name=split_name,
        total_episodes=total_episodes,
        total_skipped=total_skipped,
        total_tree_strata_filtered=total_tree_strata_filtered,
        total_budget_score_filtered=total_budget_score_filtered,
        sample_trees_by_tree_strata=sample_trees_by_tree_strata,
        tree_stratification_metadata=tree_stratification_metadata,
        max_budget_score=max_budget_score,
        reward_scale=reward_scale,
        min_halt_reward_range=min_halt_reward_range,
        min_decision_margin=min_decision_margin,
        sample_trees_by_budget_score=sample_trees_by_budget_score,
        oracle_config=oracle_config,
        entries=entries,
    )

    return packed_manifest_path, total_episodes, total_skipped, total_tree_strata_filtered, total_budget_score_filtered


def _process_one_task(
    task: Tuple[str, float, float, float, bool, float, bool, float, Tuple[str, ...], BudgetedOracleConfig, float],
) -> Optional[dict[str, Any]]:
    """Worker entry point: build a packed tree result and apply all keep-probability filters.

    Returns one of:
      - None: tree was dropped by per-tree filters or all episodes were trivial.
      - dict with ``status`` key: tree was filtered by stratification or
        budget-score sampling; reported separately so the manifest tracks why.
      - dict with ``trajectory`` and ``episodes``: tree was accepted.
    """
    (
        path_str,
        reward_scale,
        min_halt_reward_range,
        min_decision_margin,
        exclude_xaba,
        tree_keep_probability,
        sample_trees_by_budget_score,
        max_budget_score,
        feature_names,
        oracle_config,
        downsample_trivial,
    ) = task

    packed_tree_result = _build_packed_tree_result(
        path_str,
        reward_scale,
        min_halt_reward_range,
        min_decision_margin,
        exclude_xaba,
        feature_names,
        oracle_config,
    )

    if packed_tree_result is None:
        return None
    # Stratified sampling: reject when the deterministic draw exceeds the
    # tree's keep probability. Reported with a distinct status so manifest
    # accounting separates this from hard filters.
    if _deterministic_unit_interval(path_str, oracle_config.seed) >= tree_keep_probability:
        return {
            "status": "filtered_tree_strata",
            "source_path": path_str,
            "tree_keep_probability": tree_keep_probability,
        }
    # Budget-sensitivity sampling: same pattern, distinct status.
    if sample_trees_by_budget_score:
        budget_score = float(packed_tree_result["budget_score"])
        budget_score_keep_probability = _tree_budget_sensitivity_keep_probability(budget_score, max_budget_score)
        if _deterministic_unit_interval(path_str, oracle_config.seed) >= budget_score_keep_probability:
            return {
                "status": "filtered_budget_score",
                "source_path": path_str,
                "budget_score": budget_score,
                "budget_score_keep_probability": budget_score_keep_probability,
            }
    # Drop per-episode halt_rewards from the packed result before returning:
    # they were only needed for the budget-sensitivity score computation
    # and the trajectory carries the canonical copy.
    for episode in packed_tree_result["episodes"]:
        episode.pop("halt_rewards", None)
    # Optional per-episode downsampling of "trivial" stop-at-step-1 episodes,
    # which would otherwise dominate the training distribution.
    if downsample_trivial < 1.0:
        packed_tree_result["episodes"] = [
            episode for episode in packed_tree_result["episodes"]
            if episode["oracle_stop_step"] > 1
            or _deterministic_unit_interval(episode["episode_key"], oracle_config.seed) < downsample_trivial
        ]
        if not packed_tree_result["episodes"]:
            return None
    return {
        "trajectory": packed_tree_result["trajectory"],
        "episodes": packed_tree_result["episodes"],
    }


def _score_one_task(
    task: Tuple[str, float, float, float, bool, Tuple[str, ...], BudgetedOracleConfig],
) -> dict[str, Any]:
    """Worker entry point for the budget-score pre-pass; returns ``{"budget_score": float}``."""
    path_str, reward_scale, min_halt_reward_range, min_decision_margin, exclude_xaba, feature_names, oracle_config = task
    packed_tree_result = _build_packed_tree_result(
        path_str,
        reward_scale,
        min_halt_reward_range,
        min_decision_margin,
        exclude_xaba,
        feature_names,
        oracle_config,
    )
    if packed_tree_result is None:
        return {"budget_score": 0.0}
    return {"budget_score": float(packed_tree_result["budget_score"])}


def _tree_stats_one_task(
    task: Tuple[str, float, float, float, bool, Tuple[str, ...], BudgetedOracleConfig],
) -> Optional[dict[str, Any]]:
    """Worker entry point for the stratification pre-pass; returns (d, j) stats or None."""
    path_str, reward_scale, min_halt_reward_range, min_decision_margin, exclude_xaba, feature_names, oracle_config = task
    packed_tree_result = _build_packed_tree_result(
        path_str,
        reward_scale,
        min_halt_reward_range,
        min_decision_margin,
        exclude_xaba,
        feature_names,
        oracle_config,
    )
    if packed_tree_result is None:
        return None
    return {
        "source_path": path_str,
        "stop_depth_excess": float(packed_tree_result["stop_depth_excess"]),
        "budget_action_variance": float(packed_tree_result["budget_action_variance"]),
    }


def _numpy_to_torch(result: dict) -> dict:
    """Convert numpy-typed arrays inside a packed result dict to torch tensors.

    Workers return numpy because numpy serializes more cheaply across
    process boundaries; the parent then converts here just before concat-ing
    into the shard payload.
    """
    converted = {
        "num_steps": result["num_steps"],
        "source_path": result["source_path"],
    }
    # Trajectory-shaped dicts carry the tree structure; episode-shaped dicts
    # carry only the per-step oracle outputs. Each branch handles one case.
    if "node_features" in result:
        converted.update(
            {
                "node_features": torch.from_numpy(result["node_features"]),
                "parent_index": torch.from_numpy(result["parent_index"]),
                "depth": torch.from_numpy(result["depth"]),
                "child_ptr": torch.from_numpy(result["child_ptr"]),
                "edge_child": torch.from_numpy(result["edge_child"]),
                "edge_slot": torch.from_numpy(result["edge_slot"]),
                "expansion_parent_ids": torch.from_numpy(result["expansion_parent_ids"]),
                "first_decision_expansion_count": int(result["first_decision_expansion_count"]),
                "step_node_cutoffs": torch.from_numpy(result["step_node_cutoffs"]),
                "halt_rewards": torch.from_numpy(result["halt_rewards"]),
                "tree_sizes": torch.from_numpy(result["tree_sizes"]),
                "update_log_step_index": torch.from_numpy(result["update_log_step_index"]),
                "update_log_node_id": torch.from_numpy(result["update_log_node_id"]),
                "update_log_visit_count": torch.from_numpy(result["update_log_visit_count"]),
                "update_log_q_value": torch.from_numpy(result["update_log_q_value"]),
                "update_log_wdl": torch.from_numpy(result["update_log_wdl"]),
                "node_update_ptr": torch.from_numpy(result["node_update_ptr"]),
            }
        )
    if "target_advantages" in result:
        converted.update(
            {
                "target_advantages": torch.from_numpy(result["target_advantages"]),
                "oracle_stop_step": result["oracle_stop_step"],
                "oracle_value": result["oracle_value"],
                "starting_budget": result["starting_budget"],
                "bucket_index": result["bucket_index"],
                "bucket_name": result["bucket_name"],
                "episode_key": result["episode_key"],
            }
        )
    return converted


def main(config: PackControllerEpisodesConfig) -> None:
    """CLI entry point: parse args, pack the train and validation splits, print summary stats."""
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    split_root = Path(config.split_root)
    output_root = Path(config.output_root)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"

    # ``--clear`` wipes the output tree before packing. Files first, then
    # the now-empty directories in reverse-sorted order so children come
    # before parents.
    if config.clear and output_root.exists():
        for child in output_root.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(output_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()

    output_root.mkdir(parents=True, exist_ok=True)
    quality_config = _quality_config(config)
    oracle_config = _oracle_config(config)

    # Print the resolved config up front so slurm logs capture exactly what
    # was packed without having to crack open the output manifest.
    print(f"split_root={split_root}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"num_workers={config.num_workers}", flush=True)
    print(f"reward_scale={config.reward_scale}", flush=True)
    print(f"min_halt_reward_range={config.min_halt_reward_range}", flush=True)
    print(f"min_decision_margin={config.min_decision_margin}", flush=True)
    print(f"exclude_xaba={config.exclude_xaba}", flush=True)
    print(f"sample_trees_by_tree_strata={config.sample_trees_by_tree_strata}", flush=True)
    print(f"tree_stratification_mode={config.tree_stratification_mode}", flush=True)
    print(f"sample_trees_by_budget_score={config.sample_trees_by_budget_score}", flush=True)
    print(f"downsample_trivial={config.downsample_trivial}", flush=True)
    print(json.dumps(budgeted_oracle_metadata(oracle_config), sort_keys=True), flush=True)

    train_manifest_out, train_count, train_skipped, train_tree_strata_filtered, train_budget_score_filtered = _pack_split(
        train_manifest,
        output_root,
        quality_config,
        oracle_config,
        config.reward_scale,
        config.min_halt_reward_range,
        config.min_decision_margin,
        config.exclude_xaba,
        config.sample_trees_by_tree_strata,
        config.tree_stratification_mode,
        config.sample_trees_by_budget_score,
        config.shard_size,
        config.num_workers,
        config.log_interval,
        downsample_trivial=config.downsample_trivial,
    )
    validation_manifest_out, val_count, val_skipped, val_tree_strata_filtered, val_budget_score_filtered = _pack_split(
        validation_manifest,
        output_root,
        quality_config,
        oracle_config,
        config.reward_scale,
        config.min_halt_reward_range,
        config.min_decision_margin,
        config.exclude_xaba,
        config.sample_trees_by_tree_strata,
        config.tree_stratification_mode,
        config.sample_trees_by_budget_score,
        config.shard_size,
        config.num_workers,
        config.log_interval,
    )

    print(f"train_manifest={train_manifest_out}")
    print(f"validation_manifest={validation_manifest_out}")
    print(
        f"train_episodes={train_count} train_skipped={train_skipped} "
        f"train_tree_strata_filtered={train_tree_strata_filtered} "
        f"train_budget_score_filtered={train_budget_score_filtered}"
    )
    print(
        f"validation_episodes={val_count} validation_skipped={val_skipped} "
        f"validation_tree_strata_filtered={val_tree_strata_filtered} "
        f"validation_budget_score_filtered={val_budget_score_filtered}"
    )
    print(f"output_root={output_root}")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(PackControllerEpisodesConfig, main)
