"""Snapshot-style episode extraction from teacher search results.

Given a fully-built teacher tree and its pretraining targets, produce the
sequence ``(tree-at-expansion-k, halt-quality-at-k, best-move-at-k)`` for
k = 1..N. Each ``SnapshotEpisode`` represents what the controller "would
have seen" at every step of the teacher's search.

This module is a narrow utility used in tests and the ``oversearch``
analysis path. Production controller-data packing uses
``cts.data.preprocess_mc.pack`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cts.core.tree import SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    PretrainExample,
    TeacherSearchConfig,
    TeacherSearchResult,
    compute_teacher_targets,
)


@dataclass(frozen=True)
class SnapshotEpisode:
    """Per-expansion snapshots plus terminal teacher decision for one search.

    Captures the trajectory of teacher decisions as the search grows, plus
    the final root q-values so a controller can be trained against the
    teacher's terminal best move regardless of which step it halted on.
    """

    snapshots: List[SearchTree]  # tree at each expansion step k; length matches qualities and best_moves
    qualities: List[float]  # teacher's max-root-q at each snapshot k
    best_moves: List[str]  # teacher's argmax-root-q move (UCI) at each snapshot k
    final_root_q_values: Dict[str, float]  # terminal root q-values keyed by UCI move
    final_best_quality: float  # max(final_root_q_values.values()); cached for convenience


def _truncate_tree_at_expansion_count(tree: SearchTree, expansion_count: int) -> SearchTree:
    """Return a clone of ``tree`` with only the first ``expansion_count`` expansions replayed."""
    return tree.clone_expansion_prefix(expansion_count)


def _terminal_quality_from_teacher_result(tree: SearchTree, teacher_result: TeacherSearchResult) -> float:
    """Return the max root-child q-value (or root value if unexpanded)."""
    root_id = tree.root_id
    if root_id is None:
        raise ValueError("Tree must contain a root.")

    root_children = tree.root_children()
    if not root_children:
        return float(teacher_result.node_target_values[root_id])

    root_q_values = [
        teacher_result.edge_stats[(root_id, child_id)].q_value
        for child_id in root_children
    ]
    return float(max(root_q_values))


def _root_decision_from_teacher_result(
    tree: SearchTree,
    teacher_result: TeacherSearchResult,
) -> Tuple[Optional[str], float]:
    """Return ``(best_move_uci, best_q)`` for the root; ``(None, root_value)`` if unexpanded."""
    root_id = tree.root_id
    if root_id is None:
        raise ValueError("Tree must contain a root.")

    root_children = tree.root_children()
    if not root_children:
        return None, float(teacher_result.node_target_values[root_id])

    best_child_id = max(
        root_children,
        key=lambda child_id: teacher_result.edge_stats[(root_id, child_id)].q_value,
    )
    return (
        tree.get_node(best_child_id).incoming_move_uci,
        float(teacher_result.edge_stats[(root_id, best_child_id)].q_value),
    )


def _root_q_values_from_teacher_result(
    tree: SearchTree,
    teacher_result: TeacherSearchResult,
) -> Dict[str, float]:
    """Return a UCI-keyed dict of q-values for every root child."""
    root_id = tree.root_id
    if root_id is None:
        raise ValueError("Tree must contain a root.")

    root_q_values: Dict[str, float] = {}
    for child_id in tree.root_children():
        move_uci = tree.get_node(child_id).incoming_move_uci
        if move_uci is None:
            raise ValueError(f"Root child {child_id} is missing an incoming move.")
        root_q_values[str(move_uci)] = float(teacher_result.edge_stats[(root_id, child_id)].q_value)
    return root_q_values


def oracle_root_q_trace_for_example(
    example: PretrainExample,
) -> Optional[Tuple[List[int], List[Dict[str, float]], List[str], Dict[str, float]]]:
    """Pull the precomputed per-expansion oracle trace off a ``PretrainExample``, if present.

    Returns the trace as ``(expansion_counts, per_step_q_maps, best_moves,
    final_root_q_values)``, or ``None`` if the example wasn't generated with
    a stored trace. When the stored final q-values are empty, fall back to
    the last row of the trace so callers always get a usable terminal map.
    """
    if not example.oracle_root_q_trace or not example.oracle_root_moves:
        return None
    q_maps = [
        {
            move: float(value)
            for move, value in zip(example.oracle_root_moves, row)
        }
        for row in example.oracle_root_q_trace
    ]
    final_root_q_values = dict(example.oracle_final_root_q_values)
    if not final_root_q_values and q_maps:
        final_root_q_values = dict(q_maps[-1])
    return (
        list(example.oracle_trace_expansion_counts),
        q_maps,
        list(example.oracle_best_move_trace),
        final_root_q_values,
    )


def build_snapshot_episode(
    example: PretrainExample,
    quality_config: TeacherSearchConfig,
) -> Tuple[List[SearchTree], List[float]]:
    """Thin wrapper over ``build_snapshot_episode_metadata`` returning only snapshots and qualities."""
    snapshots, qualities, _, _, _ = build_snapshot_episode_metadata(example, quality_config)
    return snapshots, qualities


def build_snapshot_episode_metadata(
    example: PretrainExample,
    quality_config: TeacherSearchConfig,
) -> Tuple[List[SearchTree], List[float], List[Optional[str]], Dict[str, float], float]:
    """Replay every expansion of ``example.tree`` and emit per-step teacher decisions.

    Two paths: if the example carries a precomputed oracle trace, use it
    (the fast path used at training time). Otherwise re-run the teacher
    targeting on each truncated snapshot (the slow fallback used for
    examples without stored traces).

    Args:
        example: pretraining example containing the fully-built tree and
            optionally a precomputed oracle trace.
        quality_config: teacher config used only on the slow fallback path.

    Returns:
        Tuple of ``(snapshots, qualities, best_moves, final_root_q_values,
        final_best_quality)``. ``best_moves[0]`` is ``None`` because the
        root-only snapshot has no decision to make.
    """
    tree = example.tree
    expansion_parent_ids = tree.ordered_expansion_parent_ids()
    snapshots: List[SearchTree] = []
    qualities: List[float] = []
    best_moves: List[Optional[str]] = []
    final_root_q_values: Dict[str, float] = {}
    final_best_quality = float("nan")
    stored_trace = oracle_root_q_trace_for_example(example)
    if stored_trace is not None:
        # Fast path: stored trace covers every expansion. Validate that the
        # trace lines up with the tree's actual expansion sequence before
        # trusting it (mismatches mean data corruption or version skew).
        trace_counts, q_maps, trace_best_moves, final_root_q_values = stored_trace
        count_to_trace_idx = {int(count): idx for idx, count in enumerate(trace_counts)}
        if len(trace_counts) != len(expansion_parent_ids):
            raise ValueError(
                "Stored oracle trace must align with the original expansion sequence for generated examples."
            )
        if set(trace_counts) != set(range(1, len(expansion_parent_ids) + 1)):
            raise ValueError("Stored oracle trace must contain one row for each positive expansion count.")
        if final_root_q_values:
            final_best_quality = float(max(final_root_q_values.values()))
        else:
            final_best_quality = float(tree.get_node(tree.root_id).scalar_features["value"])

        for expansion_count in range(len(expansion_parent_ids) + 1):
            snapshot = _truncate_tree_at_expansion_count(tree, expansion_count)
            snapshots.append(snapshot)
            if expansion_count == 0:
                # Root-only snapshot: no children means no decision; use the
                # root's scalar "value" feature as the quality estimate.
                qualities.append(float(snapshot.get_node(snapshot.root_id).scalar_features["value"]))
                best_moves.append(None)
                continue
            trace_idx = count_to_trace_idx.get(expansion_count)
            if trace_idx is None:
                raise ValueError(f"Stored oracle trace is missing expansion_count={expansion_count}.")
            root_q_values = q_maps[trace_idx]
            if not root_q_values:
                raise ValueError(f"Stored oracle trace at expansion_count={expansion_count} has no root q-values.")
            qualities.append(float(max(root_q_values.values())))
            best_moves.append(str(trace_best_moves[trace_idx]))
        return snapshots, qualities, best_moves, final_root_q_values, final_best_quality

    # Slow fallback: no stored trace, so re-run the teacher targeting once
    # per snapshot. Quadratic in tree size, used only for older examples.
    for expansion_count in range(len(expansion_parent_ids) + 1):
        snapshot = _truncate_tree_at_expansion_count(tree, expansion_count)
        teacher_result = compute_teacher_targets(snapshot, quality_config, validate=False)
        snapshots.append(snapshot)
        qualities.append(_terminal_quality_from_teacher_result(snapshot, teacher_result))
        best_move, _ = _root_decision_from_teacher_result(snapshot, teacher_result)
        best_moves.append(best_move)
        if expansion_count == len(expansion_parent_ids):
            final_root_q_values = _root_q_values_from_teacher_result(snapshot, teacher_result)
            _, final_best_quality = _root_decision_from_teacher_result(snapshot, teacher_result)
    return snapshots, qualities, best_moves, final_root_q_values, final_best_quality


def trim_episode_to_first_root_decision(
    snapshots: Sequence[SearchTree],
    qualities: Sequence[float],
) -> Tuple[List[SearchTree], List[float]]:
    """Drop leading snapshots that have no root children (no decision available).

    The root-only snapshot (and any prefix with no expansions of the root
    itself) carries no actionable decision; the controller can't be trained
    on those steps. Returns the suffix starting from the first snapshot
    whose root has children.
    """
    if len(snapshots) != len(qualities):
        raise ValueError("snapshots and qualities must have the same length.")
    first_decision_index = next(
        (index for index, snapshot in enumerate(snapshots) if snapshot.root_children()),
        None,
    )
    if first_decision_index is None:
        raise ValueError("Episode contains no snapshot with root children.")
    return list(snapshots[first_decision_index:]), list(qualities[first_decision_index:])


def trim_episode_metadata_to_first_root_decision(
    snapshots: Sequence[SearchTree],
    qualities: Sequence[float],
    best_moves: Sequence[Optional[str]],
) -> Tuple[List[SearchTree], List[float], List[str]]:
    """Same trimming as ``trim_episode_to_first_root_decision`` but also for ``best_moves``.

    After trimming, every retained snapshot must have a non-None best move
    (since every retained snapshot has root children). Raises if that
    invariant is violated, which would indicate inconsistent metadata.
    """
    trimmed_snapshots, trimmed_qualities = trim_episode_to_first_root_decision(snapshots, qualities)
    first_decision_index = len(snapshots) - len(trimmed_snapshots)
    trimmed_best_moves = list(best_moves[first_decision_index:])
    if len(trimmed_best_moves) != len(trimmed_snapshots):
        raise ValueError("Trimmed episode metadata is inconsistent.")
    if not trimmed_best_moves:
        raise ValueError("Trimmed episode is empty.")

    normalized_best_moves: List[str] = []
    for index, move in enumerate(trimmed_best_moves):
        if move is None:
            raise ValueError(
                f"Trimmed snapshot at index {index} is missing a root decision; episode invariants are broken."
            )
        normalized_best_moves.append(str(move))

    return trimmed_snapshots, trimmed_qualities, normalized_best_moves


def build_trimmed_decision_episode(
    example: PretrainExample,
    quality_config: TeacherSearchConfig,
) -> SnapshotEpisode:
    """Build a ``SnapshotEpisode`` trimmed to actionable decision steps only.

    Cross-checks that every retained best move appears in
    ``final_root_q_values`` so downstream halt-reward lookups can't miss.
    """
    snapshots, qualities, best_moves, final_root_q_values, final_best_quality = build_snapshot_episode_metadata(
        example,
        quality_config,
    )
    trimmed_snapshots, trimmed_qualities, trimmed_best_moves = trim_episode_metadata_to_first_root_decision(
        snapshots,
        qualities,
        best_moves,
    )
    for move in trimmed_best_moves:
        if move not in final_root_q_values:
            raise ValueError(f"Trimmed episode move {move!r} is missing from final_root_q_values.")
    return SnapshotEpisode(
        snapshots=trimmed_snapshots,
        qualities=trimmed_qualities,
        best_moves=trimmed_best_moves,
        final_root_q_values=dict(final_root_q_values),
        final_best_quality=final_best_quality,
    )


def halt_rewards_for_snapshot_episode(episode: SnapshotEpisode) -> List[float]:
    """Per-step halt reward: terminal q-value of the move the teacher picked at step k.

    Used to score "if the controller had halted at step k, what would the
    terminal teacher have said about that move?"
    """
    return [episode.final_root_q_values[move] for move in episode.best_moves]


def build_trimmed_decision_episode_with_halt_rewards(
    example: PretrainExample,
    quality_config: TeacherSearchConfig,
) -> Tuple[SnapshotEpisode, List[float]]:
    """Convenience pairing of ``build_trimmed_decision_episode`` and its halt rewards."""
    episode = build_trimmed_decision_episode(example, quality_config)
    return episode, halt_rewards_for_snapshot_episode(episode)
