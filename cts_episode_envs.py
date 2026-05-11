from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cts_pretrain import (
    PretrainExample,
    TeacherSearchConfig,
    TeacherSearchResult,
    compute_teacher_targets,
)
from tree import SearchTree


@dataclass(frozen=True)
class SnapshotEpisode:
    snapshots: List[SearchTree]
    qualities: List[float]
    best_moves: List[str]
    final_root_q_values: Dict[str, float]
    final_best_quality: float


def _truncate_tree_at_expansion_count(tree: SearchTree, expansion_count: int) -> SearchTree:
    return tree.clone_expansion_prefix(expansion_count)


def _terminal_quality_from_teacher_result(tree: SearchTree, teacher_result: TeacherSearchResult) -> float:
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
    snapshots, qualities, _, _, _ = build_snapshot_episode_metadata(example, quality_config)
    return snapshots, qualities


def build_snapshot_episode_metadata(
    example: PretrainExample,
    quality_config: TeacherSearchConfig,
) -> Tuple[List[SearchTree], List[float], List[Optional[str]], Dict[str, float], float]:
    tree = example.tree
    expansion_parent_ids = tree.ordered_expansion_parent_ids()
    snapshots: List[SearchTree] = []
    qualities: List[float] = []
    best_moves: List[Optional[str]] = []
    final_root_q_values: Dict[str, float] = {}
    final_best_quality = float("nan")
    stored_trace = oracle_root_q_trace_for_example(example)
    if stored_trace is not None:
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
    return [episode.final_root_q_values[move] for move in episode.best_moves]


def build_trimmed_decision_episode_with_halt_rewards(
    example: PretrainExample,
    quality_config: TeacherSearchConfig,
) -> Tuple[SnapshotEpisode, List[float]]:
    episode = build_trimmed_decision_episode(example, quality_config)
    return episode, halt_rewards_for_snapshot_episode(episode)
