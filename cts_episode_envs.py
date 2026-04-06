from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cts_pretrain import PretrainExample, TeacherSearchConfig, TeacherSearchResult, compute_teacher_targets
from tree import SearchTree


@dataclass
class StepResult:
    next_tree: SearchTree
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotEpisode:
    snapshots: List[SearchTree]
    qualities: List[float]
    best_moves: List[str]
    final_root_q_values: Dict[str, float]
    final_best_quality: float


class SnapshotEpisodeCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._cache: "OrderedDict[str, SnapshotEpisode]" = OrderedDict()

    def get(self, path: str) -> Optional[SnapshotEpisode]:
        episode = self._cache.get(path)
        if episode is not None:
            self._cache.move_to_end(path)
        return episode

    def put(self, path: str, episode: SnapshotEpisode) -> None:
        self._cache[path] = episode
        self._cache.move_to_end(path)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


class ControllerOnlyEnv:
    def reset(self) -> SearchTree:
        raise NotImplementedError

    def step(self, action: int) -> StepResult:
        raise NotImplementedError


class ToyHaltEnv(ControllerOnlyEnv):
    def __init__(self, tree_snapshots: Sequence[SearchTree], continue_cost: float = 0.05) -> None:
        if not tree_snapshots:
            raise ValueError("tree_snapshots must be non-empty.")
        self.tree_snapshots = [tree.clone() for tree in tree_snapshots]
        self.continue_cost = continue_cost
        self._snapshot_index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0

    def _current_tree(self) -> SearchTree:
        return self.tree_snapshots[self._snapshot_index].clone()

    def _terminal_quality(self) -> float:
        tree = self.tree_snapshots[self._snapshot_index]
        if tree.root_children():
            best_child_id = tree.best_root_child()
            return float(tree.get_node(best_child_id).scalar_features["value"])
        return float(tree.get_node(tree.root_id).scalar_features["value"])

    def reset(self) -> SearchTree:
        self._snapshot_index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        return self._current_tree()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise ValueError("Episode already finished. Call reset().")
        if action not in (0, 1):
            raise ValueError("Action must be 0 (continue) or 1 (halt).")

        self._episode_length += 1
        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        if action == 1:
            reward = self._terminal_quality()
            done = True
        else:
            reward = -self.continue_cost
            if self._snapshot_index + 1 < len(self.tree_snapshots):
                self._snapshot_index += 1
            else:
                reward += self._terminal_quality()
                done = True

        self._episode_return += reward
        self._done = done

        if done:
            info = {
                "episode_return": self._episode_return,
                "episode_length": self._episode_length,
                "expansions": self._snapshot_index,
                "terminal_quality": self._terminal_quality(),
                "halted": action == 1,
            }

        return StepResult(next_tree=self._current_tree(), reward=reward, done=done, info=info)


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


class GeneratedTreeHaltEnv(ControllerOnlyEnv):
    def __init__(
        self,
        example_paths: Sequence[str],
        quality_config: TeacherSearchConfig,
        continue_cost: float = 0.05,
        seed: int = 0,
        shuffle: bool = True,
        max_cache_size: int = 32,
        episode_cache: Optional[SnapshotEpisodeCache] = None,
    ) -> None:
        if not example_paths:
            raise ValueError("example_paths must be non-empty.")
        self.example_paths = list(example_paths)
        self.quality_config = quality_config
        self.continue_cost = float(continue_cost)
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._next_index = 0
        self._cache = episode_cache or SnapshotEpisodeCache(max_cache_size)
        self._snapshot_index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        self._current_path: Optional[str] = None
        self._snapshots: List[SearchTree] = []
        self._qualities: List[float] = []
        self._best_moves: List[str] = []
        self._final_root_q_values: Dict[str, float] = {}
        self._final_best_quality: float = float("nan")

    def _choose_path(self) -> str:
        if self.shuffle:
            return self.example_paths[self._rng.randrange(len(self.example_paths))]
        path = self.example_paths[self._next_index % len(self.example_paths)]
        self._next_index += 1
        return path

    def _load_episode(self, path: str) -> SnapshotEpisode:
        cached = self._cache.get(path)
        if cached is not None:
            return cached

        example = torch.load(path, weights_only=False)
        if not isinstance(example, PretrainExample):
            raise ValueError(f"Expected PretrainExample at {path}, got {type(example).__name__}.")
        episode = build_trimmed_decision_episode(example, self.quality_config)
        self._cache.put(path, episode)
        return episode

    def _current_tree(self) -> SearchTree:
        return self._snapshots[self._snapshot_index].clone()

    def _terminal_quality(self) -> float:
        return float(self._qualities[self._snapshot_index])

    def _halt_reward(self) -> float:
        current_best_move = self._best_moves[self._snapshot_index]
        return float(self._final_root_q_values[current_best_move])

    def reset(self) -> SearchTree:
        self._current_path = self._choose_path()
        episode = self._load_episode(self._current_path)
        self._snapshots = episode.snapshots
        self._qualities = episode.qualities
        self._best_moves = episode.best_moves
        self._final_root_q_values = episode.final_root_q_values
        self._final_best_quality = episode.final_best_quality
        self._snapshot_index = 0
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        return self._current_tree()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise ValueError("Episode already finished. Call reset().")
        if action not in (0, 1):
            raise ValueError("Action must be 0 (continue) or 1 (halt).")

        self._episode_length += 1
        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        if action == 1:
            reward = self._halt_reward()
            done = True
        else:
            reward = -self.continue_cost
            if self._snapshot_index + 1 < len(self._snapshots):
                self._snapshot_index += 1
            else:
                reward += self._halt_reward()
                done = True

        self._episode_return += reward
        self._done = done

        if done:
            info = {
                "episode_return": self._episode_return,
                "episode_length": self._episode_length,
                "expansions": self._snapshot_index,
                "terminal_quality": self._terminal_quality(),
                "halt_reward": self._halt_reward(),
                "reference_best_quality": self._final_best_quality,
                "decision_regret": self._final_best_quality - (
                    self._final_root_q_values[self._best_moves[self._snapshot_index]]
                ),
                "halted": action == 1,
                "example_path": self._current_path,
            }

        return StepResult(next_tree=self._current_tree(), reward=reward, done=done, info=info)
