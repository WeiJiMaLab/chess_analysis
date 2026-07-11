"""Packs T_n -> T_{n+k} snapshot-pair training examples: T_n is supervised with a
target read off that same tree's own recorded state at step n + lookahead_k.

Upstream contract, confirmed against packhistory_trees's real code (``preprocess_mc/pack.py``):
structural fields match ``cts_budgeted_controller_episode_shard_v4``, plus five flat
update-log arrays (``update_log_step_index/node_id/visit_count/q_value/wdl``) and a
per-trajectory CSR index ``node_update_ptr``, decoded in ``iter_history_trajectories``.
``step_index``'s coordinate system is 0-indexed, but on the FULL (untrimmed)
expansion-position scale -- NOT re-based to ``root_rank`` the way ``step_node_cutoffs``/
``local_step = n - 1`` are. See ``_full_scale_step`` for the conversion between the two
scales (fixed 2026-07-11 -- see ``test_value_query_respects_root_rank_fix`` in
``test_canary_pack_history_structural.py`` for the regression test).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import torch
from pydantic import BaseModel, ConfigDict

from cts.core.schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeExample, collate_tensorized_examples


class PackHistoryGNNPretrainConfig(BaseModel):
    """Config for the packhistory_GNNpretrain stage (new pipeline section: ``gnn_pack_history``)."""

    model_config = ConfigDict(extra="forbid")

    pack_history_dir: str  # packhistory_trees' output root (train/validation shard manifests live directly under it)
    output_root: str  # gnnpack_history_dir
    lookahead_k: int = 12  # k-steps-ahead target horizon; v1 default per history.md (not tuned)
    snapshots_per_tree: int = 1  # distinct n's sampled per tree; v1 default keeps compute at parity with today
    search_budget: int = 96  # total root-expansion steps a full trajectory covers
    seed: int = 0  # mixed into the per-tree deterministic snapshot-step hash
    shard_size: int = 2000  # output examples per written shard
    log_interval: int = 200  # trees between progress prints
    clear: bool = False


@dataclass
class HistoryTrajectory:
    """One decoded tree's worth of packhistory_trees's packed output.

    All node/edge ids here are *local* to this trajectory (0-based), matching the
    convention every field in today's ``mc_packed/`` already uses -- see
    ``ControllerEpisodeDataset.__getitem__`` (``controller_train.py:840-907``), which
    slices the exact same fields (``node_features``, ``parent_index``, ``child_ptr``,
    ``edge_child``, ``edge_slot``, ``depth``, ``expansion_parent_ids``,
    ``step_node_cutoffs``, ``first_decision_expansion_count``) out of a shard the same
    way ``iter_history_trajectories`` does below.
    """

    source_path: str
    node_features: torch.Tensor  # [N, 5] baseline (pre-any-backprop) node features, schema column order
    parent_index: torch.Tensor  # [N] local parent id, -1 for the root
    depth: torch.Tensor  # [N]
    child_ptr: torch.Tensor  # [N+1] local CSR row pointer into edge_child/edge_slot
    edge_child: torch.Tensor  # [E] local child ids (== children_index)
    edge_slot: torch.Tensor  # [E]
    expansion_parent_ids: torch.Tensor  # [num_raw_expansions] local parent ids, full (untrimmed) expansion order
    first_decision_expansion_count: int  # root_rank + 1; see _build_compact_trajectory
    step_node_cutoffs: torch.Tensor  # [num_steps] node-count cutoff per (root-onward) step
    node_update_ptr: torch.Tensor  # [N+1] local CSR into the update_* arrays below, sorted by (node_id, step_index)
    update_step_index: torch.Tensor  # [M]
    update_wdl: torch.Tensor  # [M, 3] cumulative mean win/draw/loss as of that update
    update_visit_count: torch.Tensor  # [M] cumulative visit_count as of that update (unused by v1, carried through)
    update_q_value: torch.Tensor  # [M] cumulative mean q_value as of that update (unused by v1, carried through)


def iter_history_trajectories(payload: dict) -> Iterator[HistoryTrajectory]:
    """Decode one packhistory_trees shard payload into per-tree ``HistoryTrajectory`` records.

    See this module's docstring for the assumed shard schema (an extrapolation of
    ``preprocess_mc/pack.py``'s existing ``cts_budgeted_controller_episode_shard_v4``
    format, extended with the update log). This is the one place that assumption is
    encoded -- update it here if Task 1's real field names differ.
    """
    num_trajectories = len(payload["trajectory_source_paths"])
    node_ptr = payload["trajectory_node_ptr"]
    edge_ptr = payload["trajectory_edge_ptr"]
    child_ptr_ptr = payload["trajectory_child_ptr_ptr"]
    expansion_parent_ptr = payload["trajectory_expansion_parent_ptr"]
    step_ptr = payload["trajectory_step_ptr"]
    node_update_ptr_ptr = payload["trajectory_node_update_ptr_ptr"]
    update_log_ptr = payload["trajectory_update_log_ptr"]

    for i in range(num_trajectories):
        node_begin, node_end = int(node_ptr[i]), int(node_ptr[i + 1])
        edge_begin, edge_end = int(edge_ptr[i]), int(edge_ptr[i + 1])
        cpp_begin, cpp_end = int(child_ptr_ptr[i]), int(child_ptr_ptr[i + 1])
        exp_begin, exp_end = int(expansion_parent_ptr[i]), int(expansion_parent_ptr[i + 1])
        step_begin, step_end = int(step_ptr[i]), int(step_ptr[i + 1])
        nup_begin, nup_end = int(node_update_ptr_ptr[i]), int(node_update_ptr_ptr[i + 1])
        ulog_begin, ulog_end = int(update_log_ptr[i]), int(update_log_ptr[i + 1])

        yield HistoryTrajectory(
            source_path=payload["trajectory_source_paths"][i],
            node_features=payload["node_features"][node_begin:node_end],
            parent_index=payload["parent_index"][node_begin:node_end],
            depth=payload["depth"][node_begin:node_end],
            child_ptr=payload["child_ptr"][cpp_begin:cpp_end],
            edge_child=payload["edge_child"][edge_begin:edge_end],
            edge_slot=payload["edge_slot"][edge_begin:edge_end],
            expansion_parent_ids=payload["expansion_parent_ids"][exp_begin:exp_end],
            first_decision_expansion_count=int(payload["first_decision_expansion_counts"][i].item()),
            step_node_cutoffs=payload["step_node_cutoffs"][step_begin:step_end],
            node_update_ptr=payload["node_update_ptr"][nup_begin:nup_end],
            update_step_index=payload["update_log_step_index"][ulog_begin:ulog_end],
            update_wdl=payload["update_log_wdl"][ulog_begin:ulog_end],
            update_visit_count=payload["update_log_visit_count"][ulog_begin:ulog_end],
            update_q_value=payload["update_log_q_value"][ulog_begin:ulog_end],
        )


def deterministic_snapshot_steps(
    source_path: str,
    num_available_steps: int,
    config: PackHistoryGNNPretrainConfig,
) -> List[int]:
    """Sample ``config.snapshots_per_tree`` distinct step indices ``n``, deterministically.

    Mirrors ``deterministic_starting_budgets`` (``preprocess_mc/oracle.py:203-229``):
    every draw hashes ``(source_path, draw_index, seed[, retry-attempt])`` into a
    seeded RNG, so re-running this stage on the same config reproduces identical
    training data -- no unseeded randomness anywhere in this path.

    ``n`` ranges over ``[1, min(config.search_budget, num_available_steps) - lookahead_k]``
    (clamped so both T_n and T_{n+k} exist within *this* tree's own recorded step
    range -- trees can be shorter than the full ``search_budget`` after root-rank
    trimming). Returns an empty list when that range is empty (tree too short for
    this ``lookahead_k``).
    """
    high = min(config.search_budget, num_available_steps) - config.lookahead_k
    if high < 1:
        return []

    chosen: List[int] = []
    seen: set[int] = set()
    for draw_index in range(config.snapshots_per_tree):
        candidate: Optional[int] = None
        # Rejection-resample on collision (only matters once snapshots_per_tree > 1);
        # each retry re-hashes with an incremented, still-deterministic attempt index.
        # Bounded by `high` retries, since there can't be more distinct draws than the
        # range itself.
        for attempt in range(high):
            digest = hashlib.sha256(
                f"{source_path}|{draw_index}|{config.seed}|{attempt}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
            rng = random.Random(seed)
            candidate = rng.randint(1, high)
            if candidate not in seen:
                break
        seen.add(candidate)  # type: ignore[arg-type]
        chosen.append(candidate)  # type: ignore[arg-type]
    return chosen


def _forward_filled_wdl_at_step(
    trajectory: HistoryTrajectory,
    node_ids: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Return ``[K, 3]`` cumulative (win, draw, loss) for each of ``node_ids`` as of ``step``.

    Forward-fill lookup against the sparse update log: for each node, binary-search
    its sorted update-log slice (via ``node_update_ptr``) for the last entry with
    ``step_index <= step``. Nodes never visited by backprop, or whose first update
    hasn't happened yet as of ``step``, fall back to a zero WDL triple -- NOT
    ``node_features``'s own wdl columns, which ``pack.py``'s
    ``_build_compact_trajectory`` overwrites with each node's *final* replayed WDL
    (using that would leak the eventual answer into earlier steps). Zero matches the
    ``EdgeStats()``-zero convention used elsewhere for an unvisited node, and flows
    cleanly through ``_build_tree_n``'s renormalization below (an all-zero triple
    yields ``value=0, variance=0`` via the ``clamp_min(1e-8)`` guard there).
    """
    node_update_ptr = trajectory.node_update_ptr
    update_step_index_list = trajectory.update_step_index.tolist()
    update_wdl = trajectory.update_wdl
    out = torch.zeros((len(node_ids), 3), dtype=trajectory.node_features.dtype)

    for row, node_id in enumerate(node_ids.tolist()):
        lo = int(node_update_ptr[node_id].item())
        hi = int(node_update_ptr[node_id + 1].item())
        if hi <= lo:
            continue  # never visited by backprop; keep the zero baseline
        steps_slice = update_step_index_list[lo:hi]  # sorted ascending for this node
        idx = bisect.bisect_right(steps_slice, step) - 1
        if idx < 0:
            continue  # first update hasn't happened yet as of `step`; keep the zero baseline
        out[row] = update_wdl[lo + idx]
    return out


def _full_scale_step(trajectory: HistoryTrajectory, n: int) -> int:
    """Convert ``n`` (T_n's 1-indexed step, on the trimmed/``root_rank``-relative scale
    ``step_node_cutoffs`` and friends use) into the *full-scale* ``step_index`` convention
    the update log uses (0-indexed from the very start of ``expansion_parent_ids``, i.e.
    before ``_build_compact_trajectory``'s ``root_rank`` trimming -- see
    ``preprocess_mc/pack.py:_build_compact_trajectory``).

    ``local_step = n - 1``; ``expansion_count = trajectory.first_decision_expansion_count +
    local_step`` is the full-scale count of expansions completed as of step n (matches
    ``_build_tree_n``'s structural cutoff and ``ControllerEpisodeDataset``'s
    ``expansion_count``); the matching update-log query is ``expansion_count - 1``
    (``step_index`` is 0-indexed, tagged before increment) = ``first_decision_expansion_count
    + n - 2``.

    Collapses to plain ``n - 1`` whenever ``root_rank == 0`` (i.e.
    ``first_decision_expansion_count == 1``) -- true on every real tree sampled so far,
    since a fresh single-root PUCT search always expands the root first. That is why using
    bare ``n - 1``/``local_step`` directly against the update log (the bug this function
    fixes) never produced a visibly wrong value on real data despite being wrong in
    general -- see ``test_value_query_respects_root_rank_fix``.
    """
    return trajectory.first_decision_expansion_count + n - 2


def _delta_visits(trajectory: HistoryTrajectory, node_id: int, n: int, n_plus_k: int) -> int:
    """Count update-log entries for ``node_id`` with ``step_index`` in
    ``(_full_scale_step(n), _full_scale_step(n_plus_k)]``."""
    lo = int(trajectory.node_update_ptr[node_id].item())
    hi = int(trajectory.node_update_ptr[node_id + 1].item())
    if hi <= lo:
        return 0
    steps_slice = trajectory.update_step_index[lo:hi].tolist()
    upper = bisect.bisect_right(steps_slice, _full_scale_step(trajectory, n_plus_k))
    lower = bisect.bisect_right(steps_slice, _full_scale_step(trajectory, n))
    return upper - lower


@dataclass
class _TreeNSnapshot:
    node_features: torch.Tensor
    parent_index: torch.Tensor
    depth: torch.Tensor
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_slot: torch.Tensor


def _build_tree_n(trajectory: HistoryTrajectory, n: int) -> Optional[_TreeNSnapshot]:
    """Build T_n's structure + forward-filled features.

    Structure (which nodes/edges exist) is prefix-sliced/CSR-gathered exactly the way
    ``ControllerEpisodeDataset.__getitem__`` already does for one step
    (``controller_train.py:876-907``) -- reused, not reimplemented, just specialized to
    a single step instead of the whole ``num_steps`` range. Node feature *values* are
    then overwritten in place with the forward-filled WDL as of step ``n`` (see
    ``_forward_filled_wdl_at_step``), recomputing ``value``/``wdl_var`` from that
    triple via the same formula ``value_features_from_wdl``
    (``core/providers/parsers.py:176-204``) uses, so every column stays internally
    consistent with the forward-filled wdl rather than mixing a stale ``value`` with a
    fresh wdl triple.
    """
    num_steps = int(trajectory.step_node_cutoffs.shape[0])
    local_step = n - 1
    if local_step < 0 or local_step >= num_steps:
        return None

    node_cutoff = int(trajectory.step_node_cutoffs[local_step].item())
    expansion_count = trajectory.first_decision_expansion_count + local_step

    node_ids = torch.arange(node_cutoff, dtype=torch.long)
    # Query at `expansion_count - 1` (= _full_scale_step(trajectory, n)), NOT bare
    # `local_step`: `step_index` is on the *full*, untrimmed expansion-position scale
    # (0-indexed from the very start of `expansion_parent_ids`), while `local_step` is
    # `root_rank`-relative (trimmed). The two coincide only when `root_rank == 0` --
    # true on every real tree checked, which is why querying at bare `local_step` never
    # produced a visibly wrong value in practice despite being wrong in general. See
    # `_full_scale_step`'s docstring and `test_value_query_respects_root_rank_fix`.
    # (Also not `expansion_count` itself, which would leak the *next* expansion's
    # backprop update into T_n's own features.)
    forward_filled_wdl = _forward_filled_wdl_at_step(trajectory, node_ids, expansion_count - 1)

    win, draw, loss = forward_filled_wdl[:, 0], forward_filled_wdl[:, 1], forward_filled_wdl[:, 2]
    total = (win + draw + loss).clamp_min(1e-8)
    p_win, p_draw, p_loss = win / total, draw / total, loss / total
    value = p_win - p_loss
    variance = (p_win + p_loss) - value * value

    node_features_n = trajectory.node_features[:node_cutoff].clone()
    node_features_n[:, 0] = value
    node_features_n[:, 1] = p_win
    node_features_n[:, 2] = p_draw
    node_features_n[:, 3] = p_loss
    node_features_n[:, 4] = variance

    parent_index_n = trajectory.parent_index[:node_cutoff]
    depth_n = trajectory.depth[:node_cutoff]

    # Same "collect edges grouped by parent in sorted-parent order" pattern as
    # ControllerEpisodeDataset.__getitem__ (controller_train.py:883-907).
    active_parents = sorted(int(parent_id) for parent_id in trajectory.expansion_parent_ids[:expansion_count].tolist())
    edge_parent_parts: List[torch.Tensor] = []
    edge_child_parts: List[torch.Tensor] = []
    edge_slot_parts: List[torch.Tensor] = []
    for parent_id in active_parents:
        lo = int(trajectory.child_ptr[parent_id].item())
        hi = int(trajectory.child_ptr[parent_id + 1].item())
        if hi <= lo:
            continue
        count = hi - lo
        edge_parent_parts.append(torch.full((count,), parent_id, dtype=torch.long))
        edge_child_parts.append(trajectory.edge_child[lo:hi])
        edge_slot_parts.append(trajectory.edge_slot[lo:hi])

    if edge_parent_parts:
        edge_parent_n = torch.cat(edge_parent_parts, dim=0)
        edge_child_n = torch.cat(edge_child_parts, dim=0)
        edge_slot_n = torch.cat(edge_slot_parts, dim=0)
    else:
        edge_parent_n = torch.empty(0, dtype=torch.long)
        edge_child_n = torch.empty(0, dtype=torch.long)
        edge_slot_n = torch.empty(0, dtype=torch.long)

    return _TreeNSnapshot(
        node_features=node_features_n,
        parent_index=parent_index_n,
        depth=depth_n,
        edge_parent=edge_parent_n,
        edge_child=edge_child_n,
        edge_slot=edge_slot_n,
    )


def build_snapshot_pair_example(
    trajectory: HistoryTrajectory,
    n: int,
    lookahead_k: int,
    schema: NodeFeatureSchema,
) -> Optional[TensorizedTreeExample]:
    """Build one T_n -> T_{n+k} training example from every edge of T_n, unfiltered.

    Every edge present in T_n is both a candidate training example (matching how
    today's Child-WDL pretraining already supervises every edge of a whole tree from
    one pass -- not a new pattern) AND part of the graph topology fed into the
    encoder's message passing. Those two roles cannot be pulled apart: T_n's
    ``edge_parent``/``edge_child``/``edge_slot`` fields returned here are the literal
    edge set ``TreeEncoder`` runs upward message passing over (via
    ``collate_tensorized_examples`` -> ``TreeBatch``), and ``TreeAttMsgLayer.forward``
    (``models/tree_mha.py``) computes each parent's per-child attention via a
    SEGMENTED SOFTMAX over exactly the children present in those arrays --
    ``parent_sums.scatter_add_`` then ``exp_logits / parent_sums[edge_parent]``.
    Dropping a child changes the softmax denominator, and therefore the attention
    weight and message, for every *other* (kept) child of that same parent.

    An earlier version of this function additionally dropped edges whose child had
    zero update-log entries anywhere in the whole trajectory (the "never-visited-leaf
    filter", formerly implemented by a ``_never_visited`` helper in this module --
    since removed; a copy of the same check now lives only as a test helper in
    ``test_canary_pack_history_structural.py``, no longer part of production code) as
    a compute-savings optimization -- Delta-visits
    is provably zero for such a child regardless of (n, k), so it was assumed to be a
    pure candidate-count reduction with no effect on the surviving edges' targets.
    That assumption was wrong: because of the segmented-softmax coupling above, the
    filter *did* change the training-time prediction for every kept sibling edge under
    the same parent, in a way that never matches production inference. Inference
    (``preprocess_mc/materialize.py``, ``train/controller_train.py``) always feeds the
    encoder the FULL real edge set -- there is no such filter on that path -- so
    filtering here created a train/inference topology mismatch: with ~88.8-97.7% of a
    typical tree's nodes never touched by backprop (see history.md), the vast majority
    of each parent's children were being removed only at training time. This function
    now passes T_n's full, real edge set straight through with no filtering, so the
    training-time topology is identical to what the encoder sees at inference.

    Target for edge (u, v) = v's forward-filled WDL at step ``n + lookahead_k``.
    Weight = Delta-visits, the count of update-log entries for v with step_index in
    ``(n, n + lookahead_k]``. Delta-visits=0 edges (including the now-unfiltered
    never-visited ones) are always kept, never filtered -- this is a per-edge
    *weight*, not a filter: ``_forward_filled_wdl_at_step``'s zero-WDL fallback plus
    the ``clamp_min(1e-8)`` normalization below give a never-visited edge a
    ``[0, 0, 0]`` -> harmless normalized target, and ``_delta_visits`` returns 0 for it
    over any window by construction (zero update-log rows), exactly like any other
    kept edge that happens to have zero Delta-visits for a given (n, k) already did
    before this fix. ``gnn_pretrain.py``'s Delta-visits weighting branch already
    treats weight=0 as "contributes ~nothing to the loss", not "must be excluded" --
    see the ``elif edge_visit_weights is not None`` branch there -- so nothing
    downstream needs to change to make this safe.

    COMPUTE-COST TRADEOFF: removing the filter undoes the ~10-30x candidate-example-
    count reduction it bought (per Task 4's original design note) -- packed shards for
    this stage will now be substantially larger and slower to build/store, since most
    of a tree's nodes are leaves that were never visited by backprop. That is an
    accepted correctness-over-compute tradeoff, not an oversight; a sparse on-disk
    representation for zero-weight edges (to claw back the storage cost without
    reintroducing the topology bug) is a plausible follow-up but is explicitly NOT
    implemented here -- deferred.

    Returns None if T_n has no steps or no edges.
    """
    n_plus_k = n + lookahead_k
    snapshot = _build_tree_n(trajectory, n)
    if snapshot is None or snapshot.edge_child.numel() == 0:
        return None

    edge_parent = snapshot.edge_parent
    edge_child = snapshot.edge_child
    edge_slot = snapshot.edge_slot

    # BUGFIX: `step_index` is on the full, untrimmed expansion-position scale (see
    # `_full_scale_step`'s docstring), so "as of T_{n+k}" is `_full_scale_step(n_plus_k)`,
    # not bare `n_plus_k - 1` (which silently drops the `root_rank` shift -- dormant on
    # every real tree checked so far, since `root_rank == 0` there, but wrong in general;
    # see `test_value_query_respects_root_rank_fix`). `_delta_visits` applies the
    # same conversion internally to both ends of its window, so it's called here with the
    # plain `n`/`n_plus_k` this function already has, not pre-shifted values.
    target_wdl = _forward_filled_wdl_at_step(trajectory, edge_child, _full_scale_step(trajectory, n_plus_k))
    target_mass = target_wdl.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_wdl = target_wdl / target_mass

    weights = torch.tensor(
        [_delta_visits(trajectory, int(child_id), n, n_plus_k) for child_id in edge_child.tolist()],
        dtype=torch.float32,
    )

    return TensorizedTreeExample(
        node_features=snapshot.node_features,
        parent_index=snapshot.parent_index,
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_slot=edge_slot,
        depth=snapshot.depth,
        # Unused placeholder: ChildWdlModel's training loop (applied to T_n here,
        # unmodified -- no new head) reads only edge-level targets/weights
        # (edge_wdl_targets, edge_visit_weights), never node_targets -- see
        # ChildWdlPretrainer._tree_batch_and_edge_targets, which discards the
        # packed dataset's node-target element entirely. Kept only because
        # TensorizedTreeExample/collate_tensorized_examples require the field.
        node_targets=torch.zeros(snapshot.node_features.shape[0], dtype=torch.float32),
        feature_names=schema.feature_names,
        edge_wdl_targets=target_wdl,
        edge_visit_weights=weights,
    )


def read_history_manifest(path: Path) -> List[Path]:
    """Read packhistory_trees's packed shard manifest (JSON ``entries: [{path, ...}]``)."""
    if not path.exists():
        raise FileNotFoundError(f"packhistory_trees manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError(f"No packhistory_trees shard entries found in manifest: {path}")
    return [Path(entry["path"]) for entry in entries]


def _write_futurewdl_shard(shard_path: Path, examples: Sequence[TensorizedTreeExample], schema: NodeFeatureSchema) -> None:
    """Write one packed shard of T_n->T_{n+k} examples, mirroring ``preprocess_gnn/pack.py``'s
    ``pack_train_or_val_split_to_shards`` shard-writing convention (node_ptr/edge_ptr CSR),
    extended with ``edge_visit_weights``."""
    node_ptr = [0]
    edge_ptr = [0]
    node_features_parts, parent_index_parts = [], []
    edge_parent_parts, edge_child_parts, edge_slot_parts = [], [], []
    depth_parts, target_parts = [], []
    edge_wdl_target_parts: List[torch.Tensor] = []
    edge_visit_weight_parts: List[torch.Tensor] = []

    for tensorized in examples:
        node_features_parts.append(tensorized.node_features.cpu())
        parent_index_parts.append(tensorized.parent_index.cpu())
        edge_parent_parts.append(tensorized.edge_parent.cpu())
        edge_child_parts.append(tensorized.edge_child.cpu())
        edge_slot_parts.append(tensorized.edge_slot.cpu())
        depth_parts.append(tensorized.depth.cpu())
        target_parts.append(tensorized.node_targets.cpu())
        assert tensorized.edge_wdl_targets is not None and tensorized.edge_visit_weights is not None, (
            "build_snapshot_pair_example must always populate edge_wdl_targets/edge_visit_weights"
        )
        edge_wdl_target_parts.append(tensorized.edge_wdl_targets.cpu())
        edge_visit_weight_parts.append(tensorized.edge_visit_weights.cpu())
        node_ptr.append(node_ptr[-1] + int(tensorized.node_features.shape[0]))
        edge_ptr.append(edge_ptr[-1] + int(tensorized.edge_parent.shape[0]))

    payload = {
        "format": "cts_tensorized_futurewdl_shard_v1",
        "num_examples": len(examples),
        "feature_names": list(schema.feature_names),
        "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
        "node_features": torch.cat(node_features_parts, dim=0),
        "parent_index": torch.cat(parent_index_parts, dim=0),
        "edge_parent": torch.cat(edge_parent_parts, dim=0) if edge_parent_parts else torch.empty(0, dtype=torch.long),
        "edge_child": torch.cat(edge_child_parts, dim=0) if edge_child_parts else torch.empty(0, dtype=torch.long),
        "edge_slot": torch.cat(edge_slot_parts, dim=0) if edge_slot_parts else torch.empty(0, dtype=torch.long),
        "depth": torch.cat(depth_parts, dim=0),
        "node_targets": torch.cat(target_parts, dim=0),
        "edge_wdl_targets": torch.cat(edge_wdl_target_parts, dim=0),
        "edge_visit_weights": torch.cat(edge_visit_weight_parts, dim=0),
    }
    torch.save(payload, shard_path)


def pack_history_split_to_shards(
    manifest_path: Path,
    output_root: Path,
    config: PackHistoryGNNPretrainConfig,
    *,
    schema: NodeFeatureSchema,
) -> Tuple[Path, dict[str, Any]]:
    """Sample snapshot pairs for every tree in ``manifest_path``'s shards and pack them.

    Returns ``(packed_manifest_path, stats)``.
    """
    split_name = manifest_path.stem.replace("_manifest", "")
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = read_history_manifest(manifest_path)
    start_time = time.time()
    entries: List[dict[str, Any]] = []
    buffered: List[TensorizedTreeExample] = []
    shard_index = 0
    total_examples = 0
    trees_seen = 0
    trees_skipped_too_short = 0
    # Renamed from `trees_skipped_all_edges_filtered`: since the never-visited-leaf
    # filter was removed from build_snapshot_pair_example (see its docstring), the
    # only remaining reason it returns None is a genuinely edgeless T_n (no steps or
    # zero edges) -- "all edges filtered" no longer describes any real code path.
    trees_skipped_no_edges = 0

    def flush() -> None:
        nonlocal shard_index, total_examples
        if not buffered:
            return
        shard_path = split_output_dir / f"shard_{shard_index:05d}.pt"
        _write_futurewdl_shard(shard_path, buffered, schema)
        entries.append({"path": str(shard_path), "num_examples": len(buffered), "shard_index": shard_index})
        total_examples += len(buffered)
        shard_index += 1
        buffered.clear()

    for source_shard_path in shard_paths:
        payload = torch.load(source_shard_path, weights_only=False)
        for trajectory in iter_history_trajectories(payload):
            trees_seen += 1
            num_steps = int(trajectory.step_node_cutoffs.shape[0])
            sampled_ns = deterministic_snapshot_steps(trajectory.source_path, num_steps, config)
            if not sampled_ns:
                trees_skipped_too_short += 1
                continue
            for n in sampled_ns:
                example = build_snapshot_pair_example(trajectory, n, config.lookahead_k, schema)
                if example is None:
                    trees_skipped_no_edges += 1
                    continue
                buffered.append(example)
                if len(buffered) >= config.shard_size:
                    flush()
            if trees_seen % max(config.log_interval, 1) == 0:
                elapsed = time.time() - start_time
                print(
                    f"split={split_name} trees_seen={trees_seen} examples_packed={total_examples + len(buffered)} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
        del payload
    flush()

    stats = {
        "total_examples": total_examples,
        "trees_seen": trees_seen,
        "trees_skipped_too_short": trees_skipped_too_short,
        "trees_skipped_no_edges": trees_skipped_no_edges,
    }
    packed_manifest_path = output_root / f"{split_name}_manifest.json"
    with packed_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "cts_tensorized_futurewdl_manifest_v1",
                "split": split_name,
                "lookahead_k": config.lookahead_k,
                "snapshots_per_tree": config.snapshots_per_tree,
                "entries": entries,
                **stats,
            },
            handle,
            indent=2,
        )
    return packed_manifest_path, stats


def main(config: PackHistoryGNNPretrainConfig) -> None:
    """Pack ``train_manifest.json``/``validation_manifest.json`` from ``pack_history_dir``."""
    if config.lookahead_k <= 0:
        raise ValueError("lookahead_k must be positive.")
    if config.snapshots_per_tree <= 0:
        raise ValueError("snapshots_per_tree must be positive.")
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    pack_history_dir = Path(config.pack_history_dir)
    output_root = Path(config.output_root)

    if config.clear and output_root.exists():
        for child in output_root.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(output_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)

    schema = tree_encoder_feature_schema()
    train_manifest_path, train_stats = pack_history_split_to_shards(
        pack_history_dir / "train_manifest.json", output_root, config, schema=schema
    )
    validation_manifest_path, validation_stats = pack_history_split_to_shards(
        pack_history_dir / "validation_manifest.json", output_root, config, schema=schema
    )

    print(f"train_manifest={train_manifest_path} stats={train_stats}")
    print(f"validation_manifest={validation_manifest_path} stats={validation_stats}")
    print(f"output_root={output_root}")


class PackedFutureWdlShardDataset(Sequence[TensorizedTreeExample]):
    """Dataset over packed T_n->T_{n+k} snapshot-pair shards (this stage's own output).

    Mirrors ``PackedTensorizedShardDataset``
    (``preprocess_gnn/teacher_targets.py:987-1058``) -- same flatten-up-front strategy
    so shuffled ``DataLoader`` access doesn't thrash shards -- but also reads
    ``edge_visit_weights`` into each example. Feed this straight into
    ``ChildWdlPretrainer`` wrapping a plain, unmodified ``ChildWdlModel`` (no new
    head -- see history.md's "Stage: packhistory_GNNpretrain"): its ``collate_fn``
    produces ``(TreeBatch, node_targets)`` pairs exactly like the existing packed
    dataset, with ``TreeBatch.edge_visit_weights`` populated for
    ``gnn_pretrain.py``'s weighting branch to pick up.
    """

    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed futurewdl shard entries found in manifest: {manifest_path}")

        self._examples: List[TensorizedTreeExample] = []
        for entry in entries:
            path = entry["path"]
            payload = torch.load(path, weights_only=False)
            if not isinstance(payload, dict) or payload.get("format") != "cts_tensorized_futurewdl_shard_v1":
                raise ValueError(f"Packed futurewdl shard file has unexpected format: {path}")
            node_ptr = payload["node_ptr"]
            edge_ptr = payload["edge_ptr"]
            feature_names = tuple(payload["feature_names"])
            num_examples = int(entry["num_examples"])
            for i in range(num_examples):
                ns = int(node_ptr[i].item())
                ne = int(node_ptr[i + 1].item())
                es = int(edge_ptr[i].item())
                ee = int(edge_ptr[i + 1].item())
                self._examples.append(
                    TensorizedTreeExample(
                        node_features=payload["node_features"][ns:ne].clone(),
                        parent_index=payload["parent_index"][ns:ne].clone(),
                        edge_parent=payload["edge_parent"][es:ee].clone(),
                        edge_child=payload["edge_child"][es:ee].clone(),
                        edge_slot=payload["edge_slot"][es:ee].clone(),
                        depth=payload["depth"][ns:ne].clone(),
                        node_targets=payload["node_targets"][ns:ne].clone(),
                        feature_names=feature_names,
                        edge_wdl_targets=payload["edge_wdl_targets"][es:ee].clone(),
                        edge_visit_weights=payload["edge_visit_weights"][es:ee].clone(),
                    )
                )
            del payload

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TensorizedTreeExample:
        return self._examples[index]

    @staticmethod
    def collate_fn(batch: Sequence[TensorizedTreeExample]) -> Tuple[Any, Any]:
        """DataLoader collate that defers to the canonical batcher in ``tensorizer``."""
        return collate_tensorized_examples(batch)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(PackHistoryGNNPretrainConfig, main)
