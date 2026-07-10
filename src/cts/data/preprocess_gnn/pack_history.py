"""Pack T_n -> T_{n+k} snapshot-pair training examples ("packhistory_GNNpretrain" stage).

See ``history.md`` (repo root) for the full plan; this module implements the
"packhistory_GNNpretrain" stage described there. Summary: instead of pretraining
the GNN encoder on one example per *complete* tree (today's ``preprocess_gnn/pack.py``
+ ``ChildWdlModel``), we sample a step ``n`` per tree, build ``T_n`` (the tree as it
existed at step n), and supervise every edge of ``T_n`` with a target read off the
*same* tree's own future -- its recorded state at step ``n + lookahead_k`` -- since a
tree's own history at a later step is just a later point in its own already-completed
96-step trajectory. No new search is run; everything is read from
``packhistory_trees``'s packed output (never raw trees).

**Upstream contract (packhistory_trees's packed shard format) -- IMPORTANT**: as of
this module's authorship, Task 1 (``packhistory_trees``, the ``preprocess_mc/pack.py``
rewrite) had not yet landed real code (see ``history.md`` Progress Log, Task 1
subsection). The sparse update-log *schema* is fully fixed by the plan (the
``step_index``/``node_id``/``visit_count``/``q_value``/``wdl`` columns + the
``node_update_ptr`` CSR index, sorted by ``(node_id, step_index)``); everything else
about the packed shard's exact field layout is this module's own extrapolation,
built by mirroring the *existing* ``preprocess_mc/pack.py`` shard format
(``cts_budgeted_controller_episode_shard_v4``, see ``_serialize_shard_payload`` and
``_accumulate_shard_buffers`` in that file) -- since packhistory_trees is explicitly a
rewrite of that stage's *data source*, not its *control flow* or output structure,
and Task 2's diff-based regression test requires the structural fields
(``node_features``, ``parent_index``, ``child_ptr``, ``depth``, ``expansion_parent_ids``,
``step_node_cutoffs``, ``first_decision_expansion_counts``, the ``trajectory_*_ptr`` CSR
index arrays, etc.) to stay byte-identical to today's ``mc_packed/`` output. This
module therefore assumes packhistory_trees's shard adds exactly two new pieces on top
of that unchanged structure:

  1. Five new flat, shard-concatenated arrays holding the sparse update log, using the
     schema fixed in history.md: ``update_log_step_index``, ``update_log_node_id``,
     ``update_log_visit_count``, ``update_log_q_value``, ``update_log_wdl``.
  2. A per-trajectory-local CSR index ``node_update_ptr`` (one array per trajectory,
     shape ``[N_i + 1]``, local node ids -- mirrors how ``child_ptr`` is already
     trajectory-local) into those five arrays, sliced out via two new pointer arrays
     that follow the exact same convention as the existing ``trajectory_child_ptr_ptr``
     / ``trajectory_edge_ptr``: ``trajectory_node_update_ptr_ptr`` (slices
     ``node_update_ptr``) and ``trajectory_update_log_ptr`` (slices the five flat
     update-log arrays).

``iter_history_trajectories`` below is the single seam where this assumption lives --
if Task 1's real field names differ, only that function (plus ``HistoryTrajectory``'s
field list) needs to change; everything downstream operates on the decoded
``HistoryTrajectory`` dataclass, not the raw payload dict. This is flagged in
``history.md``'s Task 4 Progress Log entry for Wave-2 integration.

**Step-index coordinate system -- also flagged, also unconfirmed**: ``n`` here is
1-indexed into the trajectory's own root-rank-trimmed step sequence, i.e.
``local_step = n - 1`` indexes ``step_node_cutoffs`` / ``expansion_parent_ids`` the
same way ``ControllerEpisodeDataset.__getitem__`` already does
(``controller_train.py:876-907``), and the update log's ``step_index`` column is
assumed to live in this *same* coordinate system (both are produced by the same
replay loop in Task 1, so a single consistent numbering is the natural design, but
this has not been confirmed against Task 1's real code).
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

    Forward-fill lookup against the sparse update log: for each node, find the last
    update-log entry with ``step_index <= step`` (binary search over that node's own
    sorted slice via ``node_update_ptr``). Nodes with zero update-log entries at all
    (never visited by backprop), or whose first update hasn't happened yet as of
    ``step``, fall back to the node's static baseline WDL -- read straight off its own
    ``node_features`` row (``wdl_win``/``wdl_draw``/``wdl_loss`` columns, indices 1:4
    per ``TREE_ENCODER_FEATURE_NAMES``), i.e. the same static leaf WDL a node starts
    at before its first backprop update.
    """
    node_update_ptr = trajectory.node_update_ptr
    update_step_index_list = trajectory.update_step_index.tolist()
    update_wdl = trajectory.update_wdl
    baseline_wdl = trajectory.node_features[node_ids][:, 1:4].clone()

    out = baseline_wdl
    for row, node_id in enumerate(node_ids.tolist()):
        lo = int(node_update_ptr[node_id].item())
        hi = int(node_update_ptr[node_id + 1].item())
        if hi <= lo:
            continue  # never visited by backprop; keep the static baseline
        steps_slice = update_step_index_list[lo:hi]  # sorted ascending for this node
        idx = bisect.bisect_right(steps_slice, step) - 1
        if idx < 0:
            continue  # first update hasn't happened yet as of `step`; keep the static baseline
        out[row] = update_wdl[lo + idx]
    return out


def _never_visited(trajectory: HistoryTrajectory, node_id: int) -> bool:
    """True iff ``node_id`` has zero update-log entries across the *entire* tree.

    Such nodes were created as some ancestor's child but never themselves selected by
    backprop during the original search -- Delta-visits is provably zero for them
    regardless of ``n``/``lookahead_k``. This is the "never-visited-leaf" filter from
    history.md (measured there at ~90-98% of nodes in typical trees).
    """
    lo = int(trajectory.node_update_ptr[node_id].item())
    hi = int(trajectory.node_update_ptr[node_id + 1].item())
    return hi <= lo


def _delta_visits(trajectory: HistoryTrajectory, node_id: int, n: int, n_plus_k: int) -> int:
    """Count update-log entries for ``node_id`` with ``step_index`` in ``(n, n_plus_k]``."""
    lo = int(trajectory.node_update_ptr[node_id].item())
    hi = int(trajectory.node_update_ptr[node_id + 1].item())
    if hi <= lo:
        return 0
    steps_slice = trajectory.update_step_index[lo:hi].tolist()
    upper = bisect.bisect_right(steps_slice, n_plus_k)
    lower = bisect.bisect_right(steps_slice, n)
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
    # BUGFIX: query at `local_step` (= n - 1), not `n`. `step_index` in the update log
    # is 0-indexed (matches `local_step`, per _replay_backprop_history's own convention
    # -- row k of oracle_root_q_trace lines up with step_index == k), same convention
    # `node_cutoff` above already correctly uses. Querying at `n` directly leaked the
    # *next* expansion's backprop update into T_n's own features (found by due-diligence
    # review against real data, history.md Task 4 Progress Log).
    forward_filled_wdl = _forward_filled_wdl_at_step(trajectory, node_ids, local_step)

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
    """Build one T_n -> T_{n+k} training example from every (filtered) edge of T_n.

    Every edge present in T_n is a candidate training example from this one forward
    pass (matching how today's Child-WDL pretraining already supervises every edge of
    a whole tree from one pass -- not a new pattern), except edges whose child has
    zero update-log entries anywhere in the tree (the never-visited-leaf filter --
    Delta-visits is provably always zero for those, for any (n, k)).

    Target for edge (u, v) = v's forward-filled WDL at step ``n + lookahead_k``.
    Weight = Delta-visits, the count of update-log entries for v with step_index in
    ``(n, n + lookahead_k]``. Delta-visits=0 edges are kept (not filtered) -- this is
    a per-example *weight*, not a filter; note the never-visited-leaf filter above is
    a strictly stronger, k-independent condition than "this particular (n, k)
    happened to see zero visits."

    Returns None if T_n has no steps, no edges, or every edge gets filtered out.
    """
    n_plus_k = n + lookahead_k
    snapshot = _build_tree_n(trajectory, n)
    if snapshot is None or snapshot.edge_child.numel() == 0:
        return None

    keep_mask = torch.tensor(
        [not _never_visited(trajectory, int(child_id)) for child_id in snapshot.edge_child.tolist()],
        dtype=torch.bool,
    )
    if not bool(keep_mask.any()):
        return None

    kept_parent = snapshot.edge_parent[keep_mask]
    kept_child = snapshot.edge_child[keep_mask]
    kept_slot = snapshot.edge_slot[keep_mask]

    # BUGFIX (same root cause as _build_tree_n's local_step fix above): `step_index` is
    # 0-indexed, so "as of T_{n+k}" is step_index `n_plus_k - 1`, not `n_plus_k`. And for
    # Delta-visits to describe the same (T_n, T_{n+k}) pair the value target now does,
    # its window must shift by the same -1 on both ends: "new visits since T_n" means
    # step_index in (n - 1, n_plus_k - 1], not (n, n_plus_k].
    target_wdl = _forward_filled_wdl_at_step(trajectory, kept_child, n_plus_k - 1)
    target_mass = target_wdl.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_wdl = target_wdl / target_mass

    weights = torch.tensor(
        [_delta_visits(trajectory, int(child_id), n - 1, n_plus_k - 1) for child_id in kept_child.tolist()],
        dtype=torch.float32,
    )

    return TensorizedTreeExample(
        node_features=snapshot.node_features,
        parent_index=snapshot.parent_index,
        edge_parent=kept_parent,
        edge_child=kept_child,
        edge_slot=kept_slot,
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
    trees_skipped_all_edges_filtered = 0

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
                    trees_skipped_all_edges_filtered += 1
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
        "trees_skipped_all_edges_filtered": trees_skipped_all_edges_filtered,
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
