"""Pack a single :class:`SearchTree` into a torch-serializable dict.

The legacy ``ysagiv`` controller shards (``cts_budgeted_controller_episode_shard_v4``)
store many trajectories per file with RL pointer columns. For Phase~1 supervised
trees we emit a smaller, documented format that reuses the same *tensor* keys
where practical (``node_features``, ``parent_index``, ``edge_child``, …) and
adds explicit MC/GNN supervision tensors.

See the repository ``README.md`` metacontrol section for a schema comparison.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from metacontrol.core.schemas import ChessFeatureSchema, GeneratorResult
from metacontrol.core.tensorizer import TreeTensorizer
from metacontrol.data.targets_gnn import compute_gnn_targets
from metacontrol.data.targets_mc import calculate_dp_values, compute_halt_rewards, derive_snapshots


def pack_single_tree_like_legacy_shard(
    result: GeneratorResult,
    *,
    continue_cost: float = 1e-3,
    schema: ChessFeatureSchema | None = None,
) -> Dict[str, Any]:
    """Tensorize *result* and attach MC/GNN targets for one tree.

    **Compared to** ``ysagiv`` packed shards:

    - ``format`` is ``metacontrol_single_tree_v1`` (not ``cts_budgeted_…``).
    - RL replay columns (``trajectory_node_ptr``, ``episode_step_ptr``, …) are
      omitted; loaders must not assume they exist.
    - ``edge_child`` uses ``torch.int32`` like legacy shards; ``edge_parent`` is
      included for convenience (not present on all legacy shards).
    - ``target_advantages`` holds per-*snapshot* advantages (length
      ``num_snapshots``), not the flattened multi-trajectory layout of production
      controller shards.
    - ``oracle_values`` stores per-node GNN consolidated value targets (length
      ``num_nodes``), not per-episode oracle scalars from budgeted packing.
    """
    schema = schema or ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    batch = tensorizer.tensorize(result.tree)

    gnn = compute_gnn_targets(result.tree, result.edge_stats)
    halt = compute_halt_rewards(result.tree, result.edge_stats)
    dp = calculate_dp_values(halt, continue_cost)
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost)

    num_nodes = batch.num_nodes
    num_edges = batch.num_edges

    oracle_values = torch.zeros(num_nodes, dtype=torch.float32)
    for nid, val in gnn.node_values.items():
        if 0 <= int(nid) < num_nodes:
            oracle_values[int(nid)] = float(val)

    edge_wdl = torch.zeros(num_edges, 3, dtype=torch.float32)
    ep = batch.edge_parent.tolist()
    ec = batch.edge_child.tolist()
    for i, (p, c) in enumerate(zip(ep, ec)):
        wdl = gnn.edge_wdls.get((int(p), int(c)))
        if wdl is not None:
            edge_wdl[i, 0], edge_wdl[i, 1], edge_wdl[i, 2] = float(wdl[0]), float(wdl[1]), float(wdl[2])

    mc_halt = torch.tensor([s.halt_reward for s in snapshots], dtype=torch.float32)
    mc_adv = torch.tensor([s.advantage for s in snapshots], dtype=torch.float32)
    mc_dp = torch.tensor(dp, dtype=torch.float32)

    return {
        "format": "metacontrol_single_tree_v1",
        "num_trajectories": 1,
        "num_episodes": 1,
        "feature_names": list(schema.feature_names),
        "reward_scale": 1.0,
        "continue_cost": continue_cost,
        "node_features": batch.node_features,
        "parent_index": batch.parent_index,
        "edge_parent": batch.edge_parent.to(torch.int64),
        "edge_child": batch.edge_child.to(torch.int32),
        "edge_slot": batch.edge_slot,
        "depth": batch.depth,
        "root_index": batch.root_index,
        "tree_index": batch.tree_index,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_snapshots": len(snapshots),
        "oracle_values": oracle_values,
        "edge_wdl_targets": edge_wdl,
        "mc_halt_rewards": mc_halt,
        "mc_dp_values": mc_dp,
        "target_advantages": mc_adv,
        "full_fen": result.tree.root.fen,
        "num_expansions": result.num_expansions,
    }
