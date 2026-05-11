"""GNN target derivation from search history.

This module computes the supervised learning targets for the GNN:
1. Node values: visit-weighted average of child Q-values.
2. Edge WDLs: visit-weighted average of child WDLs, or flipped static WDL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from metacontrol.core.tree import SearchTree
from metacontrol.data.generator import EdgeStats, node_wdl, flip_wdl


@dataclass
class GNNTargets:
    """Supervised targets for training the GNN on a generated search tree."""
    node_values: Dict[int, float]
    edge_wdls: Dict[Tuple[int, int], Tuple[float, float, float]]


def normalize_wdl(wdl: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Ensure WDL sums to 1.0."""
    w, d, l = max(0.0, wdl[0]), max(0.0, wdl[1]), max(0.0, wdl[2])
    total = w + d + l
    if total <= 0.0:
        raise ValueError("WDL must have positive mass")
    return (w / total, d / total, l / total)


def _compute_node_value(
    tree: SearchTree,
    node_id: int,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
) -> float:
    """Compute the visit-weighted average Q-value of the node's children."""
    node = tree.get_node(node_id)
    if not node.children:
        return node.features.get("value", 0.0)

    total_visits = 0
    weighted_sum = 0.0

    for child_id in (child.node_id for child in node.children.values()):
        stats = edge_stats.get((node_id, child_id))
        if stats is not None and stats.visit_count > 0:
            total_visits += stats.visit_count
            weighted_sum += stats.visit_count * stats.q_value

    if total_visits == 0:
        return node.features.get("value", 0.0)

    return weighted_sum / total_visits


def _compute_edge_wdls(
    tree: SearchTree,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    """Compute target WDLs for every edge in the tree."""
    edge_targets = {}

    for node in tree.nodes_by_id.values():
        for child_node in node.children.values():
            edge_key = (node.node_id, child_node.node_id)
            stats = edge_stats.get(edge_key)

            if stats is not None and stats.visit_count > 0 and sum(stats.mean_wdl) > 0.0:
                edge_targets[edge_key] = normalize_wdl(stats.mean_wdl)
            else:
                # Use static child WDL, flipped to parent perspective
                child_wdl = node_wdl(child_node)
                if child_wdl is not None:
                    edge_targets[edge_key] = flip_wdl(normalize_wdl(child_wdl))

    return edge_targets


def compute_gnn_targets(
    tree: SearchTree,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
) -> GNNTargets:
    """Compute all GNN targets for the given tree."""
    node_values = {
        node_id: _compute_node_value(tree, node_id, edge_stats)
        for node_id in tree.nodes_by_id.keys()
    }
    edge_wdls = _compute_edge_wdls(tree, edge_stats)

    return GNNTargets(node_values=node_values, edge_wdls=edge_wdls)
