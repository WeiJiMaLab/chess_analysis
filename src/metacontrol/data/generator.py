"""Search tree generator: select → expand → backpropagate."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from metacontrol.core.providers import TreeExpansionProvider
from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import ChessFeatureSchema, SearchNode, ChildInfo, GeneratorConfig, EdgeStats, GeneratorResult
from metacontrol.core.tensorizer import TreeTensorizer



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_priors(raw: Sequence[float]) -> List[float]:
    """Softmax-normalise raw prior scores."""
    if not raw:
        return []
    max_val = max(raw)
    exps = [math.exp(v - max_val) for v in raw]
    total = sum(exps)
    return [e / total for e in exps]


def node_wdl(node: SearchNode) -> Optional[Tuple[float, float, float]]:
    """Extract WDL tuple from a node's features, or None if absent."""
    f = node.features
    if "wdl_win" in f:
        return (f["wdl_win"], f["wdl_draw"], f["wdl_loss"])
    return None


def flip_wdl(wdl: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (wdl[2], wdl[1], wdl[0])


# ---------------------------------------------------------------------------
# TreeSearch
# ---------------------------------------------------------------------------

class TreeSearch:
    """Grows a search tree via iterative select → expand → backpropagate.

    The provider handles chess-specific evaluation; this class implements
    the PUCT search loop.
    """

    def __init__(self, provider: TreeExpansionProvider, config: GeneratorConfig):
        self.provider = provider
        self.config = config
        self.tree: Optional[SearchTree] = None
        self.edge_stats: Dict[Tuple[int, int], EdgeStats] = {}
        self.num_expansions: int = 0
        self._next_id: int = 1

    # -- Main entry point ---------------------------------------------------

    def generate(self, root_fen: str) -> GeneratorResult:
        """Build a search tree from root_fen."""
        self.init_root(root_fen)

        while self.num_expansions < self.config.max_nodes:
            leaf_id, path = self.select_leaf()
            leaf = self.tree.get_node(leaf_id)

            if leaf.is_terminal or leaf.depth >= self.config.max_depth:
                self.backpropagate(path, leaf)
                if not self.has_expandable_frontier():
                    break
                continue

            if leaf.children:
                self.backpropagate(path, leaf)
                continue

            expanded = self.expand_leaf(leaf, path)
            if not expanded and not self.has_expandable_frontier():
                break

        return GeneratorResult(self.tree, self.edge_stats, self.num_expansions)

    def save(
        self,
        result: GeneratorResult,
        path: str | Path,
        *,
        continue_cost: float = 1e-3,
        schema: ChessFeatureSchema | None = None,
    ) -> Dict[str, Any]:
        """Save one generated tree as a torch payload and return that payload.

        This is the raw single-tree output step:

        ``TreeSearch.generate(...) -> GeneratorResult -> one .pt file``.

        The later many-file collation step should own the term "packing" and
        produce packed training shards.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        from metacontrol.data.targets_gnn import compute_gnn_targets
        from metacontrol.data.targets_mc import calculate_dp_values, compute_halt_rewards, derive_snapshots

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

        payload = {
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
        torch.save(payload, out_path)
        return payload

    # -- Search phases ------------------------------------------------------

    def init_root(self, root_fen: str) -> None:
        value, wdl = self.provider.evaluate_root(root_fen)
        features = {"value": value, "prior": 1.0}
        if wdl is not None:
            features["wdl_win"], features["wdl_draw"], features["wdl_loss"] = wdl

        self.tree = SearchTree(SearchNode(node_id=0, fen=root_fen, features=features))
        self.edge_stats = {}
        self.num_expansions = 0
        self._next_id = 1

    def select_leaf(self) -> Tuple[int, List[Tuple[int, int]]]:
        """PUCT walk from root to an unexpanded leaf."""
        path: List[Tuple[int, int]] = []
        node = self.tree.root

        while node.children:
            child_items = list(node.children.items())
            total_visits = sum(
                self.edge_stats[(node.node_id, c.node_id)].visit_count
                for _, c in child_items
            )
            explore_scale = self.config.c_puct * math.sqrt(total_visits + 1.0)

            best, best_score = child_items[0][1], float("-inf")
            for _, child in child_items:
                stats = self.edge_stats[(node.node_id, child.node_id)]
                q = stats.q_value if stats.visit_count > 0 else 0.0
                prior = child.features.get("prior", 0.0)
                score = q + prior * explore_scale / (1.0 + stats.visit_count)
                if score > best_score:
                    best_score, best = score, child

            path.append((node.node_id, best.node_id))
            node = best

        return node.node_id, path

    def expand_leaf(self, leaf: SearchNode, path: List[Tuple[int, int]]) -> bool:
        """Expand leaf via provider. Returns True if children were added."""
        children = self.provider.expand(leaf.fen, leaf.depth)
        if not children:
            leaf.is_terminal = True
            self.backpropagate(path, leaf)
            return False

        priors = normalize_priors([c.prior for c in children])
        for info, prior in zip(children, priors):
            features = {"value": info.value, "prior": prior}
            if info.wdl is not None:
                features["wdl_win"], features["wdl_draw"], features["wdl_loss"] = info.wdl

            child = SearchNode(
                node_id=self._next_id,
                fen=info.fen,
                is_terminal=info.is_terminal,
                features=features,
            )
            self.tree.add_node(leaf.node_id, info.move_uci, child)
            self.edge_stats[(leaf.node_id, child.node_id)] = EdgeStats()
            self._next_id += 1

        self.tree.search_expansion_history.append(leaf)
        self.num_expansions += 1
        self.backpropagate(path, leaf)
        return True

    def backpropagate(self, path: List[Tuple[int, int]], leaf: SearchNode) -> None:
        """Propagate leaf value (and WDL) up the selection path, negating at each level."""
        value = leaf.features.get("value", 0.0)
        wdl = node_wdl(leaf)

        for parent_id, child_id in reversed(path):
            value = -value
            stats = self.edge_stats[(parent_id, child_id)]
            stats.visit_count += 1
            stats.total_value += value
            stats.q_value = stats.total_value / stats.visit_count

            if wdl is not None:
                wdl = flip_wdl(wdl)
                stats.total_wdl = tuple(stats.total_wdl[i] + wdl[i] for i in range(3))
                stats.mean_wdl = tuple(c / stats.visit_count for c in stats.total_wdl)

    def has_expandable_frontier(self) -> bool:
        for node in self.tree.nodes_by_id.values():
            if not node.children and not node.is_terminal and node.depth < self.config.max_depth:
                return True
        return False
