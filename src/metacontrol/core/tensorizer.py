import torch
from typing import TYPE_CHECKING, Any, Dict, List

from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import ChessFeatureSchema, TreeBatch

if TYPE_CHECKING:
    from metacontrol.core.schemas import GeneratorResult


class TreeTensorizer:
    def __init__(self, schema: ChessFeatureSchema):
        self.schema = schema

    def tensorize(self, tree: SearchTree) -> TreeBatch:
        return self.tensorize_forest([tree])

    def tensorize_forest(self, trees: List[SearchTree]) -> TreeBatch:
        node_features = []
        parent_index = []
        edge_parent = []
        edge_child = []
        edge_slot = []
        depth = []
        root_index = []
        tree_index = []
        
        node_offset = 0
        for tree_idx, tree in enumerate(trees):
            root_index.append(node_offset + tree.root.node_id)
            
            # Use a deterministic order for nodes (by node_id)
            sorted_nodes = sorted(tree.nodes_by_id.values(), key=lambda n: n.node_id)
            
            for node in sorted_nodes:
                node_features.append(self.schema.vectorize(node.features))
                depth.append(node.depth)
                tree_index.append(tree_idx)
                
                if node.parent is not None:
                    parent_index.append(node_offset + node.parent.node_id)
                else:
                    parent_index.append(-1)
                
                # Sorted children for slot stability (UCI sorting)
                sorted_moves = sorted(node.children.keys())
                for slot, move in enumerate(sorted_moves):
                    child = node.children[move]
                    edge_parent.append(node_offset + node.node_id)
                    edge_child.append(node_offset + child.node_id)
                    edge_slot.append(slot)
            
            node_offset += len(tree.nodes_by_id)
            
        return TreeBatch(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            parent_index=torch.tensor(parent_index, dtype=torch.long),
            edge_parent=torch.tensor(edge_parent, dtype=torch.long),
            edge_child=torch.tensor(edge_child, dtype=torch.long),
            edge_slot=torch.tensor(edge_slot, dtype=torch.long),
            depth=torch.tensor(depth, dtype=torch.long),
            root_index=torch.tensor(root_index, dtype=torch.long),
            tree_index=torch.tensor(tree_index, dtype=torch.long),
            num_nodes=len(node_features),
            num_edges=len(edge_parent),
            batch_size=len(trees)
        )

    def pack_training_shard(
        self,
        result: "GeneratorResult",
        *,
        continue_cost: float = 1e-3,
    ) -> Dict[str, Any]:
        """Tensorize *result.tree* and attach MC/GNN targets (see :mod:`tree_pack`)."""
        from metacontrol.data.tree_pack import pack_single_tree_like_legacy_shard

        return pack_single_tree_like_legacy_shard(
            result, continue_cost=continue_cost, schema=self.schema
        )
