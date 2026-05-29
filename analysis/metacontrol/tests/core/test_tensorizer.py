import pytest
import torch
from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import ChessFeatureSchema, SearchNode
from metacontrol.core.tensorizer import TreeTensorizer

def test_single_leaf_tensorization():
    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    root = SearchNode(node_id=0, fen="start", features={"value": 0.5, "wdl_win": 0.1, "wdl_draw": 0.8, "wdl_loss": 0.1})
    tree = SearchTree(root)
    
    batch = tensorizer.tensorize(tree)
    assert batch.num_nodes == 1
    assert batch.num_edges == 0
    assert batch.node_features.shape == (1, 5)
    assert torch.allclose(
        batch.node_features[0],
        torch.tensor([0.5, 0.1, 0.8, 0.1, 0.0]),
    )

def test_tensorization_different_sized_trees():
    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    
    # Tree 1: 1 node
    root1 = SearchNode(node_id=0, fen="tree1_root")
    tree1 = SearchTree(root1)
    
    # Tree 2: 2 nodes
    root2 = SearchNode(node_id=0, fen="tree2_root")
    tree2 = SearchTree(root2)
    child2 = SearchNode(node_id=1, fen="tree2_child")
    tree2.add_node(0, "e2e4", child2)
    
    batch = tensorizer.tensorize_forest([tree1, tree2])
    assert batch.num_nodes == 3 # 1 + 2
    assert batch.num_edges == 1
    assert batch.batch_size == 2
    assert batch.root_index[0] == 0
    assert batch.root_index[1] == 1 # Offset by 1 node from tree1

def test_ids_unique():
    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    
    root1 = SearchNode(node_id=0, fen="tree1_root")
    tree1 = SearchTree(root1)
    
    root2 = SearchNode(node_id=0, fen="tree2_root")
    tree2 = SearchTree(root2)
    
    batch = tensorizer.tensorize_forest([tree1, tree2])
    # tree_index should distinguish them
    assert batch.tree_index[0] == 0
    assert batch.tree_index[1] == 1

def test_move_slot_stability():
    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    
    root = SearchNode(node_id=0, fen="start")
    tree = SearchTree(root)
    # Add moves in non-UCI order
    tree.add_node(0, "g1f3", SearchNode(node_id=1, fen="nf3"))
    tree.add_node(0, "e2e4", SearchNode(node_id=2, fen="e4"))
    
    batch = tensorizer.tensorize(tree)
    # Sorted order should be e2e4 (slot 0), g1f3 (slot 1)
    assert batch.edge_slot[0] == 0 # e2e4
    assert batch.edge_slot[1] == 1 # g1f3
    assert batch.edge_child[0] == 2 # child 2 is e4
    assert batch.edge_child[1] == 1 # child 1 is nf3

def test_round_trip():
    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    
    root = SearchNode(node_id=0, fen="start")
    tree = SearchTree(root)
    child = SearchNode(node_id=1, fen="child")
    tree.add_node(0, "e2e4", child)
    
    batch = tensorizer.tensorize(tree)
    assert batch.num_nodes == 2
    assert batch.num_edges == 1
    # Check parent/child relationship in tensor
    p_idx = batch.edge_parent[0]
    c_idx = batch.edge_child[0]
    assert p_idx == 0
    assert c_idx == 1
    assert batch.parent_index[c_idx] == p_idx
