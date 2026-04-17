from __future__ import annotations

import sys
from pathlib import Path

# This demo runs from lmcos/, but core modules live one directory up.
THIS_DIR = Path(__file__).resolve().parent
LMCOS_DIR = THIS_DIR.parent
if str(LMCOS_DIR) not in sys.path:
    sys.path.insert(0, str(LMCOS_DIR))

import torch
from schema import tree_encoder_feature_schema
from tensorizer import TreeTensorizer
from tree import SearchTree, ExpansionChild

def build_tiny_tree(root_fen: str, name: str) -> SearchTree:
    """Builds a tiny tree with a specific shape."""
    t = SearchTree()
    # GNN expects these features from tree_encoder_feature_schema()
    feats = {"value": 0.5, "wdl_win": 0.3, "wdl_draw": 0.4, "wdl_loss": 0.3, "wdl_var": 0.1, "prior": 1.0}
    t.create_root(root_fen, feats, metadata={"name": name})
    
    # Expand root with 2 children
    c1 = ExpansionChild("e2e4", "fen_e4", feats)
    c2 = ExpansionChild("d2d4", "fen_d4", feats)
    child_ids = t.add_children(t.root_id, [c1, c2])
    
    # Expand one child only (making it asymmetric)
    if name == "Tree_A":
        c3 = ExpansionChild("e7e5", "fen_e5", feats)
        t.add_children(child_ids[0], [c3])
        
    return t

def main():
    # 1. Create two different trees
    tree_a = build_tiny_tree("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Tree_A")
    tree_b = build_tiny_tree("k7/8/8/8/8/8/8/K7 w - - 0 1", "Tree_B")
    
    print(f"Tree A Nodes: {tree_a.num_nodes()}") # Should be 4
    print(f"Tree B Nodes: {tree_b.num_nodes()}") # Should be 3
    
    # 2. Tensorize them together
    schema = tree_encoder_feature_schema()
    tensorizer = TreeTensorizer(schema)
    batch = tensorizer.tensorize_forest([tree_a, tree_b])
    
    print("\n--- TreeBatch (Flattened Forest) ---")
    print(f"Total Nodes in Batch: {batch.node_features.shape[0]}") # 4 + 3 = 7
    print(f"Node Features Shape:  {batch.node_features.shape}")
    
    print("\n--- Structural Tensors ---")
    # depth tells the GNN which nodes to process in which 'Sequential' round
    print(f"Depth Vector:         {batch.depth.tolist()}") 
    
    # parent_index tells the GNN where to send downward messages
    # Note how the indices in Tree B are shifted by 4!
    print(f"Parent Index:         {batch.parent_index.tolist()}")
    
    # root_index tells the GNN which nodes represent the 'Start' of each tree
    print(f"Root Indices:         {batch.root_index.tolist()}")
    
    print("\n--- How the GNN 'Sweeps' ---")
    max_d = int(batch.depth.max())
    for d in range(max_d + 1):
        nodes_at_d = (batch.depth == d).nonzero().flatten().tolist()
        print(f"Depth {d}: Nodes {nodes_at_d}")

if __name__ == "__main__":
    main()
