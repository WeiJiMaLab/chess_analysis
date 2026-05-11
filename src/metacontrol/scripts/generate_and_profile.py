from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from pathlib import Path
import pandas as pd
import torch
import sys

from metacontrol.core.providers import UciExpansionProvider, StockfishExpansionProvider
from metacontrol.data.generator import TreeSearch, UcbNodeSelection, FullExpansion
from metacontrol.data.targets_mc import compute_mc_targets
from metacontrol.data.targets_gnn import compute_gnn_targets
from metacontrol.core.schemas import ChessFeatureSchema
from metacontrol.core.tensorizer import TreeTensorizer


def generate_and_save_tree(fen: str, out_path: Path):
    print(f"Generating tree for FEN: {fen}")
    
    # 1. Initialize Engine
    # Using StockfishExpansionProvider for stockfish directly since it's the simplest
    # We will just use the default path "stockfish" 
    expansion_strategy = StockfishExpansionProvider("stockfish", nodes=64)
    selection_strategy = UcbNodeSelection()
    
    tree_search = TreeSearch(
        expansion_strategy=expansion_strategy,
        selection_strategy=selection_strategy,
        max_nodes=64,
        max_depth=10
    )
    
    # 2. Grow Tree
    t0 = time.time()
    tree = tree_search.search(fen)
    t_grow = time.time() - t0
    print(f"Tree grown to {len(tree.nodes_by_id)} nodes in {t_grow:.2f}s")
    
    # 3. Calculate Targets
    t1 = time.time()
    mc_targets = compute_mc_targets(tree)
    gnn_targets = compute_gnn_targets(tree, mc_targets.edge_stats)
    t_targets = time.time() - t1
    print(f"Calculated MC and GNN targets in {t_targets:.2f}s")
    
    # 4. Tensorize
    t2 = time.time()
    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    batch = tensorizer.tensorize(tree)
    t_tensorize = time.time() - t2
    print(f"Tensorized tree in {t_tensorize:.2f}s")
    
    # 5. Pack matching ysagiv style
    t3 = time.time()
    
    # Extract arrays
    node_features = batch.node_features
    parent_index = batch.parent_index
    edge_child = batch.edge_child
    edge_slot = batch.edge_slot
    depth = batch.depth
    
    # Map targets
    num_nodes = len(tree.nodes_by_id)
    target_advantages = torch.zeros(num_nodes, dtype=torch.float32)
    oracle_values = torch.zeros(num_nodes, dtype=torch.float32)
    
    for node_id, stats in mc_targets.node_stats.items():
        if node_id < num_nodes:
            target_advantages[node_id] = stats.advantage
            
    for node_id, value in gnn_targets.node_values.items():
        if node_id < num_nodes:
            oracle_values[node_id] = value
    
    packed_data = {
        'format': 'metacontrol_v1',
        'num_trajectories': 0, # Omitted RL trajectory tracking
        'num_episodes': 0,
        'feature_names': ['turn', 'white_kingside_castle', 'white_queenside_castle', 'black_kingside_castle', 'black_queenside_castle'],
        'reward_scale': 1.0,
        'node_features': node_features,
        'parent_index': parent_index,
        'edge_child': edge_child,
        'edge_slot': edge_slot,
        'depth': depth,
        'target_advantages': target_advantages,
        'oracle_values': oracle_values,
        # Following omitted or mocked because they are not needed for supervised trees:
        # trajectory_node_ptr, trajectory_step_ptr, episode_step_ptr, etc.
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed_data, out_path)
    t_pack = time.time() - t3
    print(f"Packed and saved to {out_path} in {t_pack:.2f}s")
    print(f"Total time: {time.time() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv")
    parser.add_argument("--out", default="/scratch/gpfs/GRIFFITHS/hl4291/data/trees/shard_00001.pt")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("CSV is empty!")
        return

    fen = df.iloc[0]["full_fen"]
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        generate_and_save_tree(fen, Path(args.out))
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        print("\n--- Top 20 Bottlenecks ---")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
