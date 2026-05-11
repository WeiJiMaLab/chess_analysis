import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# --- Markdown Cell: Header ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
# Metacontrol Data Generation Pipeline
### A Didactic Walkthrough of Search Tree Generation and Target Derivation

This notebook demonstrates the modular pipeline for generating chess search trees and computing the associated training targets for:
1. **GNN Pretraining**: Supervised learning of value and WDL heads.
2. **Meta-Control (Planning Head)**: RL-based advantage regression for search termination.

We use the **actual LC0 engine** to provide realistic search statistics.
"""))

# --- Markdown Cell: Imports ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
## 1. Environment Setup and Imports
We import the core components from the `metacontrol` package. Note the separation between `core` (data structures), `data` (generation logic), and `utils` (visualization).
"""))

nb['cells'].append(nbf.v4.new_code_cell("""
import sys
import os
import duckdb
import chess
import random
import torch
import matplotlib.pyplot as plt
from IPython.display import display

# Add src to path
sys.path.append(os.path.abspath("../../"))

from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import GeneratorConfig
from metacontrol.core.providers import LC0ExpansionProvider
from metacontrol.data.generator import TreeSearch
from metacontrol.data.targets_mc import derive_snapshots
from metacontrol.data.targets_gnn import compute_gnn_targets
from metacontrol.utils.plotting import render_tree, render_gnn_view, plot_advantage_landscape
"""))

# --- Markdown Cell: Engine ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
## 2. The Engine Provider
The `TreeExpansionProvider` is an abstraction that allows us to swap between different engines (LC0, Stockfish, or even Mocks). 
Here we instantiate the `LC0ExpansionProvider` using production distilled weights. 

We set `nodes=64` to get a meaningful but fast evaluation for each expansion.
"""))

nb['cells'].append(nbf.v4.new_code_cell("""
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"

# Initialize provider
provider = LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=64)
print("LC0 Provider initialized successfully.")
"""))

# --- Markdown Cell: Root Selection ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
## 3. Root Selection
In production, we sample roots from a Lichess database based on ELO, piece count, and game phase. 
For this tutorial, we'll use a fixed midgame FEN.
"""))

nb['cells'].append(nbf.v4.new_code_cell("""
# A complex midgame position
root_fen = "r1bq1rk1/pp2bppp/2nppn2/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 1 9"
print(f"Starting Position: {root_fen}")
"""))

# --- Markdown Cell: Search ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
## 4. Tree Search (PUCT)
The `TreeSearch` class implements the iterative **Select → Expand → Backpropagate** loop. 
- **Select**: Uses the PUCT formula to find an unexpanded leaf.
- **Expand**: Queries LC0 for child priors (P) and values (Q).
- **Backpropagate**: Updates path statistics (N, Q, WDL), flipping the perspective at each level.

**Note on Perspective**: Engine scores (Q, WDL) are reported from the perspective of the side whose turn it is. In our tree, we ensure that each node's features and its outgoing edge statistics are perspective-correct.
"""))

nb['cells'].append(nbf.v4.new_code_cell("""
# Configure search: very small tree for clear visualization
config = GeneratorConfig(max_nodes=2, max_depth=2, c_puct=1.5)
search = TreeSearch(provider, config)

print("Growing search tree (2 expansions)...")
result = search.generate(root_fen)
print(f"Tree grown with {len(result.tree.nodes_by_id)} nodes.")

# Visualize the tree
display(render_tree(result.tree, result.edge_stats, title="LC0 Search Tree (2 Nodes)"))
"""))

# --- Markdown Cell: Meta-Control ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
## 5. Meta-Control Advantage (DP)
We derive training targets for the meta-controller (the planning head). 
This uses Dynamic Programming to calculate the "True" value of continuing search vs. halting, 
factoring in a `continue_cost`.

The **Advantage** is defined as: `Continue_Value - Halt_Reward`. 
A positive advantage suggests the search should continue.
"""))

nb['cells'].append(nbf.v4.new_code_cell("""
continue_cost = 0.05
snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=continue_cost)

# Plot the advantage landscape
plt = plot_advantage_landscape(snapshots, continue_cost=continue_cost)
plt.show()

print(f"Derived {len(snapshots)} snapshots for meta-control training.")
"""))

# --- Markdown Cell: GNN ---
nb['cells'].append(nbf.v4.new_markdown_cell("""
## 6. GNN Targets
Finally, we compute the targets for the GNN value and WDL heads. 
These are visit-weighted averages from the final search tree, representing the "wisdom of the search" 
that the neural network should learn to emulate.
"""))

nb['cells'].append(nbf.v4.new_code_cell("""
gnn_targets = compute_gnn_targets(result.tree, result.edge_stats)

# Visualize the GNN target view
display(render_gnn_view(result.tree, gnn_targets, title="GNN Supervised Targets"))
"""))

# Save the notebook
output_path = "src/metacontrol/tutorials/01_pipeline_visualization.ipynb"
with open(output_path, "w") as f:
    nbf.write(nb, f)

print(f"Tutorial notebook generated at: {output_path}")
print("You can now open it in the editor or run it via nbconvert.")
