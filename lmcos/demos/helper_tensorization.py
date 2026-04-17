from __future__ import annotations

import torch
import numpy as np
import graphviz
from schema import tree_encoder_feature_schema
from tree import SearchTree, ExpansionChild

# Symmetrical Minimalist Constants
FONT = "Inter, Arial, sans-serif"
ACCENT = "#4338ca"        # Indigo (Upward/Root)
ACCENT_DOWN = "#059669"   # Emerald (Downward/Context)
BORDER_NODE = "#94a3b8"   # Light Slate
COLOR_ROOT_BG = "#eef2ff" # Light Indigo

def build_demo_trees() -> list[SearchTree]:
    """Builds two distinct asymmetric trees for demonstration."""
    schema = tree_encoder_feature_schema()
    feats = {name: 0.5 for name in schema.feature_names}
    feats["prior"] = 1.0
    
    tree_a = SearchTree()
    tree_a.create_root("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", feats)
    tree_a.add_children(0, [
        ExpansionChild("e2e4", "fen_e4", feats),
        ExpansionChild("d2d4", "fen_d4", feats)
    ])
    
    tree_b = SearchTree()
    tree_b.create_root("k7/8/8/8/8/8/8/K7 w - - 0 1", feats)
    c_ids = tree_b.add_children(0, [ExpansionChild("Ka1a2", "fen_ka2", feats)])
    tree_b.add_children(c_ids[0], [ExpansionChild("Ka2a3", "fen_ka3", feats)])
    
    return [tree_a, tree_b]

def plot_tree(tree: SearchTree, title: str, node_data: dict = None):
    """Renders a SearchTree using graphviz with optional embedded vectors."""
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='TB', size='10,5')
    dot.attr('node', shape='rect', style='rounded,filled', fontname=FONT, fontsize='10', color=BORDER_NODE, penwidth='1.5')

    for node in tree.iter_nodes():
        node_id_int = node.node_id
        node_id_str = str(node_id_int)
        label = node.incoming_move_uci or "ROOT"
        
        # Inject data if provided
        if node_data is not None and node_id_int in node_data:
            val = node_data[node_id_int]
            if isinstance(val, torch.Tensor): val = val.detach().numpy()
            label += f"\nh: {np.around(val, 2)}"
        else:
            label += f"\nID: {node_id_int}\nDepth: {node.depth}"
            
        color = COLOR_ROOT_BG if node.parent_id is None else "#ffffff"
        fontcolor = ACCENT if node.parent_id is None else "#0f172a"
        dot.node(node_id_str, label, fillcolor=color, fontcolor=fontcolor)
        if node.parent_id is not None:
            dot.edge(str(node.parent_id), node_id_str, color="#64748b")
    return dot

def summarize_batch_structure(batch):
    """Prints a clear summary of how the trees are flattened."""
    print(f"--- FOLDED REPRESENTATION ---")
    print(f"Total Nodes in Batch: {batch.node_features.shape[0]}")
    print(f"Batch Size (Trees):   {batch.batch_size}")
    print(f"\n--- ENCODING THE HIERARCHY ---")
    print(f"Depth Vector:  {batch.depth.tolist()}")
    print(f"Parent Index: {batch.parent_index.tolist()}")
    print(f"Edge Slots:   {batch.edge_slot.tolist()}")
    print(f"  (Note: -1 means the node is a Root)")
    print(f"Root Indices: {batch.root_index.tolist()}")

def visualize_sweep_order(batch):
    """Shows the order in which the GNN would process these nodes."""
    max_d = int(batch.depth.max())
    print(f"--- GNN EXECUTION PLAN (Topological Sweep) ---")
    print("UPWARD PASS (Leaves to Root):")
    for d in range(max_d, -1, -1):
        nodes = (batch.depth == d).nonzero().flatten().tolist()
        print(f"  Round {max_d - d}: Process depth {d} nodes -> {nodes}")
    print("\nDOWNWARD PASS (Root to Leaves):")
    for d in range(max_d + 1):
        nodes = (batch.depth == d).nonzero().flatten().tolist()
        print(f"  Round {d}: Process depth {d} nodes -> {nodes}")

def visualize_batch_layout(batch):
    """Draws a 'Memory Map' of the TreeBatch."""
    dot = graphviz.Digraph(comment="Batch Layout")
    dot.attr(rankdir='LR', size='10,4')
    dot.attr('node', shape='none', fontname=FONT, fontsize='10')

    rows = []
    for i in range(batch.node_features.shape[0]):
        is_root = i in batch.root_index
        color = COLOR_ROOT_BG if is_root else "#ffffff"
        fontcolor = ACCENT if is_root else "#0f172a"
        label = f"ROOT {i}" if is_root else f"Node {i}"
        rows.append(f'<tr><td port="n{i}" bgcolor="{color}"><font color="{fontcolor}">{label}</font></td></tr>')
    
    table = f'<<table border="0" cellborder="1" cellspacing="0" color="{BORDER_NODE}">{"".join(rows)}</table>>'
    dot.node('matrix', label=table)

    for child_idx, parent_idx in enumerate(batch.parent_index):
        if parent_idx != -1:
            dot.edge(f'matrix:n{child_idx}', f'matrix:n{parent_idx}', label=" parent", color="#64748b")

    dot.attr(label="\nTreeBatch Memory Layout\n(Individual trees 'melted' into one contiguous batch)")
    return dot

def visualize_gnn_sweep_step(batch, depth_to_highlight, direction='up'):
    """Draws the flattened forest with a highlight on nodes at a specific depth."""
    dot = graphviz.Digraph(comment=f"GNN Sweep {direction} depth {depth_to_highlight}")
    dot.attr(rankdir='TB', size='10,6')
    dot.attr('node', shape='rect', style='rounded,filled', fontname=FONT, fontsize='10', color=BORDER_NODE)

    highlight_color = ACCENT if direction == 'up' else ACCENT_DOWN 
    past_color = "#f8fafc"
    future_color = "#e2e8f0"

    for i in range(batch.node_features.shape[0]):
        node_depth = batch.depth[i].item()
        if node_depth == depth_to_highlight:
            state_color = "#ffffff"
            penwidth = "2.5"
            border_color = highlight_color
            fontcolor = highlight_color
        elif (direction == 'up' and node_depth > depth_to_highlight) or (direction == 'down' and node_depth < depth_to_highlight):
            state_color = past_color
            penwidth = "1.0"
            border_color = BORDER_NODE
            fontcolor = "#64748b"
        else:
            state_color = future_color
            penwidth = "1.0"
            border_color = BORDER_NODE
            fontcolor = "#94a3b8"
            
        label = f"Node {i}\nDepth {node_depth}"
        if i in batch.root_index: label += "\n[ROOT]"
        dot.node(str(i), label, fillcolor=state_color, penwidth=penwidth, color=border_color, fontcolor=fontcolor)

    for child_idx, parent_idx in enumerate(batch.parent_index):
        if parent_idx != -1:
            child_depth = batch.depth[child_idx].item()
            parent_depth = batch.depth[parent_idx].item()
            # Highlight outgoing edges from children to parents in upward pass
            if direction == 'up' and child_depth == depth_to_highlight:
                edge_color = highlight_color
                edge_width = "2.0"
                edge_dir = "forward" # From Child to Parent
            # Highlight outgoing edges from parents to children in downward pass
            elif direction == 'down' and parent_depth == depth_to_highlight:
                edge_color = highlight_color
                edge_width = "2.0"
                edge_dir = "back"    # From Parent to Child
            else:
                edge_color = "#cbd5e1"
                edge_width = "1.0"
                edge_dir = "forward"
                
            dot.edge(str(child_idx), str(parent_idx), color=edge_color, penwidth=edge_width, dir=edge_dir)

    direction_label = "Leaves-to-Root (Summarizing)" if direction == 'up' else "Root-to-Leaves (Contextualizing)"
    dot.attr(label=f"\nGNN Execution Round: Depth {depth_to_highlight}\nDirection: {direction_label}")
    return dot
