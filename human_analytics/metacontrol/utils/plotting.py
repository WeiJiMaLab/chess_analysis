import chess
import graphviz
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any

def fen_to_emoji_board(fen: str, html: bool = False) -> str:
    """Converts a FEN string to a 2D ASCII board using unicode emojis."""
    board = chess.Board(fen)
    emoji_map = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
        None: '.'
    }
    
    rows = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = emoji_map[piece.symbol()] if piece else emoji_map[None]
            row.append(symbol)
        rows.append(" ".join(row))
    
    separator = "<BR/>" if html else "\n"
    return separator.join(rows)

def render_tree(tree, edge_stats=None, title="Search Tree"):
    """Renders a SearchTree using Graphviz with board emojis."""
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR', label=title, labelloc='t', fontsize='20')
    
    for node_id, node in tree.nodes_by_id.items():
        board_str = fen_to_emoji_board(node.fen, html=True)
        # Use HTML-like labels for better formatting
        label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        label += f'<TR><TD><B>Node {node_id}</B></TD></TR>'
        label += f'<TR><TD><FONT FACE="Courier" POINT-SIZE="10">{board_str}</FONT></TD></TR>'
        label += f'<TR><TD>Value: {node.features.get("value", 0):.2f}</TD></TR>'
        if node.is_terminal:
            label += f'<TR><TD COLOR="RED">TERMINAL</TD></TR>'
        label += '</TABLE>>'
        
        dot.node(str(node_id), label, shape='none')
        
        for move, child in node.children.items():
            edge_label = move
            if edge_stats and (node_id, child.node_id) in edge_stats:
                stats = edge_stats[(node_id, child.node_id)]
                edge_label += f"\nQ: {stats.q_value:.2f} (N={stats.visit_count})"
            dot.edge(str(node_id), str(child.node_id), label=edge_label)
            
    return dot

def render_gnn_view(tree, targets, title="GNN Targets"):
    """Visualizes the tree with GNN targets (Node values and Edge WDLs)."""
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR', label=title, labelloc='t', fontsize='20')
    
    for node_id, node in tree.nodes_by_id.items():
        val = targets.node_values.get(node_id, 0)
        label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        label += f'<TR><TD><B>Node {node_id}</B></TD></TR>'
        label += f'<TR><TD BGCOLOR="LIGHTBLUE">Target V: {val:.3f}</TD></TR>'
        label += '</TABLE>>'
        dot.node(str(node_id), label, shape='none')
        
        for move, child in node.children.items():
            edge_key = (node.node_id, child.node_id)
            wdl = targets.edge_wdls.get(edge_key, (0,0,0))
            # Graphviz standard labels handle \n fine
            edge_label = f"{move}\nTarget WDL:\n({wdl[0]:.2f}, {wdl[1]:.2f}, {wdl[2]:.2f})"
            dot.edge(str(node_id), str(child.node_id), label=edge_label)
    return dot

def plot_advantage_landscape(snapshots, continue_cost=0.0, title="Meta-Control Advantage"):
    """Plots the advantage sequence over expansion steps."""
    plt.figure(figsize=(10, 4))
    plt.plot([s.advantage for s in snapshots], marker='o', label='Advantage')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f"{title} (Cost={continue_cost})")
    plt.xlabel("Expansion Step")
    plt.ylabel("Advantage")
    plt.grid(alpha=0.3)
    plt.legend()
    return plt
