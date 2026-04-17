from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import chess

# Core imports from lmcos
from cts_pretrain import (
    TeacherSearchConfig,
    _backup_target_from_child_wdl,
    generate_partial_tree_from_provider,
    consolidate_generated_tree,
    _static_node_wdl
)
from tree import SearchTree, SearchNode

def build_tutorial_config() -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=3,
        search_budget=64,
        c_puct=1.0,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="tutorial_v1",
        search_config_id="prefix_tutorial",
    )

def _format_wdl(wdl: tuple[float, float, float]) -> str:
    w, d, l = wdl
    return f"{w:.2f}/{d:.2f}/{l:.2f}"

def _render_ascii_board(fen: str) -> str:
    """Renders a simple ASCII representation of the board."""
    board = chess.Board(fen)
    # Get a compact 8x8 grid
    rows = []
    for rank in range(7, -1, -1):
        row_chars = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            char = piece.symbol() if piece else "."
            row_chars.append(char)
        rows.append(" ".join(row_chars))
    return "<br/>".join(rows)

def visualize_unified_tree(
    oracle_tree: SearchTree,
    partial_node_ids: set[int],
    input_wdl: dict[int, tuple[float, float, float]],
    target_wdl: dict[int, tuple[float, float, float]],
):
    """Renders a unified tree using Graphviz with embedded boards."""
    import graphviz
    dot = graphviz.Digraph(comment="Unified Prefix Context")
    dot.attr(rankdir='TB', size='10,12')
    dot.attr('node', shape='none', fontname='Inter, Arial', fontsize='10')

    for node in oracle_tree.iter_nodes():
        node_id = str(node.node_id)
        is_input = node.node_id in partial_node_ids
        move = node.incoming_move_uci or "ROOT"
        
        # Color & Border
        border_color = "#3b82f6" if is_input else "#cbd5e1"
        bg_color = "#ffffff" if is_input else "#f8fafc"
        style = "solid" if is_input else "dashed"
        
        # ASCII Board (Using a monospaced font for alignment)
        board_ascii = _render_ascii_board(node.fen)
        board_cell = f'<tr><td colspan="2" bgcolor="#f1f5f9"><font face="Courier" point-size="7">{board_ascii}</font></td></tr>'

        # Values
        target_str = _format_wdl(target_wdl.get(node.node_id, (0,0,0)))
        if is_input:
            input_str = _format_wdl(input_wdl.get(node.node_id, (0,0,0)))
            val_content = f'<tr><td align="left"><font color="#64748b">INPUT:</font></td><td align="left"><b>{input_str}</b></td></tr>' \
                          f'<tr><td align="left"><font color="#64748b">TARGET:</font></td><td align="left"><font color="#3b82f6"><b>{target_str}</b></font></td></tr>'
        else:
            val_content = f'<tr><td align="left" colspan="2"><font color="#94a3b8"><i>FUTURE</i></font></td></tr>' \
                          f'<tr><td align="left" colspan="2"><font color="#94a3b8"><b>{target_str}</b></font></td></tr>'

        label = f'<<table border="0" cellborder="1" cellspacing="0" bgcolor="{bg_color}" color="{border_color}" style="{style}">' \
                f'<tr><td colspan="2"><b>{move}</b> (ID: {node_id})</td></tr>' \
                f'{board_cell}' \
                f'{val_content}' \
                f'</table>>'
        
        dot.node(node_id, label)
        
        if node.parent_id is not None:
            e_style = "solid" if node.node_id in partial_node_ids else "dashed"
            e_color = "#3b82f6" if node.node_id in partial_node_ids else "#cbd5e1"
            dot.edge(str(node.parent_id), node_id, color=e_color, style=e_style)

    dot.attr(label=f"\nUnified Search Context\nBlue = GNN Input | Gray = Future Signal")
    return dot
