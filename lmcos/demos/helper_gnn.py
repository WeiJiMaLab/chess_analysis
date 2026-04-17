from __future__ import annotations

import graphviz
import numpy as np
import torch

# Symmetrical Minimalist Constants
FONT = "Inter, Arial, sans-serif"
ACCENT = "#4338ca"        # Indigo
ACCENT_DOWN = "#059669"   # Emerald
ACCENT_WDL = "#7c3aed"    # Violet (Readout Head)
BORDER_MODULE = "#475569" # Slate
BORDER_VECTOR = "#94a3b8" # Light Slate

def fmt_h(label, h):
    """Formats a labeled vector into a clean 2pd string."""
    if h is None: return f"{label}: ???"
    if isinstance(h, torch.Tensor):
        h = h.detach().numpy()
    return f"{label}\n{np.around(h, 2)}"

def apply_formal_styles(dot):
    dot.attr(fontname=FONT, fontsize='11', rankdir='TB', splines='ortho')
    dot.attr('node', fontname=FONT, fontsize='10', shape='rect', style='rounded,filled', 
             fillcolor='#f8fafc', color=BORDER_MODULE, penwidth='1.5')
    dot.attr('edge', fontname=FONT, fontsize='9', color='#64748b', arrowsize='0.7')

def visualize_upward_step(h_parent=None, h_c1=None, h_c2=None):
    """Phase 1: Upward pass with labeled vectors."""
    dot = graphviz.Digraph(comment="Upward Pass")
    apply_formal_styles(dot)
    dot.attr(rankdir='BT') 
    
    dot.node('vec_c1', fmt_h("h_child1", h_c1), style='rounded', color=BORDER_VECTOR)
    dot.node('vec_c2', fmt_h("h_child2", h_c2), style='rounded', color=BORDER_VECTOR)
    dot.node('vec_p', fmt_h("h_parent", h_parent), style='rounded', color=ACCENT)
    
    dot.node('mod_mha', "Attention Aggregator", fillcolor='#eef2ff', color=ACCENT)
    dot.node('mod_gru', "Node GRU Update", fillcolor=ACCENT, fontcolor="white", color=ACCENT)

    dot.edge('vec_c1', 'mod_mha'); dot.edge('vec_c2', 'mod_mha')
    dot.edge('mod_mha', 'mod_gru', label=" v_children")
    dot.edge('mod_gru', 'vec_p')
    
    dot.attr(label="PHASE 1: UPWARD (Evidence Summarization)")
    return dot

def visualize_downward_step(h_parent=None, h_c1=None, h_c2=None):
    """Phase 2: Downward pass with labeled vectors."""
    dot = graphviz.Digraph(comment="Downward Pass")
    apply_formal_styles(dot)
    dot.attr(rankdir='TB') 
    
    dot.node('vec_p', fmt_h("h_parent", h_parent), style='rounded', color=BORDER_VECTOR)
    dot.node('vec_c1', fmt_h("h_child1", h_c1), style='rounded', color="#059669")
    dot.node('vec_c2', fmt_h("h_child2", h_c2), style='rounded', color="#059669")
    
    dot.node('mod_proj', "Linear Projection", fillcolor='#ecfdf5', color="#059669")
    dot.node('mod_gru', "Node GRU Update", fillcolor="#059669", fontcolor="white", color="#059669")

    dot.edge('vec_p', 'mod_proj')
    dot.edge('mod_proj', 'mod_gru', label=" v_context")
    dot.edge('mod_gru', 'vec_c1'); dot.edge('mod_gru', 'vec_c2')
    
    dot.attr(label="PHASE 2: DOWNWARD (Global Broadcasting)")
    return dot

def visualize_dual_pass(h_parent=None, h_c1=None, h_c2=None):
    """Symmetrical Summary with ALL children and labels."""
    dot = graphviz.Digraph(comment="Dual Formal Summary")
    dot.attr(rankdir='LR', size='12,12')
    apply_formal_styles(dot)

    with dot.subgraph(name='cluster_up') as u:
        u.attr(label="1. UPWARD", bgcolor="#fffbeb", style="rounded", rankdir='BT')
        u.node('u_v_c1', fmt_h("h_child1", h_c1), style='rounded')
        u.node('u_v_c2', fmt_h("h_child2", h_c2), style='rounded')
        u.node('u_mod', "Aggregation", style='rounded,filled', fillcolor='#ffffff')
        u.node('u_v_p', fmt_h("h_parent", h_parent), style='rounded', color=ACCENT)
        u.edge('u_v_c1', 'u_mod'); u.edge('u_v_c2', 'u_mod'); u.edge('u_mod', 'u_v_p')

    with dot.subgraph(name='cluster_dn') as d:
        d.attr(label="2. DOWNWARD", bgcolor="#f0fdf4", style="rounded", rankdir='TB')
        d.node('d_v_p', fmt_h("h_parent", h_parent), style='rounded')
        d.node('d_mod', "Broadcasting", style='rounded,filled', fillcolor='#ffffff')
        d.node('d_v_c1', fmt_h("h_child1", h_c1), style='rounded', color="#059669")
        d.node('d_v_c2', fmt_h("h_child2", h_c2), style='rounded', color="#059669")
        d.edge('d_v_p', 'd_mod'); d.edge('d_mod', 'd_v_c1'); d.edge('d_mod', 'd_v_c2')

    # Handshake
    dot.edge('u_v_p', 'd_v_p', label=" Sync ", style='dashed', color=ACCENT, penwidth='2')

    return dot

def visualize_wdl_readout(h_parent, slot_vector, wdl_probs=None):
    """Illustrates the Readout head: Parent State + Slot Mapping -> Predictions."""
    dot = graphviz.Digraph(comment="WDL Readout")
    apply_formal_styles(dot)
    dot.attr(rankdir='LR')

    # Input Components (Hollow)
    dot.node('vec_p', fmt_h("h_parent", h_parent), style='rounded', color=ACCENT)
    dot.node('vec_s', fmt_h("slot_encoding", slot_vector), style='rounded', color=BORDER_VECTOR)

    # Readout Head (Filled)
    dot.node('mod_head', "Child WDL Head\n(MLP)", fillcolor=ACCENT_WDL, fontcolor="white", color=ACCENT_WDL)

    # Prediction Output
    w, d, l = wdl_probs if wdl_probs is not None else (0.33, 0.33, 0.33)
    wdl_label = f"WDL Predictions\nWin: {w:.2f} | Draw: {d:.2f} | Loss: {l:.2f}"
    dot.node('out', wdl_label, style='rounded', color=ACCENT_WDL, penwidth='2.0')

    # Flow
    dot.edge('vec_p', 'mod_head')
    dot.edge('vec_s', 'mod_head')
    dot.edge('mod_head', 'out')

    dot.attr(label="FINAL READOUT: Translating Hierarchical Memory into Game Realities")
    return dot

def visualize_halt_decision(h_root, halt_prob, thinking_cost=0.01):
    """Illustrates the Halt Head decision: Root State -> [Halt | Continue]."""
    dot = graphviz.Digraph(comment="Halt Decision")
    apply_formal_styles(dot)
    dot.attr(rankdir='LR')

    # Status Color
    is_halting = halt_prob > 0.5
    status_bg = "#fee2e2" if is_halting else "#ecfdf5" # Red-ish (Stop) vs Green-ish (Go)
    status_color = "#dc2626" if is_halting else "#059669"
    status_text = "HALT (Execute Move)" if is_halting else "CONTINUE (Expand Tree)"

    # Input (Indigo Root)
    dot.node('vec_r', fmt_h("h_root summary", h_root), style='rounded', color=ACCENT)

    # Controller (Filled)
    dot.node('mod_halt', "Halt Controller\n(Policy Head)", fillcolor=BORDER_MODULE, fontcolor="white")

    # The Decision Panel
    decision_label = f"<<table border='0' cellborder='1' cellspacing='0' bgcolor='{status_bg}'>" \
                     f"<tr><td><b>{status_text}</b></td></tr>" \
                     f"<tr><td>P(Halt): {halt_prob:.4f}</td></tr>" \
                     f"<tr><td><font point-size='8'>Thinking Cost (C): {thinking_cost}</font></td></tr>" \
                     f"</table>>"
    dot.node('panel', decision_label, shape='none', color=status_color)

    # Flow
    dot.edge('vec_r', 'mod_halt')
    dot.edge('mod_halt', 'panel', color=status_color, penwidth='2.0')

    dot.attr(label="\nTHE META-CONTROLLER: Optimizing the Economy of Thought")
    return dot
