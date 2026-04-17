from __future__ import annotations

import graphviz

# Symmetrical Minimalist Constants
FONT = "Inter, Arial, sans-serif"
ACCENT = "#4338ca"
BORDER = "#e2e8f0"

def apply_clean_styles(dot):
    dot.attr(fontname=FONT, fontsize='11', rankdir='TB', splines='ortho')
    dot.attr('node', fontname=FONT, fontsize='10', shape='rect', style='rounded,filled', 
             fillcolor='#ffffff', color='#94a3b8', penwidth='1.2', width='1.6')
    dot.attr('edge', fontname=FONT, fontsize='9', color='#64748b', arrowsize='0.7')

def visualize_upward_step():
    """Formally detailed Phase 1: Children to Parent."""
    dot = graphviz.Digraph(comment="Upward Pass")
    apply_clean_styles(dot)
    dot.attr(rankdir='BT') 
    
    # Children at bottom
    dot.node('c1', "Child 1"); dot.node('c2', "Child 2"); dot.node('c3', "Child 3")
    
    # Aggregator
    dot.node('mha', "Attention Aggregator", shape='rect', color=ACCENT)
    
    # Node Compute
    dot.node('gru1', "Node Update (GRU)", fillcolor=ACCENT, fontcolor="white", color=ACCENT)
    dot.node('p', "Parent Module", shape='rect')

    # Connections
    dot.edge('c1', 'mha', label=" v_c1"); dot.edge('c2', 'mha', label=" v_c2"); dot.edge('c3', 'mha', label=" v_c3")
    dot.edge('mha', 'gru1', label=" v_children")
    dot.edge('gru1', 'p', label=" v_p_msg")
    
    dot.attr(label="PHASE 1: UPWARD (Evidence Gathering)")
    return dot

def visualize_downward_step():
    """Formally detailed Phase 2: Parent to Children."""
    dot = graphviz.Digraph(comment="Downward Pass")
    apply_clean_styles(dot)
    dot.attr(rankdir='TB')
    
    # Parent at Top
    dot.node('p', "Parent Module", shape='rect')
    dot.node('proj', "Linear Projection", shape='rect', color=ACCENT)
    
    # Node Compute
    dot.node('gru2', "Node Update (GRU)", fillcolor="#059669", fontcolor="white", color="#059669")
    
    # Children at Bottom
    dot.node('c1', "Child 1"); dot.node('c2', "Child 2"); dot.node('c3', "Child 3")

    # Connections
    dot.edge('p', 'proj', label=" v_p_msg")
    dot.edge('proj', 'gru2', label=" v_ctx")
    dot.edge('gru2', 'c1', label=" v_c1_ctx"); dot.edge('gru2', 'c2', label=" v_c2_ctx"); dot.edge('gru2', 'c3', label=" v_c3_ctx")
    
    dot.attr(label="PHASE 2: DOWNWARD (Context Distribution)")
    return dot

def visualize_dual_pass():
    """Symmetrical layout: Multiple children funneling UP and broadcasting DOWN."""
    dot = graphviz.Digraph(comment="Symmetrical Sync")
    dot.attr(rankdir='TB', size='12,12', splines='polyline')
    apply_clean_styles(dot)

    # 1. TOP RANK (Parents)
    dot.node('u_p', "Upward Parent", fillcolor="#fffbeb")
    dot.node('d_p', "Downward Parent", fillcolor="#f0fdf4")
    
    # 2. MID RANK (The Compute Junciton)
    dot.node('u_gru', "GRU Update #1", fillcolor=ACCENT, fontcolor="white")
    dot.node('d_gru', "GRU Update #2", fillcolor="#059669", fontcolor="white")
    
    # 3. BOTTOM RANK (The Fan-Out)
    with dot.subgraph(name='cluster_u_children') as sc:
        sc.attr(style='invis')
        sc.node('u_c1', "Child 1"); sc.node('u_c2', "Child 2"); sc.node('u_c3', "Child 3")
    with dot.subgraph(name='cluster_d_children') as sc:
        sc.attr(style='invis')
        sc.node('d_c1', "Child 1"); sc.node('d_c2', "Child 2"); sc.node('d_c3', "Child 3")

    # Force Ranks
    with dot.subgraph() as s:
        s.attr(rank='same'); s.node('u_p'); s.node('d_p')
    with dot.subgraph() as s:
        s.attr(rank='same'); s.node('u_gru'); s.node('d_gru')
    with dot.subgraph() as s:
        s.attr(rank='same'); s.node('u_c1'); s.node('u_c2'); s.node('u_c3')
        s.node('d_c1'); s.node('d_c2'); s.node('d_c3')

    # LEFT SIDE: UPWARD (Evidence Funnel)
    dot.edge('u_c1', 'u_gru', color=ACCENT); dot.edge('u_c2', 'u_gru', color=ACCENT); dot.edge('u_c3', 'u_gru', color=ACCENT)
    dot.edge('u_gru', 'u_p', label=" [Summary]", color=ACCENT)

    # RIGHT SIDE: DOWNWARD (Strategy Broadcast)
    dot.edge('d_p', 'd_gru', label=" [Context]", color="#059669")
    dot.edge('d_gru', 'd_c1', color="#059669"); dot.edge('d_gru', 'd_c2', color="#059669"); dot.edge('d_gru', 'd_c3', color="#059669")

    # THE HANDSHAKE
    dot.edge('u_gru', 'd_gru', label=" h_interm ", style='dashed', 
             color=ACCENT, constraint='false', penwidth='2')

    dot.attr(label="\nThe Full Bidirectional Synchronization Round\nFan-In (Upward Evidence) <---> Fan-Out (Downward Context)")
    return dot
