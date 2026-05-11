import pytest
import chess
import re
from typing import Dict, List, Tuple, Mapping, Optional

# --- Legacy Logic Simulation ---

def legacy_parse_wdl(line: str) -> Optional[Tuple[float, float, float]]:
    # Simulating WDL_RE in legacy: r"wdl\s+(?P<win>\d+)\s+(?P<draw>\d+)\s+(?P<loss>\d+)"
    m = re.search(r"wdl\s+(?P<win>\d+)\s+(?P<draw>\d+)\s+(?P<loss>\d+)", line)
    if m:
        w, d, l = float(m.group("win")), float(m.group("draw")), float(m.group("loss"))
        total = w + d + l
        return (w/total, d/total, l/total)
    return None

def legacy_backpropagate(path: List[Tuple[int, int]], leaf_val: float, leaf_wdl: Tuple[float, float, float]):
    # Simulating _backpropagate_path in lmcos/cts_pretrain.py
    edge_q = {}
    edge_wdl = {}
    
    val = leaf_val
    wdl = leaf_wdl
    
    for parent_id, child_id in reversed(path):
        val = -val # Flip value
        # Flip WDL: (win, draw, loss) -> (loss, draw, win)
        wdl = (wdl[2], wdl[1], wdl[0])
        
        edge_q[(parent_id, child_id)] = val
        edge_wdl[(parent_id, child_id)] = wdl
        
    return edge_q, edge_wdl

# --- New Pipeline Logic Simulation ---

def new_parse_child(line: str) -> Tuple[float, Tuple[float, float, float]]:
    # Simulating LC0ExpansionProvider.expand
    # Engine Q is from parent perspective. We store -Q for child.
    mq = re.search(r"Q[:=]\s*(?P<q>[\d.-]+)", line)
    q_parent = float(mq.group("q")) if mq else 0.0
    val_child = -q_parent
    
    # Engine WDL is from parent perspective. We reverse it for child.
    mwdl = re.search(r"wdl\s+(?P<win>\d+)\s+(?P<draw>\d+)\s+(?P<loss>\d+)", line)
    if mwdl:
        w, d, l = float(mwdl.group("win")), float(mwdl.group("draw")), float(mwdl.group("loss"))
        total = w + d + l
        wdl_parent = (w/total, d/total, l/total)
        wdl_child = (wdl_parent[2], wdl_parent[1], wdl_parent[0])
    else:
        wdl_child = (0.33, 0.34, 0.33)
        
    return val_child, wdl_child

def new_backpropagate(path: List[Tuple[int, int]], leaf_val: float, leaf_wdl: Tuple[float, float, float]):
    edge_q = {}
    edge_wdl = {}
    
    val = leaf_val
    wdl = leaf_wdl
    
    for parent_id, child_id in reversed(path):
        val = -val
        wdl = (wdl[2], wdl[1], wdl[0])
        
        edge_q[(parent_id, child_id)] = val
        edge_wdl[(parent_id, child_id)] = wdl
        
    return edge_q, edge_wdl

# --- Test Case ---

def test_legacy_parity_explanation():
    """
    This test demonstrates the fundamental sign/perspective difference 
    between the legacy 'lmcos' code and the new 'metacontrol' pipeline.
    """
    # Position: White at root, move e2e4 is GOOD.
    # LC0 Output (Parent Perspective): Q=0.6, WDL=(0.7, 0.2, 0.1)
    lc0_line = "info string e2e4 P: 10% Q: 0.6 wdl 700 200 100"
    path = [(0, 1)] # root -> child
    
    print("\n=== PARITY ANALYSIS (Root=White, Move=e2e4, Good for White) ===")
    
    # 1. Legacy Analysis
    leg_val = legacy_parse_lc0_child_value(lc0_line) # Returns 0.6
    leg_wdl = legacy_parse_wdl(lc0_line)             # Returns (0.7, 0.2, 0.1)
    leg_q_edges, leg_wdl_edges = legacy_backpropagate(path, leg_val, leg_wdl)
    
    print(f"Legacy Child Node Value: {leg_val}")
    print(f"Legacy Edge Q:   {leg_q_edges[(0, 1)]} (Should be +0.6 for White!)")
    print(f"Legacy Edge WDL: {leg_wdl_edges[(0, 1)]} (Should be High Win for White!)")
    
    # 2. New Analysis
    new_val, new_wdl = new_parse_child(lc0_line) # Returns -0.6, (0.1, 0.2, 0.7)
    new_q_edges, new_wdl_edges = new_backpropagate(path, new_val, new_wdl)
    
    print(f"New Child Node Value:    {new_val}")
    print(f"New Edge Q:      {new_q_edges[(0, 1)]}")
    print(f"New Edge WDL:    {new_wdl_edges[(0, 1)]}")
    
    # 3. Discrepancy explanation
    print("\n=== CONCLUSION ===")
    if leg_q_edges[(0, 1)] < 0 and new_q_edges[(0, 1)] > 0:
        print("DIAGNOSIS: Legacy code had a sign-inversion bug in target generation.")
        print("Legacy stored Parent-Perspective scores in Child Nodes, then flipped them in backprop.")
        print("This resulted in GOOD moves having NEGATIVE scores on the parent's edges.")
        print("The new pipeline correctly stores Child-Perspective scores in Child Nodes,")
        print("so that backpropagation restores the Parent-Perspective at the edge level.")

def legacy_parse_lc0_child_value(line: str) -> float:
    m = re.search(r"Q[:=]\s*(?P<q>[\d.-]+)", line)
    return float(m.group("q")) if m else 0.0

if __name__ == "__main__":
    test_legacy_parity_explanation()
