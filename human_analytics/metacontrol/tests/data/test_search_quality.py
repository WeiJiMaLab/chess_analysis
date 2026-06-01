import pytest
import chess
import os
from metacontrol.data.generator import TreeSearch
from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider
from metacontrol.core.schemas import GeneratorConfig

# Real Engine Paths (Princeton Della)
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
STOCKFISH_BIN = "/home/hl4291/stockfish-sf_15/src/stockfish"

def get_provider(kind, nodes=100):
    if kind == "lc0":
        if not os.path.exists(LC0_BIN): pytest.skip("LC0 not found")
        return LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=nodes)
    else:
        if not os.path.exists(STOCKFISH_BIN): pytest.skip("Stockfish not found")
        # MultiPV=8 is usually sufficient and avoids engine segfaults on some positions
        return StockfishExpansionProvider(STOCKFISH_BIN, nodes=nodes * 5, options={"MultiPV": "8"})

# --- Positions ---

MATE_IN_1 = [
    ("Scholar's Mate", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "f3f7"),
    ("Back Rank Mate", "k7/4R3/K7/8/8/8/8/8 w - - 0 1", "e7e8"),
    ("Arabian Mate", "7k/8/5N2/8/8/8/6R1/K7 w - - 0 1", "g2g8"),
    ("Damiano's Mate", "5rk1/5p1p/5PpQ/8/8/8/8/7K w - - 0 1", "h6g7"),
]

MATE_IN_2 = [
    ("Morphy's Puzzle", "8/8/8/8/7k/R7/3K4/6B1 w - - 0 1", "a3a6"),
    ("Philidor's Smothered Sequence", "r5nk/5ppp/5N2/8/8/8/Q7/7K w - - 0 1", "a2a8"),
    ("Rook Staircase Mate-in-2", "k7/4R3/2R5/8/8/8/8/K7 w - - 0 1", "c6c8"),
]

# --- Engine Invariant Quality Tests ---

@pytest.mark.parametrize("kind", ["lc0", "stockfish"])
@pytest.mark.parametrize("name, fen, mate_move", MATE_IN_1)
def test_search_quality_mate_in_1(kind, name, fen, mate_move):
    provider = get_provider(kind, nodes=200)
    config = GeneratorConfig(max_nodes=100, max_depth=5, c_puct=2.0)
    search = TreeSearch(provider, config)
    
    res = search.generate(fen)
    tree, edge_stats = res.tree, res.edge_stats
    
    root_id = tree.root.node_id
    assert mate_move in tree.root.children, f"{kind}: {name} ({mate_move}) not in children"
    
    stats = edge_stats[(root_id, tree.root.children[mate_move].node_id)]
    
    # Prioritize mate
    visits = [s.visit_count for (p, c), s in edge_stats.items() if p == root_id]
    assert stats.visit_count == max(visits), f"{kind}: {name} visits {stats.visit_count} < max {max(visits)}"
    assert stats.q_value > 0.8

@pytest.mark.parametrize("kind", ["lc0", "stockfish"])
@pytest.mark.parametrize("name, fen, key_move", MATE_IN_2)
def test_search_quality_mate_in_2(kind, name, fen, key_move):
    # Puzzles need more budget
    provider = get_provider(kind, nodes=500)
    config = GeneratorConfig(max_nodes=1000, max_depth=10, c_puct=3.0)
    search = TreeSearch(provider, config)
    
    res = search.generate(fen)
    tree, edge_stats = res.tree, res.edge_stats
    
    root_id = tree.root.node_id
    assert key_move in tree.root.children, f"{kind}: {name} key move {key_move} not in children"
    
    stats = edge_stats[(root_id, tree.root.children[key_move].node_id)]
    
    # Check identifying advantage
    assert stats.q_value > 0.5, f"{kind}: {name} advantage not identified. Q={stats.q_value}"
    
    # Check exploration. Relaxed for mate-in-2 but must be substantial.
    visits = [s.visit_count for (p, c), s in edge_stats.items() if p == root_id]
    assert stats.visit_count > max(visits) * 0.1, f"{kind}: {name} visits {stats.visit_count} too low vs max {max(visits)}"

def test_lc0_succeeds_with_sufficient_c_puct():
    """
    Illustrative test requested by the user:
    Show that Morphy's Puzzle (a quiet mate-in-2) might fail to be explored 
    with a low c_puct (exploration constant), but succeeds with a high c_puct.
    """
    try:
        provider = get_provider("lc0", nodes=200)
    except pytest.skip.Exception:
        pytest.skip("LC0 not found")
        
    fen = "8/8/8/8/7k/R7/3K4/6B1 w - - 0 1"
    key_move = "a3a6"
    
    # Run with very low c_puct (almost greedy wrt priors/values)
    config_low = GeneratorConfig(max_nodes=200, max_depth=10, c_puct=0.1)
    search_low = TreeSearch(provider, config_low)
    res_low = search_low.generate(fen)
    
    # Run with high c_puct (encourages exploring low-prior moves)
    config_high = GeneratorConfig(max_nodes=200, max_depth=10, c_puct=4.0)
    search_high = TreeSearch(provider, config_high)
    res_high = search_high.generate(fen)
    
    root_id_low = res_low.tree.root.node_id
    root_id_high = res_high.tree.root.node_id
    
    # It's possible the low c_puct doesn't even expand the key move, or visits it very little
    low_visits = 0
    if key_move in res_low.tree.root.children:
        low_cid = res_low.tree.root.children[key_move].node_id
        low_visits = res_low.edge_stats.get((root_id_low, low_cid)).visit_count if (root_id_low, low_cid) in res_low.edge_stats else 0
        
    # High c_puct should definitely visit it more
    assert key_move in res_high.tree.root.children
    high_cid = res_high.tree.root.children[key_move].node_id
    high_visits = res_high.edge_stats[(root_id_high, high_cid)].visit_count
    
    assert high_visits > low_visits, "High c_puct should lead to more visits for the quiet key move"
