import pytest
import chess
import os
from metacontrol.data.generator import TreeSearch
from metacontrol.core.providers import LC0ExpansionProvider
from metacontrol.core.schemas import GeneratorConfig

# Real Engine Paths (Princeton Della)
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"

@pytest.fixture
def search():
    if not os.path.exists(LC0_BIN):
        pytest.skip("LC0 binary not found")
    # Use a small node budget for fast testing, but enough to see prioritization
    config = GeneratorConfig(max_nodes=128, max_depth=10, c_puct=1.0)
    provider = LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=100)
    return TreeSearch(provider, config)

def test_real_search_identifies_scholars_mate(search):
    # Scholar's mate position (White to move, Qf7# is winning)
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    res = search.generate(fen)
    tree, edge_stats = res.tree, res.edge_stats
    
    root_id = tree.root.node_id
    mate_move_uci = "f3f7"
    
    # The real engine should definitely find f3f7 as the best move
    mate_cid = [node.node_id for move, node in tree.root.children.items() if move == mate_move_uci][0]
    stats = edge_stats[(root_id, mate_cid)]
    
    visits = [s.visit_count for (p, c), s in edge_stats.items() if p == root_id]
    assert stats.visit_count == max(visits)
    assert stats.q_value > 0.8

def test_real_search_king_rook_mate_in_1(search):
    # White to move, Rc8# is mate
    fen = "k7/8/K1R5/8/8/8/8/8 w - - 0 1"
    res = search.generate(fen)
    tree, edge_stats = res.tree, res.edge_stats
    
    root_id = tree.root.node_id
    mate_move_uci = "c6c8"
    
    mate_cid = [node.node_id for move, node in tree.root.children.items() if move == mate_move_uci][0]
    stats = edge_stats[(root_id, mate_cid)]
    
    visits = [s.visit_count for (p, c), s in edge_stats.items() if p == root_id]
    assert stats.visit_count == max(visits)
    assert stats.q_value > 0.8

def test_real_search_queen_king_mate_in_1(search):
    # White to move, Qb7# is mate
    fen = "k7/8/K7/1Q6/8/8/8/8 w - - 0 1"
    res = search.generate(fen)
    tree, edge_stats = res.tree, res.edge_stats
    
    root_id = tree.root.node_id
    mate_move_uci = "b5b7"
    
    mate_cid = [node.node_id for move, node in tree.root.children.items() if move == mate_move_uci][0]
    stats = edge_stats[(root_id, mate_cid)]
    
    visits = [s.visit_count for (p, c), s in edge_stats.items() if p == root_id]
    # We relax the visit count assertion because LC0 sometimes gives higher priors to other winning moves
    # But it MUST identify it as a decisive win.
    assert stats.q_value > 0.95, f"Mate move {mate_move_uci} was not identified as a win. Q={stats.q_value}"
