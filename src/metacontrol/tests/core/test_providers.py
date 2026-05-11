import pytest
import chess
import os
from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider

# Paths for real testing (Princeton Della)
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
STOCKFISH_BIN = "/home/hl4291/stockfish-sf_15/src/stockfish"

def get_provider(kind, nodes=100):
    if kind == "lc0":
        if not os.path.exists(LC0_BIN): pytest.skip("LC0 not found")
        return LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=nodes)
    else:
        if not os.path.exists(STOCKFISH_BIN): pytest.skip("Stockfish not found")
        return StockfishExpansionProvider(STOCKFISH_BIN, nodes=nodes)

# --- Engine Invariant Tests ---

@pytest.mark.parametrize("kind", ["lc0", "stockfish"])
def test_engine_mate_in_1_identification(kind):
    """Verify that any engine correctly identifies a mate-in-1."""
    provider = get_provider(kind, nodes=500)
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    
    # 1. Root Eval
    val, wdl = provider.evaluate_root(fen)
    assert val > 0.8
    
    # 2. Expansion
    children = provider.expand(fen, 0)
    mate_move = [c for c in children if c.move_uci == "f3f7"][0]
    # Child perspective: side being mated
    assert mate_move.value < -0.9
    assert mate_move.is_terminal is True

@pytest.mark.parametrize("kind", ["lc0", "stockfish"])
def test_engine_opening_eval(kind):
    provider = get_provider(kind, nodes=500)
    val, wdl = provider.evaluate_root(chess.STARTING_FEN)
    assert abs(val) < 0.3

@pytest.mark.parametrize("kind", ["lc0", "stockfish"])
def test_engine_perspective_consistency(kind):
    """Verify ChildPerspective = -ParentQ convention."""
    provider = get_provider(kind, nodes=500)
    root_fen = chess.STARTING_FEN
    root_val, _ = provider.evaluate_root(root_fen)
    
    children = provider.expand(root_fen, 0)
    # Pick the engine's preferred move (best child)
    best_child = min(children, key=lambda c: c.value) # Min because child wants lowest value (it's flipped)
    
    # If root_val > 0 (White better), best_child.value should be < 0 (Black worse)
    if abs(best_child.value) > 0.05:
        assert (best_child.value < 0) == (root_val > 0)

# --- Rule-based (Terminal) Tests ---

def test_terminal_states_rule_based():
    # Use any real path to avoid initialization error, but engine won't be called
    real_path = LC0_BIN if os.path.exists(LC0_BIN) else STOCKFISH_BIN
    if not os.path.exists(real_path): pytest.skip("No real engine binary found for terminal test")
    
    provider = StockfishExpansionProvider(real_path)
    
    # Fool's Mate (White is checkmated)
    fools_mate_w = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    val, wdl = provider.evaluate_root(fools_mate_w)
    assert val == -1.0, "White is mated, local value should be -1.0"
    assert wdl == (0.0, 0.0, 1.0), "WDL should be 100% loss"
    
    # Fool's Mate (Black is checkmated)
    fools_mate_b = "rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3"
    val, wdl = provider.evaluate_root(fools_mate_b)
    assert val == -1.0, "Black is mated, local value should be -1.0"
    assert wdl == (0.0, 0.0, 1.0), "WDL should be 100% loss"
    
    # Stalemate
    stalemate = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
    val, wdl = provider.evaluate_root(stalemate)
    assert val == 0.0, "Stalemate, value should be 0.0"
    assert wdl == (0.0, 1.0, 0.0), "WDL should be 100% draw"
