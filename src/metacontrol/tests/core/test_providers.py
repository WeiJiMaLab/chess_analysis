import pytest
import chess
import os
from unittest.mock import MagicMock, patch
from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider

# Paths for real testing (Princeton Della)
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"

@pytest.fixture
def mock_uci():
    """Context manager to mock UCI engine handshake."""
    with patch("subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.stdout.readline.side_effect = ["uciok", "readyok", "readyok", "readyok"]
        mock_popen.return_value = proc
        yield proc

def test_lc0_parsing_realistic_mock(mock_uci):
    """Verify parsing with LC0's actual (WL: ...) format."""
    mock_uci.stdout.readline.side_effect = [
        "uciok", "readyok",
        # Realistic LC0 VerboseMoveStats line
        "info string e2e4 (1) N: 10 (P: 10.0%) (WL: 0.200) (D: 0.500) (Q: 0.200) (V: 0.100)",
        "bestmove e2e4"
    ]
    provider = LC0ExpansionProvider("path", "weights")
    children = provider.expand(chess.STARTING_FEN, 0)
    
    assert len(children) == 1
    move = children[0]
    # Root (White) saw Q=0.2. Child (Black) should see -0.2.
    assert move.value == pytest.approx(-0.2)
    # WL=0.2, D=0.5 => W-L=0.2, W+L=0.5 => 2W=0.7 => W=0.35, L=0.15
    # For child (Black), win/loss are flipped: W=0.15, D=0.5, L=0.35
    assert move.wdl == pytest.approx((0.15, 0.5, 0.35))

def test_perspective_taking_mock(mock_uci):
    """Verify that root and child values are flipped in mock."""
    mock_uci.stdout.readline.side_effect = [
        "uciok", "readyok",
        # For evaluate_root
        "info depth 1 score cp 100 wdl 800 100 100 pv e2e4",
        "bestmove e2e4",
        # For expand
        "info string e2e4 (1) N: 10 (P: 10.0%) (WL: 0.700) (D: 0.100) (Q: 0.700) (V: 0.700)",
        "bestmove e2e4"
    ]
    provider = LC0ExpansionProvider("path", "weights")
    
    # Root: White to move, e2e4 is good (+0.7)
    val, wdl = provider.evaluate_root(chess.STARTING_FEN)
    assert val == pytest.approx(0.7)
    assert wdl == pytest.approx((0.8, 0.1, 0.1)) # From summary line
    
    # Child: Black to move, should see -0.7
    children = provider.expand(chess.STARTING_FEN, 0)
    assert children[0].value == pytest.approx(-0.7)
    # Root WL=0.7, D=0.1 => Root W=0.8, L=0.1. Child W=0.1, L=0.8.
    assert children[0].wdl == pytest.approx((0.1, 0.1, 0.8))

@pytest.mark.skipif(not os.path.exists(LC0_BIN), reason="LC0 binary not found")
def test_real_lc0_mate_in_1():
    """Verify real LC0 identifies a decisive win in mate-in-1."""
    provider = LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=256)
    # Scholar's mate position (White to move, Qf7# is mate)
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    
    # 1. Root Evaluation (White perspective)
    val, wdl = provider.evaluate_root(fen)
    assert val > 0.8 # Near 1.0
    assert wdl[0] > 0.9 # High win prob for White
    
    # 2. Expansion (Child perspective)
    children = provider.expand(fen, 0)
    mate_move = [c for c in children if c.move_uci == "f3f7"][0]
    
    # The child node for f3f7 is Black to move, and they are mated!
    # So the value for Black should be -1.0
    assert mate_move.value < -0.9
    assert mate_move.wdl[2] > 0.9 # High loss prob for Black
    assert mate_move.is_terminal is True

@pytest.mark.skipif(not os.path.exists(LC0_BIN), reason="LC0 binary not found")
def test_real_lc0_opening_uninformative():
    """Verify real LC0 gives balanced values for startpos."""
    provider = LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=128)
    
    val, wdl = provider.evaluate_root(chess.STARTING_FEN)
    # Startpos is roughly equal (0.0 to 0.15 for White)
    assert abs(val) < 0.2
    assert wdl[1] > 0.3 # Non-trivial draw prob

def test_hardcoded_terminal_states(mock_uci):
    """Verify rule-based terminal states (no engine needed)."""
    # Use any path, engine won't be called for terminal FEN
    provider = StockfishExpansionProvider("dummy")
    
    # Fool's Mate (White is checkmated)
    fools_mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    val, wdl = provider.evaluate_root(fools_mate)
    assert val == -1.0
    assert wdl == (0.0, 0.0, 1.0)
    
    # Stalemate
    stalemate = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
    val, wdl = provider.evaluate_root(stalemate)
    assert val == 0.0
    assert wdl == (0.0, 1.0, 0.0)

@pytest.mark.skipif(not os.path.exists(LC0_BIN), reason="LC0 binary not found")
def test_real_lc0_perspective_flip():
    """Verify real engine perspective flipping across multiple levels."""
    provider = LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=128)
    
    # Start (White to move)
    root_fen = chess.STARTING_FEN
    root_val, _ = provider.evaluate_root(root_fen)
    
    # Expand e2e4 (Black to move)
    children = provider.expand(root_fen, 0)
    e2e4 = [c for c in children if c.move_uci == "e2e4"][0]
    
    # If e2e4 is good for White (root_val), it should be bad for Black (e2e4.value)
    # They should have opposite signs (unless it's exactly 0)
    if abs(e2e4.value) > 0.01:
        # e2e4.value is from Black's perspective
        # LC0 Q for e2e4 was from White's perspective.
        # So e2e4.value = -Q_white.
        # If White likes e2e4 (Q > 0), Black should hate it (val < 0).
        assert (e2e4.value < 0) == (root_val > 0) or abs(root_val) < 0.05
