import pytest
import chess
import os
from unittest.mock import MagicMock, patch
from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider

# Paths for real testing
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

def test_lc0_parsing(mock_uci):
    mock_uci.stdout.readline.side_effect = [
        "uciok", "readyok",
        "info string e2e4  P: 43.2% Q: 0.123 wdl 450 400 150",
        "bestmove e2e4"
    ]
    provider = LC0ExpansionProvider("path", "weights")
    children = provider.expand(chess.STARTING_FEN, 0)
    
    assert len(children) == 1
    assert children[0].move_uci == "e2e4"
    assert children[0].prior == pytest.approx(0.432)
    # Child perspective: win/loss flipped, Q negated
    assert children[0].wdl == pytest.approx((0.15, 0.4, 0.45))
    assert children[0].value == pytest.approx(-0.123)

def test_stockfish_parsing(mock_uci):
    mock_uci.stdout.readline.side_effect = [
        "uciok", "readyok",
        "info multipv 1 wdl 410 400 190 pv e2e4",
        "bestmove e2e4"
    ]
    provider = StockfishExpansionProvider("path")
    children = provider.expand(chess.STARTING_FEN, 0)
    
    assert len(children) == 1
    assert children[0].move_uci == "e2e4"
    # Child perspective: 190 wins for Black, 410 losses
    assert children[0].wdl == pytest.approx((0.19, 0.4, 0.41))
    assert children[0].value == pytest.approx(-0.22)

def test_perspective_taking(mock_uci):
    """Explicitly verify that value is flipped between root and child."""
    # Root (White to move): e2e4 is good for White (+0.6)
    mock_uci.stdout.readline.side_effect = [
        "uciok", "readyok",
        "info multipv 1 wdl 700 200 100 pv e2e4",
        "bestmove e2e4",
        "info multipv 1 wdl 700 200 100 pv e2e4",
        "bestmove e2e4"
    ]
    provider = StockfishExpansionProvider("path")
    
    # Root eval (from White perspective)
    val, wdl = provider.evaluate_root(chess.STARTING_FEN)
    assert val == pytest.approx(0.6) 
    assert wdl == pytest.approx((0.7, 0.2, 0.1))
    
    # expand() returns child perspective (Black)
    children = provider.expand(chess.STARTING_FEN, 0)
    assert children[0].value == pytest.approx(-0.6)
    assert children[0].wdl == pytest.approx((0.1, 0.2, 0.7))

def test_checkmate_decisive_wdl(mock_uci):
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    provider = StockfishExpansionProvider("path")
    val, wdl = provider.evaluate_root(fen)
    assert val == -1.0 
    assert wdl == (0.0, 0.0, 1.0) 

def test_stalemate_draw_wdl(mock_uci):
    fen = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
    provider = StockfishExpansionProvider("path")
    val, wdl = provider.evaluate_root(fen)
    assert val == 0.0
    assert wdl == (0.0, 1.0, 0.0)

def test_castling_legality(mock_uci):
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPPBPPP/RNBQK2R w KQkq - 4 4"
    mock_uci.stdout.readline.side_effect = ["uciok", "readyok", "info multipv 1 wdl 500 400 100 pv e1g1", "bestmove e1g1"]
    provider = StockfishExpansionProvider("path")
    children = provider.expand(fen, 0)
    castle_child = [c for c in children if c.move_uci == "e1g1"][0]
    board = chess.Board(castle_child.fen)
    assert board.piece_at(chess.G1).symbol() == "K"
    assert board.piece_at(chess.F1).symbol() == "R"

def test_en_passant(mock_uci):
    fen = "rnbqkbnr/pp2pppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    mock_uci.stdout.readline.side_effect = ["uciok", "readyok", "info multipv 1 wdl 500 400 100 pv e5d6", "bestmove e5d6"]
    provider = StockfishExpansionProvider("path")
    children = provider.expand(fen, 0)
    ep_child = [c for c in children if c.move_uci == "e5d6"][0]
    board = chess.Board(ep_child.fen)
    assert board.piece_at(chess.D5) is None

def test_start_position_uninformative_wdl(mock_uci):
    mock_uci.stdout.readline.side_effect = ["uciok", "readyok", "info multipv 1 wdl 333 334 333 pv e2e4", "bestmove e2e4"]
    provider = StockfishExpansionProvider("path")
    val, wdl = provider.evaluate_root(chess.STARTING_FEN)
    assert val == pytest.approx(0.0, abs=0.01)
    assert wdl[1] == pytest.approx(0.334, abs=0.01)

@pytest.mark.skipif(not os.path.exists(LC0_BIN), reason="LC0 binary not found")
def test_real_lc0_integration():
    provider = LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=128)
    fen = "r1bq1rk1/pp2bppp/2nppn2/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 1 9"
    val, wdl = provider.evaluate_root(fen)
    assert isinstance(val, float)
    if wdl: assert len(wdl) == 3
    children = provider.expand(fen, 0)
    assert len(children) > 0
