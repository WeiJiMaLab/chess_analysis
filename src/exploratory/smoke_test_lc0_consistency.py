import os
import sys
import chess
import chess.engine
import numpy as np
from tqdm import tqdm

# Ensure src is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.helpers import get_lc0_engine

def get_e_win(info, board):
    if board.is_checkmate():
        # If the side to move is checkmated, their E[win] is 0.0
        return 0.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.5
    
    if not info: return None
    score = info.get("score")
    if not score: return None
    
    # pov(board.turn) gives the score from the perspective of the side whose turn it is.
    wdl = score.pov(board.turn).wdl()
    return (wdl.wins + 0.5 * wdl.draws) / wdl.total()

def test_consistency(fens, n_repeats=5, depth=10):
    print(f"Testing consistency over {len(fens)} FENs with {n_repeats} repeats at depth {depth}...")
    engine = get_lc0_engine()
    
    results = {}
    for fen in tqdm(fens):
        board = chess.Board(fen)
        evs = []
        for _ in range(n_repeats):
            if board.is_game_over():
                evs.append(get_e_win(None, board))
            else:
                info_list = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
                info = info_list[0] if info_list else None
                evs.append(get_e_win(info, board))
        
        results[fen] = evs
    
    engine.quit()
    
    print("\nConsistency Results:")
    for fen, evs in results.items():
        std = np.std(evs)
        mean = np.mean(evs)
        print(f"FEN: {fen[:30]}... | Mean: {mean:.4f} | Std: {std:.4e}")

def main():
    # 1. Standard positions
    test_fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    ]
    
    # 2. Checkmate / Near Checkmate positions
    checkmate_fens = [
        "R6k/R7/8/8/8/8/8/7K b - - 0 1", # Black to move, Black is checkmated.
        "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", # White to move, White is checkmated (backrank).
    ]
    
    all_fens = test_fens + checkmate_fens
    
    test_consistency(all_fens, n_repeats=5, depth=10)

if __name__ == "__main__":
    main()
