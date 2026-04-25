import os
import sys
import chess
import chess.engine

# Ensure src is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.helpers import get_lc0_engine

def test_lc0_multipv(fen: str, multipv: int = 2, nodes: int = 1000):
    print(f"Testing lc0 on FEN: {fen}")
    print(f"MultiPV: {multipv}, Nodes: {nodes}")
    
    engine = get_lc0_engine()
    board = chess.Board(fen)
    
    try:
        # Run analysis
        info = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv)
        
        print("\nAnalysis Results:")
        for i, entry in enumerate(info):
            score = entry.get("score")
            wdl = score.pov(board.turn).wdl() if score else None
            e_win = (wdl.wins + 0.5 * wdl.draws) / wdl.total() if wdl else "N/A"
            pv = entry.get("pv")
            move = pv[0] if pv else "N/A"
            print(f"Rank {i+1}: Move {move}, Score: {score}, E[win]: {e_win}")
            
    finally:
        engine.quit()

if __name__ == "__main__":
    # A slightly complex position to see diverse evaluations
    test_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    test_lc0_multipv(test_fen)
