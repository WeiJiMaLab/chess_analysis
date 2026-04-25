import os
import sys
import argparse
import duckdb
import chess
import chess.engine
import multiprocessing as mp
from tqdm import tqdm
import time

# Ensure src is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.helpers import get_lc0_engine
from utils.features import row_to_fen

# Global engine variable per worker process
_worker_engine = None

def init_worker():
    """Initialize an lc0 engine instance once per worker process."""
    global _worker_engine
    _worker_engine = get_lc0_engine()

def get_e_win(info, board):
    if board.is_checkmate():
        return 0.0
    if board.is_game_over():
        return 0.5
    
    if not info: return None
    score = info.get("score")
    if not score: return None
    
    wdl = score.pov(board.turn).wdl()
    return (wdl.wins + 0.5 * wdl.draws) / wdl.total()

def analyze_position(row_dict, depth=10):
    global _worker_engine
    try:
        if _worker_engine is None:
            _worker_engine = get_lc0_engine()
            
        fen = row_dict["fen"]
        move_taken_uci = row_dict["move_uci"]
        board = chess.Board(fen)
        
        if board.is_game_over():
            e_win = get_e_win(None, board)
            return {
                "fen": fen,
                "e_win_best": e_win,
                "e_win_second_best": e_win,
                "e_win_move_taken": e_win,
                "n_repeats": 1
            }

        # 1. MultiPV analysis for top 2
        info_list = _worker_engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
        
        e_win_best = get_e_win(info_list[0], board) if len(info_list) > 0 else None
        e_win_second_best = get_e_win(info_list[1], board) if len(info_list) > 1 else e_win_best
        
        # 2. Identify e_win_move_taken
        e_win_move_taken = None
        move_taken = chess.Move.from_uci(move_taken_uci)
        
        # Check if move_taken is in the top 2
        for info in info_list:
            if info.get("pv") and info["pv"][0] == move_taken:
                e_win_move_taken = get_e_win(info, board)
                break
        
        # If not in top 2, evaluate specifically
        if e_win_move_taken is None:
            info_move = _worker_engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move_taken])
            e_win_move_taken = get_e_win(info_move[0], board) if info_move else None
            
        return {
            "fen": fen,
            "e_win_best": e_win_best,
            "e_win_second_best": e_win_second_best,
            "e_win_move_taken": e_win_move_taken,
            "n_repeats": 1
        }
    except Exception as e:
        return {"error": str(e), "fen": row_dict.get("fen")}

def main():
    parser = argparse.ArgumentParser(description="Distributed lc0 evaluation.")
    parser.add_argument("--db", default="/scratch/gpfs/GRIFFITHS/hl4291/personal.db", help="Path to personal.db")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of positions to process")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for distributed processing")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--depth", type=int, default=10, help="Search depth")
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    
    # Create output table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lc0_evaluations (
            fen VARCHAR PRIMARY KEY,
            e_win_best DOUBLE,
            e_win_second_best DOUBLE,
            e_win_move_taken DOUBLE,
            n_repeats INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Load unique FENs and move_uci
    print("Loading positions from selected_moves...")
    query = f"""
        SELECT 
            board_position, player_white, castling_rights, en_passant_targets, move_uci
        FROM selected_moves
        WHERE (gid % {args.total_shards}) = {args.shard_id}
    """
    if args.limit:
        query += f" LIMIT {args.limit}"
        
    df = conn.execute(query).df()
    
    # Preprocess into full FENs
    df["fen"] = df.apply(row_to_fen, axis=1)
    data_list = df[["fen", "move_uci"]].to_dict('records')
    
    print(f"Processing {len(data_list)} positions in shard {args.shard_id}/{args.total_shards}...")
    
    start_time = time.time()
    with mp.Pool(processes=args.n_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(analyze_position, data_list), total=len(data_list)))
    
    duration = time.time() - start_time
    print(f"Analysis complete in {duration:.2f}s ({len(data_list)/duration:.2f} pos/s)")
    
    # Filter errors
    valid_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    if errors:
        print(f"Encountered {len(errors)} errors.")
        for e in errors[:5]:
            print(f"Error for {e['fen']}: {e['error']}")

    # Insert results
    if valid_results:
        import pandas as pd
        res_df = pd.DataFrame(valid_results)
        # Reorder to match table schema
        res_df = res_df[["fen", "e_win_best", "e_win_second_best", "e_win_move_taken", "n_repeats"]]
        conn.execute("INSERT OR REPLACE INTO lc0_evaluations (fen, e_win_best, e_win_second_best, e_win_move_taken, n_repeats) SELECT * FROM res_df")
        print(f"Inserted {len(valid_results)} records into lc0_evaluations.")

    conn.close()

if __name__ == "__main__":
    main()
