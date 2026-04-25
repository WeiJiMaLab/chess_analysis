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

from utils.helpers import get_lc0_engine, get_stockfish_engine
from utils.features import row_to_fen

# Global engine variable per worker process
_worker_engine = None
_engine_type = None

def init_worker(engine_type):
    global _worker_engine, _engine_type
    _engine_type = engine_type
    if engine_type == "stockfish":
        _worker_engine = get_stockfish_engine(threads=1)
    elif engine_type == "lc0":
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

def analyze_position(row_dict, depth=5):
    global _worker_engine, _engine_type
    try:
        fen = row_dict["fen"]
        move_taken_uci = row_dict["move_uci"]
        board = chess.Board(fen)
        
        if board.is_game_over():
            e_win = get_e_win(None, board)
            return {"fen": fen, "e_win_best": e_win, "e_win_second_best": e_win, "e_win_move_taken": e_win, "n_repeats": 1}

        if _engine_type == "stockfish":
            _worker_engine.configure({"Clear Hash": None})

        # 1. MultiPV analysis
        info_list = _worker_engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
        e_win_best = get_e_win(info_list[0], board) if len(info_list) > 0 else None
        e_win_second_best = get_e_win(info_list[1], board) if len(info_list) > 1 else e_win_best
        
        # 2. Move taken
        e_win_move_taken = None
        move_taken = chess.Move.from_uci(move_taken_uci)
        for info in info_list:
            if info.get("pv") and info["pv"][0] == move_taken:
                e_win_move_taken = get_e_win(info, board)
                break
        
        if e_win_move_taken is None:
            info_move = _worker_engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move_taken])
            e_win_move_taken = get_e_win(info_move, board) if info_move else None
            
        return {"fen": fen, "e_win_best": e_win_best, "e_win_second_best": e_win_second_best, "e_win_move_taken": e_win_move_taken, "n_repeats": 1}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}", "fen": row_dict.get("fen")}

def main():
    parser = argparse.ArgumentParser(description="Distributed Engine evaluation with FEN sharding.")
    parser.add_argument("--engine", choices=["stockfish", "lc0"], required=True)
    parser.add_argument("--db", default="/scratch/gpfs/GRIFFITHS/hl4291/personal.db")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    table_name = f"{args.engine}_evaluations"
    conn = duckdb.connect(args.db)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (fen VARCHAR PRIMARY KEY, e_win_best DOUBLE, e_win_second_best DOUBLE, e_win_move_taken DOUBLE, n_repeats INTEGER, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

    # Mutual Exclusivity: Shard by position hash to ensure unique FENs per shard
    print(f"Loading positions for {args.engine} (Shard {args.shard_id}/{args.total_shards})...")
    query = f"""
        SELECT 
            board_position, player_white, castling_rights, en_passant_targets, ANY_VALUE(move_uci) as move_uci
        FROM selected_moves
        GROUP BY 1, 2, 3, 4
        HAVING (hash(board_position) % {args.total_shards}) = {args.shard_id}
    """
    if args.limit:
        query += f" LIMIT {args.limit}"
        
    df = conn.execute(query).df()
    df["fen"] = df.apply(row_to_fen, axis=1)
    data_list = df[["fen", "move_uci"]].to_dict('records')
    
    print(f"Processing {len(data_list)} unique positions...")
    
    start_time = time.time()
    with mp.Pool(processes=args.n_workers, initializer=init_worker, initargs=(args.engine,)) as pool:
        results = list(tqdm(pool.imap(analyze_position, data_list), total=len(data_list)))
    
    duration = time.time() - start_time
    print(f"Analysis complete in {duration:.2f}s ({len(data_list)/duration:.2f} pos/s)")
    
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        import pandas as pd
        res_df = pd.DataFrame(valid_results)
        res_df = res_df[["fen", "e_win_best", "e_win_second_best", "e_win_move_taken", "n_repeats"]]
        conn.execute(f"INSERT OR REPLACE INTO {table_name} (fen, e_win_best, e_win_second_best, e_win_move_taken, n_repeats) SELECT * FROM res_df")
        print(f"Inserted {len(valid_results)} records into {table_name}.")

    conn.close()

if __name__ == "__main__":
    main()
