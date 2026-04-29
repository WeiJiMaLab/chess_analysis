import os
import argparse
import duckdb
import chess
import chess.engine
import multiprocessing as mp
from tqdm import tqdm
import time

from _bootstrap import ensure_src

ensure_src()

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
    """
    Win probability in [0, 1] for the *side to move* at the *root* position.

    Uses ``score.pov(board.turn)`` so WDL (wins/draws/losses) is always from the
    current player's perspective, consistent across multipv lines: each ``info`` row
    is a different first move (or continuation line), but the board before any
    move is the same, so best vs second are comparable in the same frame.

    For terminal root positions, multipv is skipped; the stored value is the
    game-theoretic outcome (mate => 0.0 for the mated side, draw => 0.5, etc.).
    """
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
    """
    Single-position eval for the distributed pipeline.

    Runs ``analyse`` with ``multipv=2`` at ``depth`` and records WDL of the
    first and second *root* lines (``info_list[0]`` and ``info_list[1]``) using
    :func:`get_e_win`. If only one line exists (e.g. a single legal move), the
    second is set equal to the first so the top-2 gap is zero. ``e_win_move_taken``
    is the WDL of the line whose PV starts with the move stored in the row (or
    a dedicated search over ``root_moves`` if the human move is not in the
    first two lines).
    """
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

        info_list = _worker_engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
        e_win_best = get_e_win(info_list[0], board) if len(info_list) > 0 else None
        e_win_second_best = get_e_win(info_list[1], board) if len(info_list) > 1 else e_win_best
        
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

def run_eval(args):
    os.makedirs(args.output_dir, exist_ok=True)
    shard_file = os.path.join(args.output_dir, f"results_{args.engine}_shard_{args.shard_id}.parquet")

    # Connect read-only to avoid locking issues
    conn = duckdb.connect(args.db, read_only=True)

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
    conn.close()
    
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
        res_df.to_parquet(shard_file)
        print(f"Saved {len(valid_results)} records to {shard_file}.")

def run_merge(args):
    input_path = os.path.join(args.input_dir, args.engine, "*.parquet")
    table_name = f"{args.engine}_evaluations"

    # Make sure the input directory exists
    dir_to_check = os.path.dirname(input_path.replace("*", ""))
    if not os.path.exists(dir_to_check):
        print(f"Input directory {dir_to_check} does not exist.")
        return

    print(f"Merging results for {args.engine} from {input_path} into {args.db}...")
    
    start_time = time.time()
    conn = duckdb.connect(args.db)
    
    # Rebuild the table in one pass from all shards. (Large INSERT OR REPLACE into a
    # PK-backed table can trigger DuckDB internal errors on this dataset; CTAS is stable.)
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            fen,
            any_value(e_win_best) AS e_win_best,
            any_value(e_win_second_best) AS e_win_second_best,
            any_value(e_win_move_taken) AS e_win_move_taken,
            any_value(n_repeats) AS n_repeats,
            current_timestamp AS timestamp
        FROM read_parquet('{input_path}')
        GROUP BY fen
    """)

    count_after = conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    
    duration = time.time() - start_time
    print(f"Merge complete in {duration:.2f}s.")
    print(f"Total records in {table_name}: {count_after}")

    # Now build selected_moves_with_engine
    print(f"Building selected_moves_with_engine for {args.engine}...")
    conn.execute(f"""
        CREATE OR REPLACE TABLE selected_moves_with_engine AS
        SELECT
            m.*,
            e.e_win_best,
            e.e_win_second_best,
            e.e_win_move_taken,
            e.n_repeats,
            abs(e.e_win_best - e.e_win_second_best) AS top2_wdl_diff
        FROM _selected_moves m
        INNER JOIN {table_name} e ON m.fen = e.fen
    """)
    n = conn.execute("SELECT count(*) FROM selected_moves_with_engine").fetchone()[0]
    n_base = conn.execute("SELECT count(*) FROM _selected_moves").fetchone()[0]
    n_missing = conn.execute(f"""
        SELECT count(*)
        FROM _selected_moves s
        LEFT JOIN {table_name} e ON s.fen = e.fen
        WHERE e.fen IS NULL
    """).fetchone()[0]
    
    conn.close()

    print(f"✅ selected_moves_with_engine: {n:,} rows (joining _selected_moves to {table_name})")
    print(f"   _selected_moves: {n_base:,} rows; moves with no engine row: {n_missing:,}")

def main():
    parser = argparse.ArgumentParser(description="Distributed Engine evaluation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run a single evaluation shard")
    eval_parser.add_argument("--engine", choices=["stockfish", "lc0"], required=True)
    eval_parser.add_argument("--db", default="/scratch/gpfs/GRIFFITHS/hl4291/personal.db")
    eval_parser.add_argument("--output_dir", required=True, help="Directory to save shard results")
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--n_workers", type=int, default=16)
    eval_parser.add_argument("--shard_id", type=int, default=0)
    eval_parser.add_argument("--total_shards", type=int, default=1)
    eval_parser.add_argument("--depth", type=int, default=5)

    # Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge evaluation shards and build joined table")
    merge_parser.add_argument("--engine", choices=["stockfish", "lc0"], required=True)
    merge_parser.add_argument("--db", default="/scratch/gpfs/GRIFFITHS/hl4291/personal.db")
    merge_parser.add_argument("--input_dir", default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/eval_results")

    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args)
    elif args.command == "merge":
        run_merge(args)

if __name__ == "__main__":
    main()
