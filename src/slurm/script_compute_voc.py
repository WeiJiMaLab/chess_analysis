import sys
import os
# Add parent directory to path to allow importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import multiprocessing
import resource
import time

import chess
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import get_stockfish_engine

# Constants for analysis (mirroring voc_movetime.py)
SHALLOW_DEPTH = 1
DEEP_DEPTH = 14

def score_to_wp(score, board_after):
    try:
        wdl = score.white().wdl(ply=board_after.ply())
        wp_white = wdl.wins / 1000.0
        return wp_white if not board_after.turn == chess.WHITE else 1.0 - wp_white
    except:
        return None

def compute_voc(board, engine, candidate_moves, deep_depth, time_limit, game_ply):
    scores = {}
    for move in candidate_moves:
        board_after = board.copy()
        board_after.push(move)
        if board_after.is_stalemate():
            scores[move] = 0.5
            continue
        engine.configure({"Clear Hash": True})
        try: 
            info = engine.analyse(board_after, chess.engine.Limit(depth=deep_depth))
        except Exception as e: 
            print(f"  [Warning] Engine analysis failed for move {move}: {e}")
            continue
        
        # Use the actual game ply (current ply + 1 for board_after) for WDL
        try:
            wdl = info["score"].white().wdl(ply=game_ply + 1)
            wp_white = wdl.wins / 1000.0
            wp = wp_white if not board_after.turn == chess.WHITE else 1.0 - wp_white
            if wp is not None: 
                scores[move] = wp
        except:
            continue
    return scores

def analyze_position(position_row, retries=2):
    engine = None
    attempt = 0
    game_ply = int(position_row.move_ply)
    while attempt <= retries:
        try:
            engine = get_stockfish_engine(version=14)
            board = chess.Board(position_row.board_position)
            info_shallow = engine.analyse(board, chess.engine.Limit(depth=SHALLOW_DEPTH), multipv=5)
            candidate_moves = [entry["pv"][0] for entry in info_shallow]
            shallow_move = candidate_moves[0]
            scores = compute_voc(board, engine, candidate_moves, DEEP_DEPTH, TIME_LIMIT, game_ply)
            
            if shallow_move not in scores or len(scores) == 0:
                return None
                
            voc = max(scores.values()) - scores[shallow_move]
            return {
                "board_position": position_row.board_position,
                "player_white":   position_row.player_white,
                "voc":            voc,
                "move_time":      position_row.move_time,
                "elo":            position_row.white_elo if position_row.player_white else position_row.black_elo,
                "n_candidates":   len(candidate_moves),
                "ply":            game_ply
            }
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
            attempt += 1
            print(f"  [Error] Engine terminated for position {position_row.board_position[:20]}... (Attempt {attempt}/{retries+1}): {e}")
            if engine is not None:
                try: engine.quit()
                except: pass
            if attempt <= retries:
                time.sleep(1) # Brief cooldown before retry
                continue
            return None
        except Exception as e:
            print(f"  [Fatal] Unexpected error for position {position_row.board_position[:20]}...: {e}")
            return None
        finally:
            if engine is not None:
                try: engine.quit()
                except: pass

def main():
    parser = argparse.ArgumentParser(description="Compute Value of Computation (VOC) for chess positions.")
    parser.add_argument("--n_games", type=int, default=500, help="Number of games to process")
    parser.add_argument("--n_jobs", type=int, default=40, help="Number of parallel jobs")
    parser.add_argument("--retries", type=int, default=2, help="Number of times to retry on engine failure")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for incremental saving")
    args = parser.parse_args()

    # Resource management
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Ground paths relative to the project root (grandparent of this script's location in src/slurm/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", f"moves_{args.n_games}.parquet")
    output_path = os.path.join(base_dir, "data", "processed", f"voc_{args.n_games}.csv")
    checkpoint_path = os.path.join(base_dir, "data", "processed", f"voc_{args.n_games}_partial.csv")

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    positions = df.drop_duplicates("board_position")
    
    # Check if we have partial results to resume (optional enhancement)
    processed_positions = set()
    if os.path.exists(checkpoint_path):
        try:
            df_partial = pd.read_csv(checkpoint_path)
            processed_positions = set(df_partial["board_position"].unique())
            print(f"Found existing checkpoint with {len(processed_positions)} positions. Resuming...")
        except:
            pass

    remaining_positions = positions[~positions["board_position"].isin(processed_positions)]
    
    if len(remaining_positions) == 0:
        print("All positions already processed.")
        return

    print(f"Processing {len(remaining_positions)} remaining positions using {args.n_jobs} workers...")
    
    # Process in batches to allow for incremental saving
    all_results = []
    pos_list = [row for _, row in remaining_positions.iterrows()]
    
    for i in range(0, len(pos_list), args.batch_size):
        batch = pos_list[i : i + args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{(len(pos_list)-1)//args.batch_size + 1}...")
        
        batch_results = Parallel(n_jobs=args.n_jobs)(
            delayed(analyze_position)(row, retries=args.retries) for row in batch
        )
        
        # Filter and append
        valid_results = [r for r in batch_results if r is not None]
        all_results.extend(valid_results)
        
        # Save checkpoint
        df_batch = pd.DataFrame(valid_results)
        header = not os.path.exists(checkpoint_path)
        df_batch.to_csv(checkpoint_path, mode='a', index=False, header=header)

    # Final save
    if os.path.exists(checkpoint_path):
        df_final = pd.read_csv(checkpoint_path)
        df_final.to_csv(output_path, index=False)
        print(f"✅ Computation complete. Final results saved to {output_path}")
        # Clean up partial file if desired, but keeping it for safety
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
