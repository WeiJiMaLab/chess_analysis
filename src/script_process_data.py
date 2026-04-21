import os
import sys
import time
import pandas as pd
import dask.dataframe as dd
import chess
import multiprocessing as mp
from tqdm import tqdm

# Ensure the src directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.utils import preprocess_data, voc_single_sample, voc_consideration_set, best_two_diff, get_stockfish_engine

# Global engine variable per worker process
_worker_engine = None

def init_worker():
    """Initialize a Stockfish engine instance once per worker process."""
    global _worker_engine
    _worker_engine = get_stockfish_engine(version=14)

def analyze_row(row_dict):
    """
    Process a single row using the persistent engine instance.
    """
    global _worker_engine
    try:
        if _worker_engine is None:
            # Fallback in case initializer wasn't called
            _worker_engine = get_stockfish_engine(version=14)
            
        board = chess.Board(row_dict["fen"])
        
        return {
            "voc_single_sample": voc_single_sample(_worker_engine, board),
            "voc_consideration_set": voc_consideration_set(_worker_engine, board),
            "best_two_diff": best_two_diff(_worker_engine, board),
        }
    except Exception as e:
        return {
            "error": str(e),
            "voc_single_sample": None,
            "voc_consideration_set": None,
            "best_two_diff": None,
        }

import argparse

def main():
    parser = argparse.ArgumentParser(description="Parallel VOC data processing.")
    parser.add_argument("--n_workers", type=int, default=40, help="Number of parallel workers")
    parser.add_argument("--n_rows", type=int, default=None, help="Number of rows to process (None for all)")
    args = parser.parse_args()

    n_workers = args.n_workers
    
    # 1. Load and Preprocess Data
    data_path = os.path.join(parent_dir, "data", "moves_500.parquet")
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return
    
    print(f"Loading data from {data_path}...")
    df_raw = dd.read_parquet(data_path)
    
    print("Preprocessing data...")
    df_proc = preprocess_data(df_raw).compute()
    
    if args.n_rows is not None:
        df_sub = df_proc.head(args.n_rows)
    else:
        df_sub = df_proc
        
    data_list = df_sub.to_dict('records')
    
    print(f"Starting parallel processing of {len(data_list)} rows with {n_workers} workers (reusing engines)...")
    
    start_time = time.time()
    
    # 2. Run Parallel Processing with Pool and initializer
    # We use mp.Pool instead of joblib to clearly manage engine persistence and tqdm
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        # imap provides an iterator used by tqdm for progress tracking
        results = list(tqdm(pool.imap(analyze_row, data_list), total=len(data_list)))
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 3. Combine results with original data and save
    res_df = pd.DataFrame(results)
    
    # Merge results onto the processed dataframe slice
    # Note: df_sub and res_df are aligned because pool.imap preserves order
    for col in res_df.columns:
        df_sub[col] = res_df[col].values
    
    output_dir = os.path.join(parent_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_moves_500.parquet")
    
    print(f"Exporting results to {output_path}...")
    df_sub.to_parquet(output_path, index=False)
    
    # 4. Summary
    print("\n" + "="*40)
    print("ANALYSIS SUMMARY")
    print("="*40)
    print(f"Total rows processed: {len(results)}")
    print(f"Total time:           {duration:.2f} seconds")
    print(f"Average time per row: {duration/len(results):.4f} seconds")
    print(f"Effective throughput: {len(results)/duration:.2f} rows/second")
    print(f"Output saved to:      {output_path}")
    print("="*40)

    if "error" in res_df.columns:
        errors = res_df["error"].dropna()
        if not errors.empty:
            print(f"Encountered {len(errors)} errors during processing.")
    
    print("\nSample Results (merged):")
    cols_to_show = ["fen", "voc_single_sample", "voc_consideration_set", "best_two_diff"]
    print(df_sub[cols_to_show].head())

if __name__ == "__main__":
    main()
