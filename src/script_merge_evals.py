import os
import argparse
import duckdb
import time

def main():
    parser = argparse.ArgumentParser(description="Merge evaluation shards into personal.db")
    parser.add_argument("--engine", choices=["stockfish", "lc0"], required=True)
    parser.add_argument("--db", default="/scratch/gpfs/GRIFFITHS/hl4291/personal.db")
    parser.add_argument("--input_dir", default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/eval_results")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, args.engine, "*.parquet")
    table_name = f"{args.engine}_evaluations"

    if not os.path.exists(os.path.dirname(input_path.replace("*", ""))):
        print(f"Input directory {os.path.dirname(input_path.replace('*', ''))} does not exist.")
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
    conn.close()
    
    duration = time.time() - start_time
    print(f"Merge complete in {duration:.2f}s.")
    print(f"Total records in {table_name}: {count_after}")

if __name__ == "__main__":
    main()
