import duckdb
import pandas as pd

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    print("Calculating early ply statistics (1-20)...")
    # We use _selected_moves to include T=0
    df = conn.execute("""
        SELECT 
            move_ply,
            avg(move_time) as avg_time,
            median(move_time) as median_time,
            avg(CASE WHEN move_time = 0 THEN 1 ELSE 0 END) as instant_prob,
            count(*) as n
        FROM _selected_moves
        WHERE move_ply <= 20
        GROUP BY move_ply
        ORDER BY move_ply
    """).df()
    
    print("\n--- Early Ply Stats ---")
    print(df.to_string(index=False))
    
    # Check specifically for the "spike" around ply 10
    peak_ply = df.loc[df['instant_prob'].idxmax(), 'move_ply']
    peak_val = df['instant_prob'].max()
    print(f"\nPeak Instant Move Probability: {peak_val:.2%} at Ply {peak_ply}")
    
    conn.close()

if __name__ == "__main__":
    main()
