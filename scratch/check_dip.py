import duckdb
import pandas as pd

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def check_move_time_at_300():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    print("Calculating mean move_time vs player_clock_time around 300s...")
    query = """
    SELECT 
        player_clock_time,
        AVG(move_time) as avg_move_time,
        COUNT(*) as move_count
    FROM selected_moves
    WHERE player_clock_time BETWEEN 250 AND 350
    GROUP BY player_clock_time
    ORDER BY player_clock_time;
    """
    df = conn.sql(query).df()
    
    print(df)
    
    conn.close()

if __name__ == "__main__":
    check_move_time_at_300()
