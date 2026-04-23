import duckdb
import pandas as pd

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def check_clock_distribution():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    print("Fetching player_clock_time distribution from selected_moves...")
    # Get counts of moves for each clock second
    query = """
    SELECT 
        player_clock_time,
        COUNT(*) as move_count
    FROM selected_moves
    GROUP BY player_clock_time
    ORDER BY player_clock_time;
    """
    df = conn.sql(query).df()
    
    if df.empty:
        print("No moves found.")
        return

    print("\nTop 20 clock times by move count:")
    print(df.sort_values('move_count', ascending=False).head(20))
    
    # Specifically check around 300
    around_300 = df[(df['player_clock_time'] >= 290) & (df['player_clock_time'] <= 310)]
    print("\nMove counts around 300s:")
    print(around_300)

    conn.close()

if __name__ == "__main__":
    check_clock_distribution()
