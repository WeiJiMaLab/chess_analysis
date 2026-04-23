import duckdb
import pandas as pd

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"

def check_berserk_full():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    conn.sql(f"ATTACH '{CORE_DB}' AS core (READ_ONLY);")
    
    print("Analyzing first moves for ALL games in selected_moves...")
    
    # Identify games where the first move has player_clock_time <= 300
    # but initial_clock was 600.
    query = """
    SELECT 
        m.gid,
        m.move_ply,
        m.player_clock_time,
        g.initial_clock
    FROM selected_moves m
    JOIN core.games g ON m.gid = g.gid
    WHERE m.move_ply IN (1, 2)
      AND ((m.move_ply = 1 AND m.player_white = true) OR (m.move_ply = 2 AND m.player_white = false))
    """
    df = conn.sql(query).df()
    
    if df.empty:
        print("No first moves found.")
        return

    print(f"Total games analyzed: {len(df)}")
    
    # Distribution of first move clock times
    dist = df['player_clock_time'].value_counts().sort_index(ascending=False)
    print("\nTop first move clock times:")
    print(dist.head(20))
    
    berserk_count = len(df[df['player_clock_time'] <= 300])
    print(f"\nGames starting at or below 300s: {berserk_count} ({berserk_count/len(df)*100:.2f}%)")
    
    # Check if there's a peak exactly at 300 for first moves
    if 300 in dist.index:
        print(f"Games starting exactly at 300s: {dist[300]}")
    
    conn.close()

if __name__ == "__main__":
    check_berserk_full()
