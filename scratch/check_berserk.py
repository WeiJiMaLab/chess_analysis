import duckdb
import pandas as pd

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def check_berserk_detection():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    # Let's join selected_moves with games to get initial_clock if not already there
    # Wait, the user's selected_moves seems to have player_clock_time.
    # Does it have initial_clock? Let's check schema again.
    # In my previous inspect_schema.py output:
    # Schema for table: selected_moves
    # 0: gid, 1: move_ply, 2: move_uci, 3: move_time, 4: player_white, 5: player_clock_time, 6: opponent_clock_time, ...
    # It doesn't have initial_clock.
    
    conn.sql("ATTACH '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db' AS core (READ_ONLY);")
    
    print("Checking first moves for games with initial_clock = 600...")
    query = """
    SELECT 
        m.gid,
        m.move_ply,
        m.player_white,
        m.player_clock_time,
        g.initial_clock,
        g.clock_increment
    FROM selected_moves m
    JOIN core.games g ON m.gid = g.gid
    WHERE m.move_ply IN (1, 2)
    LIMIT 10000;
    """
    df = conn.sql(query).df()
    
    if df.empty:
        print("No moves found in selected_moves.")
        return

    # For each gid, we might have ply 1 (white) and ply 2 (black)
    # Filter for ply 1 if player_white is true, ply 2 if player_white is false
    first_moves = df[((df['move_ply'] == 1) & (df['player_white'] == True)) | 
                     ((df['move_ply'] == 2) & (df['player_white'] == False))]
    
    print("\nSummary of first move clock times (Expected initial_clock = 600):")
    print(first_moves['player_clock_time'].value_counts().head(20))
    
    berserk_candidates = first_moves[first_moves['player_clock_time'] <= 310] # 300 is half of 600
    print(f"\nFound {len(berserk_candidates)} potential berserk starts out of {len(first_moves)} games.")
    
    if len(berserk_candidates) > 0:
        print("\nSample of potential berserk games:")
        print(berserk_candidates.head(10))

    conn.close()

if __name__ == "__main__":
    check_berserk_detection()
