import duckdb

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"

def check_selected_tcs():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    conn.sql(f"ATTACH '{CORE_DB}' AS core (READ_ONLY);")
    
    print("Checking initial_clock for games in selected_moves...")
    query = """
    SELECT g.initial_clock, count(distinct m.gid) as game_count
    FROM selected_moves m
    JOIN core.games g ON m.gid = g.gid
    GROUP BY g.initial_clock
    """
    res = conn.sql(query).df()
    print(res)
    
    conn.close()

if __name__ == "__main__":
    check_selected_tcs()
