import duckdb

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def check_one_game():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    print("Moves for a single game:")
    res = conn.sql("""
        SELECT move_ply, player_white, player_clock_time, move_time 
        FROM selected_moves 
        WHERE gid = (SELECT gid FROM selected_moves LIMIT 1)
        ORDER BY move_ply
    """).df()
    print(res)
    conn.close()

if __name__ == "__main__":
    check_one_game()
