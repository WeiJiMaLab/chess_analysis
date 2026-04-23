import duckdb

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def check_berserk_movetime():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    print("Checking move_time for games where player_clock_time = 300 at ply 3 or 4:")
    query = """
    SELECT m1.gid, m1.move_ply, m1.player_clock_time as p1_clock, m1.move_time as p1_time,
           m3.move_ply as p3_ply, m3.player_clock_time as p3_clock
    FROM selected_moves m1
    JOIN selected_moves m3 ON m1.gid = m3.gid AND m1.move_ply = 1 AND m3.move_ply = 3
    WHERE m3.player_clock_time = 300
    LIMIT 10
    """
    res = conn.sql(query).df()
    print(res)
    conn.close()

if __name__ == "__main__":
    check_berserk_movetime()
