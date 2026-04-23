import duckdb

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def check_low_ply_300():
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    print("Checking move_ply distribution for moves where player_clock_time = 300:")
    query = """
    SELECT move_ply, count(*) as move_count
    FROM selected_moves
    WHERE player_clock_time = 300
    GROUP BY move_ply
    ORDER BY move_ply
    LIMIT 20
    """
    res = conn.sql(query).df()
    print(res)
    conn.close()

if __name__ == "__main__":
    check_low_ply_300()
