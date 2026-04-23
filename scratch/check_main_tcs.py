import duckdb

CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"

def check_main_db_tcs():
    conn = duckdb.connect(database=CORE_DB, read_only=True)
    print("Checking initial_clock and clock_increment distribution in core.games...")
    res = conn.sql("""
        SELECT initial_clock, clock_increment, count(*) 
        FROM games 
        GROUP BY initial_clock, clock_increment 
        ORDER BY count(*) DESC 
        LIMIT 20
    """).df()
    print(res)
    conn.close()

if __name__ == "__main__":
    check_main_db_tcs()
