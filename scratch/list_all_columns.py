import duckdb

CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"

def check_all_columns():
    conn = duckdb.connect(database=CORE_DB, read_only=True)
    print("Columns in 'games' table:")
    cols = conn.sql("PRAGMA table_info('games')").df()
    print(cols['name'].tolist())
    
    print("\nColumns in 'moves' table:")
    cols = conn.sql("PRAGMA table_info('moves')").df()
    print(cols['name'].tolist())
    
    conn.close()

if __name__ == "__main__":
    check_all_columns()
