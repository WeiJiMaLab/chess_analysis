import duckdb
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
conn = duckdb.connect(database=PERSONAL_DB, read_only=True)

print("Columns in selected_moves:")
print(conn.sql("DESCRIBE selected_moves;").df())

print("\nUnique board positions:")
try:
    count = conn.sql("SELECT count(DISTINCT board_position) FROM selected_moves;").fetchone()[0]
    print(f"Count: {count}")
except Exception as e:
    print(f"Error counting: {e}")

conn.close()
