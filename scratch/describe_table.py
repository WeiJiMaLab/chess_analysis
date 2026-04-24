import duckdb
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
print(conn.execute("DESCRIBE _selected_moves_nonzero_T;").df())
conn.close()
