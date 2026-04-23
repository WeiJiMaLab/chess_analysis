import duckdb

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"

def inspect_db(db_path, name):
    print(f"--- Inspecting {name} ({db_path}) ---")
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        tables = conn.sql("SHOW ALL TABLES").df()
        
        for _, row in tables.iterrows():
            table_name = row['name']
            if table_name not in ['games', 'moves', 'players']:
                continue
            print(f"\nSchema for table: {table_name}")
            schema = conn.sql(f"DESCRIBE {table_name}").df()
            print(schema[['column_name', 'column_type']].to_string())
            
            # Check for specific keywords in column names
            keywords = ['tournament', 'berserk', 'points', 'arena', 'event']
            matching = [col for col in schema['column_name'] if any(kw in col.lower() for kw in keywords)]
            if matching:
                print(f"!!! Matching columns in {table_name}: {matching}")
        
        conn.close()
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

inspect_db(CORE_DB, "Core DB")
