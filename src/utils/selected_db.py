"""DuckDB workspace defaults for human move analysis (see ``preprocess.py`` ``merge`` + ``process_moves``)."""

SELECTED_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

# Raw merged parquets → ``merge_game_shards``.
TABLE_MOVES = "moves"

# Derived features (fen, ply_tertiles, board counts) built by ``process_moves`` after ``merge``.
TABLE_PROCESSED_MOVES = "processed_moves"

# Positive ``move_time`` only (dashboards / typical plots).
TABLE_PROCESSED_MOVES_NONZERO = "processed_moves_nonzero"
