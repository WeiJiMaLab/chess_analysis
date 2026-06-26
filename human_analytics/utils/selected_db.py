"""DuckDB workspace defaults for human move analysis (see ``preprocess.py`` ``merge`` + ``process_moves``)."""

from utils.helpers import CONFIG

SELECTED_DB_DEFAULT = CONFIG["selected_db_default"]

# Raw merged parquets → ``merge_game_shards``.
TABLE_MOVES = CONFIG["table_moves"]

# Derived features (fen, ply_tertiles, board counts) built by ``process_moves`` after ``merge``.
TABLE_PROCESSED_MOVES = CONFIG["table_processed_moves"]

# Positive ``move_time`` only (dashboards / typical plots).
TABLE_PROCESSED_MOVES_NONZERO = CONFIG["table_processed_moves_nonzero"]
