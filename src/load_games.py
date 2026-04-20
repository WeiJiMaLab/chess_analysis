import argparse
import os

import pandas as pd

from utils import get_db_connection


def main():
    parser = argparse.ArgumentParser(description="Sample chess games from the database.")
    parser.add_argument('--n_games', type=int, default=200, help='Number of games to sample')
    args = parser.parse_args()

    n_games = args.n_games
    games_cache = os.path.join(os.path.dirname(__file__), "..", "data", f"moves_{n_games}.parquet")

    need_reload = not os.path.exists(games_cache)
    conn = get_db_connection(threads=16, memory_limit="6GB")
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    db_path = '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db'
    try:
        conn.execute(f"ATTACH '{db_path}' AS core (READ_ONLY)")
    except Exception as e:
        print(f"Warning attaching database: {e}")

    # use hash-based sampling to get a random sample of 2% of the games
    sample_games = conn.sql(f"""
        WITH picked AS (
            SELECT g.gid
            FROM core.games g
            WHERE g.utc_datetime BETWEEN '{start_date}' AND '{end_date}'
            AND g.initial_clock = 600
            AND g.clock_increment = 0
            AND g.white_elo >= 2000
            AND g.black_elo >= 2000
            AND MOD(HASH(g.gid), 1000) < 50  -- ~5% slice
            LIMIT {n_games}
        )

        SELECT m.gid, m.board_position, m.move_time, m.move_ply, m.player_white,
            g.white_elo, g.black_elo, g.initial_clock, g.clock_increment,
            g.white_id, g.black_id
        FROM core.moves m
        JOIN picked p ON m.gid = p.gid
        JOIN core.games g ON m.gid = g.gid
        ORDER BY m.gid, m.move_ply
    """).df()

    conn.close()
    sample_games.to_parquet(games_cache)

    print(f"{sample_games['gid'].nunique()} games, {len(sample_games)} plys")

if __name__ == "__main__":
    main()
    print("✅ Done loading games")
