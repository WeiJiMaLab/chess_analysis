from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

DEFAULT_CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"
DEFAULT_RAW_DATA = "/scratch/gpfs/GRIFFITHS/chess-db/rawdata"
DEFAULT_OUT_DIR = "/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol/roots"


def compose_full_fen(
    board_position: str,
    player_white: bool,
    castling_rights: str,
    en_passant_targets: str,
    halfmove_clock: int,
    fullmove_number: int,
) -> str:
    """Build a full FEN string matching the DuckDB sampling query.

    ``castling_rights`` / ``en_passant_targets`` use ``'-'`` for empty, as in
    Lichess parquet exports.
    """
    side = "w" if player_white else "b"
    cr = castling_rights if castling_rights else "-"
    ep = en_passant_targets if en_passant_targets else "-"
    return (
        f"{board_position} {side} {cr} {ep} "
        f"{int(halfmove_clock)} {int(fullmove_number)}"
    )


class ChessSampler:
    def __init__(
        self,
        core_db: str = DEFAULT_CORE_DB,
        raw_data_dir: str = DEFAULT_RAW_DATA,
        memory_limit: str = "40GB",
        threads: int = 8,
    ):
        self.core_db = core_db
        self.raw_data_dir = raw_data_dir
        self.conn = duckdb.connect(database=":memory:", config={
            "memory_limit": memory_limit,
            "threads": threads
        })
        self.conn.sql(f"ATTACH '{self.core_db}' AS core (READ_ONLY);")

    def sample_positions(
        self,
        year: int = 2023,
        n_games: int = 1000,
        n_final_fens: int = 100,
        min_elo: int = 1800,
        max_elo: int = 2600,
        min_ply: int = 8,
        max_ply: int = 120,
        min_legal_moves: int = 2,
        max_legal_moves: int = 60,
        min_pieces: int = 8,
        max_pieces: int = 32,
        seed: int = 42,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Sample chess positions from the Lichess database matching criteria.
        Ensures only one position per game is sampled.
        """
        start_ts = start_date if start_date else f"{year}-01-01"
        end_ts = end_date if end_date else f"{year + 1}-01-01"

        print(f"Sampling {n_games} games from {year}...")
        
        # 1. Sample games first
        self.conn.sql(f"""
        CREATE OR REPLACE TEMP TABLE sampled_games AS (
            SELECT
                gid,
                utc_datetime,
                time_control_type,
                opening,
                white_elo,
                black_elo
            FROM
                core.games
            WHERE
                utc_datetime >= '{start_ts}'
                AND utc_datetime < '{end_ts}'
                AND white_elo BETWEEN {min_elo} AND {max_elo}
                AND black_elo BETWEEN {min_elo} AND {max_elo}
            LIMIT {n_games}
        );
        """)

        sampled_games_df = self.conn.sql("SELECT * FROM sampled_games").df()
        print(f"Sampled {len(sampled_games_df)} games.")
        if sampled_games_df.empty:
            return pd.DataFrame()

        # Group gids by partition and segment to optimize parquet reading
        # gid is typically YYYYMMNNNNNN
        sampled_games_df["partition"] = sampled_games_df["gid"].astype(str).str[:6]
        # In the original script, segment was str[6:9]. 
        # Let's check how many segments there are.
        # For simplicity and robustness, we'll just use the partition for now if possible, 
        # or follow the original logic if it's strictly necessary.
        
        file_groups = sampled_games_df.groupby("partition")["gid"].apply(list).reset_index()

        print(f"Scanning moves for {len(sampled_games_df)} games across {len(file_groups)} partitions...")

        self.conn.sql(f"""
        CREATE OR REPLACE TEMP TABLE candidate_positions (
            full_fen VARCHAR,
            gid UBIGINT,
            move_ply USMALLINT,
            n_pieces UTINYINT,
            n_possible_moves UTINYINT
        );
        """)

        for _, row in file_groups.iterrows():
            partition = row["partition"]
            gids_str = ",".join(map(str, row["gid"]))
            parquet_glob = f"{self.raw_data_dir}/partition={partition}/*-moves.parquet"

            # ``full_fen`` expression below must stay aligned with :func:`compose_full_fen`.
            self.conn.sql(f"""
            INSERT INTO candidate_positions
            SELECT
                (
                    board_position || ' ' ||
                    CASE WHEN player_white THEN 'w' ELSE 'b' END || ' ' ||
                    COALESCE(NULLIF(castling_rights, ''), '-') || ' ' ||
                    COALESCE(NULLIF(en_passant_targets, ''), '-') || ' ' ||
                    CAST(halfmove_clock AS VARCHAR) || ' ' ||
                    CAST(CAST(1 + FLOOR((move_ply - 1) / 2.0) AS BIGINT) AS VARCHAR)
                ) AS full_fen,
                gid,
                move_ply,
                n_pieces,
                n_possible_moves
            FROM
                read_parquet('{parquet_glob}')
            WHERE
                gid IN ({gids_str})
                AND board_position IS NOT NULL
                AND board_position <> ''
                AND move_ply BETWEEN {min_ply} AND {max_ply}
                AND n_possible_moves BETWEEN {min_legal_moves} AND {max_legal_moves}
                AND n_pieces BETWEEN {min_pieces} AND {max_pieces}
            """)

        n_candidates = self.conn.sql("SELECT COUNT(*) FROM candidate_positions").fetchone()[0]
        print(f"Found {n_candidates} candidate positions.")
        if n_candidates == 0:
            return pd.DataFrame()

        # 2. Pick one move per game and reservoir sample to final count
        print(f"Filtering to one move per game and sampling {n_final_fens} final FENs...")
        final_df = self.conn.sql(f"""
        WITH one_per_game AS (
            SELECT
                cp.*,
                sg.utc_datetime,
                sg.time_control_type,
                sg.opening,
                sg.white_elo,
                sg.black_elo,
                ROW_NUMBER() OVER (PARTITION BY cp.gid ORDER BY hash(cp.move_ply, {seed + 1})) as rn
            FROM
                candidate_positions cp
            JOIN
                sampled_games sg ON cp.gid = sg.gid
        ),
        filtered AS (
            SELECT * EXCLUDE (rn) FROM one_per_game WHERE rn = 1
        )
        SELECT * FROM filtered
        USING SAMPLE reservoir({n_final_fens} ROWS) REPEATABLE ({seed + 2})
        """).df()

        return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--n-fens", type=int, default=100)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sampler = ChessSampler()
    df = sampler.sample_positions(
        year=args.year,
        n_games=args.n_games,
        n_final_fens=args.n_fens,
        seed=args.seed
    )

    print(f"Sampled {len(df)} positions.")
    
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path)
        else:
            df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
    else:
        # Default behavior: export one example if requested by user
        pass

if __name__ == "__main__":
    main()
