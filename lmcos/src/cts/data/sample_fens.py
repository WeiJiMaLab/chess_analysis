"""Sample candidate root FENs from the Della Lichess game database.

First step of the data pipeline: runs the sampling SQL (see
``sql/sample_root_positions.sql``) against the on-cluster DuckDB build of
the Lichess corpus, applies Elo / ply / piece-count filters, draws a
reservoir sample of full-FEN positions, and writes them to a text file
plus a metadata-rich parquet. The text output feeds the next stage,
``validate_root_fens_with_lc0.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
from pydantic import BaseModel, ConfigDict


# Cluster paths baked in as defaults — the script is only ever run on Della.
DEFAULT_PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS_personal.db"
DEFAULT_CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"
DEFAULT_OUT_DIR = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/roots"


class SampleFensConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    personal_db: str = DEFAULT_PERSONAL_DB
    core_db: str = DEFAULT_CORE_DB
    out_dir: str = DEFAULT_OUT_DIR
    year: int = 2023
    sampled_games: int = 200_000
    final_fens: int = 100_000
    seed: int = 2023
    threads: int = 10
    memory_limit: str = "60GB"
    temp_directory: str = "."
    min_elo: int = 1800
    max_elo: int = 2600
    min_game_halfmoves: int = 20
    min_move_ply: int = 8
    max_move_ply: int = 120
    min_legal_moves: int = 2
    max_legal_moves: int = 60
    min_pieces: int = 8
    max_pieces: int = 32
    prefix: str = "sampled_root_fens"


def main(config: SampleFensConfig) -> None:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Year window is [Jan 1 year, Jan 1 year+1). final_sample_seed must
    # differ from per-game-move pick seed so the two reservoirs aren't
    # correlated.
    start_ts = f"{config.year}-01-01"
    end_ts = f"{config.year + 1}-01-01"
    final_sample_seed = config.seed + 2

    t0 = time.time()
    print(f"[1/7] Connecting to DuckDB at {config.personal_db}")
    # Personal DB holds the staging tables; core DB is attached read-only
    # so the script can never corrupt the shared Lichess build.
    conn = duckdb.connect(
        database=config.personal_db,
        config={
            "threads": config.threads,
            "memory_limit": config.memory_limit,
            "temp_directory": config.temp_directory,
        },
    )
    conn.sql(f"ATTACH '{config.core_db}' AS core (READ_ONLY);")

    print(f"[2/7] Building eligible games for {config.year}")
    # Filter games by date/length/Elo before sampling so the reservoir
    # draws only from positions we'd actually accept.
    conn.sql(f"""
    CREATE OR REPLACE TABLE eligible_games_{config.year} AS (
        SELECT
            gid,
            utc_datetime,
            time_control_type,
            opening,
            white_elo,
            black_elo,
            n_moves
        FROM
            core.games
        WHERE
            utc_datetime >= '{start_ts}'
            AND utc_datetime < '{end_ts}'
            AND n_moves >= {config.min_game_halfmoves}
            AND white_elo BETWEEN {config.min_elo} AND {config.max_elo}
            AND black_elo BETWEEN {config.min_elo} AND {config.max_elo}
    );
    """)

    print(f"[3/7] Sampling {config.sampled_games:,} games")
    # REPEATABLE makes the game-level draw deterministic given --seed.
    conn.sql(f"""
    CREATE OR REPLACE TABLE sampled_games_{config.year} AS (
        SELECT
            *
        FROM
            eligible_games_{config.year}
        USING SAMPLE reservoir({config.sampled_games} ROWS) REPEATABLE ({config.seed})
    );
    """)

    # Move data on disk is partitioned by gid prefix: the first 6 chars
    # select the parquet directory and the next 3 select the file within
    # it. Grouping sampled gids by (partition, segment) means each parquet
    # is read exactly once below.
    sampled_games = conn.sql(f"""
    SELECT
        gid
    FROM
        sampled_games_{config.year};
    """).df()
    sampled_games["partition"] = sampled_games["gid"].astype(str).str[:6]
    sampled_games["segment"] = sampled_games["gid"].astype(str).str[6:9]
    file_groups = sampled_games.groupby(["partition", "segment"])["gid"].apply(list).reset_index()

    print(f"sampled_games={len(sampled_games):,}")
    print(f"files_to_scan={len(file_groups):,}")

    print("[4/7] Creating staged full-FEN table")
    # Create an empty, fully-typed staging table. The `WHERE 1 = 0` trick
    # gives us the schema without inserting any rows; subsequent INSERT BY
    # NAME calls then match columns by name regardless of SELECT order.
    conn.sql(f"""
    CREATE OR REPLACE TABLE sampled_root_positions_{config.year} AS (
        SELECT
            CAST(NULL AS VARCHAR) AS full_fen,
            CAST(NULL AS UBIGINT) AS gid,
            CAST(NULL AS UINTEGER) AS partition,
            CAST(NULL AS USMALLINT) AS move_ply,
            CAST(NULL AS UTINYINT) AS n_pieces,
            CAST(NULL AS UTINYINT) AS n_possible_moves,
            CAST(NULL AS TIMESTAMP_S) AS utc_datetime,
            CAST(NULL AS VARCHAR) AS time_control_type,
            CAST(NULL AS VARCHAR) AS opening,
            CAST(NULL AS SMALLINT) AS white_elo,
            CAST(NULL AS SMALLINT) AS black_elo
        WHERE
            1 = 0
    );
    """)

    print("[5/7] Reading raw parquet by (partition, segment)")
    # One INSERT per parquet file; each scans only the gids that fell into
    # this (partition, segment) group, then picks one move per game.
    for i, row in file_groups.iterrows():
        partition = row["partition"]
        segment = row["segment"]
        gids = ",".join(str(int(gid)) for gid in row["gid"])

        conn.sql(f"""
        INSERT INTO sampled_root_positions_{config.year} BY NAME (
            WITH candidate_moves AS (
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
                    partition,
                    move_ply,
                    n_pieces,
                    n_possible_moves
                FROM
                    '/scratch/gpfs/GRIFFITHS/chess-db/rawdata/partition={partition}/{segment}-moves.parquet'
                WHERE
                    gid IN ({gids})
                    AND board_position IS NOT NULL
                    AND board_position <> ''
                    AND move_ply BETWEEN {config.min_move_ply} AND {config.max_move_ply}
                    AND n_possible_moves BETWEEN {config.min_legal_moves} AND {config.max_legal_moves}
                    AND n_pieces BETWEEN {config.min_pieces} AND {config.max_pieces}
            ),
            one_move_per_game AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY gid
                        ORDER BY hash(move_ply, {config.seed + 1})
                    ) AS rn
                FROM
                    candidate_moves
            )
            SELECT
                om.full_fen,
                om.gid,
                om.partition,
                om.move_ply,
                om.n_pieces,
                om.n_possible_moves,
                sg.utc_datetime,
                CAST(sg.time_control_type AS VARCHAR) AS time_control_type,
                sg.opening,
                sg.white_elo,
                sg.black_elo
            FROM
                one_move_per_game AS om
            JOIN
                sampled_games_{config.year} AS sg
            USING
                (gid)
            WHERE
                om.rn = 1
        );
        """)

        # Progress every 100 files plus the final iteration.
        if (i + 1) % 100 == 0 or i + 1 == len(file_groups):
            print(f"  processed_files={i + 1:,}/{len(file_groups):,} elapsed={time.time() - t0:.1f}s")

    print(f"[6/7] Reservoir-sampling final {config.final_fens:,} full FENs")
    # Second reservoir: every sampled game contributes one candidate
    # position, but we only want --final-fens of them. Deterministic given
    # final_sample_seed.
    conn.sql(f"""
    CREATE OR REPLACE TABLE final_root_positions_{config.year} AS (
        SELECT
            *
        FROM
            sampled_root_positions_{config.year}
        USING SAMPLE reservoir({config.final_fens} ROWS) REPEATABLE ({final_sample_seed})
    );
    """)

    # Sanity counts: total rows, distinct games (should equal rows since
    # we kept one move per game), and FENs missing the canonical 6 fields
    # (any non-zero count means the FEN-construction expression is broken).
    counts = conn.sql(f"""
    SELECT
        COUNT(*) AS n_final_positions,
        COUNT(DISTINCT gid) AS n_distinct_games,
        SUM(
            CASE
                WHEN 1 + LENGTH(full_fen) - LENGTH(REPLACE(full_fen, ' ', '')) = 6 THEN 0
                ELSE 1
            END
        ) AS n_bad_fens
    FROM
        final_root_positions_{config.year};
    """).df()
    print(counts)

    parquet_path = out_dir / f"{config.prefix}_{config.year}_with_metadata.parquet"
    txt_path = out_dir / f"{config.prefix}_{config.year}.txt"

    print("[7/7] Writing outputs")
    # Parquet keeps full metadata (Elo, opening, etc.) for later analysis;
    # the txt file is just FENs and is what the next pipeline stage reads.
    conn.sql(f"""
    COPY final_root_positions_{config.year}
    TO '{parquet_path.as_posix()}'
    (FORMAT PARQUET);
    """)

    conn.sql(f"""
    COPY (
        SELECT
            full_fen
        FROM
            final_root_positions_{config.year}
    ) TO '{txt_path.as_posix()}'
    (FORMAT CSV, HEADER FALSE, DELIMITER '\t');
    """)

    print(f"Wrote: {txt_path}")
    print(f"Wrote: {parquet_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(SampleFensConfig, main)
