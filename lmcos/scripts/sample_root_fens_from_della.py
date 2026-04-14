from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


DEFAULT_PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS_personal.db"
DEFAULT_CORE_DB = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"
DEFAULT_OUT_DIR = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/roots"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample full-FEN root positions from the Della chess DB.")
    parser.add_argument("--personal-db", default=DEFAULT_PERSONAL_DB)
    parser.add_argument("--core-db", default=DEFAULT_CORE_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--sampled-games", type=int, default=200_000)
    parser.add_argument("--final-fens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--memory-limit", default="60GB")
    parser.add_argument("--temp-directory", default=".")
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--max-elo", type=int, default=2600)
    parser.add_argument("--min-game-halfmoves", type=int, default=20)
    parser.add_argument("--min-move-ply", type=int, default=8)
    parser.add_argument("--max-move-ply", type=int, default=120)
    parser.add_argument("--min-legal-moves", type=int, default=2)
    parser.add_argument("--max-legal-moves", type=int, default=60)
    parser.add_argument("--min-pieces", type=int, default=8)
    parser.add_argument("--max-pieces", type=int, default=32)
    parser.add_argument("--prefix", default="sampled_root_fens")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = f"{args.year}-01-01"
    end_ts = f"{args.year + 1}-01-01"
    final_sample_seed = args.seed + 2

    t0 = time.time()
    print(f"[1/7] Connecting to DuckDB at {args.personal_db}")
    conn = duckdb.connect(
        database=args.personal_db,
        config={
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "temp_directory": args.temp_directory,
        },
    )
    conn.sql(f"ATTACH '{args.core_db}' AS core (READ_ONLY);")

    print(f"[2/7] Building eligible games for {args.year}")
    conn.sql(f"""
    CREATE OR REPLACE TABLE eligible_games_{args.year} AS (
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
            AND n_moves >= {args.min_game_halfmoves}
            AND white_elo BETWEEN {args.min_elo} AND {args.max_elo}
            AND black_elo BETWEEN {args.min_elo} AND {args.max_elo}
    );
    """)

    print(f"[3/7] Sampling {args.sampled_games:,} games")
    conn.sql(f"""
    CREATE OR REPLACE TABLE sampled_games_{args.year} AS (
        SELECT
            *
        FROM
            eligible_games_{args.year}
        USING SAMPLE reservoir({args.sampled_games} ROWS) REPEATABLE ({args.seed})
    );
    """)

    sampled_games = conn.sql(f"""
    SELECT
        gid
    FROM
        sampled_games_{args.year};
    """).df()
    sampled_games["partition"] = sampled_games["gid"].astype(str).str[:6]
    sampled_games["segment"] = sampled_games["gid"].astype(str).str[6:9]
    file_groups = sampled_games.groupby(["partition", "segment"])["gid"].apply(list).reset_index()

    print(f"sampled_games={len(sampled_games):,}")
    print(f"files_to_scan={len(file_groups):,}")

    print("[4/7] Creating staged full-FEN table")
    conn.sql(f"""
    CREATE OR REPLACE TABLE sampled_root_positions_{args.year} AS (
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
    for i, row in file_groups.iterrows():
        partition = row["partition"]
        segment = row["segment"]
        gids = ",".join(str(int(gid)) for gid in row["gid"])

        conn.sql(f"""
        INSERT INTO sampled_root_positions_{args.year} BY NAME (
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
                    AND move_ply BETWEEN {args.min_move_ply} AND {args.max_move_ply}
                    AND n_possible_moves BETWEEN {args.min_legal_moves} AND {args.max_legal_moves}
                    AND n_pieces BETWEEN {args.min_pieces} AND {args.max_pieces}
            ),
            one_move_per_game AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY gid
                        ORDER BY hash(move_ply, {args.seed + 1})
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
                sampled_games_{args.year} AS sg
            USING
                (gid)
            WHERE
                om.rn = 1
        );
        """)

        if (i + 1) % 100 == 0 or i + 1 == len(file_groups):
            print(f"  processed_files={i + 1:,}/{len(file_groups):,} elapsed={time.time() - t0:.1f}s")

    print(f"[6/7] Reservoir-sampling final {args.final_fens:,} full FENs")
    conn.sql(f"""
    CREATE OR REPLACE TABLE final_root_positions_{args.year} AS (
        SELECT
            *
        FROM
            sampled_root_positions_{args.year}
        USING SAMPLE reservoir({args.final_fens} ROWS) REPEATABLE ({final_sample_seed})
    );
    """)

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
        final_root_positions_{args.year};
    """).df()
    print(counts)

    parquet_path = out_dir / f"{args.prefix}_{args.year}_with_metadata.parquet"
    txt_path = out_dir / f"{args.prefix}_{args.year}.txt"

    print("[7/7] Writing outputs")
    conn.sql(f"""
    COPY final_root_positions_{args.year}
    TO '{parquet_path.as_posix()}'
    (FORMAT PARQUET);
    """)

    conn.sql(f"""
    COPY (
        SELECT
            full_fen
        FROM
            final_root_positions_{args.year}
    ) TO '{txt_path.as_posix()}'
    (FORMAT CSV, HEADER FALSE, DELIMITER '\t');
    """)

    print(f"Wrote: {txt_path}")
    print(f"Wrote: {parquet_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
