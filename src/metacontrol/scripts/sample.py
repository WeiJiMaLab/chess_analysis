"""Sample root FEN rows for metacontrol tree generation.

This script is intentionally outside ``metacontrol.data`` because root sampling
is an external data-acquisition step: it reads the Lichess DuckDB/parquet store
and writes a CSV/parquet file with ``full_fen`` rows for later tree generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

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
            "threads": threads,
        })
        self.conn.sql(f"ATTACH '{self.core_db}' AS core (READ_ONLY);")

    def sample_positions(
        self,
        year: int = 2023,
        n_games: int = 100,
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
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Sample chess positions from the Lichess database matching criteria.
        Ensures only one position per game is sampled.
        """
        start_ts = start_date if start_date else f"{year}-01-01"
        end_ts = end_date if end_date else f"{year + 1}-01-01"

        if verbose:
            print(f"Sampling {n_games} games from {start_ts} to {end_ts}...")

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
            ORDER BY hash(gid, {seed})
            LIMIT {n_games}
        );
        """)

        sampled_games_df = self.conn.sql("SELECT * FROM sampled_games").df()
        if verbose:
            print(f"Sampled {len(sampled_games_df)} games.")
        if sampled_games_df.empty:
            return pd.DataFrame()

        sampled_games_df["partition"] = sampled_games_df["gid"].astype(str).str[:6]
        file_groups = sampled_games_df.groupby("partition")["gid"].apply(list).reset_index()

        if verbose:
            print(
                f"Scanning moves for {len(sampled_games_df)} games "
                f"across {len(file_groups)} partitions..."
            )

        self.conn.sql("""
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

            # ``full_fen`` expression below must stay aligned with
            # :func:`compose_full_fen`.
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
        if verbose:
            print(f"Found {n_candidates} candidate positions.")
        if n_candidates == 0:
            return pd.DataFrame()

        if verbose:
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
        ORDER BY hash(gid, move_ply, {seed + 2})
        LIMIT {n_final_fens}
        """).df()

        return final_df


def _default_out(year: int, seed: int, n_fens: int) -> Path:
    return Path(DEFAULT_OUT_DIR) / f"sampled_fens_{year}_daily_seed{seed}_n{n_fens}.csv"


def _daily_windows(year: int, start_date: Optional[str], end_date: Optional[str]) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date if start_date else f"{year}-01-01").normalize()
    end = pd.Timestamp(end_date if end_date else f"{year + 1}-01-01").normalize()
    if end <= start:
        raise ValueError("end-date must be after start-date")
    days = pd.date_range(start, end, freq="D", inclusive="left")
    return [
        (day.date().isoformat(), (day + pd.Timedelta(days=1)).date().isoformat())
        for day in days
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-db", default=DEFAULT_CORE_DB)
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA)
    parser.add_argument("--memory-limit", default="40GB")
    parser.add_argument("--threads", type=int, default=8)

    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--n-games", type=int, default=100, help="Candidate games sampled per day before move filtering.")
    parser.add_argument("--n-fens", "--n", dest="n_fens", type=int, default=100, help="Final positions sampled per day.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--max-elo", type=int, default=2600)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=120)
    parser.add_argument("--min-legal-moves", type=int, default=2)
    parser.add_argument("--max-legal-moves", type=int, default=60)
    parser.add_argument("--min-pieces", type=int, default=8)
    parser.add_argument("--max-pieces", type=int, default=32)

    parser.add_argument(
        "--out",
        default=None,
        help="Output .csv or .parquet path. Defaults under DEFAULT_OUT_DIR.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out) if args.out else _default_out(args.year, args.seed, args.n_fens)

    sampler = ChessSampler(
        core_db=args.core_db,
        raw_data_dir=args.raw_data_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    frames = []
    windows = _daily_windows(args.year, args.start_date, args.end_date)
    progress = tqdm(windows, desc=f"Sampling {args.n_fens}/day", unit="day")
    for day_index, (day_start, day_end) in enumerate(progress):
        day_df = sampler.sample_positions(
            year=args.year,
            n_games=args.n_games,
            n_final_fens=args.n_fens,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            min_legal_moves=args.min_legal_moves,
            max_legal_moves=args.max_legal_moves,
            min_pieces=args.min_pieces,
            max_pieces=args.max_pieces,
            seed=args.seed + day_index,
            start_date=day_start,
            end_date=day_end,
            verbose=False,
        )
        if not day_df.empty:
            day_df.insert(0, "sample_date", day_start)
            frames.append(day_df)
        progress.set_postfix(day=day_start, rows=len(day_df))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path)
    else:
        df.to_csv(out_path, index=False)

    expected = len(windows) * args.n_fens
    print(f"Sampled {len(df)} positions (expected {expected}) -> {out_path}")
    if len(df) != expected:
        print(
            "WARNING: final row count differs from expected; at least one day "
            "had fewer eligible sampled positions than requested."
        )
    if not df.empty:
        print(f"First FEN: {df.iloc[0]['full_fen']}")


if __name__ == "__main__":
    main()
