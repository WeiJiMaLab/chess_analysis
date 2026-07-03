"""
Preprocess Lichess data for response-time analysis.
Includes selection of games, extraction of moves, and feature engineering.

Contract (do not subvert with optional alternate temp paths):
    DuckDB ``temp_directory`` always equals the directory named by ``work_dir`` or ``staging_dir``.
    Shard parquet outputs go under ``staging_dir``. Merge reads ``staging_dir`` only and replaces
    ``moves``. :func:`process_moves` builds ``processed_moves`` and ``processed_moves_nonzero`` from ``moves``;
    ``preprocess.sh`` runs merge then ``process_moves``.

The CLI (:func:`main`) resolves every parameter from the active config via
:func:`_load_preprocess_config` and passes them in explicitly — the yaml is the single source
of ``threads`` / ``memory_limit`` / ``moves_root`` / the selection filters and the rest.
``threads`` / ``memory_limit`` / ``total_shards`` still honor ``DUCKDB_THREADS`` /
``DUCKDB_MEMORY_LIMIT`` / ``PREPROCESS_TOTAL_SHARDS`` overrides. The ``shard`` subcommand reads
``job_id`` from ``SLURM_ARRAY_TASK_ID`` only.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import duckdb
from tqdm import tqdm

from analysis.utils.helpers import connect, load_config_section, sql_str


def get_games(
    personal_db: str,
    lichess_db: str,
    start_date: str,
    end_date: str,
    work_dir: str,
    initial_clock: int,
    clock_increment: int,
    min_elo: int,
    threads: int,
    memory_limit: str,
):
    """
    Build table ``games`` in ``personal_db`` from ``core.games`` using fixed selection filters.

    DuckDB temporary files use ``work_dir`` (same constraint as :func:`preprocess_game_shard`).
    """
    print(f"Fetching games and saving to database:\n\t{lichess_db} --> \n\t{personal_db}")
    conn = connect(personal_db, work_dir, threads, memory_limit, read_only=False)
    conn.sql(f"""ATTACH '{lichess_db}' AS core (READ_ONLY);""")

    s_start = sql_str(start_date)
    s_end = sql_str(end_date)

    start_yyyymm = int(start_date.replace("-", "")[:6])
    end_yyyymm = int(end_date.replace("-", "")[:6]) + 1
    gid_bounds = f"AND gid >= {start_yyyymm}000000000 AND gid < {end_yyyymm}000000000"

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE games AS
        SELECT
            gid,
            utc_datetime,
            substr(cast(gid AS VARCHAR), 1, 6) AS partition,
            substr(cast(gid AS VARCHAR), 7, 3) AS segment,
            initial_clock,
            clock_increment
        FROM core.games
        WHERE utc_datetime >= TIMESTAMP '{s_start}'
          AND utc_datetime < TIMESTAMP '{s_end}'
          {gid_bounds}
          AND initial_clock = {int(initial_clock)}
          AND clock_increment = {int(clock_increment)}
          AND white_elo >= {int(min_elo)}
          AND black_elo >= {int(min_elo)}
        """
    )

    count = conn.execute("SELECT count(*) FROM games").fetchone()[0]
    conn.close()
    print(
        "✅ games rebuilt | "
        f"window=[{start_date}, {end_date}) | tc={initial_clock}+{clock_increment} | "
        f"min_elo={min_elo} | n={count:,}"
    )


def preprocess_game_shard(
    personal_db: str,
    job_id: int,
    total_jobs: int,
    staging_dir: str,
    moves_root: str,
    threads: int,
    memory_limit: str,
):
    """
    Shard moves parquet by ``(partition, segment)``, join ``games``, then drop bad games.

    Writes ``selected_moves_<partition>_<segment>.parquet`` under ``staging_dir``.
    DuckDB spill uses ``staging_dir`` (same folder). Pass that same path to :func:`merge_game_shards`.
    """
    if total_jobs < 1 or job_id < 0:
        raise ValueError("need total_jobs >= 1 and job_id >= 0")

    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    conn = connect(personal_db, staging_dir, threads, memory_limit, read_only=True)

    pairs = conn.execute(
        """
        SELECT DISTINCT partition, segment
        FROM games
        ORDER BY partition, segment
        """
    ).fetchall()

    assigned = [
        (str(p), str(s))
        for p, s in pairs[job_id::total_jobs]
    ]
    print(f"preprocess_game_shard job={job_id}/{total_jobs} | shard_pairs={len(pairs)} | assigned={len(assigned)}")

    for partition, segment in tqdm(assigned, desc=f"moves {job_id}"):
        pq_path = os.path.join(moves_root, f"partition={partition}", f"{segment}-moves.parquet")
        assert os.path.isfile(pq_path), f"missing: {pq_path}"
        out = os.path.join(staging_dir, f"selected_moves_{partition}_{segment}.parquet")
        conn.execute(
            f"""
            COPY (
                WITH joined AS (
                    SELECT
                        pq.*,
                        CAST(g.initial_clock AS DOUBLE) AS initial_clock,
                        CAST(g.clock_increment AS BIGINT) AS clock_increment
                    FROM read_parquet('{sql_str(pq_path)}') pq
                    INNER JOIN games g ON pq.gid = g.gid
                    WHERE g.partition = '{sql_str(partition)}' AND g.segment = '{sql_str(segment)}'
                ),
                -- Drop entire games if any move has negative move_time.
                exclude_neg_gid AS (
                    SELECT DISTINCT gid
                    FROM joined
                    WHERE move_time < 0
                ),
                -- Drop entire games if the player chooses to berserk (half the clock time for points)
                exclude_berserk_gid AS (
                    SELECT DISTINCT gid
                    FROM joined
                    -- First full-move window where halved clock shows after berserk.
                    WHERE move_ply BETWEEN 3 AND 4
                      AND player_clock_time >= (initial_clock / 2.0) - 5.0
                      AND player_clock_time <= (initial_clock / 2.0) + 0.1
                ),
                -- Drop entire games if player chooses to grant more time to the opponent (only valid in increment-0 time controls)
                exclude_grant_more_time_gid AS (
                    SELECT DISTINCT gid
                    FROM (
                        SELECT
                            gid,
                            player_clock_time,
                            clock_increment,
                            lag(player_clock_time) OVER (
                                PARTITION BY gid, player_white ORDER BY move_ply
                            ) AS prev_clock
                        FROM joined
                    )
                    WHERE clock_increment = 0
                      AND prev_clock IS NOT NULL
                      AND player_clock_time > prev_clock
                )
                SELECT j.* EXCLUDE (initial_clock, clock_increment)
                FROM joined j
                WHERE j.gid NOT IN (SELECT gid FROM exclude_neg_gid)
                  AND j.gid NOT IN (SELECT gid FROM exclude_berserk_gid)
                  AND j.gid NOT IN (SELECT gid FROM exclude_grant_more_time_gid)
            ) TO '{sql_str(out)}' (FORMAT PARQUET)
            """
        )
        nm = conn.execute(f"SELECT count(*) FROM read_parquet('{sql_str(out)}')").fetchone()[0]
        print(f"\t{partition}/{segment} | {nm:,} moves | → {out}")
    conn.close()


def process_moves(conn: duckdb.DuckDBPyConnection, work_dir: str,
                  table: str = "processed_moves") -> None:
    """
    Build ``<table>`` (all board features) and ``<table>_nonzero`` (``move_time > 0``)
    from table ``moves``. Defaults to ``processed_moves``; pass another ``table``
    (e.g. ``full_rebuild``) to build a side-by-side candidate without touching the
    official tables.

    Bad games are already removed at shard time. Board features are first-class
    columns so downstream analysis reads them uniformly. Material / in_check /
    prev_move_was_capture are pure regex/window SQL; n_captures_avail /
    n_checks_avail need python-chess move enumeration
    (:func:`_featurize_captures_checks`, scratch in ``work_dir``). All are added
    before ``<table>_nonzero`` is derived.
    """
    # Weighted material (P/N/B/R/Q = 1/3/3/5/9, incl pawns, kings excluded) counts
    # piece letters in the placement string.
    def _piece_count(piece):
        return f"len(regexp_extract_all(replace(board_position, '/', ''), '{piece}'))"
    white_mat = f"(1*{_piece_count('P')} + 3*{_piece_count('N')} + 3*{_piece_count('B')} + 5*{_piece_count('R')} + 9*{_piece_count('Q')})"
    black_mat = f"(1*{_piece_count('p')} + 3*{_piece_count('n')} + 3*{_piece_count('b')} + 5*{_piece_count('r')} + 9*{_piece_count('q')})"
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            *,
            CASE WHEN player_white THEN {white_mat} ELSE {black_mat} END AS self_material,
            CASE WHEN player_white THEN {white_mat} - {black_mat}
                 ELSE {black_mat} - {white_mat} END AS material_imbalance,
            -- a capture strictly lowers the inc-pawns piece count (en-passant included)
            (n_pieces_on_board_inc_pawns
             < lag(n_pieces_on_board_inc_pawns) OVER (PARTITION BY gid ORDER BY move_ply))
                AS prev_move_was_capture
        FROM (
            SELECT
                gid,
                move_ply,
                board_position,
                player_white,
                player_clock_time,
                opponent_clock_time,
                n_possible_moves,
                move_time,
                player_in_check AS in_check,
                len(regexp_extract_all(replace(board_position, '/', ''), '[a-zA-Z]')) AS n_pieces_on_board_inc_pawns,
                board_position || ' ' ||
                CASE WHEN player_white THEN 'w' ELSE 'b' END || ' ' ||
                COALESCE(castling_rights, '-') || ' ' ||
                COALESCE(en_passant_targets, '-') AS fen
            FROM moves
        ) AS _base
        """
    )
    _featurize_captures_checks(conn, work_dir, table)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {table}_nonzero AS
        SELECT * FROM {table} WHERE move_time > 0
        """
    )
    n_all = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    n_nonzero = conn.execute(f"SELECT count(*) FROM {table}_nonzero").fetchone()[0]
    print(f"✅ {table}: {n_all:,} rows | {table}_nonzero: {n_nonzero:,} rows")


def calc_captures_checks(fen: str) -> tuple[int, int]:
    """(#legal captures, #legal checks) for one FEN — needs move enumeration."""
    import chess
    board = chess.Board(fen)
    caps = checks = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            caps += 1
        if board.gives_check(move):
            checks += 1
    return caps, checks


def _featurize_captures_checks(conn: duckdb.DuckDBPyConnection, work_dir: str,
                               table: str = "processed_moves") -> None:
    """Add n_captures_avail / n_checks_avail columns to ``table`` via python-chess
    over its DISTINCT FENs.

    Dumps the distinct FENs to parquet, featurizes them a row group at a time
    across the node's cores, joins the counts back. The pool uses ``spawn``: the
    open DuckDB connection has live background threads, and forking under them
    deadlocks the workers.
    """
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(work_dir, exist_ok=True)
    fens_path = os.path.join(work_dir, f"distinct_fens_{table}.parquet")
    counts_path = os.path.join(work_dir, f"captures_checks_{table}.parquet")
    conn.execute(f"COPY (SELECT DISTINCT fen FROM {table}) TO '{sql_str(fens_path)}' (FORMAT PARQUET)")

    reader = pq.ParquetFile(fens_path)
    schema = pa.schema([("fen", pa.string()),
                        ("n_captures_avail", pa.int32()), ("n_checks_avail", pa.int32())])
    # Cap the pool at the SLURM allocation (else it defaults to the node's full core
    # count, oversubscribing cores and blowing the memory ceiling).
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or None
    with pq.ParquetWriter(counts_path, schema) as writer, mp.get_context("spawn").Pool(n_workers) as pool:
        for rg in range(reader.num_row_groups):
            fens = reader.read_row_group(rg, columns=["fen"])["fen"].to_pylist()
            if not fens:
                continue
            caps, checks = zip(*pool.map(calc_captures_checks, fens, chunksize=4000))
            writer.write_table(pa.table({"fen": fens,
                                         "n_captures_avail": pa.array(caps, pa.int32()),
                                         "n_checks_avail": pa.array(checks, pa.int32())}))

    conn.execute(
        f"CREATE OR REPLACE TABLE {table} AS "
        f"SELECT p.*, c.n_captures_avail, c.n_checks_avail "
        f"FROM {table} p LEFT JOIN read_parquet('{sql_str(counts_path)}') c USING (fen)"
    )
    n = conn.execute(f"SELECT count(*) FROM read_parquet('{sql_str(counts_path)}')").fetchone()[0]
    print(f"✅ featurized captures/checks over {n:,} distinct FENs → {table}", flush=True)


def merge_game_shards(
    personal_db: str,
    staging_dir: str,
    threads: int,
    memory_limit: str,
):
    """
    Load ``selected_moves_*.parquet`` from ``staging_dir`` into ``moves``.

    DuckDB spill uses ``staging_dir`` (same folder as the parquet glob).
    Run :func:`process_moves` afterward to build ``processed_moves`` / ``processed_moves_nonzero``.
    """
    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    pattern = os.path.join(staging_dir, "selected_moves_*.parquet")

    ts = datetime.now().astimezone().replace(microsecond=0).isoformat()
    print(f"\n{'=' * 72}\n merge_game_shards started {ts}\n read_parquet → {personal_db}.moves\n{'=' * 72}\n", flush=True)
    print(f"merge_game_shards: read_parquet('{pattern}') → {personal_db}.moves", flush=True)
    conn = connect(personal_db, staging_dir, threads, memory_limit, read_only=False)
    pat = sql_str(pattern)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE moves AS
        SELECT * FROM read_parquet('{pat}')
        """
    )
    n = conn.execute("SELECT count(*) FROM moves").fetchone()[0]
    print(f"✅ merge_game_shards finished — table moves replaced ({n:,} rows).")
    conn.close()


def _load_preprocess_config() -> dict:
    """Resolve the ``preprocess`` config section (shared loader; ``personal_db`` ==
    ``human_analysis.selected_db_default``), then apply the DUCKDB_THREADS /
    DUCKDB_MEMORY_LIMIT / PREPROCESS_TOTAL_SHARDS env overrides."""
    cfg = load_config_section("preprocess")
    cfg["threads"] = int(os.environ.get("DUCKDB_THREADS", cfg["threads"]))
    cfg["memory_limit"] = os.environ.get("DUCKDB_MEMORY_LIMIT", cfg["memory_limit"])
    cfg["total_shards"] = int(os.environ.get("PREPROCESS_TOTAL_SHARDS", cfg["total_shards"]))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to the run config (else $CONFIG or the default).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("get_games", help="Build games table from Lichess core DB.")
    sub.add_parser("shard", help="One array stride; uses config staging_dir, SLURM_ARRAY_TASK_ID, PREPROCESS_TOTAL_SHARDS.")
    sub.add_parser("merge", help="Parquet shards → table moves only.")
    p_process = sub.add_parser(
        "process_moves",
        help="table moves → <table>[_nonzero] board features (default processed_moves).",
    )
    p_process.add_argument("--table", default="processed_moves",
                           help="output base table (e.g. full_rebuild to build a candidate alongside the official tables).")

    args = parser.parse_args()
    if args.config:
        os.environ["CONFIG"] = args.config
    config = _load_preprocess_config()

    if args.cmd == "get_games":
        get_games(
            personal_db=config["personal_db"],
            lichess_db=config["lichess_db"],
            work_dir=config["work_dir"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            initial_clock=config["initial_clock"],
            clock_increment=config["clock_increment"],
            min_elo=config["min_elo"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )
    elif args.cmd == "shard":
        job_raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if job_raw is None:
            raise SystemExit("shard requires SLURM_ARRAY_TASK_ID (submit as a Slurm array task).")
        preprocess_game_shard(
            personal_db=config["personal_db"],
            job_id=int(job_raw),
            total_jobs=config["total_shards"],
            staging_dir=config["staging_dir"],
            moves_root=config["moves_root"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )
    elif args.cmd == "merge":
        merge_game_shards(
            personal_db=config["personal_db"],
            staging_dir=config["staging_dir"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )
    elif args.cmd == "process_moves":
        conn = connect(config["personal_db"], config["work_dir"],
                       config["threads"], config["memory_limit"], read_only=False)
        process_moves(conn, config["work_dir"], args.table)
        conn.close()


if __name__ == "__main__":
    main()