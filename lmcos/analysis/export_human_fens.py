"""
Export human behavioral positions for Lc0 tree generation (Analysis 1).

Samples moves from personal.db with Russek filters, writes:
  - newline-separated 6-field FENs (for build_tree.py)
  - parquet manifest (gid, move_ply, move_time, index, full_fen, …)

DuckDB applies USING SAMPLE before WHERE, so sampling must be outside the
filtered subquery.

Usage:
    python analysis/export_human_fens.py --n 1000 --seed 42 \\
        --fens-out /scratch/.../human_fens_1k.txt \\
        --manifest-out /scratch/.../human_fens_1k_manifest.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import duckdb
import pandas as pd

_DB_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_MIN_PLY = 15
_MAX_PLY = 75
_MIN_OPP_CLOCK = 60


def _sample_moves(db_path: str, n: int, seed: int) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT
            gid,
            move_ply,
            move_time,
            n_possible_moves,
            n_self_pieces_exc_pawns,
            fen_4field,
            full_fen
        FROM (
            SELECT
                pm.gid,
                pm.move_ply,
                pm.move_time,
                pm.n_possible_moves,
                pm.n_self_pieces_exc_pawns,
                pm.fen AS fen_4field,
                (
                    m.board_position || ' ' ||
                    CASE WHEN pm.player_white THEN 'w' ELSE 'b' END || ' ' ||
                    COALESCE(NULLIF(m.castling_rights, ''), '-') || ' ' ||
                    COALESCE(NULLIF(m.en_passant_targets, ''), '-') || ' ' ||
                    CAST(m.halfmove_clock AS VARCHAR) || ' ' ||
                    CAST(CAST(1 + FLOOR((pm.move_ply - 1) / 2.0) AS BIGINT) AS VARCHAR)
                ) AS full_fen
            FROM processed_moves_nonzero pm
            JOIN moves m ON pm.gid = m.gid AND pm.move_ply = m.move_ply
            WHERE pm.move_ply BETWEEN {_MIN_PLY} AND {_MAX_PLY}
              AND pm.opponent_clock_time >= {_MIN_OPP_CLOCK}
              AND pm.move_time > 0
              AND m.board_position IS NOT NULL
              AND m.board_position <> ''
        ) filtered
        USING SAMPLE {n} ROWS (RESERVOIR, {seed})
    """).df()
    conn.close()
    return df


def _validate_fens(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = []
    for fen in df["full_fen"]:
        try:
            valid_mask.append(chess.Board(fen).is_valid())
        except ValueError:
            valid_mask.append(False)
    out = df.loc[valid_mask].reset_index(drop=True)
    dropped = len(df) - len(out)
    if dropped:
        print(f"Dropped {dropped} invalid FEN(s) after chess.Board validation.")
    return out


def export_human_fens(
    n: int,
    seed: int,
    fens_out: Path,
    manifest_out: Path,
    db_path: str = _DB_PATH,
) -> pd.DataFrame:
    print(f"Sampling {n} moves (seed={seed}) from {db_path}...")
    df = _sample_moves(db_path, n, seed)
    print(f"Sampled {len(df)} rows from DuckDB.")
    if len(df) != n:
        raise RuntimeError(f"Expected {n} rows, got {len(df)}")

    df = _validate_fens(df)
    if df.empty:
        raise RuntimeError("No valid FENs after validation.")
    if len(df) < n:
        print(f"Warning: only {len(df)} valid FENs remain (requested {n}).")

    df = df.copy()
    df["index"] = range(len(df))

    fens_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with fens_out.open("w", encoding="utf-8") as handle:
        for fen in df["full_fen"]:
            handle.write(f"{fen}\n")

    df.to_parquet(manifest_out, index=False)
    print(f"Wrote {len(df)} FENs to {fens_out}")
    print(f"Wrote manifest to {manifest_out}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", default=_DB_PATH)
    parser.add_argument(
        "--fens-out",
        default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k.txt",
    )
    parser.add_argument(
        "--manifest-out",
        default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k_manifest.parquet",
    )
    args = parser.parse_args()
    export_human_fens(
        n=args.n,
        seed=args.seed,
        fens_out=Path(args.fens_out),
        manifest_out=Path(args.manifest_out),
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
