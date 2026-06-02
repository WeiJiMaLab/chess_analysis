"""
End-to-end tests for preprocess.get_games → preprocess_game_shard (2-way split) → merge_game_shards.

Uses synthetic DuckDB ``games`` sources and parquet shards (no cluster paths).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "slurm" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore

if duckdb is not None:
    from preprocess import get_games, merge_game_shards, preprocess_game_shard


def _sql_str(path: str) -> str:
    return path.replace("'", "''")


def _write_lichess_db(path: str, rows: list[tuple]) -> None:
    """rows: (gid, utc_datetime str, initial_clock, clock_increment, white_elo, black_elo)."""
    if os.path.exists(path):
        os.unlink(path)
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE games (
            gid BIGINT,
            utc_datetime TIMESTAMP,
            initial_clock INTEGER,
            clock_increment INTEGER,
            white_elo INTEGER,
            black_elo INTEGER
        )
        """
    )
    con.executemany(
        "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?)",
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
    )
    con.close()


def _write_moves_parquet(out_path: str, gid: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    con = duckdb.connect()
    op = _sql_str(out_path)
    con.execute(
        f"""
        COPY (
            SELECT
                {gid}::BIGINT AS gid,
                1 AS move_ply,
                true AS player_white,
                300.0 AS player_clock_time,
                300.0 AS opponent_clock_time,
                2.5 AS move_time,
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR' AS board_position,
                20 AS n_possible_moves,
                'KQkq' AS castling_rights,
                '-' AS en_passant_targets,
                0 AS halfmove_clock
        ) TO '{op}' (FORMAT PARQUET)
        """
    )
    con.close()


@unittest.skipIf(duckdb is None, "duckdb not installed")
class TestPreprocessShardPipeline(unittest.TestCase):
    def test_two_shards_then_clean_second_run_no_contamination(self) -> None:
        """
        Run 1: four (partition, segment) pairs split across job_id 0 and 1 (total_jobs=2), merge → ``moves``.
        Delete shard staging dir entirely, rebuild ``games`` + parquets for run 2 (two new pairs).
        Run 2: merge must contain only run-2 gids (no parquet left from run 1 because that directory was removed).
        """
        base = tempfile.mkdtemp(prefix="chess_preprocess_pipeline_")
        try:
            lichess1 = os.path.join(base, "lichess_run1.duckdb")
            lichess2 = os.path.join(base, "lichess_run2.duckdb")
            personal_db = os.path.join(base, "personal.duckdb")
            moves_root1 = os.path.join(base, "moves_root_run1")
            moves_root2 = os.path.join(base, "moves_root_run2")
            shard_tmp_run1 = os.path.join(base, "shards_run1")
            shard_tmp_run2 = os.path.join(base, "shards_run2")

            run1_gids = [
                202310001000001,
                202310002000002,
                202311001000003,
                202311002000004,
            ]
            lichess_rows1 = [
                (g, "2023-10-05 12:00:00", 600, 0, 2100, 2100) for g in run1_gids
            ]
            _write_lichess_db(lichess1, lichess_rows1)
            for g in run1_gids:
                pstr = str(g)[:6]
                sstr = str(g)[6:9]
                _write_moves_parquet(
                    os.path.join(moves_root1, f"partition={pstr}", f"{sstr}-moves.parquet"),
                    g,
                )

            get_games(
                personal_db,
                lichess1,
                start_date="2023-10-01",
                end_date="2024-01-01",
                work_dir=shard_tmp_run1,
                initial_clock=600,
                clock_increment=0,
                min_elo=2000,
                threads=2,
                memory_limit="512MB",
            )

            preprocess_game_shard(
                personal_db,
                job_id=0,
                total_jobs=2,
                staging_dir=shard_tmp_run1,
                moves_root=moves_root1,
                threads=2,
                memory_limit="512MB",
            )
            preprocess_game_shard(
                personal_db,
                job_id=1,
                total_jobs=2,
                staging_dir=shard_tmp_run1,
                moves_root=moves_root1,
                threads=2,
                memory_limit="512MB",
            )
            merge_game_shards(personal_db, shard_tmp_run1, threads=2, memory_limit="512MB")

            con = duckdb.connect(personal_db)
            gids1 = {r[0] for r in con.execute("SELECT DISTINCT gid FROM moves").fetchall()}
            con.close()
            self.assertEqual(gids1, set(run1_gids))
            parquets_run1 = sorted(Path(shard_tmp_run1).glob("selected_moves_*.parquet"))
            self.assertEqual(len(parquets_run1), 4)

            shutil.rmtree(shard_tmp_run1)
            self.assertFalse(os.path.exists(shard_tmp_run1))

            run2_gids = [202312001000099, 202312002000088]
            lichess_rows2 = [
                (g, "2023-12-05 12:00:00", 600, 0, 2100, 2100) for g in run2_gids
            ]
            _write_lichess_db(lichess2, lichess_rows2)
            for g in run2_gids:
                pstr = str(g)[:6]
                sstr = str(g)[6:9]
                _write_moves_parquet(
                    os.path.join(moves_root2, f"partition={pstr}", f"{sstr}-moves.parquet"),
                    g,
                )

            get_games(
                personal_db,
                lichess2,
                start_date="2023-10-01",
                end_date="2024-01-01",
                work_dir=shard_tmp_run2,
                initial_clock=600,
                clock_increment=0,
                min_elo=2000,
                threads=2,
                memory_limit="512MB",
            )

            preprocess_game_shard(
                personal_db,
                job_id=0,
                total_jobs=2,
                staging_dir=shard_tmp_run2,
                moves_root=moves_root2,
                threads=2,
                memory_limit="512MB",
            )
            preprocess_game_shard(
                personal_db,
                job_id=1,
                total_jobs=2,
                staging_dir=shard_tmp_run2,
                moves_root=moves_root2,
                threads=2,
                memory_limit="512MB",
            )
            merge_game_shards(personal_db, shard_tmp_run2, threads=2, memory_limit="512MB")

            con = duckdb.connect(personal_db)
            gids2 = {r[0] for r in con.execute("SELECT DISTINCT gid FROM moves").fetchall()}
            n_moves = con.execute("SELECT count(*) FROM moves").fetchone()[0]
            con.close()

            self.assertEqual(gids2, set(run2_gids))
            self.assertEqual(n_moves, 2)
            self.assertTrue(gids2.isdisjoint(gids1))

            parquets_run2 = sorted(Path(shard_tmp_run2).glob("selected_moves_*.parquet"))
            self.assertEqual(len(parquets_run2), 2)
        finally:
            shutil.rmtree(base, ignore_errors=True)
            self.assertFalse(os.path.exists(base), "temp pipeline directory should be deleted")


if __name__ == "__main__":
    unittest.main()
