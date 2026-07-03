"""Filter + sample: the single run-specific step downstream of preprocess.

Both outputs come from applying the run's ply window once:
  1. filtered_moves: ``processed_moves_nonzero`` windowed to [min_ply, max_ply] with
     ``game_fraction`` added — the canonical table board/engine read.
  2. tree-root FENs: a uniform sample of N distinct FENs from filtered_moves, one per
     line → the treegen input file (roots for the normative branch).

game_fraction's denominator is the TRUE game length — max(move_ply) over the
UNWINDOWED processed_moves — so games longer than the window aren't all pinned to
the same denominator.

    python -m analysis.filter_and_sample
"""

import argparse
import os


def build_filtered_moves(conn, processed: str, nonzero: str, filtered: str,
                         min_ply: int, max_ply: int) -> None:
    conn.execute(f"""
        CREATE OR REPLACE TABLE {filtered} AS
        SELECT w.*, w.move_ply * 1.0 / g.game_len AS game_fraction
        FROM (SELECT * FROM {nonzero} WHERE move_ply BETWEEN {min_ply} AND {max_ply}) w
        JOIN (SELECT gid, max(move_ply) AS game_len FROM {processed} GROUP BY gid) g USING (gid)
    """)
    n = conn.execute(f"SELECT count(*) FROM {filtered}").fetchone()[0]
    print(f"✅ {filtered}: {n:,} windowed move-instances (ply {min_ply}-{max_ply})", flush=True)


def sample_tree_fens(conn, filtered: str, out_path: str, n: int, seed: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rows = conn.execute(
        f"SELECT fen FROM (SELECT DISTINCT fen FROM {filtered}) "
        f"USING SAMPLE {n} ROWS (reservoir, {seed})"
    ).fetchall()
    with open(out_path, "w") as f:
        f.write("".join(f"{r[0]}\n" for r in rows))
    print(f"✅ sampled {len(rows):,} tree-root FENs → {out_path}", flush=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    from analysis.utils.helpers import CONFIG, connect
    db = CONFIG["selected_db_default"]
    work_dir = os.path.join(os.path.dirname(os.path.abspath(db)), "tmp")
    conn = connect(db, work_dir, threads=int(os.environ.get("DUCKDB_THREADS", 16)),
                   memory_limit=os.environ.get("DUCKDB_MEMORY_LIMIT", "40GB"), read_only=False)
    build_filtered_moves(conn, CONFIG["table_processed_moves"], CONFIG["table_processed_moves_nonzero"],
                         CONFIG["table_filtered"], int(CONFIG["min_ply"]), int(CONFIG["max_ply"]))
    sample_tree_fens(conn, CONFIG["table_filtered"], CONFIG["fens_sample"],
                     int(CONFIG["tree_fens_n"]), int(CONFIG["tree_fens_seed"]))
    conn.close()


if __name__ == "__main__":
    main()
