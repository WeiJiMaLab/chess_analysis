"""
Gut-checks for top-2 WDL (multipv) vs production pipeline + SQL invariants on personal.db.

Run from repo root with .venv:
  python src/exploratory/top2_wdl_smoke_test.py
  python src/exploratory/top2_wdl_smoke_test.py --sql-only
  python src/exploratory/top2_wdl_smoke_test.py --no-engine
"""

import argparse
import os
import sys

import chess
import duckdb

# parent = src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script_engine_eval import get_e_win
from utils.helpers import EPSILON, get_stockfish_engine

PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

# (name, fen, description, search_depth or None to use --depth for all)
SMOKE_POSITIONS = [
    (
        "start",
        chess.STARTING_FEN,
        "Opening: expect a modest top-2 gap at low depth (many reasonable moves).",
        None,
    ),
    (
        "tactical_win",
        "7k/6R1/5K2/6R1/8/8/8/8 w - - 0 1",
        "Completely won position: top two lines can *both* have WDL 1.0, so the gap can be 0; "
        "this is not a bug. Compare the printed PV first moves (Rh5# vs a slower win).",
        18,
    ),
    (
        "k_p_endgame",
        "8/8/8/3k4/8/2K1P3/8/8 w - - 0 1",
        "K+p vs K: not a terminal position (avoids is_game_over() in python-chess for bare KvK).",
        None,
    ),
]


def run_engine_smoke(depth: int) -> None:
    print("\n--- Engine multipv smoke (same path as script_engine_eval: get_e_win + multipv=2) ---\n")
    engine = get_stockfish_engine(threads=1)
    with engine:
        for name, fen, note, depth_override in SMOKE_POSITIONS:
            d = depth if depth_override is None else depth_override
            board = chess.Board(fen)
            print(f"### {name}")
            print(f"    {note}")
            print(f"    FEN: {fen}")
            print(f"    search depth: {d}")
            if board.is_game_over():
                e = get_e_win(None, board)
                print(f"    game over, e_win = {e}")
                print()
                continue
            engine.configure({"Clear Hash": None})
            info_list = engine.analyse(board, chess.engine.Limit(depth=d), multipv=2)
            e0 = get_e_win(info_list[0], board) if len(info_list) > 0 else None
            e1 = get_e_win(info_list[1], board) if len(info_list) > 1 else e0
            gap = abs((e0 or 0) - (e1 or 0))
            pv0 = info_list[0].get("pv", [])
            pv1 = info_list[1].get("pv", []) if len(info_list) > 1 else []
            u0 = pv0[0].uci() if pv0 else None
            u1 = pv1[0].uci() if pv1 else None
            e0s = f"{e0:.6f}" if e0 is not None else "None"
            e1s = f"{e1:.6f}" if e1 is not None else "None"
            print(f"    e_win_best={e0s}  e_win_second={e1s}  |gap|={gap:.6f}")
            print(f"    PV1 first: {u0}  PV2 first: {u1}")
            print()


def run_sql_checks(db_path: str) -> None:
    print("\n--- SQL invariants on selected_moves_with_engine (move_time > 0) ---\n")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path} — skip SQL checks.")
        return
    conn = duckdb.connect(db_path, read_only=True)
    try:
        try:
            conn.execute("SELECT 1 FROM selected_moves_with_engine LIMIT 1").fetchone()
        except Exception as ex:
            if "selected_moves_with_engine" in str(ex):
                print("Table selected_moves_with_engine does not exist — run build_selected_moves_with_engine.py first.")
                return
            raise

        base = "selected_moves_with_engine"
        filt = "move_time > 0"

        n = conn.execute(f"SELECT count(*) FROM {base} WHERE {filt}").fetchone()[0]
        print(f"Rows (with filter): {n:,}")

        bad_forced = conn.execute(
            f"""
            SELECT count(*)
            FROM {base}
            WHERE {filt} AND n_possible_moves = 1 AND abs(top2_wdl_diff) > 1e-9
            """
        ).fetchone()[0]
        print(
            f"n_possible_moves = 1 but top2_wdl_diff > 1e-9: {bad_forced:,} "
            f"({'FAIL' if bad_forced else 'OK'})"
        )

        order_violation = conn.execute(
            f"""
            SELECT count(*)
            FROM {base}
            WHERE {filt} AND e_win_second_best > e_win_best + 1e-12
            """
        ).fetchone()[0]
        print(
            f"rows with e_win_second_best > e_win_best (naming inconsistency): {order_violation:,} "
            f"({'unexpected' if order_violation else 'OK'})"
        )

        pct = conn.execute(
            f"""
            WITH t AS (
              SELECT top2_wdl_diff, ln(move_time + {EPSILON}) AS ln_t
              FROM {base}
              WHERE {filt}
            ),
            ranked AS (
              SELECT *, ntile(100) OVER (ORDER BY top2_wdl_diff) AS pct_gap
              FROM t
            )
            SELECT
              pct_gap,
              avg(top2_wdl_diff) AS mean_gap,
              avg(ln_t) AS mean_ln_t,
              count(*) AS n
            FROM ranked
            WHERE pct_gap IN (1, 50, 100)
            GROUP BY pct_gap
            ORDER BY pct_gap
            """
        ).df()
        print("\nMean log move time at bottom / mid / top percentiles of top2_wdl_diff:")
        print(pct.to_string(index=False))
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description="Top-2 WDL smoke tests + SQL invariants")
    p.add_argument("--db", default=PERSONAL_DB, help="Path to personal.db")
    p.add_argument("--depth", type=int, default=5, help="Engine search depth (match production default)")
    p.add_argument("--sql-only", action="store_true", help="Only run DuckDB checks (no Stockfish)")
    p.add_argument("--no-engine", action="store_true", help="Skip Stockfish multipv checks")
    args = p.parse_args()

    if not (args.sql_only or args.no_engine):
        run_engine_smoke(depth=args.depth)
    run_sql_checks(args.db)


if __name__ == "__main__":
    main()
