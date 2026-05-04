"""Write gid_bounds vs datetime-only diagnostics to a file (stdout may be swallowed)."""
from __future__ import annotations

import argparse
import os

import duckdb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--lichess-db", default="/scratch/gpfs/GRIFFITHS/chess-db/lichess.db")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def log(msg: str) -> None:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    open(args.out, "w", encoding="utf-8").close()

    c = duckdb.connect(args.lichess_db, read_only=True)
    base = "initial_clock = 600 AND clock_increment = 0 AND white_elo >= 2000 AND black_elo >= 2000"

    def gid_bounds(start_date: str, end_date: str) -> str:
        start_yyyymm = int(start_date.replace("-", "")[:6])
        end_yyyymm = int(end_date.replace("-", "")[:6]) + 1
        return f"AND gid >= {start_yyyymm}000000000 AND gid < {end_yyyymm}000000000"

    scenarios = [
        ("prod_like", "2023-10-01", "2023-12-31"),
        ("include_dec31", "2023-10-01", "2024-01-01"),
        ("late_dec", "2023-12-28", "2023-12-31"),
        ("late_dec_incl31", "2023-12-28", "2024-01-01"),
        ("oct_start", "2023-09-28", "2023-10-05"),
    ]

    for name, sd, ed in scenarios:
        gb = gid_bounds(sd, ed)
        n_dt = c.execute(
            f"""
            SELECT count(*) FROM games
            WHERE utc_datetime >= TIMESTAMP '{sd}'
              AND utc_datetime < TIMESTAMP '{ed}'
              AND {base}
            """
        ).fetchone()[0]
        n_both = c.execute(
            f"""
            SELECT count(*) FROM games
            WHERE utc_datetime >= TIMESTAMP '{sd}'
              AND utc_datetime < TIMESTAMP '{ed}'
              {gb}
              AND {base}
            """
        ).fetchone()[0]
        log(f"[{name}] [{sd}, {ed})")
        log(f"  datetime only: {n_dt:,}")
        log(f"  datetime+gid:  {n_both:,}")
        log(f"  dropped by gid_bounds only: {n_dt - n_both:,}")

    # Gid vs calendar month mismatch in Dec 2023
    nm = c.execute(
        f"""
        SELECT count(*) FROM games
        WHERE utc_datetime >= TIMESTAMP '2023-12-01'
          AND utc_datetime < TIMESTAMP '2024-01-01'
          AND {base}
          AND cast(substr(cast(gid AS VARCHAR), 1, 6) AS INTEGER)
              <> (year(utc_datetime) * 100 + month(utc_datetime))
        """
    ).fetchone()[0]
    log(f"\nGames Dec 2023 UTC window where gid YYYYMM != year(utc)*100+month(utc): {nm:,}")
    if nm > 0:
        rows = c.execute(
            f"""
            SELECT gid, utc_datetime, substr(cast(gid AS VARCHAR), 1, 6) AS g6,
                   year(utc_datetime)*100+month(utc_datetime) AS ym
            FROM games
            WHERE utc_datetime >= TIMESTAMP '2023-12-01'
              AND utc_datetime < TIMESTAMP '2024-01-01'
              AND {base}
              AND cast(substr(cast(gid AS VARCHAR), 1, 6) AS INTEGER)
                  <> (year(utc_datetime) * 100 + month(utc_datetime))
            ORDER BY utc_datetime
            LIMIT 25
            """
        ).fetchall()
        for r in rows:
            log(f"  {r}")

    c.close()
    log("\nDone.")


if __name__ == "__main__":
    main()
