"""Single source for root-FEN handling in the tree-generation pipeline.

``fens.txt`` — ``SELECT DISTINCT fen FROM processed_moves_nonzero`` (~110.5M
4-field FENs from the human behavioral corpus) — is **ground truth**: the full,
de-duplicated pool of positions humans actually faced. Two operations live here,
and nothing else generates or samples root FENs:

  ``build-pool`` : (re)generate ``fens.txt`` from ``personal.db``. Run rarely;
                   the pool is the canonical artifact every downstream stage
                   draws from.
  ``sample``     : draw N FENs from ``fens.txt`` for one tree-generation run,
                   normalized to 6 fields so lc0 / python-chess parse them.

Why 4-field is fine. The pool stores board / side-to-move / castling / en-passant
only. The throwaway halfmove (``0``) and fullmove (``1``) counters appended at
sample time are immaterial to shallow tree generation (the 50-move rule cannot
bite within a depth-~4 search) and are *not* how reaction times are recovered:
that join uses the 4-field FEN as its key, against ``processed_moves_nonzero``,
post-hoc. So no 6-field reconstruction or RT manifest is carried through
generation — a deduped position simply maps back to a distribution of human RTs
when an analysis needs them.

Usage (from chess_analysis/):
    python -m cts.data.process_fens build-pool
    python -m cts.data.process_fens sample --n 150000 --seed 43 \\
        --out /scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens_sample_150k.txt
"""

from __future__ import annotations

import argparse
import random

import duckdb

DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
POOL_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens.txt"
SAMPLE_OUT_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens_sample_150k.txt"


def build_pool(db_path: str, out_path: str) -> None:
    """Write every distinct human position to ``out_path``, one 4-field FEN per line."""
    con = duckdb.connect(db_path, read_only=True)
    con.execute(
        f"""
        COPY (SELECT DISTINCT fen FROM processed_moves_nonzero ORDER BY fen)
        TO '{out_path}' (FORMAT CSV, HEADER false)
        """
    )
    con.close()


def _to_six_field(fen: str) -> str:
    """Append throwaway halfmove/fullmove counters so parsers accept the FEN.

    Pool FENs are 4-field; lc0 and python-chess want 6. The two appended
    values (``0 1``) are never read back — they carry no information the
    pipeline uses (see module docstring).
    """
    fields = fen.split()
    if len(fields) == 4:
        fields += ["0", "1"]
    return " ".join(fields)


def reservoir_sample(pool_path: str, n: int, seed: int) -> list[str]:
    """Uniformly sample ``n`` FENs from ``pool_path`` in a single streaming pass.

    Algorithm R over the non-blank lines, so the 5+ GB pool is never held in
    memory — only the ``n``-element reservoir. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    count = 0
    with open(pool_path, "r", encoding="utf-8") as handle:
        for line in handle:
            fen = line.strip()
            if not fen:
                continue
            count += 1
            if len(reservoir) < n:
                reservoir.append(fen)
            else:
                j = rng.randint(0, count - 1)
                if j < n:
                    reservoir[j] = fen
    if count < n:
        raise RuntimeError(f"Pool has only {count} FENs; requested {n}.")
    return reservoir


def sample(pool_path: str, n: int, seed: int, out_path: str) -> None:
    """Sample ``n`` FENs from the pool, normalize to 6 fields, write sorted."""
    print(f"Sampling {n} FENs (seed={seed}) from {pool_path}...")
    fens = reservoir_sample(pool_path, n, seed)
    # Sort so the file — and therefore the build_tree index → FEN mapping — is
    # byte-identical for a given (n, seed), which keeps --resume stable.
    fens = sorted(_to_six_field(fen) for fen in fens)
    with open(out_path, "w", encoding="utf-8") as handle:
        for fen in fens:
            handle.write(f"{fen}\n")
    print(f"Wrote {len(fens)} FENs to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pool = sub.add_parser("build-pool", help="(re)generate the ground-truth FEN pool from personal.db")
    pool.add_argument("--db", default=DB_DEFAULT)
    pool.add_argument("--out", default=POOL_DEFAULT)

    samp = sub.add_parser("sample", help="draw N FENs from the pool for one tree-generation run")
    samp.add_argument("--pool", default=POOL_DEFAULT)
    samp.add_argument("--n", type=int, default=150_000)
    samp.add_argument("--seed", type=int, default=43)
    samp.add_argument("--out", default=SAMPLE_OUT_DEFAULT)

    args = parser.parse_args()
    if args.command == "build-pool":
        print(f"exporting distinct FENs from {args.db} -> {args.out}")
        build_pool(args.db, args.out)
        print("done")
    elif args.command == "sample":
        sample(args.pool, args.n, args.seed, args.out)


if __name__ == "__main__":
    main()
