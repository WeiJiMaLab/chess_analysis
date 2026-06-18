"""Is the negative MQ↔RT slope a difficulty confound? Segment on GSS (and legal moves).

Background. ``tree_values_analysis.py`` builds ``figures/mq_vs_rt.png`` — mean MQ (y) vs
binned log RT (x) — and finds MQ DECREASING as RT grows: longer thinks coincide with
worse moves. MQ = final_Q(played) − final_Q(best) ≤ 0 (Lc0 root value loss; 0 = the
engine-best move was played, more negative = worse). The report reads the slope as a
DIFFICULTY CONFOUND: people think long on objectively harder positions, and hard
positions get worse moves regardless of time.

This script tests that claim by conditioning on a difficulty proxy. GSS (greedy stopping
step) = the first expansion at which lc0's eventual-best move is recommended — a cost-free
"search effort to find the best move," hence a per-position difficulty index. If the
negative MQ↔RT slope is purely a GSS-difficulty artefact, it should vanish or reverse
WITHIN GSS strata, and the partial Spearman ρ(MQ, logRT | GSS) should be ≈ 0. We also test
legal moves (n_possible_moves) as an alternative difficulty axis, and the joint partial.

The figure itself is the third panel of ``figures/mq_vs_rt.png`` — a second ``Analyzer`` on the
same mq_rt table built with ``segment_column='gss'`` (so it is the identical MQ-vs-RT panel,
segmented by GSS stratum instead of ply tertile); running this module just prints the
difficulty-confound statistics. Everything reads the existing parquet caches + DuckDB:
    PYTHONPATH=human_analytics python human_analytics/mq_vs_rt_by_gss.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_CACHE = Path("/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache")
_VALS = _CACHE / "vals_human_trees_200000_7.parquet"
_ROOTMOVES = _CACHE / "rootmoves_human_trees_200000_7.parquet"
_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"


def load_played_moves() -> pd.DataFrame:
    """One row per human played move on a lc0-tree FEN, with everything we condition on:
    mq (Lc0 value loss of the move played), log_rt, gss (per-FEN difficulty), and
    legal moves (n_possible_moves). The MQ↔human join is on (fen, move_uci) — the human's
    actual UCI matched to its root-move MQ — exactly as ``mq_rt`` is built upstream."""
    vals = pd.read_parquet(_VALS)[["fen", "gss"]]
    root_moves = pd.read_parquet(_ROOTMOVES)[["fen", "move_uci", "mq"]]

    conn = duckdb.connect(_DB, read_only=True)
    conn.register("_vals", vals)
    conn.register("_root_moves", root_moves)
    df = conn.execute("""
        WITH human AS (
            SELECT m.fen, m.gid, m.move_ply, m.move_time,
                   m.n_possible_moves AS legal_moves, mv.move_uci
            FROM (SELECT DISTINCT fen FROM _root_moves) f
            JOIN processed_moves_nonzero m ON m.fen = f.fen AND m.move_time > 0
            JOIN moves mv ON mv.gid = m.gid AND mv.move_ply = m.move_ply
        )
        SELECT rm.mq, ln(h.move_time) AS log_rt, v.gss, h.legal_moves
        FROM human h
        JOIN _root_moves rm ON rm.fen = h.fen AND rm.move_uci = h.move_uci
        JOIN _vals v ON v.fen = h.fen
    """).df()
    conn.close()
    return df.dropna(subset=["mq", "log_rt", "gss", "legal_moves"]).reset_index(drop=True)


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    """Spearman ρ(x, y) controlling for ``controls``: residualize rank(x) and rank(y) on
    the control ranks by least squares, then Pearson-correlate the residuals."""
    R = df[[x, y] + controls].rank()
    A = np.c_[np.ones(len(R)), R[controls].to_numpy()]

    def resid(col: str) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, R[col].to_numpy(), rcond=None)
        return R[col].to_numpy() - A @ beta

    return float(np.corrcoef(resid(x), resid(y))[0, 1])


def gss_strata(df: pd.DataFrame) -> pd.Series:
    """Label each move by a GSS difficulty tertile. GSS is heavily massed at 0–1 (the
    best move is found almost immediately on easy positions), so equal-frequency tertiles
    give: Easy (best move found fast), Medium, Hard (deep search needed)."""
    edges = df["gss"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    edges = np.unique(edges)  # collapse ties if the mass forces identical cut points
    labels = ["Easy (low GSS)", "Medium GSS", "Hard (high GSS)"][: len(edges) - 1]
    return pd.cut(df["gss"], bins=edges, labels=labels, include_lowest=True)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== MQ↔RT, conditioning on difficulty (n = {len(df):,} played moves) ===")
    print("    MQ ≤ 0 (more negative = worse); a NEGATIVE ρ(MQ, logRT) = worse moves on longer thinks.\n")

    overall = df["mq"].rank().corr(df["log_rt"].rank())
    print(f"  overall  ρ(MQ, logRT)              = {overall:+.4f}")

    df = df.assign(stratum=gss_strata(df))
    print("\n  within-GSS-stratum ρ(MQ, logRT):")
    for name, sub in df.groupby("stratum", observed=True):
        rho = sub["mq"].rank().corr(sub["log_rt"].rank())
        print(f"    {name:<18s} GSS {int(sub['gss'].min()):>2d}–{int(sub['gss'].max()):<2d} "
              f"n={len(sub):>7,}  ρ={rho:+.4f}")

    print("\n  partial Spearman ρ(MQ, logRT | …):")
    print(f"    | GSS                = {partial_spearman(df, 'mq', 'log_rt', ['gss']):+.4f}")
    print(f"    | legal moves        = {partial_spearman(df, 'mq', 'log_rt', ['legal_moves']):+.4f}")
    print(f"    | GSS + legal moves  = {partial_spearman(df, 'mq', 'log_rt', ['gss', 'legal_moves']):+.4f}")

    print("\n  [context] is GSS even a difficulty proxy here? Spearman ρ:")
    print(f"    ρ(GSS, MQ)        = {df['gss'].rank().corr(df['mq'].rank()):+.4f}   (harder → worse moves?)")
    print(f"    ρ(GSS, logRT)     = {df['gss'].rank().corr(df['log_rt'].rank()):+.4f}   (harder → longer think?)")
    print(f"    ρ(legal moves, MQ)   = {df['legal_moves'].rank().corr(df['mq'].rank()):+.4f}")
    print(f"    ρ(legal moves, logRT)= {df['legal_moves'].rank().corr(df['log_rt'].rank()):+.4f}")
    print(f"    ρ(GSS, legal moves)  = {df['gss'].rank().corr(df['legal_moves'].rank()):+.4f}")


def main() -> None:
    # Prints the difficulty-confound stats (within-stratum + partial Spearman). The figure
    # itself is the third panel of mq_vs_rt.png — a GSS-segmented Analyzer in
    # tree_values_analysis.py — there is no standalone figure.
    report(load_played_moves())


if __name__ == "__main__":
    main()
