"""Diagnostic: the Gain≈1.0 dip in figures/gain_vs_rt.png.

In gain_vs_rt.png mean human RT rises with Gain across the range, then COLLAPSES at
the top bin (Gain≈1.0) with a conspicuously TIGHT CI — strongest in late plies. This
script tests whether that dip is a real cognitive effect or an artifact of (a) a
value-head/terminal point-mass at Gain≈1.0 and (b) the tie-safe top-bin lumping that
mass together.

Gain ("voc") = final_Q(deep best) − final_Q(1-ply value-head best), both root
side-to-move Qs ∈ [−1,+1], so Gain ∈ [0,2]. The 1-ply best is a child's −value backup;
the deep best is argmax(oracle_final_root_q_values).

Reads only the parquet caches + personal.db (no slurm, no bulk .pt reload); a small
sample of .pt trees is inspected to read the final_Q vector behind the Gain≈1.0 mass.

    PYTHONPATH=human_analytics python human_analytics/gain_dip_diagnostic.py
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_VALS = "/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache/vals_human_trees_200000_7.parquet"
_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_TREES = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_FIGS = Path(__file__).resolve().parent.parent / "figures"


def joined_moves() -> pd.DataFrame:
    """One row per human move with a tree: Gain, RT, ply, legal_moves, own_material.

    Legal-moves/material come from the SAME processed_moves_nonzero row (gid, move_ply),
    not a self-join fan-out."""
    vals = pd.read_parquet(_VALS)
    conn = duckdb.connect(_DB, read_only=True)
    conn.register("_vals", vals)
    df = conn.execute("""
        SELECT v.voc AS gain, v.gss, v.h_pi, v.fen,
               m.move_ply AS ply, m.move_time AS rt,
               m.n_possible_moves AS legal_moves,
               m.n_self_pieces_exc_pawns AS own_material
        FROM _vals v
        JOIN processed_moves_nonzero m ON m.fen = v.fen
        WHERE m.move_time > 0 AND v.voc IS NOT NULL
    """).df()
    conn.close()
    df["logT"] = np.log(df["rt"])
    return df


def characterize_mass(df: pd.DataFrame) -> None:
    """Task 1: how much mass sits at Gain≈1.0, and is there a point-mass/spike?"""
    g = df["gain"].to_numpy()
    print("\n=== 1. Gain distribution & the ≈1.0 mass ===")
    print(f"  n moves={len(g):,}  min={g.min():.3f} max={g.max():.3f} mean={g.mean():.3f}")
    for lo, hi, lbl in [(0.0, 0.0, "exactly 0"), (1.0, 1.0, "exactly 1.0"),
                        (0.99, 1.01, "[0.99,1.01]"), (0.95, 2.0, ">=0.95"),
                        (1.0001, 2.0, ">1.0")]:
        m = (g >= lo) & (g <= hi) if lo == hi else (g >= lo) & (g <= hi)
        print(f"  {lbl:<12s}: {m.mean()*100:6.3f}%  (n={m.sum():,})")
    # fine histogram of the top decile of the support
    edges = np.arange(0.90, 1.06, 0.01)
    h, _ = np.histogram(g, bins=edges)
    print("  fine histogram near top (per 0.01):")
    for i, c in enumerate(h):
        bar = "#" * int(40 * c / max(h.max(), 1))
        print(f"    [{edges[i]:.2f},{edges[i+1]:.2f}) {c:5d} {bar}")


def reproduce_top_bin(df: pd.DataFrame) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Task 2: reproduce the figure's tie-safe interior binning and isolate the top bin.

    The plot uses zero_inflated (lump |x|<=0.05 → leftmost point) + tie_safe interior
    with n_bins=8: interior rows are bucketed by VALUE over the 7 interior quantile
    cut-points (quantile_cont at i/8), so the whole near-1.0 mass lands in one top bin.
    Returns (top_bin_lower_edge, top_bin_rows, all_interior_rows)."""
    print("\n=== 2. The figure's top bin: n, RT, variance, CI ===")
    g = df["gain"].to_numpy()
    interior = df[g > 0.05].copy()
    qs = np.quantile(interior["gain"], [i / 8 for i in range(1, 8)])
    # bin index = # cut-points exceeded (matches utils.analysis tie_safe list_filter)
    interior["bin"] = (interior["gain"].to_numpy()[:, None] > qs[None, :]).sum(axis=1)
    top = interior[interior["bin"] == 7]
    nbr = interior[interior["bin"] == 6]
    print(f"  interior cut-points (q1/8..q7/8): {np.round(qs, 3)}")
    print(f"  top bin = Gain >= {qs[-1]:.3f}")

    def ci_half(d):  # 95% CI half-width of mean logT
        return 1.96 * d["logT"].std() / np.sqrt(len(d))

    for name, d in [("top bin (#7)", top), ("neighbor (#6)", nbr), ("interior all", interior)]:
        print(f"  {name:<14s} n={len(d):6d}  mean RT={d['rt'].mean():5.2f}s "
              f"median={d['rt'].median():4.1f}s  logT std={d['logT'].std():.3f}  "
              f"CI±(logT)={ci_half(d):.4f}  Gain[{d['gain'].min():.2f},{d['gain'].max():.2f}]")
    return float(qs[-1]), top, interior


def late_ply_breakdown(df: pd.DataFrame, top_edge: float) -> None:
    """Task 5 / 2: is the dip concentrated in late plies? Compare top vs neighbor bin
    RT within ply tertiles."""
    print("\n=== 2b. Top-bin dip by ply tertile ===")
    c1, c2 = np.quantile(df["ply"], [1/3, 2/3])
    df = df.copy()
    df["tert"] = 1 + (df["ply"] > c1).astype(int) + (df["ply"] > c2).astype(int)
    interior = df[df["gain"] > 0.05]
    qs = np.quantile(interior["gain"], [i / 8 for i in range(1, 8)])
    for t in (1, 2, 3):
        sub = interior[interior["tert"] == t]
        top = sub[sub["gain"] >= qs[-1]]
        nbr = sub[(sub["gain"] >= qs[-2]) & (sub["gain"] < qs[-1])]
        lbl = {1: f"ply<={int(c1)}", 2: f"ply {int(c1)+1}-{int(c2)}", 3: f"ply>{int(c2)}"}[t]
        print(f"  {lbl:<14s} neighbor RT={nbr['rt'].mean():5.2f}s (n={len(nbr)})  "
              f"top RT={top['rt'].mean():5.2f}s (n={len(top)})  "
              f"drop={nbr['rt'].mean()-top['rt'].mean():+.2f}s")


def inspect_trees(df: pd.DataFrame, n: int = 40) -> None:
    """Tasks 3 & 4: read the final_Q vector behind a sample of Gain≈1.0 trees.

    For each sampled FEN with Gain in [0.99,1.01] load its .pt, recompute the 1-ply
    best and deep best, and report:
      * final_Q(deep best)  — is the deep winner a saturated ~+1 (proven win/mate)?
      * final_Q(1-ply best) — is the shallow pick ~0 (value-head says draw)?
      * n_legal, whether the deep best is the unique strong move.
    A Gain≈1.0 = (deep≈+1) − (shallow≈0): deep search flips a value-head 'draw' into a
    proven win. That is the saturation mechanism."""
    print("\n=== 3/4. Final-Q structure of Gain≈1.0 trees (sampled .pt) ===")
    # Match against the FULL Gain≈1.0 FEN set (filenames aren't FENs, so we scan .pt
    # files and keep any whose root FEN is in the target set) until we have `n` hits.
    target = set(df[(df["gain"] >= 0.99) & (df["gain"] <= 1.01)]["fen"].unique())
    names = [e.name for e in os.scandir(_TREES) if e.name.endswith(".pt")]
    import random as _r
    _r.Random(7).shuffle(names)
    rows = []
    scanned = 0
    for nm in names:
        if len(rows) >= n or scanned > 60000:
            break
        scanned += 1
        try:
            t = torch.load(os.path.join(_TREES, nm), map_location="cpu", weights_only=False)
            fen = " ".join(t["root_position_spec"].split()[:4])
            if fen not in target:
                continue
            fnames = list(t["feature_names"])
            nf = t["node_features"].numpy()
            par = t["parent_index"].numpy()
            vi = fnames.index("value")
            fq = np.asarray(t["oracle_final_root_q_values"], float).ravel()
            inc = t["incoming_moves"]
            rmoves = list(t["oracle_root_moves"])
            roots = np.where(par < 0)[0]
            kids = np.where(par == int(roots[0]))[0]
            myopic = -nf[kids, vi]
            order = np.argsort(myopic)[::-1]
            shallow_uci = inc[int(kids[order[0]])]
            a_sh = rmoves.index(shallow_uci)
            a_dp = int(np.argmax(fq))
            rows.append(dict(
                n_legal=len(rmoves),
                q_deep=fq[a_dp],          # deep best final Q
                q_shallow=fq[a_sh],       # 1-ply best's deep final Q
                gain=fq[a_dp] - fq[a_sh],
                deep_is_shallow=(a_dp == a_sh),
                # ROOT CAUSE: oracle_final_root_q_values is 0.0 for root moves the deep
                # search never visited (lc0 pours its budget into the 1-2 winning moves).
                # Here the 1-ply best move is one the search abandoned, so its "deep" Q is
                # the uninitialized 0.0, not its real value.
                shallow_q_is_unvisited_zero=bool(fq[a_sh] == 0.0),
                frac_root_q_zero=float(np.mean(fq == 0.0)),
            ))
        except Exception:
            continue
    r = pd.DataFrame(rows)
    if r.empty:
        print("  (no matching trees found in scan window)")
        return
    print(f"  matched {len(r)} Gain≈1.0 trees")
    print(f"  final_Q(deep best):    mean={r['q_deep'].mean():+.3f}  "
          f"frac>=0.99: {(r['q_deep']>=0.99).mean()*100:.0f}%  "
          f"frac>=0.95: {(r['q_deep']>=0.95).mean()*100:.0f}%")
    print(f"  final_Q(1-ply best):   mean={r['q_shallow'].mean():+.3f}  "
          f"frac in [-0.05,0.05] (≈draw): {((r['q_shallow'].abs()<=0.05)).mean()*100:.0f}%")
    print(f"  n_legal:               median={r['n_legal'].median():.0f}  "
          f"min={r['n_legal'].min()}  frac<=3: {(r['n_legal']<=3).mean()*100:.0f}%")
    print(f"  deep best == 1-ply best move: {r['deep_is_shallow'].mean()*100:.0f}% "
          f"(if high, Gain is a within-move Q jump, not a different move)")
    print(f"  saturation signature  (deep>=0.99 AND |shallow|<=0.05): "
          f"{((r['q_deep']>=0.99)&(r['q_shallow'].abs()<=0.05)).mean()*100:.0f}%")
    print("  ROOT CAUSE:")
    print(f"    frac of root moves with final_Q==0 (unvisited): mean={r['frac_root_q_zero'].mean()*100:.0f}%")
    print(f"    1-ply best move's deep final_Q is an UNVISITED 0.0: "
          f"{r['shallow_q_is_unvisited_zero'].mean()*100:.0f}%")
    print("    => Gain = final_Q(deep~+1) − final_Q(shallow=unvisited 0.0) = ~+1.0 is an")
    print("       artifact: the shallow move's deep Q was never computed, not a real draw.")


def monotonicity(df: pd.DataFrame) -> None:
    """Task 5: quantify Gain↔RT monotonicity overall and within ply tertiles."""
    print("\n=== 5. Monotonicity of Gain vs RT ===")
    c1, c2 = np.quantile(df["ply"], [1/3, 2/3])
    df = df.copy()
    df["tert"] = 1 + (df["ply"] > c1).astype(int) + (df["ply"] > c2).astype(int)
    rho, p = stats.spearmanr(df["gain"], df["logT"])
    print(f"  overall Spearman ρ(Gain, logRT) = {rho:+.4f} (p={p:.1e})")
    for t in (1, 2, 3):
        sub = df[df["tert"] == t]
        rho, _ = stats.spearmanr(sub["gain"], sub["logT"])
        # restrict to interior (drop the zero lump) — does ρ stay positive?
        si = sub[sub["gain"] > 0.05]
        rho_i, _ = stats.spearmanr(si["gain"], si["logT"])
        print(f"  tertile {t}: ρ(all)={rho:+.4f}  ρ(Gain>0.05)={rho_i:+.4f}  (n={len(sub):,})")


def plot_diagnostics(df: pd.DataFrame) -> None:
    """Side-by-side: (a) fine Gain histogram with the ≈1.0 spike; (b) binned RT with the
    saturated mass split out from the rest of the top bin."""
    _FIGS.mkdir(parents=True, exist_ok=True)
    g = df["gain"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].hist(g[(g > 0.05)], bins=np.arange(0.05, 1.05, 0.01), color="#4063A3")
    ax[0].axvline(1.0, color="crimson", ls="--", lw=1)
    ax[0].set_xlabel("Gain"); ax[0].set_ylabel("count (interior, Gain>0.05)")
    ax[0].set_title("Spike at Gain≈1.0 (value-head saturation)")

    # RT vs Gain: tie-safe 8-bin interior, then split the top bin into
    # 'saturated' (Gain in [0.99,1.01]) vs 'genuine' (rest of top bin).
    interior = df[g > 0.05].copy()
    qs = np.quantile(interior["gain"], [i / 8 for i in range(1, 8)])
    interior["bin"] = (interior["gain"].to_numpy()[:, None] > qs[None, :]).sum(axis=1)
    xs, ys, es = [], [], []
    for b in range(8):
        d = interior[interior["bin"] == b]
        xs.append(d["gain"].mean()); ys.append(d["rt"].mean())
        es.append(1.96 * d["rt"].std() / np.sqrt(len(d)))
    ax[1].errorbar(xs, ys, yerr=es, marker="o", color="#4063A3", label="tie-safe 8 bins (as plotted)")
    top = interior[interior["bin"] == 7]
    sat = top[(top["gain"] >= 0.99) & (top["gain"] <= 1.01)]
    gen = top[(top["gain"] < 0.99) | (top["gain"] > 1.01)]
    for d, c, lbl in [(sat, "crimson", "saturated [0.99,1.01]"), (gen, "green", "genuine (rest of top bin)")]:
        if len(d):
            ax[1].errorbar(d["gain"].mean(), d["rt"].mean(),
                           yerr=1.96 * d["rt"].std() / np.sqrt(len(d)),
                           marker="s", ms=10, color=c, label=f"{lbl} (n={len(d)})")
    ax[1].set_xlabel("Gain"); ax[1].set_ylabel("mean RT (s)")
    ax[1].set_title("Top-bin RT: saturated mass pulls it down")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    out = _FIGS / "gain_dip_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved {out}")


def main() -> None:
    df = joined_moves()
    print(f"joined {len(df):,} human moves with a tree.")
    characterize_mass(df)
    top_edge, _, _ = reproduce_top_bin(df)
    late_ply_breakdown(df, top_edge)
    inspect_trees(df)
    monotonicity(df)
    plot_diagnostics(df)


if __name__ == "__main__":
    main()
