"""Construal-rank PROXY — R-CONSTRUAL go/no-go (no generation).

Cheap test of the construal core on the existing n1md36 trees: order the side-to-move's
pieces by a GUT criticality (the early root-Q trace, before the 96-expansion search refines
it), and define |S*| = the gut-rank of the piece that makes the search's best move — i.e. the
smallest piece-set, in gut order, whose construal already contains the winning move. A surprising
best move (its piece ranked low by the gut) ⇒ large |S*| ⇒ predict slow; an obvious one ⇒ small.

This is the LOOSE proxy (root-restricted, depth from the existing full tree, no re-search). If
|S*| tracks RT and dissociates from legal-moves, the full restricted-move BeFS build is worth it.

  python construal_proxy.py            # compute -> parquet
  python construal_proxy.py analyze    # RT join + Spearman(|S*|,RT) + partial|legal
"""
from __future__ import annotations
import os, sys, glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, torch

from cts.stats import spearman, partial_spearman, bootstrap_ci

SET = os.environ.get("VOC_SET", "n1md36")
TREES = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/{SET}"
OUT = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/construal_proxy.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"


def _worker(path):
    try:
        torch.set_num_threads(1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        fen = " ".join(d["root_position_spec"].split()[:4])
        mv = [str(m) for m in d["oracle_root_moves"]]
        if not mv:
            return None
        bi = int(np.asarray(d["oracle_best_move_index"]).ravel()[-1])
        if bi >= len(mv):
            return None
        qt = np.asarray(d["oracle_root_q_trace"], float)
        gut = None
        for r in range(qt.shape[0]):            # first trace row that is actually evaluated
            if np.isfinite(qt[r]).all() and np.std(qt[r]) > 1e-6:
                gut = qt[r]; break
        if gut is None or gut.shape[0] != len(mv):
            return None
        bmp = mv[bi][:2]                          # best move's from-square (the critical piece)
        fq = np.asarray(d["oracle_final_root_q_values"], float).ravel()  # full value per move
        if fq.shape[0] != len(mv):
            return None
        pv = defaultdict(lambda: -9.0)            # piece -> best gut value of its moves
        for m, g in zip(mv, gut):
            pv[m[:2]] = max(pv[m[:2]], float(g))
        order = sorted(pv, key=lambda s: -pv[s])  # gut criticality order
        s_star = order.index(bmp) + 1             # gut-rank of the winning piece (loose proxy)
        # value-based reward-cost stop: V_sub(k) = best FULL value reachable using the top-k
        # gut pieces' moves (monotone ↑ in k); |S*|(c) = argmax_k [V_sub(k) - c*k].
        piece_idx = {p: j for j, p in enumerate(order)}
        rank_of = np.array([piece_idx[m[:2]] for m in mv])  # 0-based gut rank of each move's piece
        Vsub = np.maximum.accumulate(
            [fq[rank_of <= k].max() for k in range(len(order))])  # V_sub(k=1..n_pieces)
        kk = np.arange(1, len(Vsub) + 1)
        row = {"fen": fen, "legal_moves": len(mv), "n_pieces": len(pv),
               "s_star": s_star, "s_star_frac": s_star / len(pv)}
        for c in (0.02, 0.05, 0.1, 0.2, 0.4):
            row[f"sstar_c{c}"] = int(np.argmax(Vsub - c * kk) + 1)
        return row
    except Exception:  # noqa: BLE001
        return None


def compute():
    paths = sorted(glob.glob(TREES + "/*.pt"))
    print(f"SET={SET}  trees={len(paths)}", flush=True)
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    rows = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for r in ex.map(_worker, paths, chunksize=64):
            if r is not None:
                rows.append(r)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"wrote {OUT}  rows={len(rows)}", flush=True)


def analyze():
    import duckdb
    df = pd.read_parquet(OUT)
    con = duckdb.connect(DB, read_only=True); con.register("t", df)
    m = con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x "
                    "ON t.fen=x.fen WHERE x.move_time>0").df()
    con.close()
    y = m["move_time"].to_numpy(); leg = m["legal_moves"].to_numpy().astype(float)
    print(f"n={len(m):,}  med: legal={int(np.median(leg))} n_pieces={int(m.n_pieces.median())} "
          f"s_star={int(m.s_star.median())}\n", flush=True)
    cols = ["legal_moves", "n_pieces", "s_star", "s_star_frac"] + \
           [c for c in ("sstar_c0.02", "sstar_c0.05", "sstar_c0.1", "sstar_c0.2", "sstar_c0.4") if c in m.columns]
    print(f"{'feature':>14} {'ρ(.,RT)':>22} {'partial|legal':>22}", flush=True)
    for col in cols:
        c = m[col].to_numpy().astype(float)
        r, lo, hi = bootstrap_ci(spearman, c, y, n_boot=400)
        if col == "legal_moves":
            print(f"{col:>14} {r:>+8.3f}[{lo:+.3f},{hi:+.3f}] {'(floor)':>22}", flush=True)
        else:
            pr, plo, phi = bootstrap_ci(partial_spearman, y, c, leg, n_boot=400)
            print(f"{col:>14} {r:>+8.3f}[{lo:+.3f},{hi:+.3f}] {pr:>+8.3f}[{plo:+.3f},{phi:+.3f}]", flush=True)


if __name__ == "__main__":
    (analyze if len(sys.argv) > 1 and sys.argv[1] == "analyze" else compute)()
