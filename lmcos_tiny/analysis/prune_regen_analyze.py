"""R-PRUNING step 2 readout — regenerated pruned trees vs human RT.

For the unpruned baseline (elo2000_n1, first 50k) and each regenerated pruned set
(elo2000_n1_eps{0.05,0.1,0.3}): compute per-tree cost (n_total / n_expanded / n_leaf)
and legal_moves, join human RT, and report Spearman(cost, RT) + partial|legal — the
test the loose proxy could NOT do (these trees were BUILT with pruning, budget
redeployed deeper, causal early values). Bootstrap 95% CIs.

  python lmcos_tiny/analysis/prune_regen_analyze.py   # (sbatch: DuckDB join needs memory)
"""
from __future__ import annotations

import os
import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
BASE = "/scratch/gpfs/GRIFFITHS/hl4291/sf_trees"
LEVELS = ["none", "0.05", "0.1", "0.3"]  # all regenerated at max_depth=36, full 250k roots


def _dir(e: str) -> str:
    return f"{BASE}/n1md36" if e == "none" else f"{BASE}/n1md36_eps{e}"


def _counts(path: str):
    try:
        torch.set_num_threads(1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        fen = " ".join(d["root_position_spec"].split()[:4])
        pi = np.asarray(d["parent_index"])
        ie = np.asarray(d["is_expanded"]).astype(bool)
        return {
            "fen": fen,
            "legal_moves": int((pi == 0).sum()),
            "n_total": int(pi.shape[0]),
            "n_expanded": int(ie.sum()),
            "n_leaf": int((~ie).sum()),
        }
    except Exception:  # noqa: BLE001
        return None


def _spearman(a, b):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _partial_spearman(y, x, z):
    from scipy.stats import rankdata
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    rz1 = np.c_[np.ones_like(rz), rz]
    ex = rx - rz1 @ np.linalg.lstsq(rz1, rx, rcond=None)[0]
    ey = ry - rz1 @ np.linalg.lstsq(rz1, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def _boot(fn, *cols, B=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(cols[0])
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)              # one paired resample shared across columns
        vals[b] = fn(*[c[idx] for c in cols])
    return float(fn(*cols)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    import duckdb
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    con = duckdb.connect(DB, read_only=True)
    print(f"{'set':>6} {'trees':>7} {'med_ntot':>9} {'med_nexp':>9} "
          f"{'rho(ntot,RT)':>22} {'partial|legal':>22} {'rho(legal,RT)':>14}", flush=True)
    for e in LEVELS:
        paths = sorted(glob.glob(_dir(e) + "/*.pt"))
        if not paths:
            print(f"{e:>6}  (no trees yet)", flush=True)
            continue
        recs = []
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for r in ex.map(_counts, paths, chunksize=64):
                if r is not None:
                    recs.append(r)
        df = pd.DataFrame(recs)
        con.register("t", df)
        m = con.execute(
            "SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x "
            "ON t.fen = x.fen WHERE x.move_time > 0"
        ).df()
        con.unregister("t")
        y = m["move_time"].to_numpy()
        leg = m["legal_moves"].to_numpy()
        nt = m["n_total"].to_numpy().astype(float)
        r_nt, lo_nt, hi_nt = _boot(_spearman, nt, y)
        p_nt, plo, phi = _boot(_partial_spearman, y, nt, leg)
        r_leg = _spearman(leg, y)
        print(f"{e:>6} {len(m):>7} {int(np.median(df.n_total)):>9} {int(np.median(df.n_expanded)):>9} "
              f"{r_nt:>+8.3f}[{lo_nt:+.3f},{hi_nt:+.3f}] {p_nt:>+8.3f}[{plo:+.3f},{phi:+.3f}] {r_leg:>+8.3f}",
              flush=True)
    con.close()


if __name__ == "__main__":
    main()
