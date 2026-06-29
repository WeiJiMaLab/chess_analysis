"""Value-pruning PROXY — R-PRUNING step 1 (pick the eps grid, no regen).

For each existing n=1 tree, POST-HOC prune the COMPLETED tree and count surviving
nodes across an epsilon grid. Pruning rule (locked): relative-to-best by leaf-eval
value, NEGAMAX (verified: parent ~= -min(child), so the best child for the parent is
the MIN-value child). At each kept node we keep a child iff value <= min_sibling + eps,
and drop its whole subtree otherwise.

This is the LOOSE proxy (uses the final completed tree, not the causal early values, and
does not redeploy the freed budget) — valid only to CHOOSE the 2-3 eps levels worth
re-generating. The real test is the regen (R-PRUNING step 2).

  python prune_proxy.py            # heavy: compute -> FEN-keyed parquet (sbatch)
  python prune_proxy.py analyze    # light: join RT, Spearman(pruned_eps, RT) + partial|legal
"""
from __future__ import annotations

import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

TREES_DIR = "/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/elo2000_n1"
OUTDIR = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000_n1"
OUT = f"{OUTDIR}/prune_proxy.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
# eps grid in WDL value units (values live in [-1, 1]; eps is a gap above the best sibling).
# eps=0 -> only the exact-best children (satisfaction-like); eps>=2 -> no prune (= n_total).
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


def _worker(path: str):
    try:
        torch.set_num_threads(1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        fen = " ".join(d["root_position_spec"].split()[:4])
        v = np.asarray(d["node_features"], dtype=float)[:, 0]
        pi = np.asarray(d["parent_index"])
        ie = np.asarray(d["is_expanded"]).astype(bool)
        n = v.shape[0]
        if n == 0:
            return None
        # children adjacency (skip the root self/neg parent)
        children: list[list[int]] = [[] for _ in range(n)]
        for ci in range(n):
            par = int(pi[ci])
            if par != ci and 0 <= par < n:
                children[par].append(ci)

        row = {
            "fen": fen,
            "legal_moves": len(children[0]),
            "n_total": int(n),
            "n_leaf": int((~ie).sum()),
        }
        for eps in EPS_GRID:
            keep = 0
            stack = [0]
            while stack:
                node = stack.pop()
                keep += 1
                ch = children[node]
                if not ch:
                    continue
                cv = v[ch]
                thr = cv.min() + eps
                stack.extend(c for c, val in zip(ch, cv) if val <= thr)
            row[f"pruned_{eps}"] = keep
        return row
    except Exception as e:  # noqa: BLE001
        return ("ERR", path, repr(e))


def compute():
    paths = sorted(glob.glob(os.path.join(TREES_DIR, "*.pt")))
    print(f"n1 trees: {len(paths)}", flush=True)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    print(f"workers={n_workers}  eps_grid={EPS_GRID}", flush=True)
    rows, errs = [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_worker, p) for p in paths]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is None:
                continue
            if isinstance(r, tuple) and r and r[0] == "ERR":
                errs += 1
                if errs <= 5:
                    print("ERR", r[1], r[2], flush=True)
                continue
            rows.append(r)
            if i % 20000 == 0:
                print(f"  {i}/{len(paths)} kept={len(rows)} errs={errs}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}  rows={len(df)}  errs={errs}", flush=True)


def _spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial_spearman(y, x, z):
    """Spearman partial of x on y controlling z (residualize ranks linearly)."""
    from scipy.stats import rankdata
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    rz1 = np.c_[np.ones_like(rz), rz]
    bx = np.linalg.lstsq(rz1, rx, rcond=None)[0]
    by = np.linalg.lstsq(rz1, ry, rcond=None)[0]
    ex, ey = rx - rz1 @ bx, ry - rz1 @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def _boot(fn, *cols, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(cols[0])
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = fn(*[c[idx] for c in cols])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(fn(*cols)), float(lo), float(hi)


def analyze():
    import duckdb
    df = pd.read_parquet(OUT)
    con = duckdb.connect(DB, read_only=True)
    rt = con.execute(
        "SELECT fen, move_time FROM processed_moves_nonzero WHERE move_time > 0"
    ).df()
    con.close()
    m = rt.merge(df, on="fen", how="inner")
    print(f"joined human moves: {len(m):,}  (unique FENs: {m.fen.nunique():,})\n")
    y = m["move_time"].to_numpy()  # Spearman is rank-based; log is monotone, so raw RT is fine
    leg = m["legal_moves"].to_numpy()

    r, lo, hi = _boot(_spearman, leg, y)
    print(f"{'legal_moves (floor)':<26} rho={r:+.3f}  [{lo:+.3f},{hi:+.3f}]")
    r, lo, hi = _boot(_spearman, m['n_total'].to_numpy(), y)
    print(f"{'n_total (unpruned)':<26} rho={r:+.3f}  [{lo:+.3f},{hi:+.3f}]")
    print()
    print(f"{'eps':>5} {'pruned/legal':>13} {'rho(pruned,RT)':>22} {'partial|legal':>22}")
    for eps in EPS_GRID:
        p = m[f"pruned_{eps}"].to_numpy().astype(float)
        ratio = float(np.median(p / np.maximum(leg, 1)))
        r, lo, hi = _boot(_spearman, p, y)
        pr, plo, phi = _boot(_partial_spearman, y, p, leg)
        print(f"{eps:>5} {ratio:>13.2f} {r:>+8.3f} [{lo:+.3f},{hi:+.3f}] {pr:>+8.3f} [{plo:+.3f},{phi:+.3f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        compute()
