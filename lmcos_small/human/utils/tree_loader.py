"""
PyTorch Leela tree data loading and feature extraction.
Handles loading PT files in parallel to extract Greedy Stopping Step (GSS),
Value of Computation (Gain), Action Gap, Prior Entropy H(π), and Root MQ.
"""

import os
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def _tree_voc_and_gap(payload) -> tuple[float, float]:
    """Extract Gain and Action Gap from tree payload."""
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    vi = feature_names.index("value")
    final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()

    action_gap = float("nan")
    voc_val = float("nan")
    roots = np.where(par < 0)[0]
    if roots.size:
        kids = np.where(par == int(roots[0]))[0]
        if kids.size >= 2:
            myopic_q = -nf[kids, vi]
            order = np.argsort(myopic_q)[::-1]
            action_gap = float(myopic_q[order[0]] - myopic_q[order[1]])
            qt = np.asarray(payload["oracle_root_q_trace"], dtype=float)
            qt = qt.reshape(qt.shape[0], -1)
            if qt.shape[1] == final_q.size and final_q.size >= 2:
                visited = qt != 0.0
                if visited.any():
                    steps = np.where(visited, np.arange(qt.shape[0])[:, None], qt.shape[0])
                    first_visit = np.where(visited.any(axis=0), steps.min(axis=0), qt.shape[0] + 1)
                    a_shallow = int(np.argmin(first_visit))
                    a_deep = int(np.argmax(final_q))
                    voc_val = float(final_q[a_deep] - final_q[a_shallow])
    return voc_val, action_gap


def _tree_root_mq(payload) -> tuple[list[str], list[float]]:
    """Extract Lc0 root-move MQ from tree payload."""
    moves = list(payload["oracle_root_moves"])
    fq = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
    if not moves or fq.size != len(moves) or not np.isfinite(fq).all():
        return [], []
    mq = (fq - float(np.max(fq))).tolist()
    return moves, mq


def _tree_gss(payload) -> int:
    """Extract Greedy Stopping Step from tree payload."""
    bmi = np.asarray(payload["oracle_best_move_index"]).ravel()
    if bmi.size == 0:
        return -1
    return int(np.argmax(bmi == bmi[-1]))


def _tree_hpi(payload) -> float:
    """Extract policy prior entropy H(π)."""
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    pi_idx = feature_names.index("prior")
    roots = np.where(par < 0)[0]
    if not roots.size:
        return float("nan")
    kids = np.where(par == int(roots[0]))[0]
    if kids.size < 2:
        return float("nan")
    p = nf[kids, pi_idx].astype(float)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _worker(path: str):
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        gss = _tree_gss(t)
        if gss < 0:
            return None
        voc_val, gap = _tree_voc_and_gap(t)
        hpi = _tree_hpi(t)
        ucis, mqs = _tree_root_mq(t)
        fen = " ".join(t["root_position_spec"].split()[:4])
        return (fen, gss, voc_val, gap, hpi, ucis, mqs)
    except Exception:
        return None


def compute_values(trees_dir: str, n_trees: int, seed: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    tree_rows, move_rows = [], []
    with mp.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, paths, chunksize=64), total=len(paths), desc="trees"):
            if r is None:
                continue
            fen, gss, voc_val, gap, hpi, ucis, mqs = r
            tree_rows.append({"fen": fen, "gss": gss, "voc": voc_val, "action_gap": gap, "h_pi": hpi})
            for u, q in zip(ucis, mqs):
                move_rows.append({"fen": fen, "move_uci": u, "mq": q})
    return pd.DataFrame(tree_rows), pd.DataFrame(move_rows)
