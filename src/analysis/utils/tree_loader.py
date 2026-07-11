"""Deep signals are computed by REPLAYING the negamax backup over stored per-node
static values in the requested unit (pwin | cp) -- not by reading the generator's
oracle_* arrays, whose unit silently follows generation-time value_feature."""

import functools
import os
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Per-unit readout constants. frac_good ε: "within ε of the best myopic value";
# 0.1 pwin ↔ 50 cp (≈ half a pawn; the sigmoid slope at 0 is ~0.002/cp · 2 range,
# so 50 cp ≈ 0.1 value-units near balance). acceptable_thr: an ABSOLUTE satisficing
# bar — "not clearly losing", value ≥ −0.20 pwin (≈ −100 cp). The old ≥ 0 ("at least
# equal") bar was too strict: the side-to-move is routinely a hair below equal, so
# ~48% of positions had ZERO acceptable moves (n_acceptable degenerate). −0.20 pwin
# cuts that to ~32%; the residual zeros are genuinely lost positions (best move still
# on the pwin losing rail) and are handled by the hurdle split in the dashboards.
# oss_node_cost: per-node halt cost; 1e-4 was settled in pwin units (range ~2);
# the cp default scales by the ~×400 effective range ratio (±800 cp interior).
UNITS = {
    "pwin": {"feature": "value", "frac_good_eps": 0.1, "acceptable_thr": -0.20,
             "oss_node_cost": 1e-4},
    "cp": {"feature": "cp_order", "frac_good_eps": 50.0, "acceptable_thr": -100.0,
           "oss_node_cost": 0.04},
}


def _tree_arrays(payload, unit: str):
    """(static_values, parent_index, root, kids) in the unit's per-node feature.

    static values are each node's OWN-side-to-move POV; kids are the root's
    children in node-index (= insertion/expansion) order.
    """
    feature_names = list(payload["feature_names"])
    nf = payload["node_features"].numpy()
    par = payload["parent_index"].numpy()
    vi = feature_names.index(UNITS[unit]["feature"])
    static = nf[:, vi].astype(float)
    roots = np.where(par < 0)[0]
    if not roots.size:
        return None
    root = int(roots[0])
    kids = np.where(par == root)[0]
    return static, par, root, kids


def _deep_values_prefix(static: np.ndarray, par: np.ndarray, upto: int) -> np.ndarray:
    """Negamax deep values over the tree prefix of nodes [0, upto].

    v(node) = static(node) if it has no children in the prefix, else
    −min(v(children)) — the node's STM picks the child worst for the opponent.
    Children always have a higher index than their parent (insertion order), so
    one reverse pass suffices.
    """
    v = static[: upto + 1].copy()
    has_kid = np.zeros(upto + 1, dtype=bool)
    minkid = np.full(upto + 1, np.inf)
    for i in range(upto, 0, -1):
        # Finalize node i's deep value FIRST (all its children have higher index
        # and are already processed), THEN fold that value into its parent — folding
        # the raw static value would break negamax for trees deeper than one level.
        if has_kid[i]:
            v[i] = -minkid[i]
        p = par[i]
        if 0 <= p <= upto:
            if v[i] < minkid[p]:
                minkid[p] = v[i]
            has_kid[p] = True
    if has_kid[0]:
        v[0] = -minkid[0]
    return v


def _tree_signals(payload, unit: str) -> dict | None:
    """All unit-parametric signals for one tree (see module docstring)."""
    arrs = _tree_arrays(payload, unit)
    if arrs is None:
        return None
    static, par, root, kids = arrs
    if kids.size < 2 or not np.isfinite(static[kids]).all():
        return None
    cfg = UNITS[unit]
    n_nodes = static.size

    # --- myopic (pre-search) signals: mover-POV leaf evals of the root children.
    # COUNTS, not fractions: a fraction divides by the legal-move count, folding
    # board-crowding back into the signal (a confound with RT) and manufacturing a
    # non-monotone proxy; the raw count is monotone in "how many good/safe moves
    # exist" and has no denominator. n_root_children is kept so the fraction is
    # recoverable downstream (frac = count / n_root_children) if ever needed.
    myo = -static[kids]
    order = np.argsort(myo)[::-1]
    action_gap = float(myo[order[0]] - myo[order[1]])
    n_within_epsilon = int(np.sum(myo >= myo.max() - cfg["frac_good_eps"]))
    n_acceptable = int(np.sum(myo >= cfg["acceptable_thr"]))
    n_root_children = int(kids.size)

    # --- deep signals: replayed negamax trace over the expansion prefix.
    # Root-child deep values are the ROOT mover's values: −v(kid).
    # trace[t] = best root child (by deep value) if the search stopped after
    # node t entered the tree.
    best_trace = np.empty(n_nodes, dtype=int)
    final_kid_vals = None
    for t in range(n_nodes):
        v = _deep_values_prefix(static, par, t)
        kid_vals = np.where(kids <= t, -v[np.minimum(kids, t)], myo * np.nan)
        # before a kid exists in the prefix it can't be chosen; at t >= kids.max()
        # (root expansion complete) all kids are live. Root expansion inserts all
        # children consecutively, so early prefixes only matter for tiny trees.
        live = kids <= t
        if not live.any():
            best_trace[t] = -1
            continue
        vals = np.where(live, kid_vals, -np.inf)
        best_trace[t] = int(np.argmax(vals))
        if t == n_nodes - 1:
            final_kid_vals = kid_vals
    if final_kid_vals is None or not np.isfinite(final_kid_vals).all():
        return None
    final_best = best_trace[-1]

    # GSS: first step from which the running best equals the final best.
    gss = int(np.argmax(best_trace == final_best))
    # VOC/Gain: deep value of the deep-best move minus deep value of the move
    # the GREEDY (myopic) policy would pick with no search.
    voc = float(final_kid_vals[final_best] - final_kid_vals[int(np.argmax(myo))])
    # OSS: budgeted-oracle stop on the replayed trace — commit at step t to the
    # move that LOOKS best at step t; reward = that move's FINAL deep value; cost
    # grows per node. Steps before any root child is live (best_trace == -1) are
    # not stoppable → reward −inf (else the fallback-to-final-best makes t=0 the
    # trivial argmax and OSS collapses to 0).
    live_step = best_trace >= 0
    committed = np.where(live_step, best_trace, 0)
    rewards = np.where(live_step, final_kid_vals[committed], -np.inf)
    oss = int(np.argmax(rewards - cfg["oss_node_cost"] * (np.arange(n_nodes) + 1.0)))

    # MQ per root move: final deep value relative to the best (≤ 0; cp unit =
    # centipawn loss). Move labels come from oracle_root_moves, which the
    # generator writes in root-children node order.
    moves = list(payload.get("oracle_root_moves", []))
    if len(moves) == kids.size:
        mq = (final_kid_vals - final_kid_vals.max()).tolist()
    else:
        moves, mq = [], []

    return {
        "gss": gss, "voc": voc, "action_gap": action_gap,
        "n_within_epsilon": n_within_epsilon, "n_acceptable": n_acceptable,
        "n_root_children": n_root_children,
        "oss": oss, "moves": moves, "mq": mq,
    }


def _tree_hpi(payload) -> float:
    """Policy prior entropy H(π) (unit-independent; degenerate under uniform priors)."""
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


def _worker(path: str, unit: str):
    try:
        torch.set_num_threads(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        sig = _tree_signals(t, unit)
        if sig is None:
            return None
        hpi = _tree_hpi(t)
        fen = " ".join(t["root_position_spec"].split()[:4])
        return (fen, sig["gss"], sig["voc"], sig["action_gap"], hpi,
                sig["n_within_epsilon"], sig["n_acceptable"], sig["n_root_children"],
                sig["oss"], sig["moves"], sig["mq"])
    except Exception:
        return None


def compute_values(trees_dir: str, n_trees: int, seed: int, n_workers: int,
                   unit: str = "pwin") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-tree signals + per-root-move MQ in the requested unit ("pwin" | "cp")."""
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {sorted(UNITS)}")
    names = [e.name for e in os.scandir(trees_dir) if e.name.endswith(".pt")]
    names = random.Random(seed).sample(names, min(n_trees, len(names)))
    paths = [os.path.join(trees_dir, nm) for nm in names]
    tree_rows, move_rows = [], []
    with mp.Pool(n_workers) as pool:
        worker = functools.partial(_worker, unit=unit)
        for r in tqdm(pool.imap_unordered(worker, paths, chunksize=64), total=len(paths), desc="trees"):
            if r is None:
                continue
            fen, gss, voc_val, gap, hpi, n_eps, n_acc, n_kids, oss, ucis, mqs = r
            tree_rows.append({"fen": fen, "gss": gss, "voc": voc_val, "action_gap": gap,
                              "h_pi": hpi, "n_within_epsilon": n_eps,
                              "n_acceptable": n_acc, "n_root_children": n_kids,
                              "oss": oss})
            for u, q in zip(ucis, mqs):
                move_rows.append({"fen": fen, "move_uci": u, "mq": q})
    return pd.DataFrame(tree_rows), pd.DataFrame(move_rows)
