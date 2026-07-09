"""T1/T2-style sanity check for the ysagiv ``human_trees`` corpus (Phase 2, Agent 1).

Mirrors the methodology plan.md's T1 (5 example trees spanning the argmax spread, graphviz +
red-flag checks) and T2 (depth/width/argmax histograms over the population, red-flag thresholds
as vertical lines) used the same way on our own ``puctvalue_md36`` corpus -- reuses the exact
same pure helper functions from ``analysis.tree_diagnostics`` so the checks are apples-to-apples.

Unlike T1/T2 on our own corpus (which ran on packed VALIDATION EPISODES, i.e. post-oracle,
post-split trajectories), this runs directly on RAW ysagiv trees before any packing/filtering --
the corpus hasn't been through this pipeline before, so this is the gate that decides whether it's
worth spending real compute on at all (per plan.md's "sanity-check a sample first" instruction).

Usage:
    python -m analysis.ysagiv_sanity --trees-dir <sampled dir> --n-sample 3000 \
        --out-dir outputs/figures/ysagiv --trees-out-dir outputs/figures/ysagiv/trees
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.tree_diagnostics import (
    duplicate_fen_check,
    illegal_move_check,
    linear_chain_check,
    plot_diagnostic_histograms,
    prune_topk_subtree,
    render_tree_graphviz,
    root_dominance_check,
    select_percentile_episodes,
)
from cts.data.preprocess_gnn.teacher_targets import load_raw_pretrain_record
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.data.preprocess_mc.pack import build_compact_trajectory, budgeted_oracle_from_trajectory

# Same regime filter_argmax.py uses -- keeps this sanity check's argmax numbers directly
# comparable to the "thinking helps" filter that's about to be applied to this same population.
_ORACLE_CONFIG = BudgetedOracleConfig(
    time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0
)
_STARTING_BUDGET = 96


def tree_stats(path: str) -> dict | None:
    """Load one raw tree and compute (num_nodes, height, width, argmax, root_branching).

    height/width are computed on the FULL final tree (all expanded nodes), matching what T1/T2
    called "depth"/"width (final)" for our own corpus -- not per-episode step stats (this runs
    pre-packing, there are no episodes yet).
    """
    try:
        record = load_raw_pretrain_record(path)
    except Exception:
        return None
    depth = record.depth.to(dtype=torch.int64)
    num_nodes = int(depth.shape[0])
    height = int(depth.max().item())
    width = int(torch.bincount(depth).max().item())
    root_children = int((record.parent_index == 0).sum().item())

    trajectory = build_compact_trajectory(record, source_path=path)
    if trajectory is None or trajectory["num_steps"] <= 0:
        argmax = None
    else:
        policy = budgeted_oracle_from_trajectory(trajectory, _STARTING_BUDGET, _ORACLE_CONFIG)
        argmax = int(policy.optimal_stop_step)

    return {
        "path": path,
        "num_nodes": num_nodes,
        "height": height,
        "width": width,
        "root_children": root_children,
        "argmax": argmax,
    }


def collect_stats(paths: list[str]) -> list[dict]:
    rows = []
    for i, path in enumerate(paths):
        row = tree_stats(path)
        if row is not None:
            rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"[ysagiv_sanity] {i + 1}/{len(paths)} loaded, {len(rows)} valid", flush=True)
    return rows


def run_t1(rows: list[dict], out_dir: Path, trees_out_dir: Path) -> list[dict]:
    """Pick 5 trees spanning the argmax spread (p10/p50/p75/p90/max), run the 4 red-flag checks,
    render each via graphviz. Returns the per-tree summary list (mirrors t1_summary.json)."""
    argmax_vals = np.array([r["argmax"] for r in rows], dtype=np.float64)
    p10, p50, p75, p90 = np.percentile(argmax_vals, [10, 50, 75, 90])
    targets = [p10, p50, p75, p90, float(argmax_vals.max())]
    labels = ["p10", "p50", "p75", "p90", "max"]
    idxs = select_percentile_episodes(argmax_vals, targets)

    # Absolute path required: graphviz's `dot` subprocess runs with cwd set to the OUTPUT file's
    # directory, so a relative <IMG SRC="..."> reference re-resolves against that directory and
    # breaks (see t1_render_trees_svg.py's docstring for the same fix on our own corpus).
    trees_out_dir = trees_out_dir.resolve()
    trees_out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for label, idx in zip(labels, idxs):
        row = rows[idx]
        record = load_raw_pretrain_record(row["path"])
        fens = record._resolved_position_specs()
        parent_index = record.parent_index.numpy().astype(np.int64)
        incoming_moves = list(record.incoming_moves)
        node_targets = record.node_targets.numpy()
        child_counts = np.bincount(parent_index[parent_index >= 0], minlength=len(parent_index))

        dup = duplicate_fen_check(fens)
        lin = linear_chain_check(child_counts)
        root_child_sizes = []
        subtree_size = np.ones(len(parent_index), dtype=np.int64)
        for node_id in range(len(parent_index) - 1, 0, -1):
            subtree_size[parent_index[node_id]] += subtree_size[node_id]
        root_children = [i for i, p in enumerate(parent_index) if p == 0]
        root_child_sizes = [int(subtree_size[c]) for c in root_children]
        dom = root_dominance_check(root_child_sizes, len(parent_index))
        illegal = illegal_move_check(fens, parent_index, incoming_moves)

        has_children = child_counts >= 1
        selected = prune_topk_subtree(parent_index, budget=40, intermediate_only=has_children)
        out_path = str(trees_out_dir / f"ysagiv_t1_tree_{label}")
        caption = (
            f"ysagiv T1 tree: {label} (argmax={row['argmax']}, {len(parent_index)} nodes, "
            f"{len(selected)} shown)"
        )
        rendered = render_tree_graphviz(selected, parent_index, incoming_moves, node_targets, fens,
                                         out_path, caption)
        print(f"[ysagiv_sanity T1] {label}: argmax={row['argmax']} nodes={len(parent_index)} "
              f"-> {rendered}", flush=True)

        summary.append({
            "label": label, "src": row["path"], "argmax": row["argmax"],
            "num_nodes": row["num_nodes"], "node_cutoff": len(parent_index),
            "dup": dup, "linear": lin, "dominance": dom, "illegal": illegal,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ysagiv_t1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_t2(rows: list[dict], out_dir: Path) -> dict:
    depth = np.array([r["height"] for r in rows], dtype=np.float64)
    width = np.array([r["width"] for r in rows], dtype=np.float64)
    argmax = np.array([r["argmax"] for r in rows], dtype=np.float64)
    # width_flag: median root branching factor across the sample -- the closest analogue to "root
    # legal-move count" for a population of DIFFERENT root positions (unlike our own corpus's T2,
    # there's no single shared root, so no single true ceiling -- median is the representative line).
    width_flag = float(np.median([r["root_children"] for r in rows]))
    flags = plot_diagnostic_histograms(
        depth, width, argmax,
        depth_flag=2.0, width_flag=width_flag, argmax_flag=90.0,
        out_dir=out_dir / "png", base="ysagiv_t2_histograms",
    )
    result = {
        "n": len(rows),
        "depth_median": float(np.median(depth)), "depth_p10": float(np.percentile(depth, 10)),
        "depth_p90": float(np.percentile(depth, 90)),
        "width_median": float(np.median(width)), "width_p10": float(np.percentile(width, 10)),
        "width_p90": float(np.percentile(width, 90)),
        "argmax_median": float(np.median(argmax)), "argmax_p10": float(np.percentile(argmax, 10)),
        "argmax_p90": float(np.percentile(argmax, 90)), "argmax_max": float(argmax.max()),
        "width_flag_root_branching_median": width_flag,
        **flags,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ysagiv_t2_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", required=True)
    parser.add_argument("--n-sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/figures/ysagiv")
    parser.add_argument("--trees-out-dir", default="outputs/figures/ysagiv/trees")
    args = parser.parse_args()

    torch.set_num_threads(1)
    paths = sorted(str(p) for p in Path(args.trees_dir).glob("*.pt"))
    rng = np.random.RandomState(args.seed)
    if len(paths) > args.n_sample:
        paths = list(rng.choice(paths, size=args.n_sample, replace=False))
    print(f"[ysagiv_sanity] sampling {len(paths)} trees from {args.trees_dir}", flush=True)

    rows = collect_stats(paths)
    rows = [r for r in rows if r["argmax"] is not None]
    print(f"[ysagiv_sanity] {len(rows)}/{len(paths)} trees usable (valid trajectory)", flush=True)

    out_dir = Path(args.out_dir)
    t2 = run_t2(rows, out_dir)
    t1 = run_t1(rows, out_dir, Path(args.trees_out_dir))

    print("T2 summary:", json.dumps(t2, indent=2))
    print(f"T1: {len(t1)} trees checked, all red flags: "
          f"{[{'label': e['label'], 'dup_pass': e['dup']['pass'], 'lin_pass': e['linear']['pass'], 'dom_pass': e['dominance']['pass'], 'illegal_pass': e['illegal']['pass']} for e in t1]}")


if __name__ == "__main__":
    main()
