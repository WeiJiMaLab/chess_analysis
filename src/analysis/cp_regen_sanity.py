"""T1/T2-style sanity check for the ``cp_regen`` corpus (plan.md Phase 2, Agent 3).

Mirrors ``analysis.ysagiv_sanity`` (Agent 1's sanity check for the ysagiv corpus), which
itself mirrors plan.md's T1 (5 example trees spanning the argmax spread, graphviz +
red-flag checks) / T2 (depth/width/argmax histograms, red-flag thresholds) methodology
used on our own WDL-baseline ``puctvalue_md36`` corpus -- reuses the exact same pure
helper functions from ``analysis.tree_diagnostics`` so every check is apples-to-apples.

Runs directly on the RAW ``cp_regen`` trees (generated under ``tanh(cp_order/300)`` +
pruning, see ``config_minply15_maxply75.yaml``'s ``variants.cp_regen`` block and
``slurm/cp_regen_gen_trees.slurm``) BEFORE packing/filtering -- same "gate before
spending real compute" role ysagiv_sanity.py plays, plus an explicit comparison against
the WDL-baseline's known T2 numbers (median depth=12, width=709, argmax=6 -- plan.md),
since the whole point of this track is to see whether shape actually differs, which
CP's shape-frozen replay test structurally could not show.

Usage:
    python -m analysis.cp_regen_sanity --trees-dir <cp_regen trees dir> --n-sample 3000 \
        --out-dir outputs/figures/minply15_maxply75/variants/cp_regen \
        --trees-out-dir outputs/figures/minply15_maxply75/variants/cp_regen/trees
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

# Same regime T2/S1/CP/filter_argmax.py all use -- keeps these numbers directly
# comparable to every other number in this investigation.
_ORACLE_CONFIG = BudgetedOracleConfig(
    time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0
)
_STARTING_BUDGET = 96

# T2's result on the WDL-baseline corpus (plan.md, n=5,599 validation episodes),
# quoted here so run output states the comparison explicitly rather than requiring
# the reader to cross-reference plan.md by hand.
_BASELINE_T2 = {
    "depth_median": 12, "depth_p10": 7, "depth_p90": 18,
    "width_median": 709, "width_p10": 408, "width_p90": 1433,
    "argmax_median": 6, "argmax_p10": 3, "argmax_p90": 19, "argmax_max": 89,
}


def tree_stats(path: str) -> dict | None:
    """Load one raw tree and compute (num_nodes, height, width, argmax, root_branching).

    height/width are computed on the FULL final tree (all expanded nodes) -- this runs
    pre-packing, there are no per-step episode stats yet (matches ysagiv_sanity.py).
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
            print(f"[cp_regen_sanity] {i + 1}/{len(paths)} loaded, {len(rows)} valid", flush=True)
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
    # directory, so a relative <IMG SRC="..."> reference re-resolves against that directory.
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
        subtree_size = np.ones(len(parent_index), dtype=np.int64)
        for node_id in range(len(parent_index) - 1, 0, -1):
            subtree_size[parent_index[node_id]] += subtree_size[node_id]
        root_children = [i for i, p in enumerate(parent_index) if p == 0]
        root_child_sizes = [int(subtree_size[c]) for c in root_children]
        dom = root_dominance_check(root_child_sizes, len(parent_index))
        illegal = illegal_move_check(fens, parent_index, incoming_moves)

        has_children = child_counts >= 1
        selected = prune_topk_subtree(parent_index, budget=40, intermediate_only=has_children)
        out_path = str(trees_out_dir / f"cp_regen_t1_tree_{label}")
        caption = (
            f"cp_regen T1 tree: {label} (argmax={row['argmax']}, {len(parent_index)} nodes, "
            f"{len(selected)} shown)"
        )
        rendered = render_tree_graphviz(selected, parent_index, incoming_moves, node_targets, fens,
                                         out_path, caption)
        print(f"[cp_regen_sanity T1] {label}: argmax={row['argmax']} nodes={len(parent_index)} "
              f"-> {rendered}", flush=True)

        summary.append({
            "label": label, "src": row["path"], "argmax": row["argmax"],
            "num_nodes": row["num_nodes"], "node_cutoff": len(parent_index),
            "dup": dup, "linear": lin, "dominance": dom, "illegal": illegal,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cp_regen_t1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_t2(rows: list[dict], out_dir: Path) -> dict:
    depth = np.array([r["height"] for r in rows], dtype=np.float64)
    width = np.array([r["width"] for r in rows], dtype=np.float64)
    argmax = np.array([r["argmax"] for r in rows], dtype=np.float64)
    width_flag = float(np.median([r["root_children"] for r in rows]))
    # NOTE: `save_pdf_png` (called inside `plot_diagnostic_histograms`) already creates its
    # own pdf/ and png/ subfolders under `out_dir` -- passing `out_dir / "png"` here (as
    # ysagiv_sanity.py does) double-nests into png/pdf/ and png/png/. Pass `out_dir` directly.
    flags = plot_diagnostic_histograms(
        depth, width, argmax,
        depth_flag=2.0, width_flag=width_flag, argmax_flag=90.0,
        out_dir=out_dir, base="cp_regen_t2_histograms",
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
        "baseline_wdl_t2": _BASELINE_T2,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cp_regen_t2_summary.json").write_text(json.dumps(result, indent=2))
    plot_shape_vs_baseline(depth, width, argmax, out_dir)
    return result


def plot_shape_vs_baseline(depth: np.ndarray, width: np.ndarray, argmax: np.ndarray, out_dir: Path) -> None:
    """T2-style 1x3 panel, but with the WDL-baseline's median/p10/p90 (plan.md T2, n=5,599
    validation episodes) overlaid as reference lines on top of the cp_regen histograms -- the
    explicit shape-vs-baseline comparison plan.md's Agent 3 section asks for (CP's shape-frozen
    replay probe could not show this at all, since it never re-generates trees)."""
    import matplotlib.pyplot as plt

    from analysis.utils.plots import save_pdf_png

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [
        ("Tree depth (final)", depth, "depth_median", "depth_p10", "depth_p90"),
        ("Tree width (max level size, final)", width, "width_median", "width_p10", "width_p90"),
        ("Optimal stop step (argmax)", argmax, "argmax_median", "argmax_p10", "argmax_p90"),
    ]
    for ax, (title, data, med_key, p10_key, p90_key) in zip(axes, specs):
        ax.hist(data, bins=40, color="#3F4DA0", alpha=0.75, edgecolor="white", linewidth=0.3,
                label="cp_regen")
        ax.axvline(_BASELINE_T2[med_key], color="#C0392B", ls="-", lw=1.8,
                   label=f"WDL-baseline median={_BASELINE_T2[med_key]}")
        ax.axvline(_BASELINE_T2[p10_key], color="#C0392B", ls=":", lw=1.2, alpha=0.8,
                   label=f"WDL-baseline p10/p90")
        ax.axvline(_BASELINE_T2[p90_key], color="#C0392B", ls=":", lw=1.2, alpha=0.8)
        cp_regen_median = float(np.median(data))
        ax.axvline(cp_regen_median, color="#12A19A", ls="-", lw=1.8,
                   label=f"cp_regen median={cp_regen_median:.1f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(title.split(" (")[0])
        ax.set_ylabel("count")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(axis="y", color="#E3E7EB", lw=1)
    fig.suptitle("cp_regen (tanh(cp_order/300) + pruning) vs. WDL-baseline tree shape", fontsize=12)
    fig.tight_layout()
    save_pdf_png(fig, str(out_dir), "cp_regen_t2_vs_baseline", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", required=True)
    parser.add_argument("--n-sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/variants/cp_regen")
    parser.add_argument("--trees-out-dir",
                        default="outputs/figures/minply15_maxply75/variants/cp_regen/trees")
    args = parser.parse_args()

    torch.set_num_threads(1)
    paths = sorted(str(p) for p in Path(args.trees_dir).glob("*.pt"))
    rng = np.random.RandomState(args.seed)
    if len(paths) > args.n_sample:
        paths = list(rng.choice(paths, size=args.n_sample, replace=False))
    print(f"[cp_regen_sanity] sampling {len(paths)} trees from {args.trees_dir}", flush=True)

    rows = collect_stats(paths)
    rows = [r for r in rows if r["argmax"] is not None]
    print(f"[cp_regen_sanity] {len(rows)}/{len(paths)} trees usable (valid trajectory)", flush=True)

    out_dir = Path(args.out_dir)
    t2 = run_t2(rows, out_dir)
    t1 = run_t1(rows, out_dir, Path(args.trees_out_dir))

    print("T2 summary (cp_regen vs. WDL-baseline):", json.dumps(t2, indent=2))
    print(f"T1: {len(t1)} trees checked, all red flags: "
          f"{[{'label': e['label'], 'dup_pass': e['dup']['pass'], 'lin_pass': e['linear']['pass'], 'dom_pass': e['dominance']['pass'], 'illegal_pass': e['illegal']['pass']} for e in t1]}")


if __name__ == "__main__":
    main()
