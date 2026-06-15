"""
Analysis: distribution of node depths across many lc0 search trees.

Appendix figure comparing two tree sets generated with different lc0
``max_depth`` caps (e.g. depth4 vs depth30). Answers the headline question
"how deep do trees actually go" with two complementary views:

  1. Per-depth node mass with 95% CI. For each tree, the fraction of its
     nodes at each depth d; across the sampled trees, the mean fraction vs
     depth with a 95% CI band. Both tree sets overlaid.
  2. Per-tree max-depth distribution. For each tree, its maximum node depth;
     across the sample, the fraction of trees attaining each max depth, with
     a 95% CI on each bar height. Shows whether max_depth=30 trees actually
     exceed the depth-4 cap (the key question).

CI method: normal approximation across trees (mean +/- 1.96 * SEM, where
SEM = sample_std / sqrt(n_trees)). Each tree contributes one independent
observation per depth bin, so the CLT across the ~thousands of sampled trees
is well-justified and far cheaper than a bootstrap. State this in the caption.

Each tree is one ``.pt`` file (``torch.load(weights_only=False)`` -> dict with
``format="cts_raw_pretrain_example_v5"``). We only read the ``depth`` tensor
(``torch.int16 [N]``, per-node ply-depth from the root).

Usage (from lmcos/):
    python analysis/tree_depth_histogram.py \\
        --trees-dir depth4=/scratch/.../lc0_trees \\
        --trees-dir depth30=/scratch/.../lc0_trees_maxdepth_30 \\
        --sample-n 2000 --out analysis/figures/tree_depth_histogram.png

Or as a module:
    PYTHONPATH=lmcos python -m cts.analysis.tree_depth_histogram ...
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._plots import analysis_style, save_fig

_FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Tab10-ish palette; index by tree-set order.
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def list_tree_files(trees_dir: str) -> list[str]:
    """Return all .pt files directly under trees_dir (sorted, deterministic)."""
    return sorted(
        os.path.join(trees_dir, f)
        for f in os.listdir(trees_dir)
        if f.endswith(".pt")
    )


def sample_tree_files(trees_dir: str, sample_n: int, seed: int) -> list[str]:
    """Deterministically subsample up to ``sample_n`` .pt paths from trees_dir.

    Uses ``random.Random(seed)`` so the choice is reproducible and independent
    of filesystem ordering.
    """
    files = list_tree_files(trees_dir)
    if sample_n is not None and 0 < sample_n < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, sample_n)
    return files


def load_tree_depths(path: str) -> np.ndarray | None:
    """Load one tree .pt and return its per-node depth array as int.

    Returns None on any failure (corrupt file, missing key, empty tree) so the
    caller can skip + count it.
    """
    try:
        t = torch.load(path, map_location="cpu", weights_only=False)
        depth = t["depth"]
        arr = np.asarray(depth.numpy() if hasattr(depth, "numpy") else depth, dtype=np.int64)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def load_set_depths(
    trees_dir: str, sample_n: int, seed: int, label: str
) -> tuple[list[np.ndarray], int, int]:
    """Load per-tree depth arrays for one tree set.

    Returns (depth_arrays, n_loaded, n_failed).
    """
    files = sample_tree_files(trees_dir, sample_n, seed)
    depth_arrays: list[np.ndarray] = []
    n_failed = 0
    for path in tqdm(files, desc=f"Loading {label}"):
        arr = load_tree_depths(path)
        if arr is None:
            n_failed += 1
        else:
            depth_arrays.append(arr)
    return depth_arrays, len(depth_arrays), n_failed


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def _mean_ci(per_tree: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95% half-width across trees (axis 0) by normal approx.

    per_tree: [n_trees, n_bins] of per-tree statistics.
    Returns (mean[n_bins], half_width[n_bins]) where the 95% CI is
    mean +/- half_width and half_width = 1.96 * std / sqrt(n_trees).
    """
    n = per_tree.shape[0]
    mean = per_tree.mean(axis=0)
    if n < 2:
        return mean, np.zeros_like(mean)
    sem = per_tree.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, 1.96 * sem


def per_depth_node_mass(
    depth_arrays: list[np.ndarray], max_depth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean per-depth node fraction across trees, with 95% CI.

    For each tree: fraction of its nodes at each depth d in [0, max_depth].
    Returns (depths[max_depth+1], mean_frac, ci_halfwidth).
    """
    n_bins = max_depth + 1
    per_tree = np.zeros((len(depth_arrays), n_bins), dtype=np.float64)
    for i, arr in enumerate(depth_arrays):
        counts = np.bincount(arr, minlength=n_bins)[:n_bins].astype(np.float64)
        total = counts.sum()
        if total > 0:
            per_tree[i] = counts / total
    mean, ci = _mean_ci(per_tree)
    return np.arange(n_bins), mean, ci


def per_tree_max_depth_dist(
    depth_arrays: list[np.ndarray], max_depth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distribution of per-tree maximum depth, with 95% CI on each bar.

    For each tree: an indicator vector (one-hot at its max depth). Mean across
    trees = fraction of trees whose max depth == d; CI is the normal-approx CI
    on that proportion across trees.
    Returns (depths, frac_with_maxdepth, ci_halfwidth, raw_max_per_tree).
    """
    n_bins = max_depth + 1
    raw_max = np.array([int(arr.max()) for arr in depth_arrays], dtype=np.int64)
    per_tree = np.zeros((len(depth_arrays), n_bins), dtype=np.float64)
    for i, m in enumerate(raw_max):
        if 0 <= m < n_bins:
            per_tree[i, m] = 1.0
    mean, ci = _mean_ci(per_tree)
    return np.arange(n_bins), mean, ci, raw_max


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_depth_distributions(
    sets: list[dict], out_path: str
) -> None:
    """Two-panel figure: per-depth node mass (left) and max-depth dist (right).

    ``sets`` is a list of dicts with keys: label, color, depths_mass, mass_mean,
    mass_ci, depths_max, max_mean, max_ci.
    """
    analysis_style()
    fig, (ax_mass, ax_max) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: per-depth node mass, mean +/- 95% CI band.
    for s in sets:
        d, m, ci = s["depths_mass"], s["mass_mean"], s["mass_ci"]
        ax_mass.plot(d, m, "-o", ms=3, color=s["color"], label=s["label"])
        ax_mass.fill_between(d, m - ci, m + ci, color=s["color"], alpha=0.2)
    ax_mass.set_xlabel("Node depth (ply from root)")
    ax_mass.set_ylabel("Mean fraction of nodes")
    ax_mass.set_title("Per-depth node mass")
    ax_mass.legend(framealpha=0.9)

    # Panel 2: per-tree max-depth distribution, grouped bars with 95% CI.
    n_sets = len(sets)
    bar_w = 0.8 / max(n_sets, 1)
    for j, s in enumerate(sets):
        d, m, ci = s["depths_max"], s["max_mean"], s["max_ci"]
        offset = (j - (n_sets - 1) / 2) * bar_w
        ax_max.bar(
            d + offset, m, bar_w, yerr=ci, capsize=2,
            color=s["color"], alpha=0.8, label=s["label"],
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )
    ax_max.set_xlabel("Tree max depth (ply)")
    ax_max.set_ylabel("Fraction of trees")
    ax_max.set_title("Per-tree max-depth distribution")
    ax_max.legend(framealpha=0.9)

    fig.suptitle(
        "lc0 search-tree node depth distribution  (shaded/bars: 95% CI across trees, normal approx)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_fig(out_path)


# --------------------------------------------------------------------------- #
# Stdout table
# --------------------------------------------------------------------------- #
def print_mass_table(sets: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("Per-depth node mass (mean fraction +/- 95% CI), by tree set")
    print(f"{'='*70}")
    for s in sets:
        print(f"\n[{s['label']}]  n_trees={s['n_trees']:,}  n_failed={s['n_failed']:,}")
        print(f"  raw max-depth across trees: "
              f"min={int(s['raw_max'].min())} median={int(np.median(s['raw_max']))} "
              f"max={int(s['raw_max'].max())} mean={s['raw_max'].mean():.2f}")
        print(f"  {'depth':>6}  {'mean_frac':>10}  {'95%CI_hw':>10}")
        for d, m, ci in zip(s["depths_mass"], s["mass_mean"], s["mass_ci"]):
            if m > 0 or ci > 0:
                print(f"  {int(d):>6}  {m:>10.5f}  {ci:>10.5f}")
    print(f"{'='*70}\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _parse_trees_dir(spec: str) -> tuple[str, str]:
    """Parse a --trees-dir value of form ``label=path`` (or bare ``path``)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    path = spec.strip()
    return Path(path).name or path, path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--trees-dir", action="append", required=True, metavar="LABEL=PATH",
        help="Tree-set directory of .pt files, as label=path. Repeatable to overlay sets.",
    )
    p.add_argument(
        "--sample-n", type=int, default=2000,
        help="Random-subsample this many .pt per dir for speed (seeded). Default 2000.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", default=str(_FIGURES_DIR / "tree_depth_histogram.png"),
        help="Output PNG path.",
    )
    args = p.parse_args(argv)

    parsed = [_parse_trees_dir(s) for s in args.trees_dir]

    # Determine a common max-depth across all sets so panels share an x-axis.
    loaded: list[dict] = []
    global_max_depth = 0
    for label, path in parsed:
        depth_arrays, n_loaded, n_failed = load_set_depths(
            path, args.sample_n, args.seed, label
        )
        if n_loaded == 0:
            print(f"WARNING: no trees loaded for set '{label}' at {path} "
                  f"(n_failed={n_failed}); skipping.")
            continue
        set_max = max(int(arr.max()) for arr in depth_arrays)
        global_max_depth = max(global_max_depth, set_max)
        loaded.append({"label": label, "path": path,
                       "depth_arrays": depth_arrays,
                       "n_trees": n_loaded, "n_failed": n_failed})

    if not loaded:
        print("No tree sets loaded; nothing to plot.")
        return

    sets: list[dict] = []
    for k, entry in enumerate(loaded):
        d_mass, mass_mean, mass_ci = per_depth_node_mass(
            entry["depth_arrays"], global_max_depth
        )
        d_max, max_mean, max_ci, raw_max = per_tree_max_depth_dist(
            entry["depth_arrays"], global_max_depth
        )
        sets.append({
            "label": entry["label"],
            "color": _COLORS[k % len(_COLORS)],
            "n_trees": entry["n_trees"],
            "n_failed": entry["n_failed"],
            "depths_mass": d_mass, "mass_mean": mass_mean, "mass_ci": mass_ci,
            "depths_max": d_max, "max_mean": max_mean, "max_ci": max_ci,
            "raw_max": raw_max,
        })

    print_mass_table(sets)
    plot_depth_distributions(sets, args.out)


if __name__ == "__main__":
    main()
