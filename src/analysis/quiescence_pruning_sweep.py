"""T3 quiescence + pruning sweep (plan.md / diagnosis.md §1.3, 2026-07-08).

Cheap per-tree diagnostics computable directly from freshly-generated raw ``.pt`` tree
files -- no packing/materialize needed. For each grid point (a ``sf_search_limit_nodes``
value and/or a pruning config) we measure, per tree:

- ``mean_abs_delta_halt_reward``: mean |consecutive-step delta| of the oracle's
  per-step ``halt_reward`` trace -- a cheap proxy for how "jagged"/noisy the leaf-eval
  signal is (diagnosis.md §1.3's quiescence/horizon-effect hypothesis: nodes=1 gives
  Stockfish's search no room to run its own built-in quiescence extension).
- ``argmax_step`` / ``thinking_helps``: the same budgeted-oracle argmax stop step and
  argmax>2 "thinking helps" yield criterion used by ``filter_argmax.py`` (same oracle
  config, so numbers here are directly comparable to the production filter's ~24.7%).
- ``final_depth`` / ``final_width``: last-step tree height/max-level-width (T2's
  width-vs-depth axis), to check whether pruning redirects search budget from width to
  depth as diagnosis.md's companion pruning knob is meant to.

Kept as small, independently-testable pure functions (``trajectory_metrics``,
``aggregate_grid_metrics``) plus thin I/O wrappers (``load_tree_metrics``,
``scan_grid_point``), per the TDD-lite convention used elsewhere in ``analysis/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.utils.plots import save_pdf_png
from cts.data.preprocess_gnn.teacher_targets import load_raw_pretrain_record
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.data.preprocess_mc.pack import build_compact_trajectory, budgeted_oracle_from_trajectory

# Same oracle regime as filter_argmax.py / mc_pack (labnotebook 2026-07-07's validated,
# non-degenerate regime: linear time cost, zero maintenance) so the argmax/yield numbers
# produced here are directly comparable to the production "thinking helps" filter.
ORACLE_CONFIG = BudgetedOracleConfig(
    time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0
)
STARTING_BUDGET = 96


def _heights_widths_from_depth(depth: torch.Tensor, node_cutoffs: list[int]) -> tuple[list[int], list[int]]:
    """Per-step (height, width) from a trajectory's node depths + per-step node cutoffs
    (root-first order). Same logic as ``analysis.evaluate._derive_step_stats``, reimplemented
    locally so this diagnostic module doesn't pull in evaluate.py's training-time dependencies."""
    heights, widths = [], []
    for node_cutoff in node_cutoffs:
        prefix_depth = depth[:node_cutoff]
        heights.append(int(prefix_depth.max().item()))
        widths.append(int(torch.bincount(prefix_depth).max().item()))
    return heights, widths


def trajectory_metrics(
    trajectory: dict[str, Any],
    *,
    argmax_threshold: int = 2,
    oracle_config: BudgetedOracleConfig = ORACLE_CONFIG,
    starting_budget: int = STARTING_BUDGET,
) -> dict[str, Any]:
    """Pure function: per-tree metrics from a ``build_compact_trajectory()``-shaped dict.

    Requires ``trajectory`` to carry ``halt_rewards``, ``depth``, and ``step_node_cutoffs``
    (exactly what ``build_compact_trajectory`` returns), so it's testable on small synthetic
    dicts without touching disk or a real Stockfish-generated tree.
    """
    halt_rewards = np.asarray(trajectory["halt_rewards"], dtype=np.float64)
    if halt_rewards.size < 2:
        mean_abs_delta = 0.0
    else:
        mean_abs_delta = float(np.mean(np.abs(np.diff(halt_rewards))))

    depth = torch.as_tensor(trajectory["depth"])
    node_cutoffs = [int(v) for v in trajectory["step_node_cutoffs"]]
    heights, widths = _heights_widths_from_depth(depth, node_cutoffs)

    policy = budgeted_oracle_from_trajectory(trajectory, starting_budget, oracle_config)
    argmax_step = int(policy.optimal_stop_step)

    return {
        "n_steps": int(halt_rewards.size),
        "mean_abs_delta_halt_reward": mean_abs_delta,
        "argmax_step": argmax_step,
        "thinking_helps": bool(argmax_step > argmax_threshold),
        "final_depth": int(heights[-1]),
        "final_width": int(widths[-1]),
    }


def load_tree_metrics(path: str | Path, *, argmax_threshold: int = 2) -> Optional[dict[str, Any]]:
    """Load one raw tree record from disk and compute its per-tree metrics.

    Returns ``None`` if the tree has no usable root expansion -- mirrors
    ``filter_argmax.argmax_step``'s skip condition exactly, so trees this sweep drops are
    the same ones the production filter would drop.
    """
    record = load_raw_pretrain_record(str(path))
    trajectory = build_compact_trajectory(record, source_path=str(path))
    if trajectory is None or trajectory["num_steps"] <= 0:
        return None
    return trajectory_metrics(trajectory, argmax_threshold=argmax_threshold)


def thinking_helps_basenames(trees_dir: str | Path, *, argmax_threshold: int = 2) -> list[str]:
    """Basenames of trees under ``trees_dir`` whose oracle argmax stop step exceeds
    ``argmax_threshold`` -- i.e. the "thinking helps" survivors, computed with the exact same
    per-tree logic ``scan_grid_point``/``filter_argmax.py`` use. Returns basenames (not full
    paths) so the result can be written straight to a ``split.include_list`` file, matching the
    production ``argmax_filter.slurm`` -> ``split.include_list`` contract.
    """
    survivors: list[str] = []
    for path in sorted(Path(trees_dir).rglob("*.pt")):
        try:
            metrics = load_tree_metrics(path, argmax_threshold=argmax_threshold)
        except Exception:
            continue
        if metrics is not None and metrics["thinking_helps"]:
            survivors.append(path.name)
    return survivors


def aggregate_grid_metrics(per_tree: list[dict[str, Any]]) -> dict[str, Any]:
    """Grid-point-level aggregate over a list of ``trajectory_metrics()`` dicts.

    Equal-weight per tree (not pooled across all steps) so long trees don't dominate the
    noise-proxy average -- each tree contributes one "how jagged is this tree's halt-reward
    trace" number, and the grid point reports the mean of those.
    """
    if not per_tree:
        return {
            "n_trees": 0,
            "mean_abs_delta_halt_reward": float("nan"),
            "yield_pct": float("nan"),
            "median_final_depth": float("nan"),
            "median_final_width": float("nan"),
        }
    deltas = np.array([m["mean_abs_delta_halt_reward"] for m in per_tree])
    yields = np.array([m["thinking_helps"] for m in per_tree])
    depths = np.array([m["final_depth"] for m in per_tree])
    widths = np.array([m["final_width"] for m in per_tree])
    return {
        "n_trees": len(per_tree),
        "mean_abs_delta_halt_reward": float(deltas.mean()),
        "yield_pct": float(100.0 * yields.mean()),
        "median_final_depth": float(np.median(depths)),
        "median_final_width": float(np.median(widths)),
    }


def scan_grid_point(trees_dir: str | Path, *, argmax_threshold: int = 2,
                     max_trees: Optional[int] = None) -> dict[str, Any]:
    """Walk every ``.pt`` tree under ``trees_dir``, compute per-tree metrics, and aggregate.

    Skips (and counts) unloadable/unusable trees rather than failing the whole scan -- a
    handful of malformed roots shouldn't sink an otherwise-informative grid point.
    """
    paths = sorted(Path(trees_dir).rglob("*.pt"))
    if max_trees is not None:
        paths = paths[:max_trees]
    per_tree: list[dict[str, Any]] = []
    n_skipped = 0
    for path in paths:
        try:
            metrics = load_tree_metrics(path, argmax_threshold=argmax_threshold)
        except Exception:
            n_skipped += 1
            continue
        if metrics is None:
            n_skipped += 1
            continue
        per_tree.append(metrics)
    agg = aggregate_grid_metrics(per_tree)
    agg["n_files"] = len(paths)
    agg["n_skipped"] = n_skipped
    return agg


# ===========================================================================
# Plots
# ===========================================================================
def plot_quiescence_sweep(
    grid: dict[int, dict[str, Any]],
    gen_cost_sec_per_tree: dict[int, float],
    *,
    out_dir: str | Path,
    conclusion: str,
    base: str = "t3_quiescence_sweep",
) -> str:
    """1x3 panel over ``sf_search_limit_nodes``: generation cost, noise proxy, yield %.

    ``grid`` maps ``sf_search_limit_nodes -> aggregate_grid_metrics()`` output;
    ``gen_cost_sec_per_tree`` maps the same keys to measured wall-clock sec/tree.
    """
    xs = sorted(grid.keys())
    cost = [gen_cost_sec_per_tree[x] for x in xs]
    noise = [grid[x]["mean_abs_delta_halt_reward"] for x in xs]
    yield_pct = [grid[x]["yield_pct"] for x in xs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(conclusion, fontsize=10.5, y=1.04, wrap=True)

    axes[0].plot(xs, cost, marker="o", color="#3F4DA0")
    axes[0].set_title("Generation cost")
    axes[0].set_xlabel("sf_search_limit_nodes")
    axes[0].set_ylabel("sec / tree")

    axes[1].plot(xs, noise, marker="o", color="#C0392B")
    axes[1].set_title("Leaf-eval noise proxy")
    axes[1].set_xlabel("sf_search_limit_nodes")
    axes[1].set_ylabel("mean |Δhalt_reward| (consecutive steps)")

    axes[2].plot(xs, yield_pct, marker="o", color="#1E8449")
    axes[2].set_title('"Thinking helps" yield (argmax>2)')
    axes[2].set_xlabel("sf_search_limit_nodes")
    axes[2].set_ylabel("yield %")

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(color="#E3E7EB", lw=1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_pdf_png(fig, str(out_dir), base, dpi=200)


def plot_pruning_sweep(
    grid: dict[str, dict[str, Any]],
    gen_cost_sec_per_tree: dict[str, float],
    *,
    out_dir: str | Path,
    conclusion: str,
    base: str = "t3_pruning_sweep",
    x_order: Optional[list[str]] = None,
) -> str:
    """1x4 panel over prune_epsilon labels (e.g. "off","0.1","0.3"): generation cost,
    noise proxy, yield %, and the resulting median final depth/width shift (T2's
    width-vs-depth imbalance -- does pruning redirect budget from width to depth?)."""
    xs = x_order if x_order is not None else list(grid.keys())
    cost = [gen_cost_sec_per_tree[x] for x in xs]
    noise = [grid[x]["mean_abs_delta_halt_reward"] for x in xs]
    yield_pct = [grid[x]["yield_pct"] for x in xs]
    depth = [grid[x]["median_final_depth"] for x in xs]
    width = [grid[x]["median_final_width"] for x in xs]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    fig.suptitle(conclusion, fontsize=10.5, y=1.06, wrap=True)

    axes[0].plot(xs, cost, marker="o", color="#3F4DA0")
    axes[0].set_title("Generation cost")
    axes[0].set_xlabel("prune_epsilon")
    axes[0].set_ylabel("sec / tree")

    axes[1].plot(xs, noise, marker="o", color="#C0392B")
    axes[1].set_title("Leaf-eval noise proxy")
    axes[1].set_xlabel("prune_epsilon")
    axes[1].set_ylabel("mean |Δhalt_reward|")

    axes[2].plot(xs, yield_pct, marker="o", color="#1E8449")
    axes[2].set_title('"Thinking helps" yield (argmax>2)')
    axes[2].set_xlabel("prune_epsilon")
    axes[2].set_ylabel("yield %")

    ax3 = axes[3]
    ax3b = ax3.twinx()
    ax3.plot(xs, depth, marker="o", color="#8E44AD", label="median final depth")
    ax3b.plot(xs, width, marker="s", color="#D68910", label="median final width")
    ax3.set_title("Depth/width shift")
    ax3.set_xlabel("prune_epsilon")
    ax3.set_ylabel("median final depth", color="#8E44AD")
    ax3b.set_ylabel("median final width", color="#D68910")

    for ax in axes:
        ax.grid(color="#E3E7EB", lw=1)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_pdf_png(fig, str(out_dir), base, dpi=200)
