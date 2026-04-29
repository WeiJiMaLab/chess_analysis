"""Paper Section 1 figures: the model learns near-optimal metacontrol.

Generates Figures 1a-1d comparing all five candidate models against baselines.
Runs on pre-computed diagnostics JSONLs -- no GPU needed.

Usage:
    python paper_section1_figures.py \
        --diagnostics slw01:path/slw01.jsonl reweight_w4:path/rw4.jsonl ... \
        --output-dir figures/section1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from budgeted_controller_oracle import (
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
    return_for_stop_step,
    time_cost,
)
from paper_figure_utils import (
    BUDGET_BUCKET_ORDER,
    MODEL_COLORS,
    MODEL_MARKERS,
    load_multiple_diagnostics,
    padded_return_matrix,
    recover_oracle_config_from_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline computations (Fig 1a)
# ---------------------------------------------------------------------------


def _baseline_stats(
    diagnostics: List[Dict[str, Any]],
    config: BudgetedOracleConfig,
    stop_steps: np.ndarray,
) -> Tuple[float, float]:
    """(avg_expansions, avg_regret) for given per-episode stop steps."""
    oracle_values = np.empty(len(diagnostics), dtype=np.float64)
    returns = np.empty(len(diagnostics), dtype=np.float64)
    for i, ep in enumerate(diagnostics):
        oracle_values[i] = ep["oracle_value"]
        returns[i] = return_for_stop_step(
            ep["halt_rewards"], ep["tree_sizes"], ep["time_budgets"],
            int(stop_steps[i]), config,
        )
    return float(stop_steps.astype(np.float64).mean()), float((oracle_values - returns).mean())


def compute_constant_prob_curve(
    diagnostics: List[Dict[str, Any]],
    config: BudgetedOracleConfig,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep geometric stopping probability p in (0,1).

    Returns (avg_expansions[n_points], avg_regret[n_points]).
    """
    returns_mat, lengths = padded_return_matrix(diagnostics, config)
    N, T = returns_mat.shape
    oracle_values = np.array([ep["oracle_value"] for ep in diagnostics], dtype=np.float64)

    last_returns = returns_mat[np.arange(N), lengths - 1]
    last_steps = (lengths - 1).astype(np.float64)
    step_mat = np.minimum(
        np.arange(T, dtype=np.float64)[None, :],
        last_steps[:, None],
    )

    ps = np.linspace(0.01, 0.99, n_points)
    out_exp = np.empty(n_points)
    out_reg = np.empty(n_points)

    for j, p in enumerate(ps):
        w = p * (1.0 - p) ** np.arange(T, dtype=np.float64)
        tail = (1.0 - p) ** T  # mass beyond T, assigned to last valid step
        exp_ret = returns_mat @ w + last_returns * tail
        exp_stp = step_mat @ w + last_steps * tail
        out_exp[j] = exp_stp.mean()
        out_reg[j] = (oracle_values - exp_ret).mean()

    return out_exp, out_reg


def _load_q_traces(
    diagnostics: List[Dict[str, Any]],
    source_path_rewrite: Optional[Tuple[str, str]] = None,
) -> Dict[str, List[List[float]]]:
    """Load oracle_root_q_trace for unique source paths from raw examples."""
    from cts_pretrain import load_pretrain_example

    unique = sorted({ep["source_path"] for ep in diagnostics})
    logger.info("Loading Q-traces from %d unique source paths...", len(unique))
    traces: Dict[str, List[List[float]]] = {}
    failed = 0
    for i, src in enumerate(unique):
        path = src
        if source_path_rewrite:
            path = path.replace(source_path_rewrite[0], source_path_rewrite[1])
        try:
            ex = load_pretrain_example(path)
            if ex.oracle_root_q_trace:
                traces[src] = ex.oracle_root_q_trace
        except Exception:
            failed += 1
        if (i + 1) % 500 == 0:
            logger.info("  %d / %d loaded", i + 1, len(unique))
    if failed:
        logger.warning("Failed to load %d / %d source paths", failed, len(unique))
    logger.info("Q-traces loaded for %d paths", len(traces))
    return traces


def compute_value_gap_curve(
    diagnostics: List[Dict[str, Any]],
    q_traces: Dict[str, List[List[float]]],
    config: BudgetedOracleConfig,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep value-gap threshold. Returns (avg_expansions, avg_regret)."""
    valid_eps: List[Dict[str, Any]] = []
    gap_traces_list: List[np.ndarray] = []
    for ep in diagnostics:
        qt = q_traces.get(ep["source_path"])
        if qt is None:
            continue
        L = int(ep["episode_length"])
        n = min(L, len(qt))
        gaps = np.full(L, -np.inf, dtype=np.float64)
        for t in range(n):
            qs = sorted(qt[t], reverse=True)
            gaps[t] = (qs[0] - qs[1]) if len(qs) >= 2 else np.inf
        gap_traces_list.append(gaps)
        valid_eps.append(ep)

    if not valid_eps:
        return np.array([]), np.array([])

    returns_mat, lengths = padded_return_matrix(valid_eps, config)
    N, T = returns_mat.shape
    oracle_values = np.array([ep["oracle_value"] for ep in valid_eps], dtype=np.float64)

    # Pad gap traces; -inf beyond episode so those steps never trigger halt
    gap_mat = np.full((N, T), -np.inf, dtype=np.float64)
    for i, g in enumerate(gap_traces_list):
        gap_mat[i, : len(g)] = g

    # Ensure halt at last valid step as fallback
    last_mask = np.zeros((N, T), dtype=bool)
    last_mask[np.arange(N), lengths - 1] = True

    thresholds = np.linspace(0.0, 0.5, n_points)
    out_exp = np.empty(n_points)
    out_reg = np.empty(n_points)

    for j, theta in enumerate(thresholds):
        exceeds = (gap_mat >= theta) | last_mask
        stops = exceeds.argmax(axis=1)
        rets = returns_mat[np.arange(N), stops]
        out_exp[j] = stops.mean()
        out_reg[j] = (oracle_values - rets).mean()

    return out_exp, out_reg


# ---------------------------------------------------------------------------
# Fig 1a: Pareto curve
# ---------------------------------------------------------------------------


def fig_1a_pareto(
    all_diag: Dict[str, List[Dict[str, Any]]],
    config: BudgetedOracleConfig,
    output_dir: Path,
    q_traces: Optional[Dict[str, List[List[float]]]] = None,
) -> None:
    ref = next(iter(all_diag.values()))
    fig, ax = plt.subplots(figsize=(8, 5))

    # Fixed baselines
    ah = _baseline_stats(ref, config, np.zeros(len(ref), dtype=np.int64))
    nh = _baseline_stats(
        ref, config,
        np.array([len(e["halt_rewards"]) - 1 for e in ref], dtype=np.int64),
    )
    ax.plot(*ah, "kx", markersize=10, label="Always halt")
    ax.plot(*nh, "k+", markersize=10, label="Never halt")

    # Sweep baselines
    cp_x, cp_y = compute_constant_prob_curve(ref, config)
    ax.plot(cp_x, cp_y, "k--", alpha=0.6, label="Constant prob.")

    if q_traces and len(q_traces) > 0:
        vg_x, vg_y = compute_value_gap_curve(ref, q_traces, config)
        if len(vg_x) > 0:
            ax.plot(vg_x, vg_y, "k:", alpha=0.6, label="Value-gap threshold")

    # Model points
    for label, diag in all_diag.items():
        mx = float(np.mean([e["predicted_stop_step"] for e in diag]))
        my = float(np.mean([e["regret"] for e in diag]))
        ax.plot(mx, my, marker=MODEL_MARKERS.get(label, "o"),
                color=MODEL_COLORS.get(label, "gray"), markersize=12,
                markeredgecolor="black", markeredgewidth=0.5, label=label, zorder=5)

    ax.set_xlabel("Average expansions")
    ax.set_ylabel("Average regret")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_1a_pareto.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_1a_pareto.pdf")


# ---------------------------------------------------------------------------
# Fig 1b: Budget-conditioned behavior
# ---------------------------------------------------------------------------


def fig_1b_budget_behavior(
    all_diag: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    n_trees: int = 6,
) -> None:
    # Index episodes by source_path and model
    source_index: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for label, diag in all_diag.items():
        for ep in diag:
            source_index[ep["source_path"]][label].append(ep)

    # Select trees present in all models with enough episodes
    all_labels = set(all_diag.keys())
    candidates = []
    for src, per_model in source_index.items():
        if set(per_model.keys()) != all_labels:
            continue
        ref_eps = next(iter(per_model.values()))
        if len(ref_eps) < 6:
            continue
        oracle_stops = [e["oracle_stop_step"] for e in ref_eps]
        candidates.append((float(np.var(oracle_stops)), src))

    if not candidates:
        logger.warning("No trees qualify for Fig 1b; skipping")
        return

    candidates.sort(reverse=True)
    selected = [src for _, src in candidates[:n_trees]]

    ncols = 3
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, src in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]
        per_model = source_index[src]

        # Oracle (shared across models)
        ref_eps = sorted(next(iter(per_model.values())), key=lambda e: e["starting_budget"])
        budgets = [e["starting_budget"] for e in ref_eps]
        ax.plot(budgets, [e["oracle_stop_step"] for e in ref_eps],
                "k-o", markersize=4, label="Oracle", zorder=4)

        for label in all_diag:
            eps = sorted(per_model.get(label, []), key=lambda e: e["starting_budget"])
            if eps:
                ax.plot(
                    [e["starting_budget"] for e in eps],
                    [e["predicted_stop_step"] for e in eps],
                    "-", color=MODEL_COLORS.get(label, "gray"),
                    marker=".", markersize=3, label=label, alpha=0.8,
                )

        ax.set_xlabel("Budget")
        ax.set_ylabel("Stop step")
        ax.set_title(Path(src).stem[:25], fontsize=9)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_1b_budget_behavior.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_1b_budget_behavior.pdf")


# ---------------------------------------------------------------------------
# Fig 1c: Regret decomposition by budget bucket
# ---------------------------------------------------------------------------


def fig_1c_regret_decomposition(
    all_diag: Dict[str, List[Dict[str, Any]]],
    config: BudgetedOracleConfig,
    output_dir: Path,
) -> None:
    labels = list(all_diag.keys())
    n_models = len(labels)
    n_buckets = len(BUDGET_BUCKET_ORDER)
    bucket_idx = {b: i for i, b in enumerate(BUDGET_BUCKET_ORDER)}

    halt_means = np.zeros((n_models, n_buckets))
    time_means = np.zeros((n_models, n_buckets))

    for mi, label in enumerate(labels):
        by_bucket_halt: Dict[str, List[float]] = defaultdict(list)
        by_bucket_time: Dict[str, List[float]] = defaultdict(list)
        for ep in all_diag[label]:
            bname = ep["budget_bucket_name"]
            if bname not in bucket_idx:
                continue
            o = int(ep["oracle_stop_step"])
            p = int(ep["predicted_stop_step"])
            hr = [float(v) for v in ep["halt_rewards"]]
            tb = [int(v) for v in ep["time_budgets"]]
            halt_term = hr[o] - hr[p]
            oracle_tc = sum(time_cost(tb[t], config) for t in range(o))
            pred_tc = sum(time_cost(tb[t], config) for t in range(p))
            by_bucket_halt[bname].append(halt_term)
            by_bucket_time[bname].append(pred_tc - oracle_tc)

        for bname, bi in bucket_idx.items():
            if by_bucket_halt[bname]:
                halt_means[mi, bi] = float(np.mean(by_bucket_halt[bname]))
                time_means[mi, bi] = float(np.mean(by_bucket_time[bname]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(n_buckets)
    w = 0.8 / n_models

    for mi, label in enumerate(labels):
        offset = (mi - n_models / 2 + 0.5) * w
        c = MODEL_COLORS.get(label, "gray")
        ax1.bar(x + offset, halt_means[mi], w, color=c, label=label)
        ax2.bar(x + offset, time_means[mi], w, color=c, label=label)

    for ax, title in [(ax1, "Halt-reward component"), (ax2, "Time-cost component")]:
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGET_BUCKET_ORDER, rotation=15, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="k", linewidth=0.5)
    ax1.set_ylabel("Mean regret component")
    ax2.legend(fontsize=7)

    fig.suptitle("Regret decomposition by budget bucket")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_1c_regret_decomposition.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_1c_regret_decomposition.pdf")


# ---------------------------------------------------------------------------
# Fig 1d: Return CDF
# ---------------------------------------------------------------------------


def fig_1d_return_cdf(
    all_diag: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    # Oracle CDF (shared)
    ref = next(iter(all_diag.values()))
    oracle_vals = np.sort([e["oracle_value"] for e in ref])
    ax.plot(oracle_vals, np.linspace(0, 1, len(oracle_vals)),
            "k--", linewidth=2, label="Oracle", alpha=0.8)

    for label, diag in all_diag.items():
        vals = np.sort([e["predicted_value"] for e in diag])
        ax.plot(vals, np.linspace(0, 1, len(vals)),
                color=MODEL_COLORS.get(label, "gray"), linewidth=1.5, label=label)

    ax.set_xlabel("Return")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_1d_return_cdf.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_1d_return_cdf.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 1 paper figures")
    parser.add_argument(
        "--diagnostics", nargs="+", required=True, metavar="LABEL:PATH",
        help="label:path pairs for each model's diagnostics JSONL",
    )
    parser.add_argument("--oracle-log", help="Training log to recover oracle config")
    parser.add_argument("--oracle-config-json", help="JSON with oracle metadata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rewrite-from", help="Source-path prefix to replace")
    parser.add_argument("--rewrite-to", help="Replacement prefix")
    parser.add_argument(
        "--skip-value-gap", action="store_true",
        help="Skip value-gap baseline (avoids loading raw .pt files)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Parse label:path specs
    pairs = []
    for spec in args.diagnostics:
        if ":" not in spec:
            parser.error(f"Expected label:path, got: {spec}")
        label, path = spec.split(":", 1)
        pairs.append((label, path))

    logger.info("Loading diagnostics for %d models...", len(pairs))
    all_diag = load_multiple_diagnostics(pairs)
    for label, diag in all_diag.items():
        logger.info("  %s: %d episodes", label, len(diag))

    # Oracle config
    if args.oracle_config_json:
        with open(args.oracle_config_json) as f:
            config = budgeted_oracle_config_from_metadata(json.load(f))
        if config is None:
            parser.error("Could not parse oracle config from JSON")
    elif args.oracle_log:
        config = recover_oracle_config_from_log(args.oracle_log)
    else:
        config = BudgetedOracleConfig()
        logger.info("Using default oracle config")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Q-traces for value-gap baseline (optional)
    q_traces = None
    if not args.skip_value_gap:
        rewrite = None
        if args.rewrite_from and args.rewrite_to:
            rewrite = (args.rewrite_from, args.rewrite_to)
        ref = next(iter(all_diag.values()))
        try:
            q_traces = _load_q_traces(ref, rewrite)
        except Exception as exc:
            logger.warning("Q-trace loading failed: %s; skipping value-gap baseline", exc)

    fig_1a_pareto(all_diag, config, output_dir, q_traces)
    fig_1b_budget_behavior(all_diag, output_dir)
    fig_1c_regret_decomposition(all_diag, config, output_dir)
    fig_1d_return_cdf(all_diag, output_dir)

    logger.info("All Section 1 figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
