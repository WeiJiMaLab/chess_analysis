"""Meta-controller assessment plots (R-EVALUATE) — mechanism behind ``outputs/reports/normative.md``.

Does the learned ``z_t`` halting head beat hand-crafted tree-stats and a fixed stop, and in WHICH
cost regime? Four figures answer it (:func:`plot_regret_effort_frontier`, :func:`plot_r_decodability`,
:func:`plot_delta_regret_vs_zt` — the latter renders both the lambda- and maintenance-sweep panels).

The halt policies are stop-controller modules, fit on the train split and scored per-episode on eval:
  * SingleHalt* — one parameter ``k`` (fixed stop step), fit by minimizing fit-split regret
    (:func:`fit_singlehalt_stop`); the frontier of fixed-``k`` rules is its parameter space.
  * Stats-Controller / Zt-Controller — MLP readouts (``cts.models.readout.build_advantage_head``)
    trained by the EXACT expected-return objective via the DEPLOYED trainer
    (``cts.train.pg_controller_train.fit_readout_pg``) — no bespoke training loop here.

Design: readouts see STEPS-TAKEN, not budget (``T_t``). All fitting is on the VALIDATION split with a
70/30 EPISODE split (the train materialized cache is ``shuffle=True`` so its ``z_t`` rows don't align
to episodes; validation is ``shuffle=False`` and does — alignment self-checked). Cost regimes are
applied at ANALYSIS TIME: packed ``halt_rewards``/``tree_sizes`` are cost-independent, so return curves
are recomputed for any cost config without repacking.

    python -m analysis.evaluate --which all \
        --packed-root <mc_packed> --cache <validation_cache.pt> \
        --out-dir outputs/figures/minply15_maxply75/normative

``_load_split_episodes`` / ``_oracle_config`` / ``_load_materialized_cache_unchecked`` are the shared
data-loaders (also imported by ``cts.analysis.zt_probe`` and ``cts.train.pg_controller_train``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
    continue_cost,
)
from cts.models.readout import build_advantage_head, stop_step_from_advantages
from cts.stats import bootstrap_ci
from cts.train.controller_train import _load_materialized_cache_unchecked
from cts.train.pg_controller_train import fit_readout_pg
from analysis.utils.plots import save_pdf_png
from analysis.utils.helpers import MAIN_COLOR


# ===========================================================================
# Data loading (shared with cts.analysis.zt_probe + cts.train.pg_controller_train)
# ===========================================================================
def _derive_step_stats(traj_depth: torch.Tensor, node_cutoffs: list[int]) -> tuple[list[int], list[int]]:
    """Per-step (height, width) from a trajectory's node depths + per-step node cutoffs (root-first order)."""
    heights, widths = [], []
    for node_cutoff in node_cutoffs:
        prefix_depth = traj_depth[:node_cutoff]
        heights.append(int(prefix_depth.max().item()))
        widths.append(int(torch.bincount(prefix_depth).max().item()))
    return heights, widths


def _action_gaps_for_trajectory(source_path: str, first_decision_expansion_count: int, num_steps: int) -> list[float]:
    """Per-step action gap (top1 − top2 of the root's candidate-move Q values) for one trajectory,
    read directly from the RAW tree snapshot — NOT from the packed shard (mc_pack never propagates
    ``oracle_root_q_trace`` into the packed format; see AGController investigation, 2026-07-09).

    Alignment: the packed trajectory's step 0 is the raw record's row
    ``first_decision_expansion_count - 1`` (mirrors ``pack.py::_build_compact_trajectory``'s own
    ``expansion_parent_ids[root_rank]`` convention) — verified against real packed/raw data
    (packed ``num_steps`` exactly spans ``raw_step ∈ [fdec-1, fdec-1+num_steps-1]``).
    """
    from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
    record = RawPretrainExampleRecord.load(source_path)
    trace = record.oracle_root_q_trace.float()
    start = first_decision_expansion_count - 1
    window = trace[start:start + num_steps]
    if window.shape[0] < num_steps:  # pad short trailing steps (shouldn't normally happen) with the last row
        pad = window[-1:].expand(num_steps - window.shape[0], -1) if window.shape[0] else torch.zeros(num_steps, trace.shape[1])
        window = torch.cat([window, pad], dim=0)
    k = min(2, window.shape[1])
    top2 = torch.topk(window, k=k, dim=1).values
    gap = (top2[:, 0] - top2[:, 1]) if k > 1 else torch.zeros(num_steps)
    return gap.tolist()


def _load_split_episodes(packed_root: Path, split: str, max_episodes: int | None = None,
                         load_action_gaps: bool = False) -> list[dict[str, Any]]:
    """Walk one split's packed shards -> per-episode dicts (halt_rewards / tree_sizes / time_budgets /
    oracle_value / oracle_stop_step / starting_budget + derived per-step heights / widths).

    ``load_action_gaps``, if set, additionally re-opens each episode's RAW tree snapshot (path
    recovered from the packed shard's own ``trajectory_source_paths`` — see
    ``_action_gaps_for_trajectory``) to compute a per-step action-gap trace for AGController. Off by
    default since it's extra I/O the other two callers (decodability's steps/stats path, and any
    caller that doesn't need AG) don't need."""
    shard_paths = sorted((packed_root / split).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.pt under {packed_root / split}")
    episodes: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        p = torch.load(shard_path, weights_only=False)
        episode_step_ptr, episode_trajectory_index = p["episode_step_ptr"], p["episode_trajectory_index"]
        trajectory_step_ptr, trajectory_node_ptr = p["trajectory_step_ptr"], p["trajectory_node_ptr"]
        step_node_cutoffs, trajectory_halt_rewards = p["step_node_cutoffs"], p["trajectory_halt_rewards"]
        oracle_stop_steps, oracle_values = p["oracle_stop_steps"], p["oracle_values"]
        starting_budgets, depth = p["starting_budgets"], p["depth"]
        source_paths = p.get("trajectory_source_paths")
        first_decision_counts = p.get("first_decision_expansion_counts")
        gap_cache: dict[int, list[float]] = {}  # keyed by traj -- several episodes/budget-buckets can share one tree
        for i in range(int(p["num_episodes"])):
            num_steps = int(episode_step_ptr[i + 1].item()) - int(episode_step_ptr[i].item())
            traj = int(episode_trajectory_index[i].item())
            traj_step_begin = int(trajectory_step_ptr[traj].item())
            node_begin, node_end = int(trajectory_node_ptr[traj].item()), int(trajectory_node_ptr[traj + 1].item())
            starting_budget = int(starting_budgets[i].item())
            node_cutoffs = step_node_cutoffs[traj_step_begin:traj_step_begin + num_steps].tolist()
            heights, widths = _derive_step_stats(depth[node_begin:node_end], node_cutoffs)
            ep = {
                "halt_rewards": trajectory_halt_rewards[traj_step_begin:traj_step_begin + num_steps].tolist(),
                "tree_sizes": node_cutoffs, "heights": heights, "widths": widths,
                "time_budgets": list(range(starting_budget, starting_budget - num_steps, -1)),
                "oracle_stop_step": int(oracle_stop_steps[i].item()),
                "oracle_value": float(oracle_values[i].item()), "starting_budget": starting_budget,
                # Globally-unique source-tree id (shard-local trajectory index is NOT unique across
                # shards) — required so `_split` can partition by tree, not by episode, and never leak
                # a tree's other episodes across the fit/eval boundary.
                "trajectory_key": f"{shard_path.name}#{traj}",
            }
            if load_action_gaps:
                if traj not in gap_cache:
                    full_gaps = _action_gaps_for_trajectory(
                        source_paths[traj], int(first_decision_counts[traj].item()),
                        int(trajectory_step_ptr[traj + 1].item() - trajectory_step_ptr[traj].item()))
                    gap_cache[traj] = full_gaps
                # this episode's own steps are the trajectory's FIRST num_steps (budget-truncated view)
                ep["action_gaps"] = gap_cache[traj][:num_steps]
            episodes.append(ep)
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
    return episodes


def _oracle_config(packed_root: Path) -> BudgetedOracleConfig:
    """Reconstruct the exact oracle config the pack used, from the manifest metadata."""
    manifest = json.loads((packed_root / "validation_manifest.json").read_text())
    config = budgeted_oracle_config_from_metadata(manifest)
    if config is None:
        raise ValueError("manifest carries no budgeted-oracle metadata")
    return config




# ===========================================================================
# Style (bundled Helvetica Neue, fonts/ at repo root — same toolkit as monkey_4iar)
# ===========================================================================
_INK, _MUTED, _GRID = "#2C3E50", "#7A8894", "#E3E7EB"
_C = {"front": "#556270", "best": "#E4A11B", "stats": "#12A19A", "zt": "#3F4DA0",
      "always": "#5580CC",  # monkey_4iar CORNFLOWER — Always Stop (k=0) endpoint
      "never": "#8B97A3", "steps": "#B0B8C0",
      "singlehalt": "#EEA35B",  # monkey_4iar ORANGE — SingleHalt* (fixed-stop) point
      "ag": "#8E6BAF",  # AGController (action-gap + steps) — distinct purple, 2026-07-09
      "mono": MAIN_COLOR}  # board/engine's indigo-blue — monochrome base for the decodability bars
_HELVETICA_FAMILY: str | None = None


def _rcparams() -> None:
    """House style; register bundled Helvetica Neue (fonts/) via font_manager, else fall back to Arial."""
    global _HELVETICA_FAMILY
    if _HELVETICA_FAMILY is None:
        from matplotlib import font_manager
        reg = Path(__file__).resolve().parents[2] / "fonts" / "HelveticaNeue-Roman.otf"
        if reg.is_file():
            for otf in (reg, reg.with_name("HelveticaNeue-Bold.otf")):
                if otf.is_file():
                    font_manager.fontManager.addfont(str(otf))
            _HELVETICA_FAMILY = font_manager.FontProperties(fname=str(reg)).get_name()
        else:
            _HELVETICA_FAMILY = ""
    sans = ([_HELVETICA_FAMILY] if _HELVETICA_FAMILY else []) + ["Helvetica Neue", "Arial", "Nimbus Sans", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "svg.fonttype": "path",
        "font.size": 12, "text.color": _INK, "axes.edgecolor": "#B7C0C9",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        # Mathtext ($z_t$, $k^*$, $R^2$, ...) ignores font.family entirely and defaults to its own
        # "dejavusans" fontset -- every math-mode label was silently rendering in DejaVu Sans right
        # next to Helvetica Neue plain text without this, which is what actually read as "not
        # Helvetica" even though the regular text genuinely was.
        "mathtext.fontset": "custom",
        "mathtext.rm": sans[0], "mathtext.it": f"{sans[0]}:italic", "mathtext.bf": f"{sans[0]}:bold",
    })


# ===========================================================================
# Cost / regret / features
# ===========================================================================
def _load_zt_by_episode(episodes: list[dict[str, Any]], cache_path: str | Path, d_embed: int) -> list[np.ndarray]:
    """Per-episode root embedding ``z_t`` aligned to ``episodes`` (validation, ``shuffle=False``); the
    cache's ``N_t`` column must equal the packed ``n_nodes`` row-for-row or the pairing is wrong."""
    cache = _load_materialized_cache_unchecked(Path(cache_path))
    step_counts = [len(ep["halt_rewards"]) for ep in episodes]
    total = int(sum(step_counts))
    parts, produced = [], 0
    for shard_path in cache.shard_paths:
        if produced >= total:
            break
        feats = torch.load(shard_path, weights_only=False)["features"]
        parts.append(feats[:total - produced])
        produced += feats.shape[0]
    allf = torch.cat(parts, dim=0)
    z, n_t = allf[:, :d_embed].to(torch.float64).numpy(), allf[:, d_embed].to(torch.float64).numpy()
    if not np.allclose(np.concatenate([np.asarray(ep["tree_sizes"], np.float64) for ep in episodes]), n_t, atol=1e-4):
        raise ValueError("z_t cache/episode misalignment — validation cache must be shuffle=False.")
    out, off = [], 0
    for n in step_counts:
        out.append(z[off:off + n])
        off += n
    return out


def _return_curves(episodes: list[dict[str, Any]], config: BudgetedOracleConfig) -> list[np.ndarray]:
    """Per-episode return curve ``r[s] = halt_reward[s] - accumulated continue cost`` under ``config``
    (cost is applied at analysis time; the packed halt_rewards/tree_sizes are cost-independent)."""
    out = []
    for ep in episodes:
        hr, ts, tb = ep["halt_rewards"], ep["tree_sizes"], ep["time_budgets"]
        r, acc = np.empty(len(hr)), 0.0
        for s in range(len(hr)):
            r[s] = hr[s] - acc
            if s < len(hr) - 1:
                acc += continue_cost(int(ts[s]), int(tb[s]), config)
        out.append(r)
    return out


def _regret_at(curves: list[np.ndarray], stops) -> np.ndarray:
    """Per-episode regret = ``max(curve) - curve[clamp(stop)]``."""
    return np.array([c.max() - c[min(max(int(s), 0), len(c) - 1)] for c, s in zip(curves, stops)])


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    """(mean, lo, hi) 95% percentile-bootstrap CI on the mean (cts.stats.bootstrap_ci; shared seed)."""
    return bootstrap_ci(lambda v: float(v.mean()), values, n_boot=2000)


def _steps_stats_tensor(ep: dict[str, Any]) -> torch.Tensor:
    """``[steps_taken, n_nodes, height, width]`` per step (C-step + C-maint features; budget-out)."""
    n = len(ep["halt_rewards"])
    return torch.tensor(np.column_stack([np.arange(n), ep["tree_sizes"], ep["heights"], ep["widths"]]), dtype=torch.float32)


def _steps_zt_tensor(ep: dict[str, Any], z_ep: np.ndarray) -> torch.Tensor:
    """``[steps_taken, z_t]`` per step (the R feature carried by the embedding; budget-out)."""
    return torch.tensor(np.column_stack([np.arange(len(ep["halt_rewards"])), z_ep]), dtype=torch.float32)


def _steps_ag_tensor(ep: dict[str, Any]) -> torch.Tensor:
    """``[steps_taken, action_gap]`` per step — AGController: the 2-feature "is the greedy-best root
    move separated from the runner-up" readout (2026-07-09 investigation). ``action_gap`` is read
    from the raw tree snapshot at load time (``_load_split_episodes(..., load_action_gaps=True)``),
    not from the packed shard — see ``_action_gaps_for_trajectory``."""
    return torch.tensor(np.column_stack([np.arange(len(ep["halt_rewards"])), ep["action_gaps"]]), dtype=torch.float32)


# ===========================================================================
# Stop-controller modules (fit on the train split -> per-episode eval stop steps)
# ===========================================================================
def fit_singlehalt_stop(fit_curves: list[np.ndarray]) -> int:
    """SingleHalt* — the ONE-parameter fixed-stop controller: the stop step ``k`` minimizing FIT-split
    mean regret (selected on the fit split, never on the eval set it is scored over)."""
    kmax = max(len(c) for c in fit_curves)
    return int(np.argmin([_regret_at(fit_curves, [k] * len(fit_curves)).mean() for k in range(kmax)]))


def _train_readout(fit_feats, ev_feats, fit_curves, *, in_dim, epochs, lr, seed,
                   ev_curves=None, out_path=None) -> np.ndarray:
    """Instantiate a native readout head (``build_advantage_head``) and PG-train it (``fit_readout_pg``),
    then return greedy eval stop steps. Features are z-scored on the fit split (n_nodes ≫ steps).

    ``ev_curves``/``out_path`` are optional passthroughs to ``fit_readout_pg``: when given, the eval
    split doubles as a per-epoch validation set (tracked, not fit on) and its regret curve is saved
    alongside the train loss — see ``fit_readout_pg``'s own docstring.
    """
    full = torch.cat(fit_feats, 0)
    mean, std = full.mean(0), full.std(0).clamp_min(1e-6)
    torch.manual_seed(seed)  # must precede head construction — fit_readout_pg's own seed call is too
                             # late to control weight init, since the head is already built by then
    head = build_advantage_head(in_dim, 64, 2)
    ev_feats_norm = [(f - mean) / std for f in ev_feats]
    fit_readout_pg(head, [(f - mean) / std for f in fit_feats], fit_curves, epochs=epochs, lr=lr, seed=seed,
                   ev_feats=ev_feats_norm if ev_curves is not None else None,
                   ev_curves=ev_curves, out_path=out_path)
    head.eval()
    with torch.no_grad():
        return np.array([stop_step_from_advantages(head(f).reshape(-1)) for f in ev_feats_norm])


def _split(trajectory_keys: list[str], seed: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """70/30 fit/eval split, partitioned by SOURCE TREE (never by raw episode index).

    A tree can back multiple episodes (e.g. several truncation depths of the same trajectory); if the
    split were over episode indices, some of a tree's episodes could land in fit and others in eval —
    since z_t at a shared step is identical across them, that's leakage, not a held-out estimate. This
    groups episode indices by ``trajectory_key`` first, shuffles whole TREES, and assigns each tree's
    episodes entirely to one side.
    """
    keys = np.asarray(trajectory_keys)
    unique_keys = np.unique(keys)
    perm = np.random.default_rng(seed).permutation(len(unique_keys))
    n_fit_trees = int(round(train_frac * len(unique_keys)))
    fit_keys = set(unique_keys[perm[:n_fit_trees]].tolist())
    fit_mask = np.array([k in fit_keys for k in keys])
    all_idx = np.arange(len(keys))
    return all_idx[fit_mask], all_idx[~fit_mask]


def _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed, train_frac=0.7,
                          load_action_gaps=False):
    """Load validation episodes + aligned ``z_t`` once, plus the fit/eval TREE split (70/30 by
    default; ``train_frac`` is exposed so the split ratio can be swept without repacking, e.g. to
    check whether a fit/eval SingleHalt* k* discrepancy is genuine finite-sample noise (shrinks as
    the eval split grows) rather than a split bug (see outputs/reports/ysagiv.md's regime-select
    threads for the same "confirm before trusting" convention). ``load_action_gaps`` opts into the
    extra raw-tree-snapshot I/O AGController needs (off by default — see ``_load_split_episodes``)."""
    episodes = _load_split_episodes(packed_root, "validation", max_episodes=max_episodes,
                                    load_action_gaps=load_action_gaps)
    z_by_ep = _load_zt_by_episode(episodes, cache_path, d_embed)
    fit_idx, ev_idx = _split([ep["trajectory_key"] for ep in episodes], seed, train_frac=train_frac)
    return episodes, z_by_ep, fit_idx, ev_idx


def _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, d_embed, seed, *,
                          ev_curves=None, curves_out_dir=None):
    """Fit all three stop controllers on the fit split; return eval-split per-episode stop steps.

    SingleHalt* (1 param, grid) / Stats-Controller (PG MLP, 200 ep) / Zt-Controller (PG MLP, 200 ep —
    same schedule as Zt: the original 30ep/lr=1e-2 Stats schedule was an undertrained optimization
    headwind, not a representational gap (n_nodes carries real signal independent of steps; matching
    Zt's epochs/lr closes the Stats-vs-SingleHalt* gap without any feature engineering). Returns
    ``{'k_singlehalt', 'singlehalt', 'stats', 'zt'}``, plus ``'ag'`` (AGController, PG MLP, same
    200-epoch schedule) IF ``episodes`` carry an ``action_gaps`` field (i.e. were loaded with
    ``_load_split_episodes(..., load_action_gaps=True)``) — silently omitted otherwise, so this
    function still works unchanged for callers that don't need it.

    ``ev_curves``/``curves_out_dir`` are optional: when given, the PG fits track per-epoch held-out
    regret and save it (CSV + PNG) under ``<curves_out_dir>/{stats,zt,ag}_training_curve.*``.
    """
    out = {
        "k_singlehalt": (kf := fit_singlehalt_stop(fit_curves)),
        "singlehalt": np.full(len(ev_idx), kf, dtype=int),
        "stats": _train_readout([_steps_stats_tensor(episodes[i]) for i in fit_idx],
                                [_steps_stats_tensor(episodes[i]) for i in ev_idx], fit_curves,
                                in_dim=4, epochs=200, lr=1e-3, seed=seed, ev_curves=ev_curves,
                                out_path=(Path(curves_out_dir) / "stats") if curves_out_dir else None),
        "zt": _train_readout([_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in fit_idx],
                             [_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in ev_idx], fit_curves,
                             in_dim=d_embed + 1, epochs=200, lr=1e-3, seed=seed, ev_curves=ev_curves,
                             out_path=(Path(curves_out_dir) / "zt") if curves_out_dir else None),
    }
    if episodes and "action_gaps" in episodes[fit_idx[0]]:
        out["ag"] = _train_readout([_steps_ag_tensor(episodes[i]) for i in fit_idx],
                                   [_steps_ag_tensor(episodes[i]) for i in ev_idx], fit_curves,
                                   in_dim=2, epochs=200, lr=1e-3, seed=seed, ev_curves=ev_curves,
                                   out_path=(Path(curves_out_dir) / "ag") if curves_out_dir else None)
    return out


# ===========================================================================
# Plots
# ===========================================================================
_PT_SIZE, _PT_ALPHA = 4, 0.88  # one uniform marker size/alpha for every point series — color is the only encoding


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays (and tuples) to plain JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _save_json(data: dict, path: Path) -> None:
    """Save a plot's precomputed (expensive: data load + model fit) intermediate data so the figure
    can be re-rendered later -- different padding, styling, which candidates to show -- without
    redoing that work. See ``replot_saved``/``--replot``."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _padded_range(lo: float, hi: float, frac: float, min_pad: float) -> tuple[float, float]:
    """(lo, hi) padded by ``frac`` of the actual span (floored at ``min_pad`` for near-zero spans).

    Proportional, not a fixed constant -- axis padding sized in absolute units (e.g. "+30") silently
    assumes a specific reward/regret SCALE, which breaks the instant that scale changes (this is
    exactly what happened when halt_reward went from cp-scale to win-probability-scale: hardcoded
    +-20/25/30 padding swamped a regret spread of a few tenths). Scaling padding to the data's own
    span makes the plot self-adjusting regardless of what scale the underlying reward is in.
    """
    span = hi - lo
    pad = max(span * frac, min_pad)
    return lo - pad, hi + pad


def _deconflict_points(points: list[dict], x_scale: float, y_scale: float, frac: float = 0.035) -> list[dict]:
    """Nudge points that are numerically COINCIDENT (identical x AND y -- e.g. a Stats-Controller
    readout that converged to exactly SingleHalt*'s fixed-k policy, a real finding, not a bug: see
    labnotebook 2026-07-07) apart by a small amount so both stay visible as distinct markers instead
    of one hiding the other.

    Pure rendering aid: shifts x/xlo/xhi and y/lo/hi by the SAME delta per point (preserving each
    point's own CI width/shape exactly, just recentering it slightly), scaled to the plot's own
    current axis range so the nudge stays proportionate regardless of the data's magnitude (same
    scale-agnostic principle as ``_padded_range``). Never changes which point is "ahead" of another
    and never used to imply a real difference that isn't in the underlying statistics -- callers that
    need the true, un-nudged values (e.g. printed regret numbers) should keep a separate reference to
    the original points.
    """
    out = [dict(p) for p in points]
    dx, dy = x_scale * frac, y_scale * frac
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[i]["x"] == out[j]["x"] and out[i]["y"] == out[j]["y"]:
                for key, delta in (("x", -dx), ("xlo", -dx), ("xhi", -dx), ("y", -dy), ("lo", -dy), ("hi", -dy)):
                    out[i][key] += delta
                for key, delta in (("x", dx), ("xlo", dx), ("xhi", dx), ("y", dy), ("lo", dy), ("hi", dy)):
                    out[j][key] += delta
    return out


def _frontier_panel(ax, fr_x, fr, points, *, xlim, ylim, label_line=False, capsize=3.5, frontier_label="Frontier"):
    """Draw the fixed-stop frontier line and every point (SingleHalt*/Stats/z_t/AlwaysStop/AlwaysContinue) —
    all the SAME marker/size, differing only by color, each with 95% CI error bars in x AND y."""
    ax.plot(fr_x, fr, color=_C["front"], lw=1.1, label=frontier_label if label_line else None)
    for p in points:
        ax.errorbar([p["x"]], [p["y"]], xerr=[[p["x"] - p["xlo"]], [p["xhi"] - p["x"]]],
                    yerr=[[p["y"] - p["lo"]], [p["hi"] - p["y"]]], fmt="o", ms=_PT_SIZE, color=p["color"],
                    ecolor=p["color"], elinewidth=1.5, capsize=capsize, mec="none", alpha=_PT_ALPHA,
                    zorder=p.get("zorder", 6), label=p["label"] if label_line else None)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Stop Step"); ax.set_ylabel("Regret")
    ax.grid(axis="both", color=_GRID, lw=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))


def _draw_zoom_indicator(fig, ax_from, ax_to, xlim, ylim):
    """Box the (xlim, ylim) region on ``ax_from`` that ``ax_to`` zooms into, connected by two lines —
    the classic 'this panel is a zoom of that box' indicator, drawn manually since ``ax_from``/``ax_to``
    are independent side-by-side subplots rather than a true embedded inset."""
    from matplotlib.patches import Rectangle, ConnectionPatch
    x0, x1 = xlim; y0, y1 = ylim
    ax_from.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=_INK, lw=1.3, alpha=0.5, zorder=8))
    for corner_from, corner_to in [((x1, y1), (0, 1)), ((x1, y0), (0, 0))]:
        fig.add_artist(ConnectionPatch(xyA=corner_from, coordsA=ax_from.transData,
                                       xyB=corner_to, coordsB=ax_to.transAxes,
                                       color=_MUTED, lw=0.9, ls=(0, (6, 4)), alpha=0.7, zorder=1))


def _compute_frontier_data(packed_root: Path, cache_path: str | Path, *,
                           d_embed: int = 32, time_mode: str = "linear", time_lambda: float = 10.0,
                           maintenance_scale: float = 0.0, maintenance_exponent: float = 1.0,
                           max_episodes: int = 15000, seed: int = 0, curves_out_dir=None,
                           train_frac: float = 0.7) -> dict:
    """Expensive half of the frontier plot: load data, fit the stop controllers, compute every
    plotted quantity. Returns a JSON-serializable dict consumed by ``_render_frontier`` -- kept
    separate (and saved to disk by the caller) so the figure can be re-rendered later without
    redoing the (slow) model fitting.

    ``curves_out_dir``, if given, saves the Stats-/Zt-/AG-Controller PG training curves (train loss +
    held-out regret per epoch, CSV + PNG) here — see ``_fit_stop_controllers``. ``train_frac``
    overrides the default 70/30 fit/eval split (see ``_load_assessment_data``). Also fits AGController
    (action-gap + steps, 2026-07-09) — its extra per-episode raw-tree read is requested unconditionally
    here since this is the one place per run these controllers are fit at the headline regime (mirrors
    ``curves_out_dir``'s own scoping rationale)."""
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed,
                                                                train_frac=train_frac, load_action_gaps=True)
    curves = _return_curves(episodes, config)
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, [curves[i] for i in fit_idx], d_embed, seed,
                                 ev_curves=ev, curves_out_dir=curves_out_dir)

    kmax = max(len(c) for c in ev)
    fr = np.array([_regret_at(ev, [k] * len(ev)).mean() for k in range(kmax)])
    fr_x = np.array([np.mean([min(k, len(c) - 1) for c in ev]) for k in range(kmax)])
    kf = min(ctrl["k_singlehalt"], kmax - 1)

    def point(stops, color, lbl, **kw):
        xs = np.array([min(int(s), len(c) - 1) for s, c in zip(stops, ev)], dtype=float)
        xm, xlo, xhi = _mean_ci(xs)
        ym, ylo, yhi = _mean_ci(_regret_at(ev, stops))
        return dict(x=xm, xlo=xlo, xhi=xhi, y=ym, lo=ylo, hi=yhi, color=color, label=lbl, **kw)

    controllers = [point(ctrl["singlehalt"], _C["singlehalt"], "SingleHalt* (fixed stop)", zorder=7),
                  point(ctrl["stats"], _C["stats"], "Stats-Controller"),
                  point(ctrl["zt"], _C["zt"], "$z_t$-Controller")]
    if "ag" in ctrl:
        controllers.append(point(ctrl["ag"], _C["ag"], "AG-Controller"))
    ends = [point(np.zeros(len(ev_idx), dtype=int), _C["always"], "Always Stop (k=0)"),
           point(np.full(len(ev_idx), kmax - 1, dtype=int), _C["never"], "Always Continue (k=max)")]
    all_points = controllers + ends
    d_zs = bootstrap_ci(lambda d: float(d.mean()), _regret_at(ev, ctrl["zt"]) - _regret_at(ev, ctrl["stats"]), n_boot=2000)

    # Per-episode stop steps for MC(zt)/TS(stats)/AG(if fit)/FS(singlehalt) -- the raw distributions
    # behind each `point()` summary above, kept separately so `_render_frontier` can draw a violin of
    # each (the x-error bars stay a 95% CI on the MEAN; this is the actual per-episode spread instead
    # -- FS's is a single fixed k, so its "violin" is a degenerate spike, which is the accurate picture).
    stop_steps_by_controller = {
        name: np.array([min(int(s), len(c) - 1) for s, c in zip(ctrl[key], ev)], dtype=float).tolist()
        for name, key in (("MC", "zt"), ("TS", "stats"), ("FS", "singlehalt"))
    }
    if "ag" in ctrl:
        stop_steps_by_controller["AG"] = np.array(
            [min(int(s), len(c) - 1) for s, c in zip(ctrl["ag"], ev)], dtype=float).tolist()

    return {"fr_x": fr_x, "fr": fr, "controllers": controllers, "all_points": all_points, "kf": kf,
            "d_zs": d_zs, "time_mode": time_mode, "time_lambda": time_lambda,
            "maintenance_scale": maintenance_scale, "maintenance_exponent": maintenance_exponent,
            "stop_steps_by_controller": stop_steps_by_controller}


def _render_frontier(data: dict, out_dir: str | Path) -> dict:
    """Cheap half of the frontier plot: build + save the figure from ``_compute_frontier_data``'s output.

    A learned point BELOW the frontier beats every fixed stop at that effort. Left panel = full range,
    with a boxed+dotted-line indicator of the region the right panel zooms into; right panel = zoom on
    the controllers, each point with 95% CI error bars in both x (average stop step) and y (regret).
    """
    fr_x, fr = np.asarray(data["fr_x"]), np.asarray(data["fr"])
    controllers, all_points, kf = data["controllers"], data["all_points"], data["kf"]

    # Colors AND labels are baked into the CACHED json (frozen at the time `_compute_frontier_data`
    # ran), so a later edit to `_C` or to the display text has no effect on an already-computed
    # frontier_data.json unless re-applied here at render time -- refresh every point's color/label
    # from the ORIGINAL label it was built with, in both lists (a JSON round-trip means
    # `controllers`/`all_points` are independent copies, not the same dict objects `all_points =
    # controllers + ends` originally shared). `rank` orders the legend: Frontier(0) is the line,
    # handled separately in `_frontier_panel`'s `frontier_label`.
    _legend_info = {
        "SingleHalt* (fixed stop)": ("Fixed Stop", "singlehalt", 4),
        "Stats-Controller": ("Tree Stats", "stats", 2),
        "$z_t$-Controller": ("Meta-Control (Ours)", "zt", 1),
        "AG-Controller": ("Action Gap", "ag", 3),
        "Always Stop (k=0)": ("Always Stop", "always", 5),
        "Always Continue (k=max)": ("Always Continue", "never", 6),
    }
    for p in controllers + all_points:
        info = _legend_info.get(p.get("label"))
        if info:
            new_label, ckey, rank = info
            p["label"], p["color"], p["_rank"] = new_label, _C[ckey], rank

    _rcparams()
    # Explicit rects (not a single shared gridspec) so the histogram row's vertical space and the
    # legend's gap below it are both under direct control, independent of the frontier row's own
    # layout. Histogram sits at the TOP (the bimodal stop-step spread it reveals is the headline
    # finding), frontier panels in the middle, shared legend at the bottom.
    fig = plt.figure(figsize=(9.2, 6.4))
    axH = fig.add_axes([0.09, 0.76, 0.89, 0.20])
    gs_mid = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], left=0.09, right=0.98, top=0.66, bottom=0.16,
                              wspace=0.28)
    axL = fig.add_subplot(gs_mid[0, 0])
    axR = fig.add_subplot(gs_mid[0, 1])
    zoom_ylim = _padded_range(min(p["lo"] for p in controllers), max(p["hi"] for p in controllers),
                              frac=0.25, min_pad=1e-3)
    zoom_xlim = _padded_range(min(p["x"] for p in controllers), max(p["x"] for p in controllers),
                              frac=0.3, min_pad=1.0)
    zoom_xlim = (max(0, zoom_xlim[0]), zoom_xlim[1])
    full_lo, full_hi = _padded_range(min(p["lo"] for p in all_points), max(p["hi"] for p in all_points),
                                     frac=0.05, min_pad=1e-3)
    full_ylim = (min(0, full_lo), full_hi)
    # Order for the LEGEND only (axR is the one that actually collects labels) — Frontier, Meta-
    # Control (Ours), Tree Stats, Fixed Stop, Always Stop, Always Continue; axL's draw order is
    # unaffected (it shows no legend) since it still plots the untouched `all_points`.
    legend_points = sorted(all_points, key=lambda p: p["_rank"])
    _frontier_panel(axL, fr_x, fr, all_points, xlim=(-1, fr_x[-1] + 1), ylim=full_ylim, capsize=0)
    _frontier_panel(axR, fr_x, fr, legend_points, ylim=zoom_ylim, xlim=zoom_xlim, label_line=True, capsize=2.5)
    _draw_zoom_indicator(fig, axL, axR, zoom_xlim, zoom_ylim)
    handles, labels = axR.get_legend_handles_labels()
    # 4 columns (not 3): with AG-Controller present this is 7 entries -- ncol=3 makes a 3rd row that
    # collides with the frontier panels' "Stop Step" xlabel just above it. ncol=4 keeps it to 2 rows
    # for both the 6-entry (no AG) and 7-entry (with AG) case.
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.045), ncol=4,
              fontsize=10, frameon=False)

    # Secondary panel: the ACTUAL per-episode stop-step distribution for each controller (MC/TS/AG/
    # FS) — the x-error bars above stay a 95% CI on the MEAN (comparable across controllers); this
    # shows the real per-episode spread the mean summarizes. FS (SingleHalt*, one fixed k for every
    # episode) renders as a degenerate spike — that IS its distribution, not a plotting artifact.
    by_ctrl = data.get("stop_steps_by_controller")
    if by_ctrl:
        order = [n for n in ("MC", "TS", "AG", "FS") if n in by_ctrl]
        ckey_by_name = {"MC": "zt", "TS": "stats", "AG": "ag", "FS": "singlehalt"}
        series = [np.asarray(by_ctrl[n]) for n in order]
        parts = axH.violinplot(series, positions=range(len(order)), vert=False, widths=0.8,
                               showmeans=True, showextrema=True)
        for body, name in zip(parts["bodies"], order):
            body.set_facecolor(_C[ckey_by_name[name]]); body.set_edgecolor(_C[ckey_by_name[name]])
            body.set_alpha(0.75)
        for key in ("cmeans", "cmaxes", "cmins", "cbars"):
            parts[key].set_color([_C[ckey_by_name[n]] for n in order])
            parts[key].set_linewidth(1.2)
        axH.set_yticks(range(len(order))); axH.set_yticklabels(order)
        axH.set_xlabel("Stop Step")
        axH.grid(axis="x", color=_GRID, lw=1)
        axH.xaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    else:
        axH.text(0.5, 0.5, "stop-step distributions unavailable (re-run to populate)",
                 transform=axH.transAxes, ha="center", va="center", fontsize=9, color=_MUTED)
        axH.set_xticks([]); axH.set_yticks([])
        for spine in axH.spines.values():
            spine.set_visible(False)

    save_pdf_png(fig, str(out_dir), "frontier", dpi=200, bbox_extra_artists=(fig.legends[0],))
    fm = controllers[0]["y"]
    print(f"[assess] frontier {data['time_mode']} λ={data['time_lambda']} m={data['maintenance_scale']} "
          f"p={data['maintenance_exponent']}: SingleHalt*(k={kf})={fm:.1f} "
          f"Stats={controllers[1]['y']:.1f} z_t={controllers[2]['y']:.1f}  "
          f"paired z_t-Stats={tuple(data['d_zs'])}", flush=True)
    return {"k_singlehalt": kf, "paired_zt_minus_stats": tuple(data["d_zs"])}


def plot_regret_effort_frontier(packed_root: Path, cache_path: str | Path, out_dir: str | Path, *,
                                d_embed: int = 32, time_mode: str = "linear", time_lambda: float = 10.0,
                                maintenance_scale: float = 0.0, maintenance_exponent: float = 1.0,
                                max_episodes: int = 15000, seed: int = 0):
    """(1) The fixed-stop frontier (search effort vs regret) with the Stats-/Zt-Controllers as points.
    ``maintenance_exponent=1.0`` makes the per-step maintenance cost EXACTLY linear in node count
    (``scale * n / ref_nodes``); overridden here rather than left to whatever's baked into the packed
    manifest, matching the analysis-time-cost design (see module docstring).

    Saves the computed data to ``<out_dir>/frontier_data.json`` so the figure can be re-rendered later
    (different padding/styling/candidates) without redoing the expensive model fitting -- see
    ``replot_saved`` / ``--replot``. Also saves the Stats-/Zt-Controller PG training curves (train loss
    + held-out regret per epoch) to ``<out_dir>/curves/{stats,zt}_training_curve.{csv,png}`` -- the one
    place per run these controllers are fit at the headline regime, so this is where the curves are
    captured (not the delta-regret sweep's many extra regime fits, or the decodability probe, which
    trains different, non-PG models).
    """
    data = _compute_frontier_data(packed_root, cache_path, d_embed=d_embed, time_mode=time_mode,
                                  time_lambda=time_lambda, maintenance_scale=maintenance_scale,
                                  maintenance_exponent=maintenance_exponent, max_episodes=max_episodes, seed=seed,
                                  curves_out_dir=Path(out_dir) / "curves")
    _save_json(data, Path(out_dir) / "frontier_data.json")
    return _render_frontier(data, out_dir)


def _compute_decodability_data(packed_root: Path, cache_path: str | Path, *,
                               d_embed: int = 32, max_episodes: int = 12000, seed: int = 0) -> dict:
    """Expensive half of the decodability plot: load data, fit the linear/MLP R^2 probes. Returns a
    JSON-serializable dict consumed by ``_render_decodability``."""
    from cts.analysis.zt_probe import _linear_r2, _mlp_r2, _episode_split_mask  # local: avoids circular import
    episodes = _load_split_episodes(packed_root, "validation", max_episodes=max_episodes)
    z_by_ep = _load_zt_by_episode(episodes, cache_path, d_embed)
    step_counts = [len(ep["halt_rewards"]) for ep in episodes]
    total = int(sum(step_counts))
    cols = {k: [] for k in ("R", "steps", "heights", "widths", "nnodes", "z")}
    for ep, z_ep in zip(episodes, z_by_ep):
        hr = np.asarray(ep["halt_rewards"], np.float64)
        cols["R"].append(np.maximum.accumulate(hr[::-1])[::-1] - hr)
        cols["steps"].append(np.arange(len(hr), dtype=np.float64))
        cols["heights"].append(np.asarray(ep["heights"], np.float64))
        cols["widths"].append(np.asarray(ep["widths"], np.float64))
        cols["nnodes"].append(np.asarray(ep["tree_sizes"], np.float64))
        cols["z"].append(z_ep)
    R = np.concatenate(cols["R"])
    steps, heights = np.concatenate(cols["steps"]), np.concatenate(cols["heights"])
    widths, nnodes, z = np.concatenate(cols["widths"]), np.concatenate(cols["nnodes"]), np.concatenate(cols["z"], axis=0)
    traj_keys = [ep["trajectory_key"] for ep in episodes]
    is_tr = _episode_split_mask(step_counts, total, seed=seed, trajectory_keys=traj_keys); is_te = ~is_tr

    # Each feature set is tested ALONE (not nested/cumulative): "steps" was
    # previously baked into every row, including the one labeled "z_t" — so
    # the reported R^2 mostly reflected R(t)'s near-tautological relationship
    # with trajectory position (fewer remaining steps -> mechanically less
    # room for R(t)'s forward-max to be large), not what z_t itself encodes.
    # "all" is kept as a reference upper bound (every signal combined).
    feats = {"steps": np.column_stack([steps]),
             "stats": np.column_stack([nnodes, heights, widths]),
             "z_t": np.column_stack([z]),
             "all (steps+stats+z_t)": np.column_stack([steps, nnodes, heights, widths, z])}
    rows = [(name, _linear_r2(X[is_tr], R[is_tr], X[is_te], R[is_te]),
             _mlp_r2(X[is_tr], R[is_tr], X[is_te], R[is_te], seed=seed)) for name, X in feats.items()]
    Xall = np.column_stack([steps, nnodes, heights, widths, z])
    Xs = Xall[np.random.default_rng(seed + 7).permutation(total)]
    shuf_r2 = _linear_r2(Xs[is_tr], R[is_tr], Xs[is_te], R[is_te])

    # Orthogonality check: how much does "stats" or "z_t" ALONE already
    # decode "steps" itself? High values here mean stats/z_t are entangled
    # with trajectory position, not independent of it — the entanglement
    # that made the old nested-feature design misleading in the first place.
    orthogonality = {
        "stats_predicts_steps": _linear_r2(
            np.column_stack([nnodes, heights, widths])[is_tr], steps[is_tr],
            np.column_stack([nnodes, heights, widths])[is_te], steps[is_te]),
        "z_t_predicts_steps": _linear_r2(z[is_tr], steps[is_tr], z[is_te], steps[is_te]),
    }
    return {"rows": rows, "shuf_r2": shuf_r2, "orthogonality": orthogonality}


def _render_decodability(data: dict, out_dir: str | Path) -> dict:
    """Cheap half of the decodability plot: build + save the figure from
    ``_compute_decodability_data``'s output. Can ``R(t)`` = value of continuing
    (``max_{s>=t} halt_reward(s) - halt_reward(t)``) be decoded from ``steps``, ``stats``, or ``z_t``
    ALONE (each an independent probe, not nested) vs a row-shuffle floor and an "all combined"
    reference upper bound."""
    rows, shuf_r2 = data["rows"], data["shuf_r2"]
    orthogonality = data.get("orthogonality", {})
    _rcparams()
    mono = _C["mono"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    y, h = np.arange(len(rows)), 0.30
    ax.barh(y - h / 2, [r[1] for r in rows], h, color=mono, alpha=0.5,
           edgecolor="none", linewidth=0, label="Linear (Ridge)")
    ax.barh(y + h / 2, [r[2] for r in rows], h, color=mono, edgecolor="none", linewidth=0, label="MLP (2×64)")
    ax.axvline(shuf_r2, color=_MUTED, ls="--", lw=1.3, label=f"Shuffle Floor ({shuf_r2:+.2f})")
    ax.axvline(0, color=_MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(["steps", "stats", "$z_t$", "combined"], fontsize=10.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.set_xlabel(r"R(t) Variance Explained (held-out $R^2$)")
    ax.grid(axis="x", color=_GRID, lw=1)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    save_pdf_png(fig, str(out_dir), "decodability", dpi=200)
    out = {r[0]: {"linear": r[1], "mlp": r[2]} for r in rows}
    print(f"[assess] R-decodability  " + "  ".join(f"{k}={v['mlp']:.3f}" for k, v in out.items()), flush=True)
    if orthogonality:
        print(f"[assess] orthogonality (R^2 predicting steps FROM each): " +
              "  ".join(f"{k}={v:.3f}" for k, v in orthogonality.items()), flush=True)
    return out


def plot_r_decodability(packed_root: Path, cache_path: str | Path, out_dir: str | Path, *,
                        d_embed: int = 32, max_episodes: int = 12000, seed: int = 0):
    """(2) Can ``R(t)`` = value of continuing be decoded, and only from ``z_t``? See
    ``_render_decodability`` for the full docstring. Saves computed data to
    ``<out_dir>/decodability_data.json`` for cheap re-rendering -- see ``replot_saved``/``--replot``."""
    data = _compute_decodability_data(packed_root, cache_path, d_embed=d_embed, max_episodes=max_episodes, seed=seed)
    _save_json(data, Path(out_dir) / "decodability_data.json")
    return _render_decodability(data, out_dir)


_DELTA_CANDIDATES = [("singlehalt", _C["singlehalt"], "SingleHalt*"), ("stats", _C["stats"], "Stats-Controller"),
                    ("ag", _C["ag"], "AG-Controller"),
                    ("always_stop", _C["always"], "Always Stop"), ("always_continue", _C["never"], "Always Continue")]
# Plotted subset for _delta_ci_panel: always_stop/always_continue are frequently degenerate (huge or
# exactly-zero delta) and, since SingleHalt* is fit by argmin over EVERY fixed stop step (including
# k=0=Always Stop and k=kmax-1=Always Continue -- see fit_singlehalt_stop), it's guaranteed at least
# as good as either extreme on the fit split -- plotting them alongside just forces the x-axis to
# whatever huge/degenerate range they occupy, hiding the actual singlehalt/stats spread. Still printed
# in the [assess] summary line (loops over the full _DELTA_CANDIDATES), just not plotted.
_DELTA_CANDIDATES_DISPLAY = _DELTA_CANDIDATES[:3]


def _regime_deltas_vs_zt(episodes, z_by_ep, fit_idx, ev_idx, base_cfg, mode, lam, mnt, exponent, d_embed, seed):
    """Per-episode paired delta = candidate_regret − z_t_regret for {singlehalt, stats, ag,
    always_stop, always_continue} under one cost regime (mode/lam/mnt/exponent). Positive = z_t
    beats the candidate. ``ag`` is included only if ``episodes`` carry ``action_gaps`` (see
    ``_fit_stop_controllers``) — silently omitted otherwise."""
    curves = _return_curves(episodes, replace(base_cfg, time_mode=mode, time_lambda=lam,
                                              maintenance_scale=mnt, maintenance_exponent=exponent))
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, [curves[i] for i in fit_idx], d_embed, seed)
    kmax = max(len(c) for c in ev)
    zt_regret = _regret_at(ev, ctrl["zt"])
    stops = {"singlehalt": ctrl["singlehalt"], "stats": ctrl["stats"],
            "always_stop": np.zeros(len(ev_idx), dtype=int),
            "always_continue": np.full(len(ev_idx), kmax - 1, dtype=int)}
    if "ag" in ctrl:
        stops["ag"] = ctrl["ag"]
    return {name: _regret_at(ev, s) - zt_regret for name, s in stops.items()}


def _delta_ci_panel(ax, regime_labels, deltas_by_regime, candidates=_DELTA_CANDIDATES_DISPLAY):
    """Horizontal dot-and-whisker: one row per regime, one 95% CI point per candidate (offset within
    the row) — mean Δ regret vs z_t, same marker/size convention as the frontier plot's points.

    Does NOT draw its own legend -- an axes-fraction ``bbox_to_anchor`` legend scales its absolute gap
    with the axes' height, which blows up on the many-row (maintenance) panel and cramps the few-row
    (lambda) panel. The caller (``_render_delta_regret``) builds one shared ``fig.legend`` per figure
    instead, matching the frontier plot's figure-level legend convention (gap sized off the whole
    figure, not the axes)."""
    n, m = len(regime_labels), len(candidates)
    step = 0.68 / m
    for ci, (key, color, lbl) in enumerate(candidates):
        positions = np.arange(n) + (ci - (m - 1) / 2) * step
        data = [deltas_by_regime[i][key] for i in range(n)]
        means, los, his = zip(*(_mean_ci(d) for d in data))
        xerr = [[me - lo for me, lo in zip(means, los)], [hi - me for me, hi in zip(means, his)]]
        ax.errorbar(means, positions, xerr=xerr, fmt="o", ms=_PT_SIZE, color=color, ecolor=color,
                    elinewidth=1.6, capsize=3.5, mec="none", alpha=_PT_ALPHA, zorder=6, label=lbl)
    ax.axvline(0, color=_MUTED, lw=1.2, ls="--")
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(regime_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Δ regret  (model − $z_t$)", fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.grid(axis="both", color=_GRID, lw=1)


def _compute_delta_regret_data(packed_root: Path, cache_path: str | Path, *,
                               d_embed: int = 32, max_episodes: int = 15000, seed: int = 0,
                               lambda_grid: tuple[float, ...] = (0.003, 0.01, 0.03), maint_lambda: float = 0.01,
                               maint_grid: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.15, 0.2, 0.3),
                               maintenance_exponent: float = 1.0) -> dict:
    """Expensive half of the delta-regret plots: load data, fit stop controllers for every regime in
    both grids. Returns a JSON-serializable dict consumed by ``_render_delta_regret``. Loads action
    gaps unconditionally (``_DELTA_CANDIDATES_DISPLAY`` includes ``ag``, 2026-07-09)."""
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed,
                                                                load_action_gaps=True)
    base_cfg = _oracle_config(packed_root)

    def regime(mode, lam, mnt):
        return _regime_deltas_vs_zt(episodes, z_by_ep, fit_idx, ev_idx, base_cfg, mode, lam, mnt,
                                    maintenance_exponent, d_embed, seed)

    labels_a = [f"linear λ={lam:g}" for lam in lambda_grid]
    deltas_a = [regime("linear", lam, 0.0) for lam in lambda_grid]
    labels_b = [f"maint={mnt:g}" for mnt in maint_grid]
    deltas_b = [regime("linear", maint_lambda, mnt) for mnt in maint_grid]
    return {"labels_a": labels_a, "deltas_a": deltas_a, "labels_b": labels_b, "deltas_b": deltas_b,
            "maint_lambda": maint_lambda}


def _render_delta_regret(data: dict, out_dir: str | Path) -> dict:
    """Cheap half of the delta-regret plots: build + save both figures from
    ``_compute_delta_regret_data``'s output.

    (3) Δ mean regret (model − $z_t$) for {SingleHalt*, Stats-Controller} (Always Stop/Continue are
    computed and printed but not plotted -- see ``_DELTA_CANDIDATES_DISPLAY``) — z_t is always the
    baseline, so positive = z_t wins. Two horizontal-violin figures: (A) sweep the linear time cost λ
    at maintenance=0 (isolates the C-step/R edge); (B) sweep maintenance_scale at a fixed λ (isolates
    the weaker C-maint edge).
    """
    labels_a, labels_b = data["labels_a"], data["labels_b"]
    # After a JSON round-trip, deltas are lists of dicts of LISTS -- convert back to arrays.
    deltas_a = [{k: np.asarray(v) for k, v in d.items()} for d in data["deltas_a"]]
    deltas_b = [{k: np.asarray(v) for k, v in d.items()} for d in data["deltas_b"]]

    def _finish(fig, ax, title, base):
        # A fixed bbox_to_anchor FRACTION scales its absolute gap with the figure's height, which
        # overlapped the x-axis label on the short (3-row) lambda figure while looking fine on the
        # tall (7-row) maintenance one. Target a constant ~0.5in gap in every figure instead by
        # dividing that inch target by this particular figure's height.
        gap_frac = 0.5 / fig.get_size_inches()[1]
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -gap_frac), ncol=len(handles),
                  fontsize=9.5, frameon=False)
        ax.set_title(title, fontsize=12, loc="left")
        save_pdf_png(fig, str(out_dir), base, dpi=200, bbox_extra_artists=(fig.legends[0],))

    _rcparams()
    # Constant + per-row height shrunk (and made a fixed-per-figure fig.legend below take the place of
    # the old axes-fraction legend) so the many-row maintenance sweep doesn't render with a huge
    # vertical gap of empty gridline rows above/below the data and before the legend -- an
    # axes-fraction bbox_to_anchor scales its absolute gap with the axes' height, which is fine for the
    # 3-row lambda panel but balloons on the 7-row maintenance panel.
    figA, axA = plt.subplots(figsize=(8.2, 1.1 + 0.62 * len(labels_a)))
    _delta_ci_panel(axA, labels_a, deltas_a)
    _finish(figA, axA, "varying linear cost λ  (maintenance = 0)", "delta_mean_regret_lambda")

    figB, axB = plt.subplots(figsize=(8.2, 1.1 + 0.62 * len(labels_b)))
    _delta_ci_panel(axB, labels_b, deltas_b)
    _finish(figB, axB, f"varying maintenance scale  (linear λ={data['maint_lambda']:g})",
           "delta_mean_regret_maintenance")

    for group_label, labels, deltas in [("lambda", labels_a, deltas_a), ("maintenance", labels_b, deltas_b)]:
        for lbl, d in zip(labels, deltas):
            summary = "  ".join(f"{k}={_mean_ci(d[k])[0]:+.1f}" for k, _, _ in _DELTA_CANDIDATES)
            print(f"[assess] delta[{group_label}] {lbl:16s} {summary}", flush=True)
    return {"lambda": dict(zip(labels_a, deltas_a)), "maintenance": dict(zip(labels_b, deltas_b))}


def plot_delta_regret_vs_zt(packed_root: Path, cache_path: str | Path, out_dir: str | Path, *,
                            d_embed: int = 32, max_episodes: int = 15000, seed: int = 0,
                            lambda_grid: tuple[float, ...] = (0.003, 0.01, 0.03), maint_lambda: float = 0.01,
                            maint_grid: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.15, 0.2, 0.3),
                            maintenance_exponent: float = 1.0):
    """(3) Δ mean regret (model − $z_t$) sweeps over cost regimes -- see ``_render_delta_regret`` for
    the full docstring.

    ``lambda_grid`` and ``maint_grid``/default ``maint_lambda`` recalibrated 2026-07-07 -- both used
    to be sized for the old cp-scale halt_reward (``lambda_grid=(5.0, 10.0, 40.0)``,
    ``maint_grid=(0.0, 0.05, 0.10, 0.30, 1.0)`` at ``maint_lambda=10.0``). Now that halt_reward is
    win-probability-scaled, those collapsed every episode to "stop at step 0" regardless of method
    (verified on real PUCT data, labnotebook 2026-07-07). New values are real points from dedicated
    sweeps on real PUCT episodes: ``lambda_grid`` spans the recommended calibration (~0.01).
    ``maint_grid`` (at ``time_lambda`` held fixed at the same 0.01) was widened from the original
    5-point 0-0.1 sweep to 7 points spanning 0-0.3, chosen from a retraining-free headroom sweep
    (``fit_singlehalt_stop``'s regret vs the true per-episode oracle, no model fitting needed): the
    fraction of achievable regret SingleHalt* already recovers falls smoothly across this grid
    (0.84 -> 0.73 -> 0.58 -> 0.39 -> 0.15 -> 0.00 at maint 0/0.1/0.15/0.2/0.25/0.3), showing the full
    transition from "little room for a learned halter" through the chosen sweet spot (0.1, ~73%
    recovered -- see ``eval.maintenance_scale`` in the config, used by the frontier plot's headline
    numbers) to full collapse. ``maintenance_exponent=1.0`` makes the per-step maintenance cost
    exactly linear in node count (see ``oracle.maintenance_cost``).

    Saves computed data to ``<out_dir>/delta_regret_data.json`` for cheap re-rendering -- see
    ``replot_saved``/``--replot``.
    """
    data = _compute_delta_regret_data(packed_root, cache_path, d_embed=d_embed, max_episodes=max_episodes,
                                      seed=seed, lambda_grid=lambda_grid, maint_lambda=maint_lambda,
                                      maint_grid=maint_grid, maintenance_exponent=maintenance_exponent)
    _save_json(data, Path(out_dir) / "delta_regret_data.json")
    return _render_delta_regret(data, out_dir)


def replot_saved(out_dir: str | Path, which: str = "all") -> None:
    """Re-render figures from previously-saved ``<out_dir>/*_data.json`` files, skipping the expensive
    data-load + model-fit step entirely. Use this after changing plotting-only code (axis padding,
    which candidates to show, colors, ...) -- no need to rerun the whole eval pipeline just to fix a
    plot. Requires an earlier non-replot run to have populated ``<out_dir>`` first."""
    out_dir = Path(out_dir)
    if which in ("frontier", "all"):
        _render_frontier(_load_json(out_dir / "frontier_data.json"), out_dir)
    if which in ("decodability", "all"):
        _render_decodability(_load_json(out_dir / "decodability_data.json"), out_dir)
    if which in ("separation", "all"):
        _render_delta_regret(_load_json(out_dir / "delta_regret_data.json"), out_dir)


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="R-EVALUATE meta-controller assessment plots")
    ap.add_argument("--packed-root", help="MC packed dir with validation/ + validation_manifest.json "
                    "(not needed with --replot)")
    ap.add_argument("--cache", help="validation materialized cache (.pt); MUST be shuffle=False "
                    "(not needed with --replot)")
    ap.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/normative")
    ap.add_argument("--which", choices=["frontier", "decodability", "separation", "all"], default="all")
    ap.add_argument("--replot", action="store_true",
                    help="re-render from --out-dir's saved *_data.json instead of recomputing -- "
                    "skips data loading + model fitting entirely, use after changing plotting-only code")
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--time-mode", default="linear")
    ap.add_argument("--time-lambda", type=float, default=10.0)
    ap.add_argument("--maintenance-scale", type=float, default=0.0)
    ap.add_argument("--maintenance-exponent", type=float, default=1.0,
                    help="1.0 = maintenance cost exactly linear in node count")
    ap.add_argument("--maint-lambda", type=float, default=10.0, help="lambda held fixed in the maintenance sweep")
    args = ap.parse_args()

    if args.replot:
        replot_saved(args.out_dir, which=args.which)
        return
    if not args.packed_root or not args.cache:
        ap.error("--packed-root and --cache are required unless --replot is set")

    packed_root, out = Path(args.packed_root), args.out_dir
    if args.which in ("frontier", "all"):
        plot_regret_effort_frontier(packed_root, args.cache, out, d_embed=args.d_embed, time_mode=args.time_mode,
                                    time_lambda=args.time_lambda, maintenance_scale=args.maintenance_scale,
                                    maintenance_exponent=args.maintenance_exponent, max_episodes=args.max_episodes)
    if args.which in ("decodability", "all"):
        plot_r_decodability(packed_root, args.cache, out, d_embed=args.d_embed, max_episodes=min(args.max_episodes, 12000))
    if args.which in ("separation", "all"):
        plot_delta_regret_vs_zt(packed_root, args.cache, out, d_embed=args.d_embed, max_episodes=args.max_episodes,
                                maint_lambda=args.maint_lambda, maintenance_exponent=args.maintenance_exponent)


if __name__ == "__main__":
    main()
