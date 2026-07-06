"""Meta-controller assessment plots (R-EVALUATE) — mechanism behind ``outputs/reports/normative.md``.

Does the learned ``z_t`` halting head beat hand-crafted tree-stats and a fixed stop, and in WHICH
cost regime? Four figures answer it (:func:`plot_regret_effort_frontier`, :func:`plot_r_decodability`,
:func:`plot_delta_regret_vs_zt` — the latter renders both the lambda- and maintenance-sweep panels).

The halt policies are stop-controller modules, fit on the train split and scored per-episode on eval:
  * Frac*   — one parameter ``k`` (fixed stop step), fit by minimizing fit-split regret
    (:func:`fit_fraction_stop`); the frontier of fixed-``k`` rules is its parameter space.
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


def _load_split_episodes(packed_root: Path, split: str, max_episodes: int | None = None) -> list[dict[str, Any]]:
    """Walk one split's packed shards -> per-episode dicts (halt_rewards / tree_sizes / time_budgets /
    oracle_value / oracle_stop_step / starting_budget + derived per-step heights / widths)."""
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
        for i in range(int(p["num_episodes"])):
            num_steps = int(episode_step_ptr[i + 1].item()) - int(episode_step_ptr[i].item())
            traj = int(episode_trajectory_index[i].item())
            traj_step_begin = int(trajectory_step_ptr[traj].item())
            node_begin, node_end = int(trajectory_node_ptr[traj].item()), int(trajectory_node_ptr[traj + 1].item())
            starting_budget = int(starting_budgets[i].item())
            node_cutoffs = step_node_cutoffs[traj_step_begin:traj_step_begin + num_steps].tolist()
            heights, widths = _derive_step_stats(depth[node_begin:node_end], node_cutoffs)
            episodes.append({
                "halt_rewards": trajectory_halt_rewards[traj_step_begin:traj_step_begin + num_steps].tolist(),
                "tree_sizes": node_cutoffs, "heights": heights, "widths": widths,
                "time_budgets": list(range(starting_budget, starting_budget - num_steps, -1)),
                "oracle_stop_step": int(oracle_stop_steps[i].item()),
                "oracle_value": float(oracle_values[i].item()), "starting_budget": starting_budget,
            })
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
      "always": "#C0392B", "never": "#8B97A3", "steps": "#B0B8C0",
      "frac": "#F2A9A6",  # pastel red — Frac* (fixed-stop) point, distinct from the deep-red AlwaysStop endpoint
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


# ===========================================================================
# Stop-controller modules (fit on the train split -> per-episode eval stop steps)
# ===========================================================================
def fit_fraction_stop(fit_curves: list[np.ndarray]) -> int:
    """Frac* — the ONE-parameter fixed-stop controller: the stop step ``k`` minimizing FIT-split mean
    regret (selected on the fit split, never on the eval set it is scored over)."""
    kmax = max(len(c) for c in fit_curves)
    return int(np.argmin([_regret_at(fit_curves, [k] * len(fit_curves)).mean() for k in range(kmax)]))


def _train_readout(fit_feats, ev_feats, fit_curves, *, in_dim, epochs, lr, seed) -> np.ndarray:
    """Instantiate a native readout head (``build_advantage_head``) and PG-train it (``fit_readout_pg``),
    then return greedy eval stop steps. Features are z-scored on the fit split (n_nodes ≫ steps)."""
    full = torch.cat(fit_feats, 0)
    mean, std = full.mean(0), full.std(0).clamp_min(1e-6)
    torch.manual_seed(seed)  # must precede head construction — fit_readout_pg's own seed call is too
                             # late to control weight init, since the head is already built by then
    head = build_advantage_head(in_dim, 64, 2)
    fit_readout_pg(head, [(f - mean) / std for f in fit_feats], fit_curves, epochs=epochs, lr=lr, seed=seed)
    head.eval()
    with torch.no_grad():
        return np.array([stop_step_from_advantages(head(((f - mean) / std)).reshape(-1)) for f in ev_feats])


def _split(n_episodes: int, seed: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """70/30 episode split (indices) for held-out controller fitting/eval."""
    perm = np.random.default_rng(seed).permutation(n_episodes)
    n_fit = int(round(train_frac * n_episodes))
    return perm[:n_fit], perm[n_fit:]


def _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed):
    """Load validation episodes + aligned ``z_t`` once, plus the 70/30 fit/eval episode split."""
    episodes = _load_split_episodes(packed_root, "validation", max_episodes=max_episodes)
    z_by_ep = _load_zt_by_episode(episodes, cache_path, d_embed)
    fit_idx, ev_idx = _split(len(episodes), seed)
    return episodes, z_by_ep, fit_idx, ev_idx


def _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, d_embed, seed):
    """Fit all three stop controllers on the fit split; return eval-split per-episode stop steps.

    Frac* (1 param, grid) / Stats-Controller (PG MLP, 200 ep) / Zt-Controller (PG MLP, 200 ep — same
    schedule as Zt: the original 30ep/lr=1e-2 Stats schedule was an undertrained optimization headwind,
    not a representational gap (n_nodes carries real signal independent of steps; matching Zt's epochs/lr
    closes the Stats-vs-Frac* gap without any feature engineering). Returns ``{'k_frac', 'fraction',
    'stats', 'zt'}``.
    """
    return {
        "k_frac": (kf := fit_fraction_stop(fit_curves)),
        "fraction": np.full(len(ev_idx), kf, dtype=int),
        "stats": _train_readout([_steps_stats_tensor(episodes[i]) for i in fit_idx],
                                [_steps_stats_tensor(episodes[i]) for i in ev_idx], fit_curves,
                                in_dim=4, epochs=200, lr=1e-3, seed=seed),
        "zt": _train_readout([_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in fit_idx],
                             [_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in ev_idx], fit_curves,
                             in_dim=d_embed + 1, epochs=200, lr=1e-3, seed=seed),
    }


# ===========================================================================
# Plots
# ===========================================================================
_PT_SIZE, _PT_ALPHA = 4, 0.88  # one uniform marker size/alpha for every point series — color is the only encoding


def _frontier_panel(ax, fr_x, fr, points, *, xlim, ylim, label_line=False):
    """Draw the fixed-stop frontier line and every point (Frac*/Stats/z_t/AlwaysStop/AlwaysContinue) —
    all the SAME marker/size, differing only by color, each with 95% CI error bars in x AND y."""
    ax.plot(fr_x, fr, color=_C["front"], lw=1.1, label="fixed-stop frontier" if label_line else None)
    for p in points:
        ax.errorbar([p["x"]], [p["y"]], xerr=[[p["x"] - p["xlo"]], [p["xhi"] - p["x"]]],
                    yerr=[[p["y"] - p["lo"]], [p["hi"] - p["y"]]], fmt="o", ms=_PT_SIZE, color=p["color"],
                    ecolor=p["color"], elinewidth=1.5, capsize=3.5, mec="none", alpha=_PT_ALPHA,
                    zorder=p.get("zorder", 6), label=p["label"] if label_line else None)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("average stop step"); ax.set_ylabel("Regret")
    ax.grid(axis="both", color=_GRID, lw=1)


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


def plot_regret_effort_frontier(packed_root: Path, cache_path: str | Path, out_dir: str | Path, *,
                                d_embed: int = 32, time_mode: str = "linear", time_lambda: float = 10.0,
                                maintenance_scale: float = 0.0, max_episodes: int = 15000, seed: int = 0):
    """(1) The fixed-stop frontier (search effort vs regret) with the Stats-/Zt-Controllers as points.

    A learned point BELOW the frontier beats every fixed stop at that effort. Left panel = full range,
    with a boxed+dotted-line indicator of the region the right panel zooms into; right panel = zoom on
    the controllers, each point with 95% CI error bars in both x (average stop step) and y (regret).
    """
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, [curves[i] for i in fit_idx], d_embed, seed)

    kmax = max(len(c) for c in ev)
    fr = np.array([_regret_at(ev, [k] * len(ev)).mean() for k in range(kmax)])
    fr_x = np.array([np.mean([min(k, len(c) - 1) for c in ev]) for k in range(kmax)])
    kf = min(ctrl["k_frac"], kmax - 1)

    def point(stops, color, lbl, **kw):
        xs = np.array([min(int(s), len(c) - 1) for s, c in zip(stops, ev)], dtype=float)
        xm, xlo, xhi = _mean_ci(xs)
        ym, ylo, yhi = _mean_ci(_regret_at(ev, stops))
        return dict(x=xm, xlo=xlo, xhi=xhi, y=ym, lo=ylo, hi=yhi, color=color, label=lbl, **kw)

    controllers = [point(ctrl["fraction"], _C["frac"], "Frac* (fixed stop)", zorder=7),
                  point(ctrl["stats"], _C["stats"], "Stats-Controller"),
                  point(ctrl["zt"], _C["zt"], "$z_t$-Controller")]
    ends = [point(np.zeros(len(ev_idx), dtype=int), _C["always"], "Always Stop (k=0)"),
           point(np.full(len(ev_idx), kmax - 1, dtype=int), _C["never"], "Always Continue (k=max)")]
    all_points = controllers + ends

    _rcparams()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})
    ylo, yhi = min(p["lo"] for p in controllers) - 25, max(p["hi"] for p in controllers) + 30
    zoom_xlim = (max(0, min(p["x"] for p in controllers) - 8), max(p["x"] for p in controllers) + 10)
    zoom_ylim = (ylo, yhi)
    full_ylim = (min(0, min(p["lo"] for p in all_points) - 20), max(p["hi"] for p in all_points) * 1.05)
    _frontier_panel(axL, fr_x, fr, all_points, xlim=(-1, fr_x[-1] + 1), ylim=full_ylim)
    _frontier_panel(axR, fr_x, fr, all_points, ylim=zoom_ylim, xlim=zoom_xlim, label_line=True)
    _draw_zoom_indicator(fig, axL, axR, zoom_xlim, zoom_ylim)
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              fontsize=10, frameon=False)
    save_pdf_png(fig, str(out_dir), "frontier", dpi=200, bbox_extra_artists=(fig.legends[0],))
    fm, fhi_ = controllers[0]["y"], controllers[0]["hi"]
    d_zs = bootstrap_ci(lambda d: float(d.mean()), _regret_at(ev, ctrl["zt"]) - _regret_at(ev, ctrl["stats"]), n_boot=2000)
    print(f"[assess] frontier {time_mode} λ={time_lambda} m={maintenance_scale}: Frac*(k={kf})={fm:.1f} "
          f"Stats={controllers[1]['y']:.1f} z_t={controllers[2]['y']:.1f}  paired z_t-Stats={d_zs}", flush=True)
    return {"k_frac": kf, "paired_zt_minus_stats": d_zs}


def plot_r_decodability(packed_root: Path, cache_path: str | Path, out_dir: str | Path, *,
                        d_embed: int = 32, max_episodes: int = 12000, seed: int = 0):
    """(2) Can ``R(t)`` = value of continuing (``max_{s>=t} halt_reward(s) - halt_reward(t)``) be decoded,
    and only from ``z_t``? Held-out ``R^2`` from steps / +stats / +``z_t`` vs a row-shuffle floor. ``z_t``
    decodes R several-fold better than structure, and only NONLINEARLY (linear ``R^2`` ~ shuffle floor)."""
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
    is_tr = _episode_split_mask(step_counts, total, seed=seed); is_te = ~is_tr

    feats = {"steps": np.column_stack([steps]),
             "stats": np.column_stack([steps, nnodes, heights, widths]),
             "z_t": np.column_stack([steps, z]),
             "z_t+stats": np.column_stack([steps, nnodes, heights, widths, z])}
    rows = [(name, _linear_r2(X[is_tr], R[is_tr], X[is_te], R[is_te]),
             _mlp_r2(X[is_tr], R[is_tr], X[is_te], R[is_te], seed=seed)) for name, X in feats.items()]
    Xs = np.column_stack([steps, z])[np.random.default_rng(seed + 7).permutation(total)]
    shuf_r2 = _linear_r2(Xs[is_tr], R[is_tr], Xs[is_te], R[is_te])

    _rcparams()
    mono = _C["mono"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y, h = np.arange(len(rows)), 0.30
    ax.barh(y - h / 2, [r[1] for r in rows], h, color=mono, alpha=0.5, hatch="///",
           edgecolor=_INK, linewidth=0.3, label="linear (Ridge)")
    ax.barh(y + h / 2, [r[2] for r in rows], h, color=mono, edgecolor="none", linewidth=0, label="MLP (2×64)")
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.006, i + h / 2, f"{r[2]:.2f}", va="center", fontsize=10, fontweight="bold", color=_INK)
    ax.axvline(shuf_r2, color=_MUTED, ls="--", lw=1.3, label=f"shuffle floor ({shuf_r2:+.2f})")
    ax.axvline(0, color=_MUTED, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(["steps\n(C-step)", "+ stats\n(C-maint)", "+ $z_t$\n(R via GNN)", "$z_t$ + stats"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("held-out $R^2$ — predicting R(t) = value of continuing")
    ax.grid(axis="x", color=_GRID, lw=1)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    save_pdf_png(fig, str(out_dir), "decodability", dpi=200)
    out = {r[0]: {"linear": r[1], "mlp": r[2]} for r in rows}
    print(f"[assess] R-decodability  " + "  ".join(f"{k}={v['mlp']:.3f}" for k, v in out.items()), flush=True)
    return out


_DELTA_CANDIDATES = [("frac", _C["frac"], "Frac*"), ("stats", _C["stats"], "Stats-Controller"),
                    ("always_stop", _C["always"], "Always Stop"), ("always_continue", _C["never"], "Always Continue")]


def _regime_deltas_vs_zt(episodes, z_by_ep, fit_idx, ev_idx, base_cfg, mode, lam, mnt, d_embed, seed):
    """Per-episode paired delta = candidate_regret − z_t_regret for {frac, stats, always_stop,
    always_continue} under one cost regime (mode/lam/mnt). Positive = z_t beats the candidate."""
    curves = _return_curves(episodes, replace(base_cfg, time_mode=mode, time_lambda=lam, maintenance_scale=mnt))
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, [curves[i] for i in fit_idx], d_embed, seed)
    kmax = max(len(c) for c in ev)
    zt_regret = _regret_at(ev, ctrl["zt"])
    stops = {"frac": ctrl["fraction"], "stats": ctrl["stats"],
            "always_stop": np.zeros(len(ev_idx), dtype=int),
            "always_continue": np.full(len(ev_idx), kmax - 1, dtype=int)}
    return {name: _regret_at(ev, s) - zt_regret for name, s in stops.items()}


def _delta_ci_panel(ax, regime_labels, deltas_by_regime, candidates=_DELTA_CANDIDATES):
    """Horizontal dot-and-whisker: one row per regime, one 95% CI point per candidate (offset within
    the row) — mean Δ regret vs z_t, same marker/size convention as the frontier plot's points."""
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
    ax.grid(axis="both", color=_GRID, lw=1)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(candidates), frameon=False)


def plot_delta_regret_vs_zt(packed_root: Path, cache_path: str | Path, out_dir: str | Path, *,
                            d_embed: int = 32, max_episodes: int = 15000, seed: int = 0,
                            lambda_grid: tuple[float, ...] = (5.0, 10.0, 40.0), maint_lambda: float = 10.0,
                            maint_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.30, 1.0)):
    """(3) Δ mean regret (model − $z_t$) for {Frac*, Stats-Controller, AlwaysStop, AlwaysContinue} —
    z_t is always the baseline, so positive = z_t wins. Two horizontal-violin figures: (A) sweep the
    linear time cost λ at maintenance=0 (isolates the C-step/R edge); (B) sweep maintenance_scale at a
    fixed λ (isolates the weaker C-maint edge). Grids match the constant-cost regime and maintenance
    probe already reported in R-EVALUATE (see outputs/reports/normative.md)."""
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed)
    base_cfg = _oracle_config(packed_root)

    def regime(mode, lam, mnt):
        return _regime_deltas_vs_zt(episodes, z_by_ep, fit_idx, ev_idx, base_cfg, mode, lam, mnt, d_embed, seed)

    _rcparams()
    labels_a = [f"linear λ={lam:g}" for lam in lambda_grid]
    deltas_a = [regime("linear", lam, 0.0) for lam in lambda_grid]
    figA, axA = plt.subplots(figsize=(9.5, 1.9 + 1.05 * len(labels_a)))
    _delta_ci_panel(axA, labels_a, deltas_a)
    axA.set_title("varying linear cost λ  (maintenance = 0)", fontsize=12, loc="left")
    save_pdf_png(figA, str(out_dir), "delta_mean_regret_lambda", dpi=200)

    labels_b = [f"maint={mnt:g}" for mnt in maint_grid]
    deltas_b = [regime("linear", maint_lambda, mnt) for mnt in maint_grid]
    figB, axB = plt.subplots(figsize=(9.5, 1.9 + 1.05 * len(labels_b)))
    _delta_ci_panel(axB, labels_b, deltas_b)
    axB.set_title(f"varying maintenance scale  (linear λ={maint_lambda:g})", fontsize=12, loc="left")
    save_pdf_png(figB, str(out_dir), "delta_mean_regret_maintenance", dpi=200)

    for group_label, labels, deltas in [("lambda", labels_a, deltas_a), ("maintenance", labels_b, deltas_b)]:
        for lbl, d in zip(labels, deltas):
            summary = "  ".join(f"{k}={_mean_ci(d[k])[0]:+.1f}" for k, _, _ in _DELTA_CANDIDATES)
            print(f"[assess] delta[{group_label}] {lbl:16s} {summary}", flush=True)
    return {"lambda": dict(zip(labels_a, deltas_a)), "maintenance": dict(zip(labels_b, deltas_b))}


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="R-EVALUATE meta-controller assessment plots")
    ap.add_argument("--packed-root", required=True, help="MC packed dir with validation/ + validation_manifest.json")
    ap.add_argument("--cache", required=True, help="validation materialized cache (.pt); MUST be shuffle=False")
    ap.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/normative")
    ap.add_argument("--which", choices=["frontier", "decodability", "separation", "all"], default="all")
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--time-mode", default="linear")
    ap.add_argument("--time-lambda", type=float, default=10.0)
    ap.add_argument("--maintenance-scale", type=float, default=0.0)
    ap.add_argument("--maint-lambda", type=float, default=10.0, help="lambda held fixed in the maintenance sweep")
    args = ap.parse_args()

    packed_root, out = Path(args.packed_root), args.out_dir
    if args.which in ("frontier", "all"):
        plot_regret_effort_frontier(packed_root, args.cache, out, d_embed=args.d_embed, time_mode=args.time_mode,
                                    time_lambda=args.time_lambda, maintenance_scale=args.maintenance_scale,
                                    max_episodes=args.max_episodes)
    if args.which in ("decodability", "all"):
        plot_r_decodability(packed_root, args.cache, out, d_embed=args.d_embed, max_episodes=min(args.max_episodes, 12000))
    if args.which in ("separation", "all"):
        plot_delta_regret_vs_zt(packed_root, args.cache, out, d_embed=args.d_embed, max_episodes=args.max_episodes,
                                maint_lambda=args.maint_lambda)


if __name__ == "__main__":
    main()
