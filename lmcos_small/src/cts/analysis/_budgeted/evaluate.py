"""Five-model readout comparison on the budgeted oracle (SF-2000 ladder trees).

Every model emits a per-step advantage and STOPS at the first step with ``A <= 0``
(continue iff ``A > 0``) — ONE decision rule across all five tiers, so the comparison
isolates the *advantage signal*. The five tiers:

  1. Always Stop          -- stop@0                                (no fit)
  2. Never Stop           -- full budget                           (no fit)
  3. Fraction-of-Budget   -- stop at round(theta*B)               (theta fit on regret)
  4. Readout(tree-stats)  -- MLP on [height, width, n_nodes, T_t]  (policy gradient)
  5. Readout(GNN-z)       -- MLP on [z_t, T_t]                      (policy gradient; checkpoint)

Tiers 4 and 5 are trained by POLICY GRADIENT on the exact expected return — the SAME
objective (see ``cts.train.pg_controller_train``) — so 4-vs-5 isolates the
REPRESENTATION (hand-crafted tree stats vs learned ``z_t``), not the objective. Both
read the per-step remaining budget ``T_t``. (``--stats-objective regret_threshold``
recovers the legacy §10b BCE-backbone + regret-tuned-threshold fit for tier 4.)

Metrics on the held-out validation split:
  * Regret         = oracle_value - return_for_stop_step(...) (minimize); mean +
                     bootstrap 95% CI, plus median / p90 / p99 (the tail the mean hides)
  * P(stop == OSS) = fraction whose chosen stop step == the DP-optimal stop
  * Expansions     = mean steps before halt = the COMPUTE spent; regret-vs-expansions
                     is the real tradeoff (regret_vs_compute.png: the Fraction-θ curve
                     with its θ* optimum, plus the learned readouts off the curve)

height / width are NOT stored per-snapshot in the packed shards; they are derived here
from the trajectory ``depth`` array + ``step_node_cutoffs`` (``height = max depth over
the first n_nodes``, ``width = max nodes at any one depth``). ``n_nodes`` is present.

    python -m cts.analysis._budgeted.evaluate \
        --packed-root /scratch/gpfs/GRIFFITHS/hl4291/sf_mc_packed/elo2000 \
        --controller-checkpoint /scratch/.../sf_mchalt_pg.pt \
        --materialized-validation-cache /scratch/.../validation_cache.pt \
        --out-dir figures/normative
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
    return_for_stop_step,
)
from cts.analysis._budgeted.baselines import _evaluate_baseline
from cts.stats import bootstrap_mean_ci as _bootstrap_ci
from cts.models.readout import StatsReadout, stop_step_from_advantages


def _derive_step_stats(traj_depth: torch.Tensor, node_cutoffs: list[int]) -> tuple[list[int], list[int]]:
    """Per-step (height, width) from a trajectory's node depths + per-step node cutoffs.

    ``traj_depth`` is the depth of every node in the trajectory's full tree, in
    node order (root first, so the first ``node_cutoff`` entries are exactly the
    prefix tree visible at that step). For each step: ``height`` = the deepest
    node in the prefix; ``width`` = the most nodes sharing any single depth.
    """
    heights: list[int] = []
    widths: list[int] = []
    for node_cutoff in node_cutoffs:
        prefix_depth = traj_depth[:node_cutoff]
        heights.append(int(prefix_depth.max().item()))
        widths.append(int(torch.bincount(prefix_depth).max().item()))
    return heights, widths


def _load_split_episodes(packed_root: Path, split: str, max_episodes: int | None = None) -> list[dict[str, Any]]:
    """Walk one split's packed shards -> per-episode dicts with derived tree stats.

    Mirrors controller_train._extract_all_episode_metadata but emits the plain
    dict the baselines consume (halt_rewards / tree_sizes / time_budgets /
    oracle_value / oracle_stop_step) PLUS per-step ``heights`` / ``widths``
    derived from the trajectory ``depth`` array (the StatsReadout inputs).
    """
    shard_paths = sorted((packed_root / split).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.pt under {packed_root / split}")
    episodes: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        p = torch.load(shard_path, weights_only=False)
        episode_step_ptr = p["episode_step_ptr"]
        episode_trajectory_index = p["episode_trajectory_index"]
        trajectory_step_ptr = p["trajectory_step_ptr"]
        trajectory_node_ptr = p["trajectory_node_ptr"]
        step_node_cutoffs = p["step_node_cutoffs"]
        trajectory_halt_rewards = p["trajectory_halt_rewards"]
        oracle_stop_steps = p["oracle_stop_steps"]
        oracle_values = p["oracle_values"]
        starting_budgets = p["starting_budgets"]
        depth = p["depth"]
        num_episodes = int(p["num_episodes"])
        for i in range(num_episodes):
            num_steps = int(episode_step_ptr[i + 1].item()) - int(episode_step_ptr[i].item())
            traj = int(episode_trajectory_index[i].item())
            traj_step_begin = int(trajectory_step_ptr[traj].item())
            node_begin = int(trajectory_node_ptr[traj].item())
            node_end = int(trajectory_node_ptr[traj + 1].item())
            traj_depth = depth[node_begin:node_end]
            starting_budget = int(starting_budgets[i].item())
            node_cutoffs = step_node_cutoffs[traj_step_begin:traj_step_begin + num_steps].tolist()
            heights, widths = _derive_step_stats(traj_depth, node_cutoffs)
            episodes.append({
                "halt_rewards": trajectory_halt_rewards[traj_step_begin:traj_step_begin + num_steps].tolist(),
                "tree_sizes": node_cutoffs,
                "heights": heights,
                "widths": widths,
                "time_budgets": list(range(starting_budget, starting_budget - num_steps, -1)),
                "oracle_stop_step": int(oracle_stop_steps[i].item()),
                "oracle_value": float(oracle_values[i].item()),
                "starting_budget": starting_budget,
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


# --- the trivial / one-parameter models as stop-step rules ------------------
def _always_stop(episode: dict[str, Any]) -> int:
    return 0


def _never_stop(episode: dict[str, Any]) -> int:
    return len(episode["halt_rewards"]) - 1


def _fraction_rule(f: float) -> Callable[[dict[str, Any]], int]:
    """Tier-3 rule: continue iff N_t < f*B, i.e. stop at round(f*B). (== FractionStop)."""
    def rule(episode: dict[str, Any]) -> int:
        return int(round(f * int(episode["starting_budget"])))
    return rule


def _mean_regret(episodes: list[dict[str, Any]], config: BudgetedOracleConfig,
                 stop_rule: Callable[[dict[str, Any]], int]) -> float:
    """Mean regret of an arbitrary stop rule over ``episodes`` (the fit objective)."""
    total = 0.0
    for episode in episodes:
        stop = max(0, min(stop_rule(episode), len(episode["halt_rewards"]) - 1))
        ret = return_for_stop_step(
            episode["halt_rewards"], episode["tree_sizes"], episode["time_budgets"], stop, config
        )
        total += float(episode["oracle_value"]) - ret
    return total / len(episodes)


def _fit_fraction(train: list[dict[str, Any]], config: BudgetedOracleConfig) -> tuple[float, list[tuple[float, float]]]:
    """Grid-search f in (0,1) to minimize mean TRAIN regret (regret-direct selection)."""
    grid = [round(0.01 * k, 2) for k in range(1, 100)]
    curve = [(f, _mean_regret(train, config, _fraction_rule(f))) for f in grid]
    best_f, _ = min(curve, key=lambda fr: fr[1])
    return best_f, curve


# --- tier 4: StatsReadout, fit then SELECTED on regret directly -------------
def _stats_features(episode: dict[str, Any]) -> torch.Tensor:
    """Per-step ``[height, width, n_nodes]`` feature matrix for one episode."""
    return torch.tensor(
        [list(triple) for triple in zip(episode["heights"], episode["widths"], episode["tree_sizes"])],
        dtype=torch.float32,
    )


def _fit_stats_readout(
    train: list[dict[str, Any]],
    config: BudgetedOracleConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[StatsReadout, float]:
    """Fit a StatsReadout MLP, then SELECT its decision threshold on TRAIN regret.

    The MLP backbone is trained against the oracle ``continue?`` label (step <
    oracle_stop_step) so it learns which steps are worth continuing, but — the
    §10b fix — the model that gets *deployed* is chosen by sweeping an additive
    threshold on the predicted advantage to minimize mean TRAIN regret directly.
    The continue rule becomes ``A - tau > 0``; tau is picked on regret, never on
    the surrogate loss.
    """
    torch.manual_seed(seed)
    model = StatsReadout()
    # Build the full per-snapshot input matrix ONCE (vectorized; the per-episode
    # python loop only runs here, not every epoch). path_lengths lets us split
    # the flat predictions back into episodes for the regret threshold sweep.
    path_lengths = [len(ep["heights"]) for ep in train]
    feats = torch.cat([_stats_features(ep) for ep in train], dim=0)
    budgets = torch.cat(
        [torch.tensor(ep["time_budgets"], dtype=torch.float32).unsqueeze(1)  # per-step remaining T_t (parity with GNN-z)
         for n, ep in zip(path_lengths, train)], dim=0
    )
    full = torch.cat([feats, budgets], dim=-1)
    # z-score so the MLP trains stably (height/width/n_nodes/B differ by orders
    # of magnitude). Stats computed on TRAIN only.
    mean = full.mean(dim=0)
    std = full.std(dim=0).clamp_min(1e-6)
    normed = (full - mean) / std  # [total_snapshots, 4], reused every epoch

    target_all = torch.cat(
        [torch.tensor([1.0 if c < ep["oracle_stop_step"] else 0.0 for c in range(n)], dtype=torch.float32)
         for n, ep in zip(path_lengths, train)],
        dim=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model.head(normed).squeeze(-1)
        loss = loss_fn(preds, target_all)
        loss.backward()
        opt.step()

    # --- regret-direct selection: tune the additive threshold tau on TRAIN ---
    model.eval()
    with torch.no_grad():
        flat_adv = model.head(normed).squeeze(-1).detach()
    cached = list(torch.split(flat_adv, path_lengths))

    best_tau, best_regret = 0.0, float("inf")
    for tau in [round(0.5 * k, 1) for k in range(-8, 9)]:
        total = 0.0
        for idx, ep in enumerate(train):
            stop = max(0, min(stop_step_from_advantages(cached[idx] - tau), len(ep["halt_rewards"]) - 1))
            ret = return_for_stop_step(ep["halt_rewards"], ep["tree_sizes"], ep["time_budgets"], stop, config)
            total += float(ep["oracle_value"]) - ret
        mean_regret = total / len(train)
        if mean_regret < best_regret:
            best_tau, best_regret = tau, mean_regret

    # Bake the normalization + tuned threshold onto the model for the eval wrapper.
    model._norm_mean = mean  # type: ignore[attr-defined]
    model._norm_std = std  # type: ignore[attr-defined]
    model._tau = best_tau  # type: ignore[attr-defined]
    return model, best_tau


def _fit_stats_readout_pg(
    train: list[dict[str, Any]],
    config: BudgetedOracleConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-2,
    seed: int = 0,
    episode_batch: int = 512,
) -> tuple[StatsReadout, float]:
    """Fit the StatsReadout by POLICY GRADIENT (exact expected-return) — the SAME
    objective the GNN-z controller's pg trainer uses. The head's scalar output is the
    continue logit; the stop step is marginalized in closed form over the full trace.
    So tier-4 (this) vs tier-5 (GNN-z) isolates the REPRESENTATION (hand-crafted tree
    stats vs learned z_t), not the training objective. Deploys greedily (stop at first
    advantage <= 0; no threshold tau).
    """
    from cts.train.pg_controller_train import expected_regret_batched, _scatter_to_padded, _pad_scalars

    torch.manual_seed(seed)
    model = StatsReadout()
    path_lengths = [len(ep["heights"]) for ep in train]
    feats = torch.cat([_stats_features(ep) for ep in train], dim=0)
    budgets = torch.cat(
        [torch.tensor(ep["time_budgets"], dtype=torch.float32).unsqueeze(1)  # per-step remaining T_t (parity with GNN-z)
         for n, ep in zip(path_lengths, train)], dim=0
    )
    full = torch.cat([feats, budgets], dim=-1)
    mean = full.mean(dim=0)
    std = full.std(dim=0).clamp_min(1e-6)

    ep_x, ep_g, ep_ov, off = [], [], [], 0
    for n, ep in zip(path_lengths, train):
        ep_x.append((full[off:off + n] - mean) / std)
        off += n
        g = [return_for_stop_step(ep["halt_rewards"], ep["tree_sizes"], ep["time_budgets"], s, config)
             for s in range(n)]
        ep_g.append(torch.tensor(g, dtype=torch.float32))
        ep_ov.append(float(ep["oracle_value"]))

    g_pad, mask, lengths, ov = _pad_scalars(ep_g, ep_ov, torch.device("cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_ep = len(ep_x)
    for _ in range(epochs):
        perm = torch.randperm(n_ep)
        for i in range(0, n_ep, episode_batch):
            idxs = perm[i:i + episode_batch]
            opt.zero_grad()
            cat = torch.cat([ep_x[j] for j in idxs.tolist()], dim=0)
            A_flat = model.head(cat).reshape(-1)
            L = lengths[idxs]
            A_pad = _scatter_to_padded(A_flat, L)
            t = A_pad.shape[1]
            loss = expected_regret_batched(A_pad, g_pad[idxs, :t], mask[idxs, :t], L, ov[idxs])
            loss.backward()
            opt.step()

    model.eval()
    model._norm_mean = mean  # type: ignore[attr-defined]
    model._norm_std = std  # type: ignore[attr-defined]
    model._tau = 0.0  # type: ignore[attr-defined]  # greedy: stop at advantage <= 0
    return model, 0.0


def _stats_rule(model: StatsReadout) -> Callable[[dict[str, Any]], int]:
    """Deployable stop rule for a fitted StatsReadout (applies its tuned threshold)."""
    mean = model._norm_mean  # type: ignore[attr-defined]
    std = model._norm_std  # type: ignore[attr-defined]
    tau = model._tau  # type: ignore[attr-defined]

    def rule(episode: dict[str, Any]) -> int:
        f = _stats_features(episode)
        b = torch.tensor(episode["time_budgets"], dtype=torch.float32).unsqueeze(1)  # per-step T_t (parity with GNN-z)
        x = (torch.cat([f, b], dim=-1) - mean) / std
        with torch.no_grad():
            adv = model.head(x).squeeze(-1) - tau
        return stop_step_from_advantages(adv)

    return rule


# --- figure style (matches the human_analytics engine/board plots) ----------
_MAIN_COLOR = "#2E86C1"      # steel blue (human_analytics MAIN_COLOR)
_LEARNED_COLOR = "#C0392B"   # brick red — learned readouts / over-search
_STAR_COLOR = "#F1C40F"      # gold — the fitted operating point (theta*)
_PENDING_COLOR = "#cccccc"

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


def _style_ax(ax) -> None:
    ax.set_axisbelow(True)  # grid behind the data (spines/grid come from rcParams)


def _save_fig(fig, out_path: Path) -> None:
    """Save PNG + PDF at dpi 300 (matches human_analytics save_figure)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  saved {out_path.with_suffix('.png')} (+ .pdf)")


def _fraction_operating_curve(episodes: list[dict[str, Any]], config: BudgetedOracleConfig,
                              thetas: list[float]) -> list[tuple[float, float]]:
    """(mean_expansions, mean_regret) per theta for the Fraction-θ family.

    This is the one-parameter operating curve whose endpoints are Always-Stop
    (θ=0, 0 compute) and Never-Stop (θ=1, full budget) and whose minimum is θ*.
    Drawn behind the discrete models so the vertical gap a learned readout opens
    BELOW it (= how much it beats blind fraction at the same compute) is visible.
    """
    n = len(episodes)
    pts: list[tuple[float, float]] = []
    for th in thetas:
        rule = _fraction_rule(th)
        regret = _mean_regret(episodes, config, rule)
        exps = sum(max(0, min(rule(ep), len(ep["halt_rewards"]) - 1)) for ep in episodes) / n
        pts.append((exps, regret))
    return pts


def _log_axis(ax, which: str) -> None:
    """Log-scale the given axis with clean 1/2/5-per-decade decimal tick labels (regret is
    > 0, so the near-zero models separate out without the minor-tick label clutter)."""
    from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
    if which == "x":
        ax.set_xscale("log")
        axis = ax.xaxis
    else:
        ax.set_yscale("log")
        axis = ax.yaxis
    axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))   # 0.1,0.2,0.5,1,2,...
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(1, 10))))
    axis.set_minor_formatter(NullFormatter())                            # gridlines only, no labels


def _plot(labels: list[str], values: list[float], pending_last: bool, *, title: str,
          xlabel: str, out_path: Path, cis: list[tuple[float, float]] | None = None,
          log_x: bool = False) -> None:
    """Horizontal comparison, models top->bottom in `labels` order (tier 1->5).

    Default: bars + value labels. If ``cis`` (per-model 95% CIs) is given, render points
    with horizontal 95%-CI error bars instead and drop the value labels — used for regret,
    where the CI matters more than the number. ``log_x`` log-scales the value axis.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    y = list(range(n))[::-1]  # reverse so labels[0] is at the top
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if cis is not None:
        for yi, val, ci in zip(y, values, cis):
            lo, hi = ci
            if lo == lo and hi == hi:  # finite CI -> point + horizontal error bar
                ax.errorbar([val], [yi], xerr=[[max(val - lo, 0.0)], [max(hi - val, 0.0)]],
                            fmt="o", color=_MAIN_COLOR, ecolor=_MAIN_COLOR, capsize=4,
                            markersize=6, zorder=3)
            else:
                ax.scatter([val], [yi], color=_PENDING_COLOR, zorder=3)  # scalar / pending: no CI
        finite = [c for c in cis if c[0] == c[0]]
        if log_x:
            _log_axis(ax, "x")
            if finite:  # tight bounds so the axis doesn't run off to 0.01
                ax.set_xlim(min(c[0] for c in finite) * 0.8, max(c[1] for c in finite) * 1.3)
        ax.margins(y=0.15)
    else:
        colors = [_MAIN_COLOR] * n
        plot_values = list(values)
        if pending_last:
            colors[-1] = _PENDING_COLOR
            plot_values[-1] = 0.0
        bars = ax.barh(y, plot_values, color=colors)
        for idx, (bar, val) in enumerate(zip(bars, values)):
            if pending_last and idx == n - 1:
                bar.set_hatch("//")
                ax.annotate("pending", (0.0, bar.get_y() + bar.get_height() / 2),
                            va="center", ha="left", fontsize=9, color="#888")
            # numeric value labels omitted — the bar + CI carry the magnitude

    if cis is None:
        ax.margins(x=0.15)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_tradeoff(labels: list[str], regrets: list[float], expansions: list[float],
                   pending_last: bool, frac_curve: list[tuple[float, float]], *, out_path: Path,
                   cis: list[tuple[float, float]] | None = None) -> None:
    """Regret vs compute, against the Fraction-θ operating curve.

    The Fraction-θ family is drawn as its continuous curve (θ: 0=Always → 1=Never,
    minimum at θ*), with the fitted θ* marked as a star. The learned readouts
    (tree-stats, GNN-z) sit as points; their vertical gap BELOW the curve is how
    much better than blind fraction they are AT THE SAME COMPUTE. Indices are the
    fixed tier order: 0 always, 1 never, 2 fraction(θ*), 3 stats, 4 mchalt.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    _style_ax(ax)

    # Fraction-θ sweep curve (the U). The STAR is its optimal MEMBER (min-regret θ*),
    # highlighted but part of the sweep — not a separate model.
    fx = [e for e, _ in frac_curve]
    fy = [r for _, r in frac_curve]
    ax.plot(fx, fy, "-", color=_MAIN_COLOR, lw=1.8, zorder=2,
            label=r"Fraction-$\theta$ sweep ($\theta$: 0$\to$1, step 0.1)")
    ax.scatter(fx, fy, color=_MAIN_COLOR, s=14, zorder=3)
    opt = min(range(len(frac_curve)), key=lambda k: fy[k])  # optimal member of the sweep
    ax.scatter([fx[opt]], [fy[opt]], marker="X", s=140, color=_STAR_COLOR,
               edgecolor="#7d6608", linewidth=0.8, zorder=6, label=r"$\theta^*$ (sweep optimum)")

    # Learned readouts as off-curve points (the comparison): tree-stats, GNN-z, with
    # 95% CI bars on regret (the y-axis is log, so error bars are asymmetric there).
    learned = [3] + ([] if pending_last else [4])
    for i in learned:
        ci = cis[i] if cis else None
        yerr = ([[max(regrets[i] - ci[0], 0.0)], [max(ci[1] - regrets[i], 0.0)]]
                if ci and ci[0] == ci[0] else None)
        ax.errorbar([expansions[i]], [regrets[i]], yerr=yerr, fmt="o", color=_LEARNED_COLOR,
                    ecolor=_LEARNED_COLOR, capsize=4, markersize=7, zorder=5)
        ax.annotate(labels[i], (expansions[i], regrets[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    # The sweep endpoints are Always (θ=0) and Never (θ=1).
    ax.annotate(r"Always ($\theta$=0)", (fx[0], fy[0]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.annotate(r"Never ($\theta$=1)", (fx[-1], fy[-1]), textcoords="offset points",
                xytext=(-6, 6), fontsize=8, ha="right")

    _log_axis(ax, "y")  # regret on log scale (>0; spreads the near-zero models)
    ax.set_xlabel("Mean Expansions\n" + r"$\leftarrow$ cheaper")
    ax.set_ylabel("Mean Regret  (log)\n" + r"$\leftarrow$ better")
    ax.set_title(r"Regret vs compute — learned readouts vs the Fraction-$\theta$ curve")
    ax.margins(0.16)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_regret_distribution(labels: list[str], metric_dicts: list[dict[str, Any] | None],
                              *, out_path: Path) -> None:
    """Per-model regret distribution: median dot, mean diamond, p90/p99 tail ticks.

    Surfaces what the mean alone hides — when the mean sits far right of the median,
    a few catastrophic stops carry the regret. Models without a per-episode fit
    (the scalar/pending GNN-z slot) are skipped.
    """
    items = [(lab, d) for lab, d in zip(labels, metric_dicts) if d and "regret_median" in d]
    if not items:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labs = [lab for lab, _ in items]
    n = len(items)
    y = list(range(n))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, (yi, (_, d)) in enumerate(zip(y, items)):
        med, p90, p99, mean = d["regret_median"], d["regret_p90"], d["regret_p99"], d["average_regret"]
        first = k == 0
        ax.plot([med, p99], [yi, yi], color="#B8B8B8", lw=2, zorder=1)
        ax.scatter([med], [yi], color=_MAIN_COLOR, s=45, zorder=3, label="median" if first else None)
        ax.scatter([p90], [yi], marker="|", color="#444", s=130, zorder=3, label="p90" if first else None)
        ax.scatter([p99], [yi], marker="|", color="#999", s=130, zorder=3, label="p99" if first else None)
        ax.scatter([mean], [yi], marker="D", color=_LEARNED_COLOR, s=35, zorder=4, label="mean" if first else None)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("regret  (median ● · mean ◆ · p90/p99 ticks · tail line median→p99)")
    ax.set_title("Regret distribution by model  (mean ≫ median ⇒ heavy tail)")
    ax.legend(loc="lower right", fontsize=8)
    ax.margins(x=0.12)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_regret_decomposition(labels: list[str], metric_dicts: list[dict[str, Any] | None],
                               *, out_path: Path) -> None:
    """Stacked bars: mean regret split into under-search (stop<OSS) + over-search (stop>OSS).

    The two pieces sum to mean regret by construction, so this reads off *why* a
    policy loses value — stopping too soon (missed value) vs too late (wasted cost).
    """
    items = [(lab, d) for lab, d in zip(labels, metric_dicts) if d and "regret_from_early" in d]
    if not items:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labs = [lab for lab, _ in items]
    early = [d["regret_from_early"] for _, d in items]
    late = [d["regret_from_late"] for _, d in items]
    n = len(items)
    y = list(range(n))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(y, early, color=_MAIN_COLOR, label="under-search (stop < OSS)")
    ax.barh(y, late, left=early, color=_LEARNED_COLOR, label="over-search (stop > OSS)")
    for yi, e, l in zip(y, early, late):
        ax.annotate(f"{e + l:.3f}", (e + l, yi), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("mean regret  ( = under-search + over-search )")
    ax.set_title("Regret decomposition: under- vs over-searching")
    ax.legend(loc="lower right", fontsize=8)
    ax.margins(x=0.12)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-root", required=True, help="MC packed dir with train/ validation/ + *_manifest.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap episodes loaded per split (means/fractions are stable on a large sample)")
    ap.add_argument("--gnn-regret", type=float, default=None,
                    help="tier-5 MCHalt mean regret SCALAR; fills its slot (back-compat; "
                         "ignored when --controller-checkpoint is given)")
    ap.add_argument("--gnn-stop", type=float, default=None,
                    help="tier-5 MCHalt P(stop==OSS) SCALAR; fills its slot (back-compat; "
                         "ignored when --controller-checkpoint is given)")
    ap.add_argument("--controller-checkpoint", default=None,
                    help="trained MCHalt MetaController checkpoint (.pt). When given, tier 5 is "
                         "scored for real over the validation materialized cache and takes "
                         "precedence over --gnn-regret/--gnn-stop.")
    ap.add_argument("--materialized-validation-cache", default=None,
                    help="validation materialized-cache index (.pt) the controller is scored over; "
                         "required with --controller-checkpoint")
    ap.add_argument("--controller-device", default="cpu",
                    help="device for the controller forward pass (default cpu)")
    ap.add_argument("--results-json", default=None,
                    help="if set, write {model: {regret, stop_acc}} for all scored tiers to this "
                         "JSON path (consumed by the cross-Elo ladder plotter)")
    ap.add_argument("--stats-objective", choices=["pg", "regret_threshold"], default="pg",
                    help="tier-4 StatsReadout fit: 'pg' (policy-gradient expected return — "
                         "matches the GNN-z controller's objective, so tier4-vs-tier5 isolates the "
                         "representation) or 'regret_threshold' (legacy §10b BCE backbone + regret-tuned threshold).")
    args = ap.parse_args()

    # The fit is a tiny full-batch MLP; cap BLAS threads so it doesn't thrash on
    # a shared/oversubscribed node (a single thread is faster here than many).
    torch.set_num_threads(1)

    packed_root = Path(args.packed_root)
    config = _oracle_config(packed_root)
    train = _load_split_episodes(packed_root, "train", args.max_episodes)
    test = _load_split_episodes(packed_root, "validation", args.max_episodes)
    print(f"loaded train={len(train):,} episodes  test={len(test):,} episodes")

    # Tier 3: Fraction theta, fit on TRAIN regret directly.
    best_f, _ = _fit_fraction(train, config)
    print(f"tier 3 Fraction theta* = {best_f:.2f} (min mean TRAIN regret)")

    # Tier 4: StatsReadout — PG (same objective as the GNN-z controller) by default, so
    # tier4-vs-tier5 isolates the representation; --stats-objective regret_threshold
    # recovers the legacy §10b BCE+threshold fit.
    if args.stats_objective == "pg":
        stats_model, _ = _fit_stats_readout_pg(train, config)
        stats_tier_label = "4. Readout(tree-stats, PG)"
        print("tier 4 StatsReadout: policy-gradient expected-return fit (greedy stop)")
    else:
        stats_model, stats_tau = _fit_stats_readout(train, config)
        stats_tier_label = "4. Readout(tree-stats)"
        print(f"tier 4 StatsReadout tau* = {stats_tau:+.1f} (decision threshold tuned on TRAIN regret)")

    labels = ["1. Always Stop", "2. Never Stop", f"3. Fraction theta*={best_f:.2f}",
              stats_tier_label, "5. Readout(GNN-z)"]
    rules: list[Callable[[dict[str, Any]], int]] = [
        _always_stop, _never_stop, _fraction_rule(best_f), _stats_rule(stats_model),
    ]

    model_keys = ["always", "never", "fraction", "stats", "mchalt"]
    metric_dicts: list[dict[str, Any] | None] = []

    print(f"\n{'model':<26s} {'regret':>10s} {'P(stop==OSS)':>14s} {'avg_expansions':>15s}")
    for name, rule in zip(labels[:4], rules):
        m = _evaluate_baseline(test, config, rule)
        metric_dicts.append(m)
        print(f"{name:<26s} {m['average_regret']:>10.4f} {m['exact_stop_step_accuracy']:>14.3f} {m['average_expansions']:>15.2f}")

    # Tier 5 (MCHalt): scored for real from a trained controller checkpoint when
    # one is supplied; else the back-compat scalar slot; else 'pending'.
    mc: dict[str, Any] | None = None
    if args.controller_checkpoint is not None:
        if args.materialized_validation_cache is None:
            raise ValueError(
                "--controller-checkpoint requires --materialized-validation-cache "
                "(the cache the controller is scored over)."
            )
        from cts.analysis._budgeted.mchalt_scorer import score_mchalt_checkpoint
        mc = score_mchalt_checkpoint(
            checkpoint_path=args.controller_checkpoint,
            validation_manifest=str(packed_root / "validation_manifest.json"),
            validation_cache_index=args.materialized_validation_cache,
            oracle_config=config,
            max_episodes=args.max_episodes,
            device=args.controller_device,
        )
        print(f"{labels[4]:<26s} {mc['average_regret']:>10.4f} {mc['exact_stop_step_accuracy']:>14.3f} "
              f"{mc['average_expansions']:>15.2f}  (checkpoint, n={int(mc['evaluated_episodes'])})")
    elif args.gnn_regret is not None:
        mc = {"average_regret": args.gnn_regret, "exact_stop_step_accuracy": args.gnn_stop or 0.0}
        print(f"{labels[4]:<26s} {mc['average_regret']:>10.4f} {mc['exact_stop_step_accuracy']:>14.3f}  (scalar)")
    metric_dicts.append(mc)

    pending = mc is None
    regret_full = [float(d["average_regret"]) if d else 0.0 for d in metric_dicts]
    stop_full = [float(d["exact_stop_step_accuracy"]) if d else 0.0 for d in metric_dicts]
    expansions_full = [float(d["average_expansions"]) if d else 0.0 for d in metric_dicts]

    # Tier-1 goodness-of-fit table: how well each policy's stop matches the oracle's
    # OSS, decomposed into under-/over-search (reg_early + reg_late == mean regret).
    print(f"\n{'model':<26s} {'stop_bias':>9s} {'stop_MAE':>8s} {'tol@1':>6s} {'tol@2':>6s} "
          f"{'%early':>7s} {'%late':>6s} {'reg_early':>9s} {'reg_late':>9s}")
    for label, d in zip(labels, metric_dicts):
        if not d or "stop_bias" not in d:
            print(f"{label:<26s} {'(no per-episode fit — scalar/pending)':>}")
            continue
        print(f"{label:<26s} {d['stop_bias']:>+9.2f} {d['stop_mae']:>8.2f} {d['tol_acc_1']:>6.3f} "
              f"{d['tol_acc_2']:>6.3f} {100*d['frac_early']:>6.1f}% {100*d['frac_late']:>5.1f}% "
              f"{d['regret_from_early']:>9.4f} {d['regret_from_late']:>9.4f}")

    # Regret DISTRIBUTION + compute: the mean alone hides tails and ignores the
    # cost actually paid. expansions = mean steps before halt (the compute axis).
    # Bootstrap 95% CI on each model's mean regret.
    cis = [_bootstrap_ci(d.get("per_episode_regrets") or []) if d else (float("nan"), float("nan"))
           for d in metric_dicts]
    print(f"\n{'model':<26s} {'mean_reg':>9s} {'regret 95% CI':>20s} {'med_reg':>9s} {'p90_reg':>9s} "
          f"{'p99_reg':>9s} {'expansions':>11s}")
    for label, d, ci in zip(labels, metric_dicts, cis):
        if not d:
            print(f"{label:<26s} {'(pending)':>}")
            continue
        med = d.get("regret_median"); p90 = d.get("regret_p90"); p99 = d.get("regret_p99")
        dist = (f"{med:>9.4f} {p90:>9.4f} {p99:>9.4f}" if med is not None
                else f"{'(scalar — no dist)':>29s}")
        ci_str = f"[{ci[0]:.4f},{ci[1]:.4f}]" if d.get("per_episode_regrets") else f"{'—':>18s}"
        print(f"{label:<26s} {d['average_regret']:>9.4f} {ci_str:>20s} {dist} {d['average_expansions']:>11.2f}")

    out_dir = Path(args.out_dir)
    _plot(labels, regret_full, pending, title="Regret by readout model (lower = better)",
          xlabel="Mean Regret  (log scale; bars = 95% CI)", out_path=out_dir / "regret_by_model.png",
          cis=cis, log_x=True)
    _plot(labels, stop_full, pending, title="P(stop == OSS) by readout model",
          xlabel="fraction stop == OSS", out_path=out_dir / "oss_by_model.png")
    frac_curve = _fraction_operating_curve(test, config, [round(0.1 * k, 1) for k in range(11)])
    _plot_tradeoff(labels, regret_full, expansions_full, pending, frac_curve,
                   out_path=out_dir / "regret_vs_compute.png", cis=cis)
    _plot_regret_distribution(labels, metric_dicts, out_path=out_dir / "regret_distribution.png")
    _plot_regret_decomposition(labels, metric_dicts, out_path=out_dir / "regret_decomposition.png")

    # Results JSON for the cross-Elo ladder plotter: one {model: {regret, stop_acc}}
    # mapping per rung. Keyed by stable short model names (not the numbered/themed
    # bar labels) so the ladder plotter can join across rungs.
    if args.results_json is not None:
        tier1_keys = ["stop_bias", "stop_mae", "tol_acc_1", "tol_acc_2",
                      "frac_early", "frac_late", "regret_from_early", "regret_from_late",
                      "regret_median", "regret_p90", "regret_p99"]
        results: dict[str, dict[str, float | None]] = {}
        for key, d, ci in zip(model_keys, metric_dicts, cis):
            if d is None:
                results[key] = {"regret": None, "stop_acc": None, "expansions": None,
                                "regret_ci_lo": None, "regret_ci_hi": None}
                continue
            entry: dict[str, float | None] = {
                "regret": float(d["average_regret"]),
                "stop_acc": float(d["exact_stop_step_accuracy"]),
                "expansions": float(d["average_expansions"]),  # compute axis (mean steps before halt)
            }
            if d.get("per_episode_regrets"):  # bootstrap 95% CI on mean regret
                entry["regret_ci_lo"], entry["regret_ci_hi"] = float(ci[0]), float(ci[1])
            for t in tier1_keys:  # absent for the scalar MCHalt slot
                if t in d:
                    entry[t] = float(d[t])
            results[key] = entry
        out_json = Path(args.results_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))
        print(f"  wrote results JSON {out_json}")


if __name__ == "__main__":
    main()
