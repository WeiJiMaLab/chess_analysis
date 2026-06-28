"""D0 — the five-model readout comparison on the human_trees budgeted oracle.

Every model is a :class:`cts.models.readout.Readout`: it emits a per-step
advantage and STOPS at the first step with ``A <= 0`` (continue iff ``A > 0``)
— ONE decision rule shared across all five tiers, so the comparison isolates
the *advantage signal* and nothing else. The five tiers:

  1. Always Stop          -- stop@0                              (no fit)
  2. Never Stop           -- full budget                         (no fit)
  3. Fraction-of-Budget   -- continue iff N_t < theta*B          (theta fit on regret)
  4. Readout(tree-stats)  -- MLP on [height, width, n_nodes, B]  (fit, threshold tuned on regret)
  5. Readout(GNN-z)       -- MLP on [z, B]                       (pending: needs the P2 tiny encoder)

THE §10b FIX: every fitted model is SELECTED on regret directly (theta by a
regret grid; the stats MLP by tuning its decision threshold on TRAIN regret),
NOT on the surrogate advantage-MSE / sign-BCE that broke the old controller.

Metrics on the TEST split (the 'validation' shard slot is the held-out test):
  * Regret         = oracle_value - return_for_stop_step(...)  (minimize)
  * P(stop == OSS) = fraction whose chosen stop step == the DP-optimal stop

height / width are NOT stored per-snapshot in the packed shards; they are
derived here from the trajectory ``depth`` array + ``step_node_cutoffs``
(``height = max depth over the first n_nodes``, ``width = max nodes at any one
depth``). ``n_nodes`` (== ``step_node_cutoffs``) is already present.

    PYTHONPATH=lmcos/src python -m cts.analysis._budgeted.alt_models_eval \
        --packed-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc \
        --out-dir figures --max-episodes 2000
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
        [torch.full((n, 1), float(ep["starting_budget"])) for n, ep in zip(path_lengths, train)], dim=0
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


def _stats_rule(model: StatsReadout) -> Callable[[dict[str, Any]], int]:
    """Deployable stop rule for a fitted StatsReadout (applies its tuned threshold)."""
    mean = model._norm_mean  # type: ignore[attr-defined]
    std = model._norm_std  # type: ignore[attr-defined]
    tau = model._tau  # type: ignore[attr-defined]

    def rule(episode: dict[str, Any]) -> int:
        f = _stats_features(episode)
        n = f.shape[0]
        b = torch.full((n, 1), float(episode["starting_budget"]))
        x = (torch.cat([f, b], dim=-1) - mean) / std
        with torch.no_grad():
            adv = model.head(x).squeeze(-1) - tau
        return stop_step_from_advantages(adv)

    return rule


# --- D0 figure (horizontal bars, models top->bottom 1->5) -------------------
_TIER_COLOR = "#4063A3"
_PENDING_COLOR = "#cccccc"


def _plot(labels: list[str], values: list[float], pending_last: bool, *, title: str,
          xlabel: str, out_path: Path) -> None:
    """One horizontal bar chart: models top->bottom in `labels` order (tier 1->5).

    ``barh`` plots bottom->top, so we reverse the y positions to put labels[0]
    at the top. The last model (GNN-z) is drawn hatched/grey + 'pending' when
    ``pending_last``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    y = list(range(n))[::-1]  # reverse so labels[0] is at the top
    colors = [_TIER_COLOR] * n
    plot_values = list(values)
    if pending_last:
        colors[-1] = _PENDING_COLOR
        plot_values[-1] = 0.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(y, plot_values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for idx, (bar, val) in enumerate(zip(bars, values)):
        if pending_last and idx == n - 1:
            bar.set_hatch("//")
            ax.annotate("pending", (0.0, bar.get_y() + bar.get_height() / 2),
                        va="center", ha="left", fontsize=9, color="#888")
        else:
            ax.annotate(f"{val:.3f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        va="center", ha="left", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


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

    # Tier 4: StatsReadout MLP, threshold SELECTED on TRAIN regret directly.
    stats_model, stats_tau = _fit_stats_readout(train, config)
    print(f"tier 4 StatsReadout tau* = {stats_tau:+.1f} (decision threshold tuned on TRAIN regret)")

    labels = ["1. Always Stop", "2. Never Stop", f"3. Fraction theta*={best_f:.2f}",
              "4. Readout(tree-stats)", "5. Readout(GNN-z)"]
    rules: list[Callable[[dict[str, Any]], int]] = [
        _always_stop, _never_stop, _fraction_rule(best_f), _stats_rule(stats_model),
    ]

    regrets, stop_acc = [], []
    print(f"\n{'model':<26s} {'regret':>10s} {'P(stop==OSS)':>14s} {'avg_expansions':>15s}")
    for name, rule in zip(labels[:4], rules):
        m = _evaluate_baseline(test, config, rule)
        regrets.append(m["average_regret"])
        stop_acc.append(m["exact_stop_step_accuracy"])
        print(f"{name:<26s} {m['average_regret']:>10.4f} {m['exact_stop_step_accuracy']:>14.3f} {m['average_expansions']:>15.2f}")

    # Tier 5 (MCHalt): scored for real from a trained controller checkpoint when
    # one is supplied; else the back-compat scalar slot; else 'pending'.
    mchalt_regret: float | None = None
    mchalt_stop: float | None = None
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
        mchalt_regret = mc["average_regret"]
        mchalt_stop = mc["exact_stop_step_accuracy"]
        print(f"{labels[4]:<26s} {mchalt_regret:>10.4f} {mchalt_stop:>14.3f} "
              f"{mc['average_expansions']:>15.2f}  (checkpoint, n={int(mc['evaluated_episodes'])})")
    elif args.gnn_regret is not None:
        mchalt_regret = args.gnn_regret
        mchalt_stop = args.gnn_stop or 0.0
        print(f"{labels[4]:<26s} {mchalt_regret:>10.4f} {mchalt_stop:>14.3f}  (scalar)")

    pending = mchalt_regret is None
    regret_full = regrets + [mchalt_regret if not pending else 0.0]
    stop_full = stop_acc + [mchalt_stop if not pending else 0.0]

    out_dir = Path(args.out_dir)
    _plot(labels, regret_full, pending, title="Regret by readout model (lower = better)",
          xlabel="mean regret", out_path=out_dir / "regret_by_model.png")
    _plot(labels, stop_full, pending, title="P(stop == OSS) by readout model",
          xlabel="fraction stop == OSS", out_path=out_dir / "oss_by_model.png")

    # Results JSON for the cross-Elo ladder plotter: one {model: {regret, stop_acc}}
    # mapping per rung. Keyed by stable short model names (not the numbered/themed
    # bar labels) so the ladder plotter can join across rungs.
    if args.results_json is not None:
        model_keys = ["always", "never", "fraction", "stats", "mchalt"]
        results: dict[str, dict[str, float | None]] = {}
        for key, regret, stop in zip(model_keys, regret_full, stop_full):
            if key == "mchalt" and pending:
                results[key] = {"regret": None, "stop_acc": None}
            else:
                results[key] = {"regret": float(regret), "stop_acc": float(stop)}
        out_json = Path(args.results_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))
        print(f"  wrote results JSON {out_json}")


if __name__ == "__main__":
    main()
