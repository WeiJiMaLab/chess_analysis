"""Part 3 — alternative stop-policy baselines on the new human_trees oracle episodes.

Evaluates three snapshot -> {HALT, CONTINUE} policies, all matching the MC
readout head's I/O (given a tree snapshot + starting budget B, emit HALT or
CONTINUE):

  * Always Stop          -- HALT at step 0                       (no fit)
  * Never Stop           -- CONTINUE to the last step            (no fit)
  * Fraction of Budget   -- HALT iff expansions >= round(f*B)    (fit f on TRAIN)

Metrics on the TEST split (see eval.md):
  * Regret            = oracle_value - return_for_stop_step(...)   (minimize)
  * P(stop == OSS)    = fraction of episodes whose chosen stop step equals the
                        budget-aware DP-optimal stop (`oracle_stop_step`).

Fitting `f`: regret is piecewise-constant in `f` (the induced stop step is an
integer round(f*B)), so we grid-search `f` to minimize MEAN TRAIN REGRET — which
is exactly the criterion the MC controller uses for checkpoint selection
(`average_regret`, controller_train.py). We do NOT fit an advantage-MSE/sign-BCE
term: a hard-threshold policy emits no per-step continuous advantage, so those MC
loss terms are undefined for it (eval.md [Q1] note).

    PYTHONPATH=lmcos/src python -m cts.analysis._budgeted.alt_models_eval \
        --packed-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc_smoke \
        --out-dir figures
"""
from __future__ import annotations

import argparse
import json
import statistics
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


def _load_split_episodes(packed_root: Path, split: str, max_episodes: int | None = None) -> list[dict[str, Any]]:
    """Walk one split's packed shards -> per-episode diagnostics dicts.

    Mirrors controller_train._extract_all_episode_metadata but emits the plain
    dict shape the baselines consume (halt_rewards / tree_sizes / time_budgets /
    oracle_value / oracle_stop_step), reading only the scalar/pointer arrays — no
    per-step encoder tree reconstruction.
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
        step_node_cutoffs = p["step_node_cutoffs"]
        trajectory_halt_rewards = p["trajectory_halt_rewards"]
        oracle_stop_steps = p["oracle_stop_steps"]
        oracle_values = p["oracle_values"]
        starting_budgets = p["starting_budgets"]
        num_episodes = int(p["num_episodes"])
        for i in range(num_episodes):
            num_steps = int(episode_step_ptr[i + 1].item()) - int(episode_step_ptr[i].item())
            traj = int(episode_trajectory_index[i].item())
            traj_step_begin = int(trajectory_step_ptr[traj].item())
            starting_budget = int(starting_budgets[i].item())
            episodes.append({
                "halt_rewards": trajectory_halt_rewards[traj_step_begin:traj_step_begin + num_steps].tolist(),
                "tree_sizes": step_node_cutoffs[traj_step_begin:traj_step_begin + num_steps].tolist(),
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


# --- the three models as stop-step rules -----------------------------------
def _always_stop(episode: dict[str, Any]) -> int:
    return 0


def _never_stop(episode: dict[str, Any]) -> int:
    return len(episode["halt_rewards"]) - 1


def _fraction_rule(f: float) -> Callable[[dict[str, Any]], int]:
    def rule(episode: dict[str, Any]) -> int:
        return int(round(f * int(episode["starting_budget"])))
    return rule


def _mean_train_regret(episodes: list[dict[str, Any]], config: BudgetedOracleConfig, f: float) -> float:
    """Mean regret of Fraction-of-Budget(f) over `episodes` (the fit objective)."""
    rule = _fraction_rule(f)
    total = 0.0
    for episode in episodes:
        stop = max(0, min(rule(episode), len(episode["halt_rewards"]) - 1))
        ret = return_for_stop_step(
            episode["halt_rewards"], episode["tree_sizes"], episode["time_budgets"], stop, config
        )
        total += float(episode["oracle_value"]) - ret
    return total / len(episodes)


def _fit_fraction(train: list[dict[str, Any]], config: BudgetedOracleConfig) -> tuple[float, list[tuple[float, float]]]:
    """Grid-search f in (0,1) to minimize mean train regret (piecewise-constant in f)."""
    grid = [round(0.01 * k, 2) for k in range(1, 100)]
    curve = [(f, _mean_train_regret(train, config, f)) for f in grid]
    best_f, _ = min(curve, key=lambda fr: fr[1])
    return best_f, curve


def _plot(models: list[str], regrets: list[float], stop_acc: list[float], out_dir: Path,
          controller_regret: float | None = None, controller_stop: float | None = None) -> None:
    """Two bar charts (Regret, P(stop==OSS)). The GNN/MC controller slot is filled with a
    highlighted bar when its metrics are supplied, else left greyed ('pending')."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = models + ["GNN/MC controller"]
    for values, ctrl, title, fname, ylab in [
        (regrets, controller_regret, "Regret on test (lower = better)", "regret_by_model.png", "mean regret"),
        (stop_acc, controller_stop, "P(stop == OSS) on test", "stop_eq_oss_by_model.png", "fraction stop == OSS"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        has_ctrl = ctrl is not None
        colors = ["#4063A3"] * len(models) + ["#B5475B" if has_ctrl else "#cccccc"]
        bars = ax.bar(labels, values + [ctrl if has_ctrl else 0.0], color=colors)
        for b, v in zip(bars[:len(models)], values):  # annotate baseline values
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
        if has_ctrl:
            ax.annotate(f"{ctrl:.3f}", (len(models), ctrl), ha="center", va="bottom", fontsize=9,
                        color="#B5475B", fontweight="bold")
        else:
            bars[-1].set_hatch("//")
            ax.annotate("pending", (len(models), 0.0), ha="center", va="bottom", fontsize=9, color="#888")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_dir / fname}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-root", required=True, help="MC packed dir with train/ validation/ + *_manifest.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap episodes loaded per split (means/fractions are stable on a large sample)")
    ap.add_argument("--controller-regret", type=float, default=None,
                    help="trained GNN/MC controller mean regret (from controller_train greedy eval); fills its slot")
    ap.add_argument("--controller-stop", type=float, default=None,
                    help="trained GNN/MC controller P(stop==OSS) (exact_stop_step_accuracy); fills its slot")
    args = ap.parse_args()

    packed_root = Path(args.packed_root)
    config = _oracle_config(packed_root)
    train = _load_split_episodes(packed_root, "train", args.max_episodes)
    test = _load_split_episodes(packed_root, "validation", args.max_episodes)  # held-out test in the 'validation' slot
    print(f"loaded train={len(train):,} episodes  test={len(test):,} episodes")

    best_f, curve = _fit_fraction(train, config)
    print(f"fitted Fraction-of-Budget f* = {best_f:.2f} (min mean train regret)")

    models = ["Always Stop", "Never Stop", f"Fraction f*={best_f:.2f}"]
    rules: list[Callable[[dict[str, Any]], int]] = [_always_stop, _never_stop, _fraction_rule(best_f)]
    regrets, stop_acc = [], []
    print(f"\n{'model':<22s} {'regret':>10s} {'P(stop==OSS)':>14s} {'avg_expansions':>15s}")
    for name, rule in zip(models, rules):
        m = _evaluate_baseline(test, config, rule)
        regrets.append(m["average_regret"])
        stop_acc.append(m["exact_stop_step_accuracy"])
        print(f"{name:<22s} {m['average_regret']:>10.4f} {m['exact_stop_step_accuracy']:>14.3f} {m['average_expansions']:>15.2f}")

    _plot(models, regrets, stop_acc, Path(args.out_dir),
          controller_regret=args.controller_regret, controller_stop=args.controller_stop)

    # Appendix: the fraction-rho grid (test regret) for context.
    print("\nAppendix — Fraction-of-Budget test regret across the rho grid:")
    for f in (0.1, 0.25, 0.5, 0.75):
        m = _evaluate_baseline(test, config, _fraction_rule(f))
        print(f"  f={f:<4} regret={m['average_regret']:.4f}  P(stop==OSS)={m['exact_stop_step_accuracy']:.3f}")


if __name__ == "__main__":
    main()
