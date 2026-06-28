"""P1-prep — relabel the budgeted DP oracle under altered cost regimes (R-MC-COST).

Re-runs ``compute_budgeted_oracle`` over the *cached* MC episode traces
(``trajectory_halt_rewards`` + ``step_node_cutoffs`` + ``starting_budgets`` from the
existing packed shards) with overridden cost parameters, producing a new set of
oracle labels (``oracle_values``, ``oracle_stop_steps``, ``target_advantages``)
WITHOUT any tree generation or encoder materialization.

This is the data-production half of P1 (mc_minimal_plan.md §P1). The final regret
measurement runs later through the P0 regret-eval harness; this module additionally
ships a *self-contained* quick regret check (Always-Stop / Never-Stop / fitted
Fraction-of-Budget) so each regime can be sanity-read in isolation.

Two sweep modes (the only cost knobs that change vs. the production oracle):

  A. FLAT / LINEAR cost: ``time_mode="linear"`` makes the per-expansion time cost a
     constant ``c_bar`` (== ``time_lambda``), replacing the convex power-law. ``p``/``tau``
     are unused in this mode.
  B. LAMBDA sweep: ``time_mode="power_law"`` with ``time_lambda`` overridden
     (e.g. {0.1, 1.0, 5.0, 18.537}); ``p=2.8``, ``tau=2.5`` kept fixed.

The output shards keep the SAME schema (and the same graph/feature tensors) as the
source packed shards, so they drop straight into ``alt_models_eval`` / the P0 harness
(both read via ``*_manifest.json`` + ``shard_*.pt``). Only the three oracle-label
fields and the oracle-config metadata change.

Usage (single regime, capped smoke):
    PYTHONPATH=lmcos/src python -m cts.data.preprocess_mc.costsweep_relabel \
        --source-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc \
        --output-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc_costsweep/lambda_1.0 \
        --time-mode power_law --time-lambda 1.0 \
        --max-episodes-per-split 4000

Full run (drop --max-episodes-per-split; documented, NOT auto-submitted to SLURM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    budgeted_oracle_metadata,
    budgeted_oracle_config_from_metadata,
    compute_budgeted_oracle,
    return_for_stop_step,
)


# --------------------------------------------------------------------------- #
# Oracle-config construction (only cost knobs differ from the source pack)
# --------------------------------------------------------------------------- #
def build_oracle_config(
    base: BudgetedOracleConfig,
    *,
    time_mode: str,
    time_lambda: float,
    time_p: Optional[float] = None,
    time_tau: Optional[float] = None,
) -> BudgetedOracleConfig:
    """Clone ``base`` (bucket partition, seed, maintenance) but override the cost regime.

    ``time_mode="linear"`` -> constant per-step cost ``c_bar = time_lambda`` (flat cost).
    ``time_mode="power_law"`` -> convex cost with overridden ``time_lambda`` (lambda sweep);
    ``time_p`` / ``time_tau`` default to the base config when not supplied.
    """
    return BudgetedOracleConfig(
        maintenance_scale=base.maintenance_scale,
        maintenance_ref_nodes=base.maintenance_ref_nodes,
        maintenance_exponent=base.maintenance_exponent,
        time_lambda=float(time_lambda),
        time_p=base.time_p if time_p is None else float(time_p),
        time_tau=base.time_tau if time_tau is None else float(time_tau),
        time_delta=base.time_delta,
        timeout_value=base.timeout_value,
        time_mode=time_mode,
        budget_buckets=base.budget_buckets,
        samples_per_bucket=base.samples_per_bucket,
        seed=base.seed,
    )


# --------------------------------------------------------------------------- #
# Per-shard relabel
# --------------------------------------------------------------------------- #
def _episode_inputs(payload: Dict[str, Any], i: int):
    """Return (halt_rewards, tree_sizes, starting_budget) for episode i in a packed shard.

    Mirrors the join in ``alt_models_eval._load_split_episodes``: episode i pulls its
    trajectory's halt-reward / node-cutoff slice and truncates to the episode budget.
    """
    esp = payload["episode_step_ptr"]
    traj_idx = int(payload["episode_trajectory_index"][i].item())
    tsp = payload["trajectory_step_ptr"]
    traj_begin = int(tsp[traj_idx].item())
    num_steps = int(esp[i + 1].item()) - int(esp[i].item())
    hr = payload["trajectory_halt_rewards"][traj_begin : traj_begin + num_steps].tolist()
    sizes = payload["step_node_cutoffs"][traj_begin : traj_begin + num_steps].tolist()
    budget = int(payload["starting_budgets"][i].item())
    return [float(x) for x in hr], [int(x) for x in sizes], budget


def relabel_shard(
    payload: Dict[str, Any],
    config: BudgetedOracleConfig,
) -> Dict[str, Any]:
    """Recompute oracle labels in-place on a loaded shard payload under ``config``.

    Overwrites ``oracle_values``, ``oracle_stop_steps`` (per-episode) and
    ``target_advantages`` (per-step, concatenated in episode order), plus the oracle
    metadata. All graph/feature/pointer tensors are left untouched so the shard stays
    byte-compatible with the materializer and the P0 eval loader. Returns the same dict.
    """
    n = int(payload["num_episodes"])
    new_values: List[float] = []
    new_stops: List[int] = []
    new_advs: List[float] = []
    for i in range(n):
        hr, sizes, budget = _episode_inputs(payload, i)
        policy = compute_budgeted_oracle(hr, sizes, budget, config)
        new_values.append(policy.oracle_value)
        new_stops.append(policy.optimal_stop_step)
        new_advs.extend(policy.target_advantages)

    payload["oracle_values"] = torch.tensor(new_values, dtype=torch.float32)
    payload["oracle_stop_steps"] = torch.tensor(new_stops, dtype=torch.long)
    payload["target_advantages"] = torch.tensor(new_advs, dtype=torch.float32)
    # Refresh the embedded oracle metadata so the manifest round-trips correctly.
    payload.update(budgeted_oracle_metadata(config))
    return payload


# --------------------------------------------------------------------------- #
# Self-contained quick regret check (Always / Never / fitted Fraction f*)
# --------------------------------------------------------------------------- #
def _episodes_from_shards(
    shard_payloads: List[Dict[str, Any]],
    max_episodes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Flatten relabeled shards into the per-episode dicts the regret rules consume."""
    out: List[Dict[str, Any]] = []
    for payload in shard_payloads:
        n = int(payload["num_episodes"])
        for i in range(n):
            hr, sizes, budget = _episode_inputs(payload, i)
            out.append(
                {
                    "halt_rewards": hr,
                    "tree_sizes": sizes,
                    "time_budgets": list(range(budget, budget - len(hr), -1)),
                    "oracle_value": float(payload["oracle_values"][i].item()),
                    "oracle_stop_step": int(payload["oracle_stop_steps"][i].item()),
                    "starting_budget": budget,
                }
            )
            if max_episodes is not None and len(out) >= max_episodes:
                return out
    return out


def _regret_of_rule(
    episodes: List[Dict[str, Any]],
    config: BudgetedOracleConfig,
    stop_rule: Callable[[Dict[str, Any]], int],
) -> float:
    """Mean regret = oracle_value - return_for_stop_step(rule), clamped in-episode."""
    total = 0.0
    for ep in episodes:
        stop = max(0, min(stop_rule(ep), len(ep["halt_rewards"]) - 1))
        ret = return_for_stop_step(ep["halt_rewards"], ep["tree_sizes"], ep["time_budgets"], stop, config)
        total += float(ep["oracle_value"]) - ret
    return total / len(episodes)


def _fit_fraction(
    train: List[Dict[str, Any]],
    config: BudgetedOracleConfig,
    grid: Optional[List[float]] = None,
) -> tuple[float, float]:
    """Grid-search f in (0,1) minimizing mean TRAIN regret of Fraction(f). Returns (f*, train_regret)."""
    if grid is None:
        grid = [round(0.01 * k, 2) for k in range(1, 100)]
    best_f, best_r = grid[0], float("inf")
    for f in grid:
        r = _regret_of_rule(train, config, lambda ep, f=f: int(round(f * int(ep["starting_budget"]))))
        if r < best_r:
            best_f, best_r = f, r
    return best_f, best_r


def quick_regret_check(
    train_episodes: List[Dict[str, Any]],
    val_episodes: List[Dict[str, Any]],
    config: BudgetedOracleConfig,
) -> Dict[str, Any]:
    """Always / Never / fitted-Fraction mean regret on the val episodes (f* fit on train).

    The Fraction-vs-{Always,Never} gap previews whether tree value can emerge as cost
    relaxes: a fitted budget-only rule that strongly beats both floors means stopping
    is still budget-dominated; the gap shrinking is the signal that tree-state matters.
    """
    best_f, train_frac_regret = _fit_fraction(train_episodes, config)
    always = _regret_of_rule(val_episodes, config, lambda ep: 0)
    never = _regret_of_rule(val_episodes, config, lambda ep: len(ep["halt_rewards"]) - 1)
    fraction = _regret_of_rule(
        val_episodes, config, lambda ep: int(round(best_f * int(ep["starting_budget"])))
    )
    best_floor = min(always, never)
    return {
        "n_train": len(train_episodes),
        "n_val": len(val_episodes),
        "fraction_f_star": best_f,
        "regret_always_stop": always,
        "regret_never_stop": never,
        "regret_fraction": fraction,
        "fraction_train_regret": train_frac_regret,
        # Positive gap => fitted budget rule beats the better trivial floor.
        "gap_fraction_vs_best_floor": best_floor - fraction,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load_manifest(source_root: Path, split: str) -> Dict[str, Any]:
    return json.loads((source_root / f"{split}_manifest.json").read_text())


def _shard_paths(source_root: Path, split: str, manifest: Dict[str, Any]) -> List[Path]:
    """Resolve shard paths: prefer manifest entries, fall back to sorted glob of the split dir."""
    entries = manifest.get("entries")
    if entries:
        paths = [Path(e["path"]) for e in entries]
        if all(p.exists() for p in paths):
            return paths
    return sorted((source_root / split).glob("shard_*.pt"))


def relabel_split(
    source_root: Path,
    output_root: Path,
    split: str,
    config: BudgetedOracleConfig,
    max_episodes: Optional[int],
) -> tuple[List[Dict[str, Any]], int]:
    """Relabel one split's shards under ``config``; write shards + manifest. Returns (payloads, n_episodes)."""
    manifest = _load_manifest(source_root, split)
    shard_paths = _shard_paths(source_root, split, manifest)
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    total = 0
    for shard_index, sp in enumerate(shard_paths):
        payload = torch.load(sp, weights_only=False)
        relabel_shard(payload, config)
        out_path = out_dir / f"shard_{shard_index:05d}.pt"
        torch.save(payload, out_path)
        payloads.append(payload)
        n = int(payload["num_episodes"])
        entries.append({"path": str(out_path), "num_episodes": n, "shard_index": shard_index})
        total += n
        print(f"  [{split}] shard {shard_index}: {n} episodes -> {out_path.name}", flush=True)
        if max_episodes is not None and total >= max_episodes:
            break

    out_manifest = {k: v for k, v in manifest.items() if k not in ("entries",)}
    out_manifest.update(budgeted_oracle_metadata(config))
    out_manifest["entries"] = entries
    out_manifest["total_episodes"] = total
    out_manifest["costsweep_capped"] = max_episodes is not None
    (output_root / f"{split}_manifest.json").write_text(json.dumps(out_manifest, indent=2))
    return payloads, total


def run(
    source_root: Path,
    output_root: Path,
    config: BudgetedOracleConfig,
    max_episodes_per_split: Optional[int],
) -> Dict[str, Any]:
    """Relabel both splits, write a manifest + cost-param manifest, run the quick regret check."""
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"=== relabel train ({config.time_mode}, lambda={config.time_lambda}) ===", flush=True)
    train_payloads, n_train = relabel_split(source_root, output_root, "train", config, max_episodes_per_split)
    print("=== relabel validation ===", flush=True)
    val_payloads, n_val = relabel_split(source_root, output_root, "validation", config, max_episodes_per_split)

    train_eps = _episodes_from_shards(train_payloads, max_episodes_per_split)
    val_eps = _episodes_from_shards(val_payloads, max_episodes_per_split)
    regret = quick_regret_check(train_eps, val_eps, config)

    manifest = {
        "regime": output_root.name,
        "source_root": str(source_root),
        "max_episodes_per_split": max_episodes_per_split,
        "n_train_episodes_written": n_train,
        "n_val_episodes_written": n_val,
        "cost_params": budgeted_oracle_metadata(config),
        "quick_regret_check": regret,
    }
    (output_root / "costsweep_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n--- quick regret check (val; f* fit on train) ---", flush=True)
    print(
        f"  f*={regret['fraction_f_star']:.2f}  "
        f"Always={regret['regret_always_stop']:.4f}  "
        f"Never={regret['regret_never_stop']:.4f}  "
        f"Fraction={regret['regret_fraction']:.4f}  "
        f"gap(Fraction vs best floor)={regret['gap_fraction_vs_best_floor']:.4f}",
        flush=True,
    )
    print(f"  wrote manifest: {output_root / 'costsweep_manifest.json'}", flush=True)
    return manifest


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True, help="packed MC dir with {train,validation}_manifest.json + shards")
    ap.add_argument("--output-root", required=True, help="output dir for the relabeled regime (created)")
    ap.add_argument("--time-mode", choices=["power_law", "linear"], required=True)
    ap.add_argument("--time-lambda", type=float, required=True,
                    help="lambda for power_law, or the constant per-step cost c_bar for linear")
    ap.add_argument("--time-p", type=float, default=None, help="power_law convexity (default: source pack value)")
    ap.add_argument("--time-tau", type=float, default=None, help="power_law offset (default: source pack value)")
    ap.add_argument("--max-episodes-per-split", type=int, default=None,
                    help="cap episodes per split for a fast smoke; omit for the full run")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    source_root = Path(args.source_root)
    base = budgeted_oracle_config_from_metadata(_load_manifest(source_root, "validation"))
    if base is None:
        raise ValueError(f"source manifest under {source_root} carries no budgeted-oracle metadata")
    config = build_oracle_config(
        base,
        time_mode=args.time_mode,
        time_lambda=args.time_lambda,
        time_p=args.time_p,
        time_tau=args.time_tau,
    )
    run(source_root, Path(args.output_root), config, args.max_episodes_per_split)


if __name__ == "__main__":
    main()
