"""
Rebuild a ``SearchTree`` from a raw v1 dict (same path as ``visualize_tree_expansion``)
and estimate the **never-stop** episode return under the **ToyHaltEnv** planning-cost
regime: always ``continue`` until the final snapshot, paying ``incremental_planning_cost``
at each step, then receiving **terminal quality** (best root child ``value``, else root).

For a tree with ``E`` expansion edges in order, there are ``E + 1`` snapshots; the total
continue cost equals ``cumulative_planning_cost(E + 1, config)`` (same as summing
``incremental_planning_cost(i)`` for ``i = 0 .. E``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
_LMCOS = _REPO / "lmcos"
for p in (str(_REPO), str(_LMCOS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from planning_cost import (  # noqa: E402
    cumulative_planning_cost,
    planning_cost_config,
    planning_cost_config_from_metadata,
)
from tree import SearchTree  # noqa: E402

from src.performance.visualize_tree_expansion import to_search_tree  # noqa: E402


def terminal_quality_toy_halt(tree: SearchTree) -> float:
    """Match ``ToyHaltEnv._terminal_quality`` (scalar ``value`` on best root child or root)."""
    if tree.root_children():
        best_child_id = tree.best_root_child()
        return float(tree.get_node(best_child_id).scalar_features["value"])
    return float(tree.get_node(tree.root_id).scalar_features["value"])


def never_stop_planning_return(
    full_tree: SearchTree,
    *,
    continue_cost: float,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> tuple[float, float, int]:
    """
    Returns ``(return_value, terminal_quality, num_snapshots)`` for always-continue
    until the last snapshot (ToyHaltEnv semantics).
    """
    n_expansions = len(list(full_tree.ordered_expansion_parent_ids()))
    n_snapshots = n_expansions + 1
    cfg = planning_cost_config(
        continue_cost,
        kind=planning_cost_kind,
        exponent=planning_cost_exponent,
    )
    full = full_tree.clone_expansion_prefix(n_expansions)
    term = terminal_quality_toy_halt(full)
    paid = cumulative_planning_cost(n_snapshots, cfg)
    return term - paid, term, n_snapshots


def _maybe_oracle_terminal(raw: dict[str, Any]) -> float | None:
    v = raw.get("oracle_final_root_q_values")
    if v is None or not isinstance(v, torch.Tensor):
        return None
    return float(torch.max(v.detach().cpu()).item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pt_path", type=Path, help="torch.load path (raw v1 dict or PretrainExample)")
    ap.add_argument(
        "--continue-cost",
        type=float,
        default=None,
        help="Per-step planning base (ToyHalt default 0.05). If omitted, use raw metadata when present.",
    )
    ap.add_argument("--planning-cost-kind", type=str, default="linear")
    ap.add_argument("--planning-cost-exponent", type=float, default=1.0)
    ap.add_argument(
        "--print-oracle-terminal",
        action="store_true",
        help="If raw has oracle_final_root_q_values, print max Q as an alternate terminal signal.",
    )
    args = ap.parse_args()

    loaded: Any = torch.load(args.pt_path, weights_only=False)
    tree = to_search_tree(loaded)

    continue_cost = args.continue_cost
    kind = args.planning_cost_kind
    exponent = args.planning_cost_exponent
    meta = None
    if isinstance(loaded, dict):
        meta = loaded.get("metadata") or loaded
        cfg_meta = planning_cost_config_from_metadata(meta)
        if continue_cost is None and cfg_meta is not None:
            continue_cost = float(cfg_meta.base_cost)
            kind = str(cfg_meta.kind)
            exponent = float(cfg_meta.exponent)
    if continue_cost is None:
        continue_cost = 0.05

    ret, term, n_snap = never_stop_planning_return(
        tree,
        continue_cost=continue_cost,
        planning_cost_kind=kind,
        planning_cost_exponent=exponent,
    )
    paid = term - ret

    print(
        f"path={args.pt_path}\n"
        f"n_expansions={n_snap - 1} n_snapshots={n_snap}\n"
        f"continue_cost={continue_cost} kind={kind} exponent={exponent}\n"
        f"terminal_quality_tree_value={term:.6f}\n"
        f"total_planning_cost_paid={paid:.6f}\n"
        f"never_stop_return={ret:.6f}"
    )

    if args.print_oracle_terminal and isinstance(loaded, dict):
        oq = _maybe_oracle_terminal(loaded)
        if oq is not None:
            cfg = planning_cost_config(continue_cost, kind=kind, exponent=exponent)
            paid_o = cumulative_planning_cost(n_snap, cfg)
            print(f"oracle_max_final_root_q={oq:.6f}\nnever_stop_return_if_oracle_terminal={oq - paid_o:.6f}")


if __name__ == "__main__":
    main()
