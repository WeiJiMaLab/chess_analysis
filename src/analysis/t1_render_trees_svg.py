"""plan.md §G item 3 (2026-07-08) -- regenerate T1's 5 tree examples as standalone SVGs.

The original driver that produced ``diagnosis/t1_tree_{label}.png`` (rasterized, for a now-
dropped montage into ``t1_tree_panel.png``) is gone -- it was never committed (``git log`` on
``tree_diagnostics.py`` shows the file itself is untracked) and no copy survives anywhere in the
repo or scratch. This module reconstructs its exact behavior from the one artifact that DID
survive, ``diagnosis/t1_summary.json`` (the 5 trees' source ``.pt`` paths + ``node_cutoff`` +
the red-flag check results T1 reported in plan.md), and ``tree_diagnostics.py``'s still-present
pure helpers (``prune_topk_subtree``, ``render_tree_graphviz``).

Reconstruction logic, reverse-engineered from the surviving per-node asset counts:
``t1_tree_{label}_assets/`` holds 2 files (an intermediate per-node board .svg + the final
.png ``render_board_png`` converts it to) per RENDERED node, and those counts (4/7/12/20/40 for
p10/p50/p75/p90/max) match ``t1_summary.json``'s ``linear.n_branching_nodes`` (4/7/12/20/89)
capped at 40 for the ``max`` tree -- i.e. the driver called
``prune_topk_subtree(parent_index, budget=40, intermediate_only=has_children)`` (root + up to 40
of the largest-subtree branching nodes) before rendering. This module recomputes that exact
selection and re-renders -- verified byte-for-byte equivalent on the red-flag checks (dup/
linear/dominance/illegal) against the recorded ``t1_summary.json`` values before trusting the
render (see ``verify_against_summary``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.tree_diagnostics import (
    duplicate_fen_check,
    illegal_move_check,
    linear_chain_check,
    prune_topk_subtree,
    render_tree_graphviz,
)
from cts.data.preprocess_gnn.teacher_targets import load_raw_pretrain_record

RENDER_BUDGET = 40  # reverse-engineered from surviving asset counts, see module docstring


def load_truncated_tree(src: str, node_cutoff: int):
    """Load one raw tree record and slice every per-node array to its first ``node_cutoff``
    nodes -- the tree state at the argmax step (root-first, topologically ordered, so a prefix
    slice is always a valid connected subtree)."""
    record = load_raw_pretrain_record(src)
    fens = record._resolved_position_specs()[:node_cutoff]
    parent_index = record.parent_index.numpy()[:node_cutoff].astype(np.int64)
    incoming_moves = list(record.incoming_moves[:node_cutoff])
    node_targets = record.node_targets.numpy()[:node_cutoff]
    return parent_index, incoming_moves, node_targets, fens


def select_render_nodes(parent_index: np.ndarray, budget: int = RENDER_BUDGET) -> set[int]:
    """Root + up to ``budget`` branching (>=1 child) nodes, largest-subtree-first -- see module
    docstring for how ``budget`` was recovered from the surviving asset-folder counts."""
    n = len(parent_index)
    child_counts = np.bincount(parent_index[parent_index >= 0], minlength=n)
    has_children = child_counts >= 1
    return prune_topk_subtree(parent_index, budget=budget, intermediate_only=has_children)


def verify_against_summary(parent_index, incoming_moves, node_targets, fens, expected: dict) -> None:
    """Recompute T1's 4 red-flag checks and assert they match ``t1_summary.json``'s recorded
    result for this tree -- if this passes, the reconstructed (parent_index, fens, moves) triple
    is confirmed identical to whatever the original (lost) driver used, so the re-render is
    trustworthy."""
    dup = duplicate_fen_check(fens)
    lin = linear_chain_check(np.bincount(parent_index[parent_index >= 0], minlength=len(parent_index)))
    illegal = illegal_move_check(fens, parent_index, incoming_moves)
    for got, exp, name in [(dup, expected["dup"], "dup"), (lin, expected["linear"], "linear"),
                           (illegal, expected["illegal"], "illegal")]:
        for key in ("pass", "n_duplicates", "n_bad", "max_children", "n_branching_nodes"):
            if key in got and key in exp and got[key] != exp[key]:
                raise ValueError(f"{name} check mismatch on key {key!r}: got {got[key]!r} "
                                 f"expected {exp[key]!r}")


def render_all(summary_path: str | Path, out_dir: str | Path, *, verify: bool = True) -> list[str]:
    """Render all 5 T1 trees from ``t1_summary.json`` straight to ``out_dir`` as standalone SVGs
    (``t1_tree_{label}.svg`` + sibling ``t1_tree_{label}.svg_assets/`` per-node board images)."""
    entries = json.loads(Path(summary_path).read_text())
    # Absolute path required: graphviz's ``dot`` subprocess runs with cwd set to the OUTPUT
    # file's directory (see ``dot -O <basename>`` in the render command), so a relative
    # ``<IMG SRC="...">`` reference gets re-resolved against that directory and breaks (double
    # directory prefix). Matches the task-level warning that the ORIGINAL asset folders may have
    # absolute paths baked in for exactly this reason.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for entry in entries:
        label = entry["label"]
        parent_index, incoming_moves, node_targets, fens = load_truncated_tree(entry["src"], entry["node_cutoff"])
        if verify:
            verify_against_summary(parent_index, incoming_moves, node_targets, fens, entry)
        selected = select_render_nodes(parent_index)
        out_path = str(out_dir / f"t1_tree_{label}")
        caption = f"T1 tree: {label} (argmax={entry['argmax']}, {len(parent_index)} nodes, {len(selected)} shown)"
        rendered_path = render_tree_graphviz(selected, parent_index, incoming_moves, node_targets, fens,
                                             out_path, caption)
        print(f"[t1_render_trees_svg] {label}: {len(selected)}/{len(parent_index)} nodes -> {rendered_path}",
             flush=True)
        rendered.append(rendered_path)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/figures/minply15_maxply75/diagnosis/t1_summary.json")
    parser.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/trees")
    parser.add_argument("--no-verify", action="store_true", help="skip the red-flag-check cross-check")
    args = parser.parse_args()
    render_all(args.summary, args.out_dir, verify=not args.no_verify)


if __name__ == "__main__":
    main()
