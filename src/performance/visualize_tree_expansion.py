"""
Tree checkpoint visualization.

Public API (three functions):

  1. ``load_tree_from_pt(path) -> SearchTree``
  2. ``tree_iter_expand(full_tree, n_expansions) -> Iterator[tuple[int, SearchTree]]``
  3. ``tree_to_graphviz(tree, stem, *, show_board=True, title=None) -> str`` (writes PNG/GV)

``to_search_tree(loaded)`` is still exported so callers that already have a loaded
object (e.g. ``estimate_never_stop_planning_return``) do not need a file path.

A ``.pt`` file is a PyTorch blob from ``torch.load`` (checkpoints), not Parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_LMCOS_ROOT = Path(__file__).resolve().parent.parent.parent / "lmcos"
if str(_LMCOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LMCOS_ROOT))

import chess  # noqa: E402
import graphviz  # noqa: E402
import torch  # noqa: E402
from tree import ExpansionChild, SearchTree  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "performance"

# Canonical CTS tree shards on Griffiths (same root as lmcos/scripts/* defaults for generated_trees / pretrain).
# This path is one example shard; override with --pt-path. ``load_tree_from_pt`` accepts ``PretrainExample``
# (uses ``.tree``) or a raw v1 dict (detensorized).
_DEFAULT_PT = os.path.join(
    "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data",
    "generated_trees_oracle96_trace_filtered",
    "shard_00000",
    "000000_root_0.pt",
)

_MONO = "Consolas, DejaVu Sans Mono, Liberation Mono, monospace"
_SANS = "Noto Color Emoji, Apple Color Emoji, Segoe UI Emoji, Inter, Arial, sans-serif"
_ACCENT = "#4338ca"
_BORDER = "#94a3b8"
_ROOT_BG = "#eef2ff"


def _is_real_fen(fen: str) -> bool:
    return isinstance(fen, str) and fen.count("/") == 7 and " " in fen and not fen.startswith("packed:")


def to_search_tree(loaded: Any) -> SearchTree:
    """Wrap checkpoint object as ``SearchTree`` (legacy ``PretrainExample`` or raw v1 dict)."""
    if getattr(loaded, "tree", None) is not None:
        return loaded.tree
    return _search_tree_from_raw_v1(loaded)


def load_tree_from_pt(pt_path: str) -> SearchTree:
    """Load a ``.pt`` checkpoint and return a ``SearchTree``."""
    loaded = torch.load(pt_path, weights_only=False)
    return to_search_tree(loaded)


def tree_iter_expand(full_tree: SearchTree, n_expansions: int) -> Iterator[tuple[int, SearchTree]]:
    """
    Yield ``(k, partial_tree)`` where ``partial_tree = clone_expansion_prefix(k)`` for
    ``k = 0 .. min(n_expansions, E)`` and ``E = len(ordered_expansion_parent_ids())``.
    """
    e_full = len(full_tree.ordered_expansion_parent_ids())
    k_max = min(n_expansions, e_full)
    for k in range(k_max + 1):
        yield k, full_tree.clone_expansion_prefix(k)


def tree_to_graphviz(
    tree: SearchTree,
    stem: str,
    *,
    show_board: bool = True,
    title: str | None = None,
) -> str:
    """
    Render ``tree`` to ``{stem}.gv`` and ``{stem}.png``.

    Labels: one loop over nodes to derive FEN from root + incoming moves (parent cached);
    second loop attaches Graphviz styling and edges.
    Returns path to the PNG file.
    """
    expansion_parent_ids = tree.ordered_expansion_parent_ids()
    path_index = {nid: i for i, nid in enumerate(expansion_parent_ids)}

    fen_by_node: dict[int, str | None] = {}
    for node in tree.iter_nodes():
        nid = node.node_id
        if nid == 0:
            root = tree.get_node(0)
            try:
                fen_by_node[0] = chess.Board(root.fen).fen() if _is_real_fen(root.fen) else None
            except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError):
                fen_by_node[0] = None
            continue
        pid = node.parent_id
        assert pid is not None
        parent_fen = fen_by_node.get(pid)
        uci = node.incoming_move_uci
        if parent_fen is None or uci is None:
            fen_by_node[nid] = None
            continue
        try:
            board = chess.Board(parent_fen)
            board.push_uci(str(uci))
            fen_by_node[nid] = board.fen()
        except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError):
            fen_by_node[nid] = None

    n_nodes = tree.num_nodes()
    if n_nodes > 400 and show_board:
        print(f"Note: {n_nodes} nodes with Unicode boards may be slow; use show_board=False if needed.")

    dot = graphviz.Digraph(comment=title or stem)
    dot.attr(rankdir="TB", charset="UTF-8")
    dot.attr(
        "graph",
        label=f"{title or stem} · {n_nodes} nodes · {len(expansion_parent_ids)} expansion event(s)",
        labelloc="t",
        fontsize="10",
        fontname=_SANS,
    )
    dot.attr("node", shape="rect", style="rounded,filled", color=_BORDER)

    for node in tree.iter_nodes():
        nid = node.node_id
        leaf = len(tree.child_ids(nid)) == 0
        is_path = nid in path_index
        inc = node.incoming_move_uci or "—"
        head = "ROOT" if node.parent_id is None else inc

        if leaf:
            label = inc
        elif not show_board:
            label = head
        else:
            fen = fen_by_node.get(nid)
            if fen is None:
                label = f"{head}\n(position N/A: UCI line does not match root FEN)"
            else:
                u_fmt = chess.Board(fen).unicode(borders=False, empty_square="·")
                label = f"ROOT\n{u_fmt}" if node.parent_id is None else f"{head}\n{u_fmt}"

        fill_root = node.parent_id is None
        fcolor = _ACCENT if fill_root else "#0f172a"
        fill = _ROOT_BG if fill_root else ("#fff7ed" if is_path else "#ffffff")
        if is_path:
            dot.node(
                str(nid),
                label,
                fillcolor=fill,
                fontcolor=fcolor,
                color="#c2410c",
                penwidth="2.5",
                fontname=_MONO,
                fontsize="8" if leaf else "9",
            )
        elif leaf:
            dot.node(
                str(nid),
                label,
                fillcolor="#f8fafc",
                fontcolor="#64748b",
                color=_BORDER,
                penwidth="0.9",
                fontname=_MONO,
                fontsize="8",
            )
        else:
            dot.node(
                str(nid),
                label,
                fillcolor=fill,
                fontcolor=fcolor,
                color=_BORDER,
                penwidth="1.0",
                fontname=_MONO,
                fontsize="9",
            )

    for node in tree.iter_nodes():
        if node.parent_id is None:
            continue
        dot.edge(str(node.parent_id), str(node.node_id), color="#94a3b8", penwidth="1.0")

    for a, b in zip(expansion_parent_ids, expansion_parent_ids[1:]):
        dot.edge(
            str(a),
            str(b),
            color="#dc2626",
            penwidth="2.0",
            style="dashed",
            fontname=_MONO,
            fontsize="9",
            label="trace order",
        )

    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    dot.save(stem + ".gv")
    dot.render(stem, format="png", cleanup=True)
    return f"{stem}.png"


def _scalar_row(node_features: torch.Tensor, j: int, feature_names: list[str]) -> dict[str, float]:
    row = node_features[j]
    return {name: float(row[k].item()) for k, name in enumerate(feature_names)}


def _search_tree_from_raw_v1(data: dict[str, Any]) -> SearchTree:
    parent_index = data["parent_index"].detach().cpu().view(-1).long()
    node_features = data["node_features"].detach().cpu()
    feature_names = list(data["feature_names"])
    is_terminal = data["is_terminal"].detach().cpu().view(-1).long()
    incoming: list = data["incoming_moves"]
    n = int(parent_index.shape[0])
    root_fen = str(data["root_position_spec"])
    meta0 = data.get("metadata")
    if not isinstance(meta0, dict):
        meta0 = {}

    tree = SearchTree()
    tree.create_root(
        root_fen,
        _scalar_row(node_features, 0, feature_names),
        metadata={**meta0},
        is_terminal=bool(is_terminal[0].item()),
    )
    for parent_id in range(n):
        child_ids = [i for i in range(n) if int(parent_index[i].item()) == parent_id]
        if not child_ids:
            continue
        child_ids.sort(key=lambda j: (incoming[j] is None, str(incoming[j] or "")))
        specs: list[ExpansionChild] = []
        for i in child_ids:
            move = incoming[i]
            specs.append(
                ExpansionChild(
                    move_uci=str(move),
                    fen=f"packed:{i}",
                    scalar_features=_scalar_row(node_features, i, feature_names),
                    metadata={},
                    is_terminal=bool(is_terminal[i].item()),
                )
            )
        tree.add_children(parent_id, specs)
    return tree


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render tree expansion prefixes to PNG under figures/performance.")
    p.add_argument("--pt-path", default=_DEFAULT_PT, help="Path to a .pt tree checkpoint")
    p.add_argument(
        "n_expansions",
        type=int,
        help="Maximum expansion index k (inclusive steps): k in 0..min(n_expansions, E).",
    )
    p.add_argument("--no-boards", action="store_true", help="Omit Unicode boards in labels")
    args = p.parse_args()

    path = args.pt_path
    full_tree = load_tree_from_pt(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(path).stem
    show_board = not args.no_boards

    for k, partial in tree_iter_expand(full_tree, args.n_expansions):
        stem = str(OUTPUT_DIR / f"{base}_expansion_{k:04d}")
        print(tree_to_graphviz(partial, stem, show_board=show_board, title=f"{base} expansion_prefix k={k}"))
