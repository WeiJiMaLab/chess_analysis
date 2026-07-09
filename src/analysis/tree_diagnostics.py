"""T1/T2 tree-shape sanity diagnostics (plan.md, 2026-07-08) -- one-off but reusable helpers.

T1: pick 5 real argmax-filtered validation trees spanning the argmax distribution (p10/p50/p75/
p90/max), render each via graphviz, and check the red flags plan.md lists: duplicate FEN at two
nodes, a fully-linear chain, one child dominating the node budget before enough of the tree has
been explored to justify it, and illegal-looking moves.

T2: histogram depth/width/argmax across all validation episodes with the red-flag threshold lines
plan.md specifies (depth<=2, width pinned at the root's legal-move count, argmax>=90).

Kept as small, independently-testable pure functions (see ``tests/test_tree_diagnostics.py``)
rather than one monolithic script, per the TDD-lite convention used elsewhere in ``analysis/``.
"""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any

import chess
import matplotlib.pyplot as plt
import numpy as np

from analysis.utils.plots import save_pdf_png
from cts.core.providers.common import board_from_position_spec


# ===========================================================================
# T1 -- episode selection + red-flag checks
# ===========================================================================
def select_percentile_episodes(values: np.ndarray, targets: list[float]) -> list[int]:
    """For each target value, return the index of the episode whose ``values`` entry is closest
    to it (ties broken by first occurrence). Used to pick 5 trees spanning the argmax spread at
    approximately p10/p50/p75/p90/max, rather than literal percentile interpolation (argmax is
    integer-valued and we want a REAL episode, not an interpolated one)."""
    values = np.asarray(values)
    return [int(np.argmin(np.abs(values - t))) for t in targets]


def duplicate_fen_check(fens: list[str]) -> dict:
    """Red flag (a): does any FEN repeat at two different nodes (transposition/generation bug)?
    ``fens`` may be plain FEN strings OR the packed "position spec" format (``<fen> ||moves|| ...``);
    both are resolved via ``board_from_position_spec`` then compared by ``board.epd()``
    (piece placement+turn+castling+ep only, no halfmove/fullmove counters) since two nodes
    reaching the identical position via different move orders IS the transposition this flag
    looks for, even if move counters differ."""
    boards = [board_from_position_spec(f).epd() for f in fens]
    seen: dict[str, int] = {}
    dups = []
    for i, b in enumerate(boards):
        if b in seen:
            dups.append((seen[b], i))
        else:
            seen[b] = i
    return {"pass": len(dups) == 0, "n_duplicates": len(dups), "example_pairs": dups[:3]}


def linear_chain_check(child_counts: np.ndarray) -> dict:
    """Red flag (b): a fully linear chain = every node has at most 1 child (zero branching
    anywhere in the tree)."""
    child_counts = np.asarray(child_counts)
    return {"pass": bool((child_counts >= 2).any()), "max_children": int(child_counts.max(initial=0)),
            "n_branching_nodes": int((child_counts >= 2).sum())}


def root_dominance_check(root_child_subtree_sizes: list[int], total_nodes: int,
                         dominance_frac: float = 0.9, min_total_nodes: int = 20) -> dict:
    """Red flag (c) proxy: one child capturing (almost) the whole node budget while sibling(s)
    were never even expanded (subtree size 1 = leaf), on a tree big enough that this reflects a
    real allocation choice rather than just being small (``min_total_nodes``).

    Node-level visit counts aren't stored on disk (see ``RawPretrainExampleRecord.feature_names``:
    value/wdl/cp_order/prior, no visit count) -- subtree size (descendant count) is used as the
    visit proxy, since PUCT only grows a subtree by continuing to select and expand within it.
    """
    sizes = np.asarray(root_child_subtree_sizes)
    if sizes.size == 0 or total_nodes < min_total_nodes:
        return {"pass": True, "reason": "tree too small to judge", "max_child_frac": None,
                "n_unexpanded_siblings": None}
    max_frac = float(sizes.max()) / float(total_nodes)
    n_unexpanded = int((sizes == 1).sum())
    fires = max_frac >= dominance_frac and n_unexpanded >= 1
    return {"pass": not fires, "max_child_frac": max_frac, "n_unexpanded_siblings": n_unexpanded,
            "n_root_children": int(sizes.size)}


def illegal_move_check(fens: list[str], parent_index: np.ndarray, moves: list[str | None]) -> dict:
    """Red flag (d): does any edge's UCI move fail to be a legal move from its parent's position?
    ``fens`` may be plain FEN strings or packed "position spec" strings (``<fen> ||moves|| ...``)."""
    bad = []
    for node_id, (pid, move) in enumerate(zip(parent_index.tolist(), moves)):
        if pid < 0 or move is None:
            continue
        board = board_from_position_spec(fens[pid])
        if board is None:
            bad.append((node_id, move, "unparseable-parent-fen"))
            continue
        try:
            mv = chess.Move.from_uci(move)
        except ValueError:
            bad.append((node_id, move, "unparseable"))
            continue
        if mv not in board.legal_moves:
            bad.append((node_id, move, "illegal"))
    return {"pass": len(bad) == 0, "n_bad": len(bad), "examples": bad[:5]}


def prune_topk_subtree(parent_index: np.ndarray, budget: int, *,
                       intermediate_only: np.ndarray | None = None) -> set[int]:
    """Select a connected, root-containing subset of <= ``budget`` node ids for legible graphviz
    rendering: greedily grow the SUBTREE ROOTED AT THE MOST-VISITED (largest-subtree-size) node
    at each step, so the rendered view prioritizes the tree's real structure (the branches PUCT
    actually explored) over an arbitrary prefix. Pure function of ``parent_index`` (root=-1) so
    it's independently testable on tiny synthetic trees.

    ``intermediate_only``: optional boolean array, True where a node has >=1 child in the FULL
    tree (i.e. was actually expanded). When given, leaf nodes are never pushed onto the frontier
    at all, so ``selected`` only ever contains genuinely-intermediate nodes plus the root (even if
    the root itself is a leaf in a degenerate zero-expansion tree, it's always kept so the
    returned set is never empty). This shows tree SHAPE (what PUCT chose to expand) without the
    (usually much larger) leaf-node clutter of every considered-but-never-expanded child.
    """
    parent_index = np.asarray(parent_index)
    n = len(parent_index)
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for node_id, pid in enumerate(parent_index.tolist()):
        if pid >= 0:
            children[pid].append(node_id)
    # subtree size via one reverse pass (topological: parent id < child id, per _validate).
    subtree_size = np.ones(n, dtype=np.int64)
    for node_id in range(n - 1, 0, -1):
        subtree_size[parent_index[node_id]] += subtree_size[node_id]

    def eligible(node_id: int) -> bool:
        return intermediate_only is None or node_id == 0 or bool(intermediate_only[node_id])

    selected: set[int] = {0}
    # max-heap (via negated size) of frontier nodes eligible to be pulled into `selected` next.
    pending = [(-int(subtree_size[c]), c) for c in children[0] if eligible(c)]
    heapq.heapify(pending)
    while pending and len(selected) < budget:
        neg_size, node_id = heapq.heappop(pending)
        if node_id in selected:
            continue
        selected.add(node_id)
        for c in children[node_id]:
            if eligible(c):
                heapq.heappush(pending, (-int(subtree_size[c]), c))
    return selected


# ===========================================================================
# T1 -- graphviz rendering
# ===========================================================================
def render_board_png(fen: str, out_path: str, size: int = 120) -> str:
    """Render one FEN (or packed ``<fen> ||moves|| ...`` position spec, per
    ``board_from_position_spec``) to a PNG board diagram via chess.svg + rsvg-convert (no cairosvg
    in this env). Returns ``out_path``. Raises if rsvg-convert isn't on PATH or the position spec
    fails to parse -- fail loud rather than silently falling back to a blank/placeholder image."""
    import subprocess
    import chess.svg
    from cts.core.providers.common import board_from_position_spec
    board = board_from_position_spec(fen)
    if board is None:
        raise ValueError(f"could not resolve position spec into a board: {fen!r}")
    svg_text = chess.svg.board(board, size=size)
    svg_path = out_path.replace(".png", ".svg")
    Path(svg_path).write_text(svg_text)
    subprocess.run(["rsvg-convert", "-o", out_path, svg_path], check=True, capture_output=True)
    return out_path


def render_tree_graphviz(node_ids: set[int], parent_index: np.ndarray, incoming_moves: list[str | None],
                         node_targets: np.ndarray, fens: list[str], out_path: str, caption: str,
                         board_size: int = 90) -> str:
    """Render the pruned node set as a graphviz tree: each node is an HTML-like label showing
    ``Val: <backed-up value>`` as text above the actual rendered board position for that node's
    FEN (no color/size-as-value encoding -- the board diagram + value text carry that information
    directly and unambiguously). Edges labeled with the UCI move. Returns the path of the rendered
    file (graphviz appends its own format extension). Per-node board PNGs are written to
    ``<out_path>_assets/`` alongside the final render.

    Output format is SVG, not PNG: a fixed-resolution raster (especially once several of these get
    montaged into one combined figure) makes the many-small-boards trees (p75/p90/max) illegible.
    SVG stays crisp at any zoom/print size -- the board sub-images are still embedded PNGs (raster
    inside vector, graphviz's SVG backend handles this natively via <image> tags), only the tree
    structure itself (lines, text, layout) is vector."""
    import graphviz
    assets_dir = Path(f"{out_path}_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    dot = graphviz.Digraph(comment=caption, format="svg")
    dot.attr(rankdir="TB", label=caption, labelloc="t", fontsize="14")
    dot.attr("node", shape="none", fontsize="10", fontname="Helvetica", margin="0")
    for node_id in node_ids:
        board_png = str(assets_dir / f"node_{node_id}.png")
        render_board_png(fens[node_id], board_png, size=board_size)
        val = float(node_targets[node_id])
        label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f'<TR><TD>Val: {val:+.2f}</TD></TR>'
            f'<TR><TD><IMG SRC="{board_png}"/></TD></TR>'
            '</TABLE>>'
        )
        dot.node(str(node_id), label=label)
    for node_id in node_ids:
        pid = int(parent_index[node_id])
        if pid in node_ids:
            mv = incoming_moves[node_id] or "?"
            dot.edge(str(pid), str(node_id), label=mv, fontsize="8")
    return dot.render(out_path, cleanup=True)


# ===========================================================================
# T2 -- histograms
# ===========================================================================
def plot_diagnostic_histograms(depth: np.ndarray, width: np.ndarray, argmax: np.ndarray, *,
                               depth_flag: float, width_flag: float, argmax_flag: float,
                               out_dir: str | Path, base: str = "t2_histograms") -> dict:
    """1x3 panel: depth / width / argmax histograms with the plan.md red-flag threshold drawn as
    a vertical line on each. Returns the fired/not-fired summary for each flag."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [("Tree depth (final)", depth, depth_flag, "depth<=2 (breadth-starvation)"),
            ("Tree width (max level size, final)", width, width_flag, "root legal-move count"),
            ("Optimal stop step (argmax)", argmax, argmax_flag, "argmax>=90 (right-censoring)")]
    for ax, (title, data, flag, flag_label) in zip(axes, specs):
        ax.hist(data, bins=40, color="#3F4DA0", alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(flag, color="#C0392B", ls="--", lw=1.6, label=flag_label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(title.split(" (")[0])
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", color="#E3E7EB", lw=1)
    fig.tight_layout()
    save_pdf_png(fig, str(out_dir), base, dpi=200)

    return {
        "depth_le_2_frac": float((depth <= depth_flag).mean()),
        "width_pinned_frac_at_flag_or_above": float((width >= width_flag).mean()),
        "argmax_ge_90_frac": float((argmax >= argmax_flag).mean()),
    }
