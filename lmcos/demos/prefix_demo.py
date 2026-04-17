from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import chess

# This demo runs from lmcos/, but core modules live one directory up.
THIS_DIR = Path(__file__).resolve().parent
LMCOS_DIR = THIS_DIR.parent
CHESS_ANALYSIS_DIR = LMCOS_DIR.parent
if str(LMCOS_DIR) not in sys.path:
    sys.path.insert(0, str(LMCOS_DIR))

from cts_pretrain import (
    EdgeStats,
    NodeBudgetDistribution,
    TeacherSearchConfig,
    _backup_target_from_child_wdl,
    generate_partial_tree_from_provider,
    consolidate_generated_tree,
)
from tree import SearchTree


def default_lc0_engine_path() -> str:
    return str(CHESS_ANALYSIS_DIR / "engines" / "lc0-src" / "build" / "release" / "lc0")


def default_lc0_weights_path() -> str:
    return str(CHESS_ANALYSIS_DIR / "weights" / "t1-256x10-distilled-swa-2432500.pb.gz")


def build_teacher_config(args: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="demo_v1",
        search_config_id="demo_partial_tree_generation",
    )


class BreadthLimitedProvider:
    """Wraps a provider to only return the top-N children by prior score."""
    def __init__(self, base_provider: TreeExpansionProvider, max_children: int = 2):
        self.base_provider = base_provider
        self.max_children = max_children

    def root_features(self, fen: str):
        return self.base_provider.root_features(fen)

    def root_metadata(self, fen: str):
        return self.base_provider.root_metadata(fen)

    def expand_node(self, fen: str, depth: int):
        children = list(self.base_provider.expand_node(fen, depth))
        # Sort by 'prior' score from lc0 (higher is better)
        children.sort(key=lambda c: c.scalar_features.get("prior", 0), reverse=True)
        return children[:self.max_children]

    def provider_metadata(self):
        return self.base_provider.provider_metadata()


def build_provider(*, engine_path: str, weights_path: Optional[str], max_breadth: int = 2):
    from cts_uci_process import UciEngineConfig, UciEngineProcess
    from uci_provider import Lc0DirectEvalProvider

    # Provider role:
    # - query lc0 for priors/children (classic mode)
    # - query lc0 for value + WDL features (valuehead mode)
    prior_config = UciEngineConfig(
        engine_path=engine_path,
        engine_kind="lc0",
        engine_mode="classic",
        movetime_ms=0,
        multipv=8,
        nodes=1,
        weights_path=weights_path,
        uci_options={},
        set_multipv=False,
        enable_verbose_move_stats=True,
    )
    value_config = UciEngineConfig(
        engine_path=engine_path,
        engine_kind="lc0",
        engine_mode="valuehead",
        movetime_ms=0,
        multipv=1,
        nodes=1,
        weights_path=weights_path,
        uci_options={"UCI_ShowWDL": "true"},
        set_multipv=False,
        enable_verbose_move_stats=False,
    )
    prior_engine = UciEngineProcess(prior_config)
    value_engine = UciEngineProcess(value_config)
    provider = Lc0DirectEvalProvider(
        prior_engine=prior_engine,
        value_engine=value_engine,
        metadata={"engine_kind": "lc0"},
    )
    return (prior_engine, value_engine), provider


def position_spec_to_board(position_spec: str) -> chess.Board:
    if "||moves||" not in position_spec:
        return chess.Board(position_spec)
    fen_part, moves_part = position_spec.split("||moves||", maxsplit=1)
    board = chess.Board(fen_part.strip())
    for move in moves_part.strip().split():
        board.push_uci(move)
    return board


def board_lines(position_spec: str) -> list[str]:
    board = position_spec_to_board(position_spec)
    return board.unicode(borders=False, empty_square=".").splitlines()


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def teacher_line_uses_child_backup(
    tree: SearchTree,
    edge_stats: dict[tuple[int, int], EdgeStats],
    node_id: int,
) -> bool:
    """True iff consolidate's _backup_target_from_child_q uses visit-weighted child Q (not static fallback)."""
    child_ids = tree.child_ids(node_id)
    if not child_ids:
        return False
    total_visits = 0
    for child_id in child_ids:
        key = (node_id, child_id)
        if key not in edge_stats:
            return False
        total_visits += edge_stats[key].visit_count
    return total_visits > 0


def _static_wdl_triple(node) -> tuple[float, float, float] | None:
    try:
        return (
            float(node.scalar_features["wdl_win"]),
            float(node.scalar_features["wdl_draw"]),
            float(node.scalar_features["wdl_loss"]),
        )
    except KeyError:
        return None


def _format_wdl(triple: tuple[float, float, float]) -> str:
    w, d, l = triple
    return f"{w:.2f}/{d:.2f}/{l:.2f}"


def render_unified_prefix_svg(
    oracle_tree: SearchTree,
    partial_node_ids: set[int],
    out_path: Path,
    *,
    input_wdl: dict[int, tuple[float, float, float]],
    target_wdl: dict[int, tuple[float, float, float]],
) -> None:
    """Renders one unified tree where Partial nodes are highlighted and Future nodes are ghosted."""
    node_w, node_h = 240, 260
    col_gap, depth_step_y = 30, 320
    margin_x, margin_y = 40, 80

    # 1. Standard Layout calculation for the Oracle tree
    nodes_by_depth: dict[int, list[int]] = {}
    def visit(node_id: int):
        node = oracle_tree.get_node(node_id)
        nodes_by_depth.setdefault(node.depth, []).append(node_id)
        for child_id in oracle_tree.child_ids(node_id):
            visit(child_id)
    if oracle_tree.root_id is not None:
        visit(oracle_tree.root_id)
    
    positions: dict[int, tuple[float, float]] = {}
    max_row_width = 0
    for depth in range(max(nodes_by_depth or [0]) + 1):
        ids = nodes_by_depth.get(depth, [])
        row_width = len(ids) * node_w + max(0, len(ids) - 1) * col_gap
        max_row_width = max(max_row_width, row_width)
        x = margin_x
        y = margin_y + depth * depth_step_y
        for node_id in ids:
            positions[node_id] = (x, y)
            x += node_w + col_gap

    canvas_w = int(max_row_width + 2 * margin_x)
    max_d = max(node.depth for node in oracle_tree.iter_nodes()) if oracle_tree.num_nodes() > 0 else 0
    canvas_h = int(margin_y + (max_d + 1) * depth_step_y + node_h)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" /></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{margin_x:.0f}" y="35" font-family="Inter, sans-serif" font-size="24" font-weight="bold" fill="#0f172a">Unified Search Context: Input vs Target</text>',
        f'<text x="{margin_x:.0f}" y="60" font-family="Inter, sans-serif" font-size="14" fill="#64748b">'
        f'<tspan fill="#3b82f6" font-weight="bold">Blue Solid</tspan> = Known by GNN (Input) | '
        f'<tspan fill="#94a3b8" font-style="italic">Gray Dashed</tspan> = The "Future" (Oracle Targets)'
        f'</text>',
    ]

    # Edges
    for node in oracle_tree.iter_nodes():
        px, py = positions[node.node_id]
        x1, y1 = px + node_w / 2, py + node_h
        for child_id in oracle_tree.child_ids(node.node_id):
            cx, cy = positions[child_id]
            x2, y2 = cx + node_w / 2, cy
            stroke = "#cbd5e1" if child_id in partial_node_ids else "#e2e8f0"
            dash = 'stroke-dasharray="4 2"' if child_id not in partial_node_ids else ""
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="1.4" {dash} marker-end="url(#arrow)"/>')
    
    # Nodes
    for node in oracle_tree.iter_nodes():
        x, y = positions[node.node_id]
        move = node.incoming_move_uci or "<root>"
        is_input = node.node_id in partial_node_ids
        
        border_color = "#3b82f6" if is_input else "#cbd5e1"
        bg_color = "#ffffff" if is_input else "#f8fafc"
        stroke_dash = "" if is_input else 'stroke-dasharray="8 4"'
        opacity = "1.0" if is_input else "0.6"

        svg.append(f'<g opacity="{opacity}">')
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w}" height="{node_h}" rx="12" ry="12" fill="{bg_color}" stroke="{border_color}" stroke-width="2.5" {stroke_dash}/>')
        
        # Header / Move
        svg.append(f'<text x="{x + node_w / 2:.1f}" y="{y + 25:.1f}" text-anchor="middle" font-family="Inter, sans-serif" font-size="12" font-weight="bold" fill="#0f172a">{escape_xml(move)}</text>')
        
        # Values
        if is_input:
            svg.append(f'<text x="{x + 15}" y="{y + 45}" font-family="Inter, sans-serif" font-size="10" fill="#64748b">INPUT:</text>')
            svg.append(f'<text x="{x + 65}" y="{y + 45}" font-family="Inter, sans-serif" font-size="10" font-weight="600" fill="#1e293b">{_format_wdl(input_wdl[node.node_id])}</text>')
            svg.append(f'<text x="{x + 15}" y="{y + 60}" font-family="Inter, sans-serif" font-size="10" fill="#64748b">TARGET:</text>')
            svg.append(f'<text x="{x + 65}" y="{y + 60}" font-family="Inter, sans-serif" font-size="10" font-weight="600" fill="#3b82f6">{_format_wdl(target_wdl[node.node_id])}</text>')
        else:
            svg.append(f'<text x="{x + 15}" y="{y + 50}" font-family="Inter, sans-serif" font-size="10" fill="#94a3b8" font-style="italic">Future Node</text>')
            svg.append(f'<text x="{x + 15}" y="{y + 65}" font-family="Inter, sans-serif" font-size="10" font-weight="600" fill="#94a3b8">{_format_wdl(target_wdl[node.node_id])}</text>')

        svg.append(f'<line x1="{x+10}" y1="{y+75}" x2="{x+node_w-10}" y2="{y+75}" stroke="#e2e8f0" />')
        lines = board_lines(node.fen)
        for i, line in enumerate(lines):
            svg.append(f'<text x="{x + node_w / 2:.1f}" y="{y + 95 + i * 19:.1f}" text-anchor="middle" font-family="Menlo, monospace" font-size="16" fill="#1e293b">{escape_xml(line)}</text>')
        svg.append('</g>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side demo of Prefix Sampling.")
    parser.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="Root FEN to expand.")
    parser.add_argument("--engine-path", default=default_lc0_engine_path())
    parser.add_argument("--weights-path", default=default_lc0_weights_path())
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--oracle-nodes", type=int, default=12, help="Size of the deep tree.")
    parser.add_argument("--prefix-nodes", type=int, default=4, help="Size of the chopped tree.")
    parser.add_argument("--out-svg", default="demos/figures/prefix_demo.svg")
    args = parser.parse_args()

    config = build_teacher_config(args)
    engines, provider = build_provider(engine_path=args.engine_path, weights_path=args.weights_path)
    
    try:
        for engine in engines: engine.start()
        
        # 1. Generate the Oracle Tree (The Deep Future)
        oracle_gen = generate_partial_tree_from_provider(
            root_fen=args.fen.strip(),
            provider=provider,
            config=config,
            node_budget_distribution=NodeBudgetDistribution(args.oracle_nodes, args.oracle_nodes)
        )
        
        # 2. Chop the Tree (Perform Prefix Sampling)
        # We simulate what the tree looked like after only 'prefix_nodes' expansions.
        partial_tree = oracle_gen.tree.clone_expansion_prefix(args.prefix_nodes)
        
        # 3. Calculate WDL for both
        # Oracle WDL uses the full results. Partial WDL uses just the static heuristics of the root-prefix.
        oracle_res = consolidate_generated_tree(oracle_gen, config)
        target_wdl = {node.node_id: _backup_target_from_child_wdl(oracle_gen.tree, node.node_id, oracle_gen.edge_stats) 
                      for node in oracle_gen.tree.iter_nodes()}
        
        # For the partial nodes, we show what they looked like BEFORE the search deepened.
        input_wdl = {node.node_id: _static_wdl_triple(node) or (0,0,0) 
                     for node in partial_tree.iter_nodes()}
        
        # Mapping: partial_tree nodes to oracle_tree nodes (since IDs might shift during cloning)
        # However, clone_expansion_prefix preserves IDs for the prefix nodes.
        partial_node_ids = {node.node_id for node in partial_tree.iter_nodes()}

        render_unified_prefix_svg(
            oracle_gen.tree, partial_node_ids, Path(args.out_svg),
            input_wdl=input_wdl, target_wdl=target_wdl
        )
        print(f"Rendered unified prefix demo to {args.out_svg}")

    finally:
        for engine in engines: engine.close()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

