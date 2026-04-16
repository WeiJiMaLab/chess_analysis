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


def build_provider(*, engine_path: str, weights_path: Optional[str]):
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


def render_tree_svg(
    tree: SearchTree,
    out_path: Path,
    *,
    teacher_line_is_backup: dict[int, bool],
    node_teacher_wdl: dict[int, tuple[float, float, float]],
) -> None:
    # Preorder per depth keeps subtrees contiguous and avoids crossing lines.
    if tree.root_id is None:
        raise ValueError("Tree has no root.")
    nodes_by_depth: dict[int, list[int]] = {}

    def visit(node_id: int) -> None:
        node = tree.get_node(node_id)
        nodes_by_depth.setdefault(node.depth, []).append(node_id)
        for child_id in tree.child_ids(node_id):
            visit(child_id)

    visit(tree.root_id)

    max_depth = max(nodes_by_depth)
    node_w, node_h = 220, 268
    col_gap, depth_step_y = 20, 360
    margin_x, margin_y = 40, 40
    legend_h = 28

    positions: dict[int, tuple[float, float]] = {}
    max_row_width = 0.0
    for depth in range(max_depth + 1):
        ids = nodes_by_depth.get(depth, [])
        row_width = len(ids) * node_w + max(0, len(ids) - 1) * col_gap
        max_row_width = max(max_row_width, row_width)
        x = margin_x
        y = margin_y + legend_h + depth * depth_step_y
        for node_id in ids:
            positions[node_id] = (x, y)
            x += node_w + col_gap

    canvas_w = int(max_row_width + 2 * margin_x)
    canvas_h = int(margin_y + legend_h + (max_depth + 1) * depth_step_y + node_h)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" /></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{margin_x:.0f}" y="18" font-family="Inter, Helvetica, Arial, sans-serif" font-size="10" fill="#64748b">'
        f'{escape_xml("teacher WDL: ")}'
        f'<tspan fill="#1d4ed8" font-weight="600">{escape_xml("blue")}</tspan>'
        f'{escape_xml(" = visit-weighted from children; ")}'
        f'<tspan fill="#94a3b8" font-weight="600">{escape_xml("gray")}</tspan>'
        f'{escape_xml(" = static (network) fallback")}'
        f"</text>",
    ]

    for node in tree.iter_nodes():
        px, py = positions[node.node_id]
        x1, y1 = px + node_w / 2, py + node_h
        for child_id in tree.child_ids(node.node_id):
            cx, cy = positions[child_id]
            x2, y2 = cx + node_w / 2, cy
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#cbd5e1" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )

    for node in tree.iter_nodes():
        x, y = positions[node.node_id]
        move = node.incoming_move_uci if node.incoming_move_uci is not None else "<root>"
        is_backup = teacher_line_is_backup[node.node_id]
        wdl_fill = "#1d4ed8" if is_backup else "#94a3b8"
        static_wdl = _static_wdl_triple(node)
        line1 = f"move: {move}"
        line2 = (
            f"wdl(s): {_format_wdl(static_wdl)}"
            if static_wdl is not None
            else "wdl(s): (missing)"
        )
        tw = node_teacher_wdl[node.node_id]
        line3 = f"teacher WDL: {_format_wdl(tw)}"
        lines = board_lines(node.fen)

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w}" height="{node_h}" rx="10" ry="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.0"/>'
        )
        svg.append(
            f'<text x="{x + node_w / 2:.1f}" y="{y + 18:.1f}" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="12" fill="#0f172a">{escape_xml(line1)}</text>'
        )
        svg.append(
            f'<text x="{x + node_w / 2:.1f}" y="{y + 36:.1f}" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="10" fill="#334155">{escape_xml(line2)}</text>'
        )
        svg.append(
            f'<text x="{x + node_w / 2:.1f}" y="{y + 54:.1f}" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="10" font-weight="500" fill="{wdl_fill}">{escape_xml(line3)}</text>'
        )
        svg.append(
            f'<line x1="{x + 8:.1f}" y1="{y + 62:.1f}" x2="{x + node_w - 8:.1f}" y2="{y + 62:.1f}" stroke="#e2e8f0" stroke-width="1"/>'
        )

        start_y = y + 80
        for i, line in enumerate(lines):
            svg.append(
                f'<text x="{x + node_w / 2:.1f}" y="{start_y + i * 22:.1f}" text-anchor="middle" font-family="Menlo, Monaco, monospace" font-size="17" fill="#1e293b">{escape_xml(line)}</text>'
            )

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demo: build a budgeted partial tree (provider + PUCT), then render WDL (static vs "
            "visit-weighted teacher backup) as SVG.\n"
            "Important: this builds a Python-side tree; it is not lc0's internal tree."
        )
    )
    parser.add_argument("--fen", required=True, help="Root FEN to expand.")
    parser.add_argument("--engine-path", default=default_lc0_engine_path(), help="Path to lc0 binary.")
    parser.add_argument("--weights-path", default=default_lc0_weights_path(), help="Path to lc0 weights.")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-svg", default="demos/out.svg")
    args = parser.parse_args()

    config = build_teacher_config(args)
    node_budget_distribution = NodeBudgetDistribution(args.min_nodes, args.max_nodes)
    rng = random.Random(args.seed)

    engines, provider = build_provider(engine_path=args.engine_path, weights_path=args.weights_path)
    try:
        for engine in engines:
            engine.start()
        generated = generate_partial_tree_from_provider(
            root_fen=args.fen.strip(),
            provider=provider,
            config=config,
            node_budget_distribution=node_budget_distribution,
            rng=rng,
        )
    finally:
        for engine in engines:
            try:
                engine.close()
            except Exception:
                pass

    teacher_line_is_backup = {
        node.node_id: teacher_line_uses_child_backup(generated.tree, generated.edge_stats, node.node_id)
        for node in generated.tree.iter_nodes()
    }
    node_teacher_wdl = {
        node.node_id: _backup_target_from_child_wdl(generated.tree, node.node_id, generated.edge_stats)
        for node in generated.tree.iter_nodes()
    }

    out_svg = Path(args.out_svg)
    render_tree_svg(
        generated.tree,
        out_svg,
        teacher_line_is_backup=teacher_line_is_backup,
        node_teacher_wdl=node_teacher_wdl,
    )
    print(f"wrote_svg={out_svg} nodes={generated.tree.num_nodes()} edges={generated.tree.num_edges()}")


if __name__ == "__main__":
    main()

