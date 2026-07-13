"""Stage 2 (no cluster resources needed): render board/return-curve/root-children figures for
3 hand-picked Action-Gap-vs-Meta-Controller divergence cases (see scripts_tmp_case_study_extract.py
+ the leader-change-point inspection that picked these 3), and write outputs/reports/casestudy.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CANDIDATES_PATH = Path("/home/hl4291/chess_analysis/outputs/reports/case_study_candidates.json")
FIG_DIR = Path("/home/hl4291/chess_analysis/outputs/reports/figures/casestudy")
REPORT_PATH = Path("/home/hl4291/chess_analysis/outputs/reports/casestudy.md")

_LIGHT, _DARK = "#EEEED2", "#769656"
_HILITE = "#F6F669"
_AG_COLOR, _ZT_COLOR, _OR_COLOR = "#8E6BAF", "#3F4DA0", "#999999"
_UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def draw_board(ax, fen: str, *, title: str, highlight_squares=None, subtitle: str | None = None):
    board = chess.Board(fen)
    highlight_squares = highlight_squares or set()
    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, rank)
            color = _HILITE if sq in highlight_squares else (_LIGHT if (rank + file) % 2 else _DARK)
            ax.add_patch(mpatches.Rectangle((file, rank), 1, 1, facecolor=color, edgecolor="none"))
            piece = board.piece_at(sq)
            if piece:
                ax.text(file + 0.5, rank + 0.5, _UNICODE[piece.symbol()], ha="center", va="center",
                        fontsize=22, zorder=3)
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, pad=6)
    if subtitle:
        ax.text(4, -0.6, subtitle, ha="center", va="top", fontsize=9, color="#555")


def san_of(fen: str, uci_move: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    return board.san(move)


def board_after(fen: str, uci_move: str) -> tuple[str, set[int]]:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    hl = {move.from_square, move.to_square}
    board.push(move)
    return board.fen(), hl


def leader_change_points(best_idx: list[int]) -> list[int]:
    return [0] + [i for i in range(1, len(best_idx)) if best_idx[i] != best_idx[i - 1]]


def render_case(cand: dict, tag: str, title: str) -> dict:
    root_fen = cand["root_fen"]
    moves = cand["oracle_root_moves"]
    return_curve = np.array(cand["return_curve"])
    best_idx = cand["best_move_index_per_step"]
    num_steps = cand["num_steps"]
    stop_ag, stop_zt, stop_or = cand["stop_ag"], cand["stop_zt"], cand["oracle_stop_step"]
    stop_or_clamped = min(max(stop_or, 0), num_steps - 1)

    move_ag_uci, move_zt_uci = moves[best_idx[stop_ag]], moves[best_idx[stop_zt]]
    san_ag, san_zt = san_of(root_fen, move_ag_uci), san_of(root_fen, move_zt_uci)
    fen_ag, hl_ag = board_after(root_fen, move_ag_uci)
    fen_zt, hl_zt = board_after(root_fen, move_zt_uci)

    changes = leader_change_points(best_idx)
    board0 = chess.Board(root_fen)
    side = "White" if board0.turn else "Black"

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.6, wspace=0.35)

    ax_root = fig.add_subplot(gs[0, 0])
    draw_board(ax_root, root_fen, title=f"Root position ({side} to move)")

    ax_curve = fig.add_subplot(gs[0, 1])
    steps = np.arange(num_steps)
    ax_curve.plot(steps, return_curve, color="#556270", lw=1.6)
    ax_curve.axvline(stop_ag, color=_AG_COLOR, ls="--", lw=2, label=f"Action Gap halts ({stop_ag})")
    ax_curve.axvline(stop_zt, color=_ZT_COLOR, ls=":", lw=2.2, label=f"Meta Controller halts ({stop_zt})")
    ax_curve.axvline(stop_or_clamped, color=_OR_COLOR, ls="-.", lw=1.4, label=f"Oracle-optimal halt ({stop_or})")
    # Only label the leader changes that matter to the story (dedupe transient blips that revert
    # to an already-seen move) -- annotating every raw change_point on a tight x-range clutters
    # into unreadable overlapping arrows.
    story_idx = sorted(set(best_idx[s] for s in {0, stop_ag, stop_zt, stop_or_clamped, num_steps - 1}))
    # One label per distinct story move, at its FIRST appearance as leader (dedupes re-entries,
    # e.g. a move that leads, briefly loses the lead, then regains it).
    first_change_for = {}
    for ci in changes:
        first_change_for.setdefault(best_idx[ci], ci)
    labeled_changes = sorted(first_change_for[mi] for mi in story_idx)
    for k, ci in enumerate(labeled_changes):
        san = san_of(root_fen, moves[best_idx[ci]])
        ax_curve.annotate(san, xy=(ci, return_curve[ci]), xytext=(10, 14 if k % 2 == 0 else -18),
                          textcoords="offset points", fontsize=9, ha="left", color="#333",
                          arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))
    ax_curve.margins(y=0.2)  # headroom so top-anchored annotate() labels never collide with the title
    ax_curve.set_xlabel("Search step (of 96-step budget)")
    ax_curve.set_ylabel("Cost-adjusted return")
    ax_curve.set_title("Return curve, with each leader-move change labeled")
    ax_curve.legend(fontsize=7, loc="lower right")

    ax_tree = fig.add_subplot(gs[0, 2])
    ax_tree.axis("off")
    involved = sorted(set(best_idx[s] for s in {0, stop_ag, stop_zt, stop_or_clamped, num_steps - 1}))
    ax_tree.text(0.5, 0.95, "Root", ha="center", fontsize=11, fontweight="bold", transform=ax_tree.transAxes)
    n = len(involved)
    for k, mi in enumerate(involved):
        y = 0.78 - k * (0.7 / max(n - 1, 1)) if n > 1 else 0.5
        san = san_of(root_fen, moves[mi])
        final_q = cand["oracle_final_root_q_values"][mi]
        tags = []
        if moves[mi] == move_ag_uci:
            tags.append("Action Gap picks this")
        if moves[mi] == move_zt_uci:
            tags.append("Meta Controller picks this")
        if moves[mi] == moves[best_idx[stop_or_clamped]]:
            tags.append("oracle-optimal pick")
        label = f"{san}   (final Q={final_q:+.3f})" + ("\n" + " / ".join(tags) if tags else "")
        color = "#2C3E50" if tags else "#7A8894"
        ax_tree.annotate("", xy=(0.5, y + 0.05), xytext=(0.5, 0.92),
                         arrowprops=dict(arrowstyle="-", color="#ccc", lw=1))
        ax_tree.text(0.5, y, label, ha="center", va="center", fontsize=8.5, color=color,
                    transform=ax_tree.transAxes,
                    bbox=dict(boxstyle="round", fc="white", ec=color, lw=1))
    ax_tree.set_title("Root's decisive candidate moves", fontsize=11)

    ax_ag = fig.add_subplot(gs[1, 0])
    draw_board(ax_ag, fen_ag, title=f"Action Gap plays {san_ag}",
              highlight_squares=hl_ag,
              subtitle=f"halts step {stop_ag}/95, regret={cand['regret_ag']:.3f}")

    ax_zt = fig.add_subplot(gs[1, 1])
    draw_board(ax_zt, fen_zt, title=f"Meta Controller plays {san_zt}",
              highlight_squares=hl_zt,
              subtitle=f"halts step {stop_zt}/95, regret={cand['regret_zt']:.3f}")

    ax_or = fig.add_subplot(gs[1, 2])
    move_or_uci = moves[best_idx[stop_or_clamped]]
    san_or = san_of(root_fen, move_or_uci)
    fen_or, hl_or = board_after(root_fen, move_or_uci)
    draw_board(ax_or, fen_or, title=f"Oracle-optimal plays {san_or}",
              highlight_squares=hl_or,
              subtitle=f"halts step {stop_or_clamped}/95 (regret = 0 by definition)")

    fig.suptitle(title, fontsize=13, y=1.0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(san_ag=san_ag, san_zt=san_zt, san_or=san_or, side=side,
               fig_path=out_path.relative_to(REPORT_PATH.parent), changes=changes,
               move_ag_uci=move_ag_uci, move_zt_uci=move_zt_uci)


def main():
    data = json.loads(CANDIDATES_PATH.read_text())
    by_key = {c["trajectory_key"]: c for c in data["candidates"]}

    picks = [
        ("shard_00013.pt#320", "case_a_knife_edge", "Case A — a one-step knife-edge (Meta Controller wins)"),
        ("shard_00000.pt#309", "case_b_patient_search", "Case B — patience pays off (Meta Controller wins)"),
        ("shard_00018.pt#190", "case_c_premature_stop", "Case C — Meta Controller stops too early (Action Gap wins)"),
    ]
    results = {}
    for key, tag, title in picks:
        cand = by_key[key]
        results[tag] = dict(cand=cand, **render_case(cand, tag, title))
        print(f"{tag}: saved {results[tag]['fig_path']}")

    json.dump({k: {kk: (str(vv) if isinstance(vv, Path) else vv) for kk, vv in v.items() if kk != "cand"}
              for k, v in results.items()},
             open(FIG_DIR / "_render_summary.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
