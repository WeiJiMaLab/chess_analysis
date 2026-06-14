"""lc0 112-plane board encoding (Phase 2 scaffold, report §2 / T-enc).

The in-process net consumes the same 112-plane input lc0 builds internally:
8 history positions x (6 own piece planes + 6 opponent piece planes + 2
repetition planes) = 104 planes, followed by 8 constant planes (4 castling, 1
side-to-move, 1 fifty-move counter, 1 move counter / all-ones, 1 all-ones).
Side-to-move orientation: the board is flipped so the side to move is always
"us" at the bottom — values/policies are therefore side-to-move-relative,
matching ``PositionEval`` (report contract).

**The decision this module encodes (report §2, resolved 2026-06-14):** our
pipeline today evaluates positions **history-free** (bare ``position fen``), so
lc0 fills its 7 history planes by its own convention. The project priority is
**bit-for-bit replication first** — so the *default* is to match lc0's actual
bare-FEN behavior, and the cleaner convention is gated behind ``re_baseline``
(flip to ``True`` later). The candidate conventions:

  (a) zero-fill — history planes all zero (no prior positions known);
  (b) repeat-fill — copy the current position into all 8 history slots;
  (c) ``LC0_BARE_FEN`` — lc0's actual bare-FEN behavior, to be confirmed
      against the binary; this is the **default** (``re_baseline=False``).

When ``re_baseline=True`` we switch to the defined canonical fill
(``REPEAT``) and re-baseline (re-run the A0 oracle-direction checks). Use
:func:`history_fill_for` to map the ``re_baseline`` flag to a fill mode so the
encoding, the numeric-parity test (T-fwd), and the re-baseline switch all agree.

Strongly preferred implementation (report §6): delegate to lczerolens's
lc0-faithful ``LczeroBoard`` encoder rather than hand-rolling the planes, which
removes most of the encoding-parity risk (T-enc). The heavy import stays inside
the methods so importing this module never requires lczerolens.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime here.
    import numpy as np


LC0_NUM_PLANES = 112
LC0_HISTORY_LENGTH = 8  # number of half-move positions lc0 stacks in its input


class BareFenHistoryFill(str, Enum):
    """How to fill lc0's history planes when a position has no move history.

    See the module docstring (§2). ``LC0_BARE_FEN`` is the replication default
    (``re_baseline=False``); ``REPEAT`` is the canonical re-baseline fill.
    """

    ZERO = "zero"  # history planes all zero
    REPEAT = "repeat"  # current position copied into every history slot
    LC0_BARE_FEN = "lc0_bare_fen"  # match lc0's actual bare-FEN fill (replication)


def history_fill_for(re_baseline: bool) -> "BareFenHistoryFill":
    """Map the project-wide ``re_baseline`` switch to a history-fill mode.

    ``re_baseline=False`` (default, priority): replicate lc0's bare-FEN behavior
    bit-for-bit. ``re_baseline=True``: use the defined canonical fill.
    """
    return BareFenHistoryFill.REPEAT if re_baseline else BareFenHistoryFill.LC0_BARE_FEN


def encode_board_planes(
    fen: str, *, history_fill: "BareFenHistoryFill" = BareFenHistoryFill.LC0_BARE_FEN
) -> "np.ndarray":
    """Encode a single FEN into the lc0 112-plane input tensor.

    Returns a ``[112, 8, 8]`` float32 array, side-to-move oriented, with the
    history planes filled per ``history_fill`` (default: replicate lc0's
    bare-FEN behavior; callers pass ``history_fill_for(re_baseline)``).

    TODO(Phase 2.2): implement via lczerolens's ``LczeroBoard`` encoder
    (preferred, report §6) so the plane layout is lc0-faithful by construction:

        from lczerolens import LczeroBoard            # lazy import here
        board = LczeroBoard(fen)
        planes = board.to_input_tensor(...)           # 112-plane lc0 layout
        return _apply_history_fill(planes, history_fill)

    If hand-rolling instead (ONNX path, no lczerolens), build the 6+6 piece
    planes per history slot + 2 repetition planes, then the 8 constant planes,
    and apply the side-to-move flip. T-enc (report §4) is the acceptance test:
    known positions, side-to-move flip, castling/en-passant/repetition planes,
    and the bare-FEN history fill.
    """
    raise NotImplementedError(
        "encode_board_planes is a Phase-2 scaffold; implement via lczerolens "
        "LczeroBoard (preferred) or a hand-rolled 112-plane builder, then gate "
        "with the T-enc encoding test."
    )


def encode_board_planes_batch(
    fens: Sequence[str], *, history_fill: "BareFenHistoryFill" = BareFenHistoryFill.LC0_BARE_FEN
) -> "np.ndarray":
    """Encode a batch of FENs into a ``[B, 112, 8, 8]`` float32 array.

    The batch axis is the whole point (report §1.2): the net wants hundreds to
    thousands of positions per forward pass. This stacks per-FEN encodings into
    one contiguous tensor ready for a single net call. ``history_fill`` is
    forwarded to every per-FEN encoding (default: replicate lc0).

    TODO(Phase 2.2): stack ``encode_board_planes`` results (or vectorize the
    lczerolens encoder over the batch if it exposes a batched path).
    """
    raise NotImplementedError(
        "encode_board_planes_batch is a Phase-2 scaffold; stack per-FEN "
        "encodings once encode_board_planes is implemented."
    )


__all__ = [
    "LC0_NUM_PLANES",
    "LC0_HISTORY_LENGTH",
    "BareFenHistoryFill",
    "history_fill_for",
    "encode_board_planes",
    "encode_board_planes_batch",
]
