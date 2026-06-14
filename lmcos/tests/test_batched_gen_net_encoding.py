"""T-enc: lc0 112-plane board encoding (report §4).

``encode_board_planes(fen)`` must return the lc0 input tensor: a
``[112, 8, 8]`` float array, side-to-move oriented, with the history planes
filled per the chosen convention (default ``LC0_BARE_FEN``). These are light
sanity checks (shape / dtype / finiteness / side-to-move orientation), not a
byte-level plane audit — the heavier plane-layout parity comes from delegating
to lczerolens's lc0-faithful encoder (report §6), so this guards on
``pytest.importorskip("lczerolens")`` and skips while the encoder is scaffolded.

Side-to-move orientation check: lc0 flips the board so the side to move is
always "us" at the bottom. A position and its exact color-flip (mirror ranks +
swap colors + flip side-to-move) therefore present the SAME own/opponent piece
geometry to the net, so their piece-plane stacks should match — while a position
and its same-color vertical mirror should NOT. We assert the flip-invariance
direction (robust) and only treat the discriminating direction as informative.
"""

from __future__ import annotations

import pytest

from cts.data.batched_gen.encoding import (
    LC0_HISTORY_LENGTH,
    LC0_NUM_PLANES,
    BareFenHistoryFill,
    encode_board_planes,
    history_fill_for,
)

# A simple, asymmetric position so a color-flip is meaningfully different.
WHITE_TO_MOVE_FEN = "8/8/8/3k4/8/3K4/4P3/8 w - - 0 1"
# Same geometry mirrored top<->bottom with colors swapped and black to move:
# the side-to-move-relative view should be identical to WHITE_TO_MOVE_FEN.
BLACK_TO_MOVE_COLORFLIP_FEN = "8/4p3/3k4/8/3K4/8/8/8 b - - 0 1"


def _encode_or_skip(fen, **kwargs):
    """Import-or-skip lczerolens, then encode; skip while encoder is scaffolded."""
    pytest.importorskip("lczerolens")
    try:
        return encode_board_planes(fen, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"encode_board_planes not implemented yet: {exc}")


def test_history_fill_constants_are_consistent() -> None:
    """Pure (no skip): the fill-mode mapping agrees with the module constants.

    This piece needs no lczerolens, so it always runs: the default
    ``encode_board_planes`` fill is ``LC0_BARE_FEN`` and matches
    ``history_fill_for(re_baseline=False)``.
    """
    assert LC0_NUM_PLANES == 112
    assert LC0_HISTORY_LENGTH == 8
    assert history_fill_for(False) is BareFenHistoryFill.LC0_BARE_FEN
    assert history_fill_for(True) is BareFenHistoryFill.REPEAT


def test_encode_shape_is_112_8_8() -> None:
    """encode_board_planes returns a [112, 8, 8] tensor."""
    planes = _encode_or_skip(WHITE_TO_MOVE_FEN)
    assert tuple(planes.shape) == (LC0_NUM_PLANES, 8, 8), planes.shape


def test_encode_is_float_and_finite() -> None:
    """All plane values are finite floats."""
    import numpy as np

    planes = _encode_or_skip(WHITE_TO_MOVE_FEN)
    arr = np.asarray(planes)
    assert np.issubdtype(arr.dtype, np.floating), arr.dtype
    assert np.all(np.isfinite(arr)), "encoding must be finite"


def test_encode_explicit_history_fill_kwarg_accepted() -> None:
    """The ``history_fill`` kwarg is honored and still yields a [112,8,8] tensor."""
    planes = _encode_or_skip(
        WHITE_TO_MOVE_FEN, history_fill=BareFenHistoryFill.LC0_BARE_FEN
    )
    assert tuple(planes.shape) == (LC0_NUM_PLANES, 8, 8)


def test_encode_side_to_move_orientation_flip_invariance() -> None:
    """A position and its color+rank flip present the same side-to-move geometry.

    lc0 orients planes from the mover's perspective, so the current position's
    own/opponent piece planes (the first 12 of the most-recent history slot)
    should coincide between a white-to-move position and its black-to-move exact
    color-flip. If the encoder instead emitted absolute (white-at-bottom)
    geometry, these would differ.
    """
    import numpy as np

    a = np.asarray(_encode_or_skip(WHITE_TO_MOVE_FEN))
    b = np.asarray(_encode_or_skip(BLACK_TO_MOVE_COLORFLIP_FEN))
    # The 12 current-position piece planes (6 own + 6 opponent) are the leading
    # planes of the most-recent history slot in lc0's layout.
    a_pieces = a[:12]
    b_pieces = b[:12]
    assert a_pieces.shape == b_pieces.shape
    # Side-to-move-relative encodings of a position and its color-flip match.
    assert np.allclose(a_pieces, b_pieces, atol=1e-6), (
        "side-to-move-relative piece planes should match across an exact "
        "color+rank flip; encoder may not be orienting to the mover"
    )


def test_encode_distinct_positions_differ() -> None:
    """Sanity: two genuinely different positions encode to different tensors."""
    import numpy as np

    a = np.asarray(_encode_or_skip(WHITE_TO_MOVE_FEN))
    other = np.asarray(_encode_or_skip("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    assert a.shape == other.shape
    assert not np.allclose(a, other), "different positions must encode differently"
