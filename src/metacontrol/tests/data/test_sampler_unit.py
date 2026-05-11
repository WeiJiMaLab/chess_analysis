"""Fast unit tests for :mod:`metacontrol.scripts.sample` (no DuckDB / parquet)."""

from metacontrol.scripts.sample import compose_full_fen


def test_compose_full_fen_basic():
    fen = compose_full_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        player_white=False,
        castling_rights="KQkq",
        en_passant_targets="e3",
        halfmove_clock=0,
        fullmove_number=2,
    )
    assert fen.endswith("0 2")
    assert " b " in fen
    assert "KQkq" in fen
    assert " e3 " in fen


def test_compose_full_fen_empty_castling_ep():
    fen = compose_full_fen("8/8/8/8/8/8/8/8", True, "", "", 0, 1)
    assert fen == "8/8/8/8/8/8/8/8 w - - 0 1"


def test_compose_matches_example_csv_row():
    # Row from ``metacontrol_example.csv`` (fullmove = ply//2 ceiling for black to move)
    fen = compose_full_fen(
        "rnbq1rk1/ppp1npbp/3pp1p1/8/3PP2P/2N1BP2/PPPQ2P1/R3KBNR",
        player_white=False,
        castling_rights="KQ",
        en_passant_targets="",
        halfmove_clock=0,
        fullmove_number=7,
    )
    expected = (
        "rnbq1rk1/ppp1npbp/3pp1p1/8/3PP2P/2N1BP2/PPPQ2P1/R3KBNR b KQ - 0 7"
    )
    assert fen == expected
