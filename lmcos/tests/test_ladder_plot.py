"""Tests for the cross-Elo ladder plotter (stage-9 ladder comparison).

Cover, on synthetic per-rung results JSONs (no models / caches needed):

  - rungs are sorted by Elo regardless of input order;
  - a model with a ``null`` metric (pending MCHalt) is DROPPED from that rung's
    line rather than plotted as 0;
  - the figure file is actually written.

Run from lmcos/: pytest tests/test_ladder_plot.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis._budgeted.ladder_plot import _load_rungs, plot_ladder


def _write(path: Path, mchalt: dict | None) -> None:
    payload = {
        "always": {"regret": 0.45, "stop_acc": 0.30},
        "never": {"regret": 0.60, "stop_acc": 0.10},
        "fraction": {"regret": 0.30, "stop_acc": 0.40},
        "stats": {"regret": 0.28, "stop_acc": 0.42},
        "mchalt": mchalt if mchalt is not None else {"regret": None, "stop_acc": None},
    }
    path.write_text(json.dumps(payload))


def test_rungs_sorted_and_null_dropped(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write(a, {"regret": 0.23, "stop_acc": 0.31})
    _write(b, {"regret": 0.25, "stop_acc": 0.29})
    _write(c, None)  # pending MCHalt at the top rung
    # Deliberately out of Elo order on input.
    elos, table = _load_rungs([(2200, str(c)), (1800, str(a)), (2000, str(b))])

    assert elos == [1800, 2000, 2200]  # sorted
    # MCHalt present only for the two non-null rungs, in Elo order.
    assert [elo for elo, _ in table["mchalt"]["regret"]] == [1800, 2000]
    # A always-present model spans all three rungs.
    assert [elo for elo, _ in table["always"]["regret"]] == [1800, 2000, 2200]


def test_plot_writes_file(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, {"regret": 0.23, "stop_acc": 0.31})
    _write(b, {"regret": 0.25, "stop_acc": 0.29})
    out = plot_ladder([(1800, str(a)), (2000, str(b))], tmp_path / "figs")
    assert out.exists() and out.stat().st_size > 0
