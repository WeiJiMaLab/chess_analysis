"""Tests that ``run_epoch`` returns the per-batch loss trace, not just the mean."""
from __future__ import annotations

import csv

from cts.train.e2e_controller_train import _write_within_epoch_trace


def test_write_within_epoch_trace_writes_csv_with_expected_rows(tmp_path):
    trace = [
        {"epoch": 1, "phase": "train", "batch_index": 1, "loss": 0.5},
        {"epoch": 1, "phase": "train", "batch_index": 2, "loss": 0.4},
        {"epoch": 1, "phase": "validation", "batch_index": 1, "loss": 0.45},
    ]
    out_path = tmp_path / "e2e_step1"
    csv_path = _write_within_epoch_trace(trace, out_path)
    assert csv_path == str(tmp_path / "e2e_step1_within_epoch_trace.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["phase"] == "train"
    assert float(rows[1]["loss"]) == 0.4


def test_write_within_epoch_trace_also_writes_plot(tmp_path):
    trace = [{"epoch": 1, "phase": "train", "batch_index": i, "loss": 1.0 / i} for i in range(1, 6)]
    trace += [{"epoch": 1, "phase": "validation", "batch_index": 1, "loss": 0.2}]
    out_path = tmp_path / "e2e_step2"
    _write_within_epoch_trace(trace, out_path)
    assert (tmp_path / "png" / "e2e_step2_within_epoch_trace.png").is_file()
    assert (tmp_path / "pdf" / "e2e_step2_within_epoch_trace.pdf").is_file()


def test_write_within_epoch_trace_empty_is_a_clean_noop(tmp_path):
    out_path = tmp_path / "e2e_step_empty"
    result = _write_within_epoch_trace([], out_path)
    assert result == ""
    assert not (tmp_path / "e2e_step_empty_within_epoch_trace.csv").exists()
