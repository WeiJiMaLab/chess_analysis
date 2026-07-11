"""Root FEN format is 4-field (board turn castling ep, no halfmove/fullmove clocks),
matching filtered_moves_minply15_maxply75.fen directly."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--restrict-to", type=Path, default=None,
                        help="optional existing include_list.txt (e.g. exclude_xaba's) -- only "
                             "trees already in this list are checked/kept, producing the intersection")
    parser.add_argument("--table", default="filtered_moves_minply15_maxply75",
                        help="DB table of in-window FENs to check membership against")
    args = parser.parse_args()

    torch.set_num_threads(1)  # matches filter_argmax.py/filter_xaba.py

    from analysis.utils.helpers import CONFIG, connect
    db = CONFIG["selected_db_default"]
    work_dir = os.path.join(os.path.dirname(os.path.abspath(db)), "tmp")
    conn = connect(db, work_dir, threads=int(os.environ.get("DUCKDB_THREADS", 16)),
                   memory_limit=os.environ.get("DUCKDB_MEMORY_LIMIT", "40GB"), read_only=True)
    print(f"[ply-filter] loading distinct FENs from {args.table}", flush=True)
    fens = {r[0] for r in conn.execute(f"SELECT DISTINCT fen FROM {args.table}").fetchall()}
    conn.close()
    print(f"[ply-filter] {len(fens):,} distinct in-window FENs loaded", flush=True)

    restrict = None
    if args.restrict_to:
        restrict = set(args.restrict_to.read_text().split())
        print(f"[ply-filter] restricting to {len(restrict):,} names from {args.restrict_to}", flush=True)

    paths = sorted(args.trees_dir.rglob("*.pt"))
    kept = []
    n_skipped = 0
    n_restricted_out = 0
    for i, path in enumerate(paths):
        if restrict is not None and path.name not in restrict:
            n_restricted_out += 1
            continue
        try:
            spec = RawPretrainExampleRecord.load(str(path)).root_position_spec
        except Exception:
            n_skipped += 1
            continue
        if spec is not None and spec in fens:
            kept.append(path.name)
        if (i + 1) % 20000 == 0:
            print(f"[{i + 1}/{len(paths)}] kept={len(kept)} skipped={n_skipped} "
                 f"restricted_out={n_restricted_out}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + ("\n" if kept else ""))
    checked = len(paths) - n_restricted_out
    print(
        f"DONE: {len(kept)}/{checked} kept of checked ({len(kept) / max(1, checked):.3f}), "
        f"{n_restricted_out} restricted-out, {n_skipped} skipped -> {args.output}"
    )


if __name__ == "__main__":
    main()
