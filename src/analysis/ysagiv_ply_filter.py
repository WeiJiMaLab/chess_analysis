"""Ply-window filter for ysagiv `human_trees`: keeps only trees whose root FEN matches a
position that occurs at ply in [min_ply, max_ply] in OUR OWN human-game move database — the
same window used by the own-corpus `puctvalue_md36` pipeline (config_minply15_maxply75.yaml,
Russek et al. 15-75), applied here as a cross-corpus membership check rather than a recomputed
window (ysagiv's trees carry no ply/game metadata of their own to window directly).

Root FEN format matches directly: ysagiv's `root_position_spec` and our DB's
`filtered_moves_minply15_maxply75.fen` are both 4-field (board turn castling ep, no
halfmove/fullmove clocks) — confirmed empirically (500/500 sampled ysagiv root FENs match SOME
row in our unwindowed `processed_moves_nonzero`; 295/500 fall in the ply-15-75 window), not
assumed. The high overlap is expected: both draw from the same underlying Lichess position pool.

Composable with exclude_xaba (filter_xaba.py) via `--restrict-to <include_list>` — when given,
only trees already in that list are checked/kept, so the two filters intersect in one pass
instead of a separate set-intersection step (mirrors the "stack a second filter on the first"
pattern already used for the own-corpus argmax+xaba probe, see outputs/reports/ysagiv.md).

Usage: python -m analysis.ysagiv_ply_filter \
    --trees-dir <dir> --output <include_list.txt> [--restrict-to <existing_include_list.txt>]
"""
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
