"""Build the temporary 50/50 train/test partition over the ysagiv human_trees set.

Part 1A of the 2026-06-19 plan (see ../partition.md). The split is **example
level** (each tree file independently assigned), pinned to the CURRENT ysagiv
snapshot — ``human_trees`` is still being populated, so this partition is
TEMPORARY and superseded by any later re-run. Filenames (``NNNNNN_root_N.pt``)
are the join key the downstream packers (MC + GNN) consume; there is one tree
per root FEN, so a per-file split is a per-FEN split.

    python partition/make_partition.py            # default: human_trees, seed 0, 0.5
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
from pathlib import Path

SOURCE_DEFAULT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
OUT_DEFAULT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_DEFAULT, help="dir of *.pt tree files")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="partition/ output root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=0.5)
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    # Deterministic enumeration: sort basenames first, THEN shuffle with the
    # pinned seed, so the split is reproducible from (snapshot file set, seed).
    names = sorted(e.name for e in os.scandir(source) if e.name.endswith(".pt"))
    n_total = len(names)
    rng = random.Random(args.seed)
    rng.shuffle(names)

    n_train = round(args.train_fraction * n_total)
    train, test = sorted(names[:n_train]), sorted(names[n_train:])

    for split, items in (("train", train), ("test", test)):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.txt").write_text("\n".join(items) + "\n")

    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    (out / "README.md").write_text(
        f"""# partition/ — TEMPORARY train/test split (Part 1A)

**Generated:** {stamp}
**Source:** `{source}`
**Snapshot tree count:** {n_total:,} `*.pt` files (one tree per root FEN)
**Split:** {args.train_fraction:.0%} train / {1 - args.train_fraction:.0%} test,
example-level, seed `{args.seed}`.
**Train:** {len(train):,}   **Test:** {len(test):,}

> ⚠️ TEMPORARY. `human_trees` is still being populated by ysagiv; this split is
> pinned to the snapshot above. Files added later are unassigned. Re-running
> `make_partition.py` with the same seed reproduces a split over whatever the
> file set is *at that time* — it does NOT preserve assignments as the set grows.
> Regenerate (and bump downstream packs) when the dataset is finalized.

Manifests list `.pt` basenames (the join key the MC + GNN packers consume).
Reproduce: `python partition/make_partition.py --seed {args.seed}`.
"""
    )
    print(f"partitioned {n_total:,} trees -> train {len(train):,} / test {len(test):,}")
    print(f"wrote {out}/train/manifest.txt, {out}/test/manifest.txt, {out}/README.md")


if __name__ == "__main__":
    main()
