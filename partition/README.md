# partition/ — TEMPORARY train/test split (Part 1A)

**Generated:** 2026-06-21T11:56:00
**Source:** `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees`
**Snapshot tree count:** 640,928 `*.pt` files (one tree per root FEN)
**Split:** 50% train / 50% test,
example-level, seed `0`.
**Train:** 320,464   **Test:** 320,464

> ⚠️ TEMPORARY. `human_trees` is still being populated by ysagiv; this split is
> pinned to the snapshot above. Files added later are unassigned. Re-running
> `make_partition.py` with the same seed reproduces a split over whatever the
> file set is *at that time* — it does NOT preserve assignments as the set grows.
> Regenerate (and bump downstream packs) when the dataset is finalized.

Manifests list `.pt` basenames (the join key the MC + GNN packers consume).
Reproduce: `python partition/make_partition.py --seed 0`.
