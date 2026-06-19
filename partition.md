# partition.md — Part 1: train/test split + eval-plot rerun

**Status:** plan draft. Awaiting line-by-line answers on the open questions (marked
**[Q#]**) before execution. Parts 3 ([[eval]]) and 4 ([[gnn_train]]) consume the
partition produced here, so 1A is on the critical path.

Related: [[tree-set-and-gss]] (canonical tree set = `human_trees`, join by FEN),
[[fens-unification-plan]], [[diagnose_gain]], [[eval]], [[gnn_train]].

---

## 1A. The partition

**Decision (from interview):** split unit is the **ysagiv tree files / FENs** —
the flat `human_trees/*.pt` set, ~182K and still growing as ysagiv runs — at a
**50/50 train/test** ratio.

### What we build
```
partition/
  README.md          # provenance: snapshot timestamp, tree count, seed, "TEMPORARY"
  train/
    manifest.txt     # newline-delimited .pt basenames (or FENs) in train
  test/
    manifest.txt     # newline-delimited .pt basenames (or FENs) in test
```
- We **record**, not copy — manifests list which files belong to each side. The
  source `human_trees/` folder is left untouched (matches "no need to edit these
  folders").
- **Temporariness is recorded explicitly** in `partition/README.md`: this split is
  pinned to the *current* ysagiv snapshot (file count + a timestamp). As ysagiv
  keeps writing trees, files not present at snapshot time are simply unassigned;
  re-running the split later supersedes this one. We never silently re-shuffle.

### How
- Reuse / adapt [split.py](lmcos/src/data/preprocess_gnn/split.py), which already
  shuffles per-tree `.pt` files with a fixed seed and writes train/val manifests.
  Changes needed: `validation_fraction = 0.5` (→ rename to test), `source_root =
  human_trees`, `split_root = ./partition`, emit the `train/` & `test/` subfolders
  + README. The split is **example-level** (every FEN independently assigned),
  which matches the chosen unit.
- **Seed pinned** (default 0) and written into the README so the split is exactly
  reproducible from the manifest + seed + snapshot count.

### Resolved + status
- **DONE.** Built by [make_partition.py](partition/make_partition.py) over
  `human_trees` (**405,659** `.pt` at the 2026-06-19 snapshot, more than 2× the
  181,992 in [[tree_gen]]) → **202,830 train / 202,829 test**, disjoint, union ==
  total. Manifests = `.pt` basenames; provenance + TEMPORARY warning in
  `partition/README.md`.
- **[Q1] RESOLVED — `human_trees`.** The literal `generated_trees` (the
  `split.py`/packer default) does **not exist**; the only live, large set is
  `human_trees` (405K, updated today). The historical `generated_trees_*` dirs are
  empty/stale (`_combined` = 0 files; `_oracle96_trace_filtered` ≈ 40K, Apr).
  **Flow (per user):** `human_trees` → **pack into
  `/scratch/gpfs/GRIFFITHS/hl4291/packed/`** → parts 3 & 4 read from there. We do
  NOT use or create `generated_trees`.

### Remaining open questions
- **[Q2] FEN-level dedup / leakage.** You chose to split by FEN directly. A FEN can
  recur across human games; if the same FEN has multiple `.pt` (it shouldn't —
  trees are keyed by FEN — but worth confirming), or if near-duplicate positions
  exist, train/test could leak. **Is plain per-file 50/50 acceptable, or do you
  want me to dedup by the 4-field FEN first?** *Assumption:* one `.pt` per FEN, no
  extra dedup.
- **[Q3] Manifest contents:** `.pt` **basenames** (what the packers consume) vs
  **FENs** (human-readable, join key). *Assumption:* store basenames, with a side
  `fen` column in a small parquet for joins.

---

## 1B. Rerun human + engine eval plots on the fuller dataset

**Decision (from interview):** regenerate **all** human+engine eval figures on
**all currently-available trees, no n cap** (today the analyses sample
`n_trees=200000`, which is the cache key `vals_human_trees_200000_7.parquet`; the
total is ~182K so 200K already covers the present set, but ysagiv keeps adding —
we lift the cap so the rerun tracks the live count).

### Figures in scope (from `git show c903646 --stat` + `figures/`)
- `gain_vs_rt.png`, `mq_vs_rt.png` (1×3: global | ply tertile | GSS stratum),
  `gss_vs_rt.png`, `actiongap_vs_rt.png`, `correlation_matrix.png`,
  `board_feature_corr.png`, `legal_moves_vs_movetime.png`.
- Drivers: [tree_values_analysis.py](human_analytics/tree_values_analysis.py),
  [movetime_analysis.py](human_analytics/movetime_analysis.py),
  [mq_vs_rt_by_gss.py](human_analytics/mq_vs_rt_by_gss.py).

### Plan
1. Bump the sampling cap above the live tree count (or add an "all trees" mode) in
   `compute_values` and the rootmoves/p1prime caches. Caches are keyed
   `{n_trees}_{seed}` so a new cap writes fresh parquet files without clobbering
   the 200K ones.
2. Re-run each driver; confirm the **Gain fix holds at the larger n** (no ≈1.0
   spike) — this dovetails with [[diagnose_gain]].
3. Diff the regenerated figures against committed ones; report any qualitative
   change (correlation signs, monotonicity, the MQ↔RT partial).

### Open questions
- **[Q4] "No cap" mechanics:** set `n_trees` to a large sentinel (e.g. 10_000_000)
  so it always exceeds the live count, or add an explicit `--all` flag that scans
  the whole dir? *Assumption:* large sentinel — minimal code change, deterministic
  cache key. Large sentinel is fine
- **[Q5] Eligible.** This is the part explicitly flagged "can be handled by a
  subagent." Confirm you want me to dispatch 1B to a subagent once 1A's manifest
  exists (the figures don't strictly need the partition, but pinning to the same
  snapshot keeps every number in today's docs consistent).
  This is correct. 
