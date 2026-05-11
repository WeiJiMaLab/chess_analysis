# Metacontrol data generation pipeline — Phase 1 plan

## 1. Current state (completed)

- [x] Modular `core/` + `data/` layout (`TreeSearch`, providers, `targets_mc`, `targets_gnn`, `TreeTensorizer`).
- [x] **PUCT expansion timeline:** `SearchTree.search_expansion_history` records one entry per successful `expand_leaf` (expanded leaf). Meta-control snapshots in `targets_mc` use this timeline so snapshot count stays **O(num expansions)**, not O(number of child nodes created). Legacy hand-built trees still use `expansion_history` (one entry per `add_node`).
- [x] **Root sampling:** `data/sampler.py` (`ChessSampler`) + DuckDB integration tests (`test_sampler.py`, marked `integration`).
- [x] **Fast sampler tests:** `compose_full_fen` and related checks in `test_sampler_unit.py` (no DB).
- [x] **Single-tree end-to-end export:** `data/tree_pack.pack_single_tree_like_legacy_shard` and `TreeTensorizer.pack_training_shard` produce `metacontrol_single_tree_v1` dicts saved via `scripts/generate_and_profile.py` to `/scratch/gpfs/GRIFFITHS/hl4291/data/trees/` (default `example_tree_00001.pt`). Example FEN CSV: `/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv`.
- [x] **Profiling:** `cProfile` + `.pstats` from `generate_and_profile.py`; documented bottlenecks in `numba_speedup_plan.md`.

## 2. Profiling summary (2026-05-11, one FEN)

On a representative row from `metacontrol_example.csv` with `max_nodes=64`, LC0 `nodes=64`, total wall time is **~3–4 s** per tree (well under the 120 s smoke budget).

| Stage | Approx. wall share |
|-------|---------------------|
| Engine UCI (`readline` / `go nodes`) | **~70–75%** |
| `python-chess` board/FEN/outcome | **~10–15%** |
| Target pack (`tree_pack` + MC/GNN) | **~5–8%** after timeline fix |
| `torch.save` + teardown | **small** |

## 3. Stage 2.5 — Root selection (done)

Criteria documented earlier (Elo, ply, legal moves, piece count, one FEN per game) are implemented in `ChessSampler.sample_positions`.

## 4. Stage 5 — Large-scale production (next)

- [ ] **Sharding:** append many trees into packed shards compatible with training loaders (either extend `metacontrol_single_tree_v1` into multi-episode tensors or match a chosen legacy layout more closely).
- [ ] **Slurm:** job array over FEN CSV chunks calling a thin CLI wrapper around `TreeSearch` + `tree_pack`.
- [ ] **Engine workers:** long-lived LC0 processes per worker to remove startup overhead (see `numba_speedup_plan.md`).

## 5. Next optimization goal — Numba / native kernels

See **`numba_speedup_plan.md`**: Numba is **not** the first lever (UCI dominates). Milestone: after worker pooling, re-profile; if MC/GNN packing exceeds ~20% wall, consider Numba on a numpy snapshot representation or native board updates.
