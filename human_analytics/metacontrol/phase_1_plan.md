# Metacontrol data generation pipeline — Phase 1 plan

## 1. Current state (completed)

- [x] Modular `core/` + `data/` layout (`TreeSearch`, providers, `targets_mc`, `targets_gnn`, `TreeTensorizer`).
- [x] **PUCT expansion timeline:** `SearchTree.search_expansion_history` records one entry per successful `expand_leaf` (expanded leaf). Meta-control snapshots in `targets_mc` use this timeline so snapshot count stays **O(num expansions)**, not O(number of child nodes created). Legacy hand-built trees still use `expansion_history` (one entry per `add_node`).
- [x] **Root sampling:** `scripts/sample.py` (`ChessSampler`) + DuckDB integration tests (`test_sampler.py`, marked `integration`).
- [x] **Fast sampler tests:** `compose_full_fen` and related checks in `test_sampler_unit.py` (no DB).
- [x] **Single-tree end-to-end export:** `TreeSearch.save` produces `metacontrol_single_tree_v1` dicts saved via `scripts/generate_and_profile.py` to `/scratch/gpfs/GRIFFITHS/hl4291/data/trees/` (default `example_tree_00001.pt`). Example FEN CSV: `/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv`.
- [x] **Profiling:** `cProfile` + `.pstats` from `generate_and_profile.py`; current bottleneck is LC0/UCI process time, not single-tree payload construction.

## 2. Profiling summary (2026-05-11, one FEN)

On a representative row from `metacontrol_example.csv` with `max_nodes=64`, LC0 `nodes=64`, total wall time is **~3–4 s** per tree (well under the 120 s smoke budget).

| Stage | Approx. wall share |
|-------|---------------------|
| Engine UCI (`readline` / `go nodes`) | **~70–75%** |
| `python-chess` board/FEN/outcome | **~10–15%** |
| Payload build/save (`TreeSearch.save` + MC/GNN) | **~5–8%** after timeline fix |
| `torch.save` + teardown | **small** |

## 3. Stage 2.5 — Root selection (done)

Criteria documented earlier (Elo, ply, legal moves, piece count, one FEN per game) are implemented in `scripts/sample.py` via `ChessSampler.sample_positions`.

## 4. Stage 5 — Large-scale production (next)

- [ ] **Raw chunks:** job array over FEN CSV chunks, writing one raw `.pt` tree per root FEN into directories named like `chunk_00012/`.
- [ ] **Packed shards:** collate many raw chunk `.pt` files into fewer training-readable `shard_00000.pt` dataset containers.
- [ ] **Engine workers:** long-lived LC0 processes per worker to remove startup overhead and amortize UCI setup across many FENs.

## 5. Optimization note

Numba/JIT prototypes for MC targets and PUCT traversal were removed: they did not improve end-to-end latency at current tree sizes because engine I/O dominates and array conversion overhead offsets small Python-loop gains. The next useful optimization is worker pooling / long-lived LC0 processes, then re-profile.
