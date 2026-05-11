# Numba / native acceleration plan (data generation)

Profiling `generate_and_profile.py` on one real FEN (`max_nodes=64`, LC0 `nodes=64`) shows where time actually goes. This document estimates where **Numba** (or other native speedups) could help and where they cannot.

## 1. Measured bottlenecks (single tree, ~3.5 s wall)

| Share (approx.) | Component | Notes |
|-----------------|-----------|--------|
| **~70–75%** | UCI I/O | `readline` inside `_wait_for` / `_get_analysis` while LC0 runs `go nodes …` |
| **~10–15%** | `python-chess` | `Board.fen`, `push_uci`, `is_game_over` / `outcome` in `StockfishExpansionProvider` and LC0 child handling |
| **~5–8%** | Target packing | `pack_single_tree_like_legacy_shard` → `compute_halt_rewards` / `derive_snapshots` / `compute_gnn_targets` (after PUCT timeline fix, snapshot count ≈ `num_expansions + 1`, not per-child) |
| **Remainder** | Tensor assembly, `torch.save`, process teardown | Small |

So **the engine subprocess boundary dominates**. Numba cannot accelerate `readline` or LC0 itself.

## 2. Where Numba *might* help (incremental)

1. **`targets_mc.compute_halt_rewards` / prefix DP**  
   If rewritten as **dense NumPy arrays** over a fixed-size support (e.g. cap nodes or snapshot index), the inner loops over `present_ids` / edge filtering could be `@njit`’d. **Expected gain:** modest (currently sub-second after timeline fix); worthwhile only at **massive batch** sizes or if snapshots are recomputed many times per tree.

2. **`targets_gnn` edge aggregation**  
   Visit-weighted means over edges are already simple Python; converting to packed COO + Numba reduction could shave a little **if** called in a tight loop over millions of trees. **Expected gain:** small vs engine.

3. **Custom bitboard FEN / legality** (not Numba-only)  
   Replacing hot `python-chess` paths with a **C extension / Rust / shakmaty-syzygy-style** board would help the ~10–15% slice. That is a larger engineering change than sprinkling Numba.

## 3. Higher ROI than Numba (recommended first)

- **Amortize engine startup:** reuse one `Popen` across many FENs in a worker process.
- **Batch or pipelining:** multiple workers × one engine each on the cluster.
- **Reduce `Board.fen` calls** if any are redundant (cache FEN per node id where legal).
- **Lower `max_nodes` / `nodes`** for ablations; linear effect on UCI round-trips.

## 4. Suggested milestone

Implement **worker pool + long-lived LC0** and re-profile; only if target packing re-emerges as >20% wall time, prototype Numba on a **numpy-backed** snapshot representation (see `phase_1_plan.md`).
