# `human_analytics/metacontrol`

Modular **search tree → targets → tensors** pipeline for meta-control research.

| Path | Role |
|------|------|
| `core/` | `SearchTree`, `TreeTensorizer`, engine `providers`, `schemas` |
| `data/` | `TreeSearch` (PUCT), `TreeSearch.save`, `targets_mc`, `targets_gnn` |
| `scripts/` | `sample.py` (root FEN CSV/parquet sampling), `generate_and_profile.py` (one FEN from CSV → `.pt`) |
| `tests/` | `pytest -m "not integration"` for fast runs; `test_sampler.py` hits DuckDB |

**Docs:** `phase_1_plan.md` (roadmap) and `migration.md` (lmcos → here). Cluster example FEN CSV: `/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv`.

## Root Sampling

`python -m metacontrol.scripts.sample` is the root-FEN acquisition step. By default it walks each day of `--year 2023`, samples `--n 100` positions per day using deterministic `ORDER BY hash(...)`, shows a `tqdm` progress bar, and writes one combined parquet file under `/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol/positions/`. For a non-leap year with enough eligible games, that is **36,500 rows**.

## Single FEN → `tree.pt` latency (profiled)

End-to-end path: read first row of a FEN CSV → `TreeSearch.generate` (LC0-backed PUCT) → `TreeSearch.save` to a `.pt` file (`metacontrol_single_tree_v1`).

Run `python -m metacontrol.scripts.generate_and_profile` from `human_analytics/` with `PYTHONPATH` set to `.../chess_analysis/human_analytics`. **Defaults** in that script: LC0, `nodes=64` per expansion, `max_nodes=64` PUCT budget, `max_depth=10`, `c_puct=1.25`, example CSV and output paths under `/scratch/gpfs/GRIFFITHS/...`.

**Representative wall time (2026-05-11, one row from `metacontrol_example.csv`, hardware as on cluster login/compute node):**

| Stage | Approx. time | Notes |
|--------|----------------|--------|
| Tree growth (`TreeSearch.generate`) | **~2.9–3.4 s** | dominated by UCI / waiting on LC0 |
| Build payload + `torch.save` | **~0.2–0.4 s** | `TreeSearch.save` + MC/GNN targets + disk |
| **Total** | **~3.3–3.8 s** | same run order as printed by the script: `grow=… save=… total=…` |

On that run the grown tree had on the order of **~2.5k nodes**, **~2.5k edges**, **65 snapshots** for **64** expansions (defaults above). Absolute seconds vary with load, I/O, and engine; the **shape** of the profile is stable: **`cProfile` shows most cumulative time in UCI I/O (`readline` / `go nodes` round-trips), then `python-chess` (FEN / `push_uci` / terminal checks), then payload construction/saving (`TreeSearch.save`) at a minority of wall time.

**Optimization note (recorded):** Numba-style JIT was explored for MC target loops and for a parallel PUCT implementation; it did **not** improve end-to-end latency in practice because payload construction and PUCT traversals are small versus engine time, and array conversion added overhead on typical tree sizes. The codebase stays **pure Python** for `TreeSearch` and targets; the actionable lever remains **amortizing LC0** (long-lived workers, batching FENs).

## Raw Chunks vs Packed Shards

The legacy `lmcos` data layout uses two different grouping concepts:

- **Raw generation chunks:** a Slurm job owns a range of root FENs and writes many one-tree `.pt` files into a directory. Prefer naming these directories `chunk_00012/` (or `raw_chunk_00012/`) because they are just generation/job chunks. Each raw `.pt` corresponds to one root FEN / one generated search tree.
- **Packed training shards:** a later packing step collates many raw tree `.pt` files into fewer `shard_00000.pt` dataset containers. These packed shards are what training loaders should read efficiently.

So the intended flow is:

```text
FEN CSV range -> generated_trees/chunk_00012/*.pt -> packed/train/shard_00000.pt -> DataLoader minibatches
```
