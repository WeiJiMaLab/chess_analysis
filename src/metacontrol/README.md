# `src/metacontrol`

Modular **search tree → targets → tensors** pipeline for meta-control research.

| Path | Role |
|------|------|
| `core/` | `SearchTree`, `TreeTensorizer`, engine `providers`, `schemas` |
| `data/` | `TreeSearch` (PUCT), `targets_mc`, `targets_gnn`, Lichess `sampler`, `tree_pack` |
| `scripts/` | `generate_and_profile.py` — one FEN from CSV → `.pt` under `/scratch/.../hl4291/data/trees/` |
| `tests/` | `pytest -m "not integration"` for fast runs; `test_sampler.py` hits DuckDB |

**Docs:** `phase_1_plan.md` (roadmap), `migration.md` (lmcos → here), `numba_speedup_plan.md` (JIT ideas). Cluster example FEN CSV: `/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv`.
