# R-U1 — U1.0 three-engine timing smoke (sprint action #1)

**Ref:** `R-U1` · [Index](README.md) · Plan: [unify.md §U1.0/§5](../unify.md)

Pins the per-tree generation cost, clears the CPU-lane gate, and profiles Stockfish for the
engine-swap decision — the prerequisites for the committed **50K** reunification run.

## Summary (after implementation)

| | |
|---|---|
| **Description** | Steady-state tree-gen cost on 100 warm human FENs (budget 96, multipv 8, max_depth 4 — *timing-smoke config only; faithful ysagiv production regime is **max_depth 10***), three engine/node configs: Lc0-GPU (A100), Lc0-CPU/blas (pure-CPU node), Stockfish (CPU reference). |
| **Rationale** | The 50K plan is CPU-led (≈3 GPUs effective); must confirm the cost number and that lc0 even runs on CPU-only nodes. |
| **Finding** | **CPU gate: BLOCKED then FIXED.** Lc0-GPU ≈ **16.6 s/tree**; Lc0-CPU ≈ **~738 s/tree** (4 cores) once unblocked; Stockfish ≈ 0.45 s/pos (d12) … 3 s/pos (d16). **50K is feasible CPU-led in ~1 day.** |

## Procedure

| Step | Status |
|------|--------|
| 100-FEN warm set (`smoke_u1/smoke_fens_100.txt`, head of human_fens_10k) | ✅ done |
| Lc0-GPU arm (cuda, 100 FENs) | ✅ job 9278289 |
| Lc0-CPU arm on pure-CPU node (blas) | ✅ job 9278287 → **failed (libcublas)** → 9278353 **fixed** |
| Stockfish throughput reference (d12/d16/d20/nodes) | ✅ `smoke_u1/sf_probe.out` |
| Recompute §5 tiers; go/no-go on CPU lane + SF provider | ✅ below |

---

## The CPU-lane gate: blocked, then fixed (key operational finding)

The ysagiv lc0 binary is **CUDA-linked** (`ldd` shows `libcublas.so.12`, `libcublasLt.so.12`,
`libcudart.so.12`), resolving to `/usr/local/cuda-12.8/lib64` — a path that exists **only on GPU
nodes**. On a pure-CPU node the first CPU arm (9278287) died at launch:

```
lc0: error while loading shared libraries: libcublas.so.12: cannot open shared object file
RuntimeError: Unexpected EOF from engine process … return_code=127
```

**Fix (cheap, no rebuild):** the three needed CUDA libs already ship in the project venv
(`/home/hl4291/venv/lib/python3.11/site-packages/nvidia/{cublas,cuda_runtime}/lib`). Pointing
`LD_LIBRARY_PATH` there lets the dynamic loader satisfy the link **without a GPU** (libs load,
blas backend never uses them). Re-run 9278353 on a pure-CPU node (`nvidia-smi` → `no_gpu`)
produced trees normally. **The CPU lane — which the 50K plan depends on — is unblocked.**

> Wire this into every CPU-lane job: prepend the venv `nvidia/*/lib` dirs to `LD_LIBRARY_PATH`.

---

## Per-tree cost (the pinned numbers)

| Engine / node | Backend | Cores/GPU | s/tree | Source |
|---|---|---|---|---|
| **Lc0-GPU** (A100 80GB) | cuda | 1 GPU | **16.87** (steady mean, trees 5–83) | 9278289 |
| **Lc0-CPU** (pure-CPU node) | blas | 4 cores | **~738** (tree 1; warmup-inclusive) | 9278353 |
| Stockfish 14 (reference) | — | 4 cores | 0.45 (d12) · 2.93 (d16) · 11.65 (d20) · **0.026 (100k nodes)** s/pos | sf_probe |

Lc0-GPU steady-state settled at `roots_per_s ≈ 0.06` from tree ~5 onward (warmup ≈ tree 1–4).

---

## Feasibility recompute (committed 50K)

Throughput at pinned numbers (3 GPUs effective; CPU `short`/`cpu` ≈ 1,400 cores → ~350 × 4-core jobs):

| Lane | Concurrency | Trees/hr |
|---|---|---|
| Lc0-GPU | 3 | 3 × 3600/16.6 ≈ **650** |
| Lc0-CPU | 350 × 4-core | 350 × 3600/738 ≈ **1,707** |
| **Combined** | | **≈ 2,357** |

| Tier | Trees | CPU-only | GPU-only (3) | **Combined** |
|---|---|---|---|---|
| 10K | 10K | 5.9 h | 15.4 h | **~4.2 h** |
| **50K** | **50K** | 29 h | 77 h (3.2 d) | **~21 h (<1 day) ✅** |
| 100K | 100K | 59 h | 154 h | **~42 h (~1.8 d)** |

**Verdict: 50K feasible in ~1 day, CPU-led.** Confirms the §5 strategy; GPU-only would be ~3 days.

---

## Sharding recipe (the fix for the empty `human_trees_10k/`)

`submit_generate_dataset_shards.py` takes `--shard-size` and `--time` as **independent** flags;
the empty-dir failure was `--shard-size 500 --time 01:00:00` at ≥16 s/tree (needs 2–6 h). **Guard:
`shard_FENs × s_per_tree × 1.3 < wall`.** Split 50K by lane throughput:

- **GPU lane** (650/2357 ≈ 28% → ~14K): **3 shards** (one per GPU), `gpu-medium` (3-day wall),
  ~4,700 FENs/shard × 16.6 s ≈ **21.7 h/shard** ✅.
- **CPU lane** (~36K): array on the **`cpu` partition** (15-day wall) with the venv-lib
  `LD_LIBRARY_PATH`, ~350 tasks × ~103 FENs × 738 s ≈ **21.1 h/task** ✅ (or smaller tasks that
  cycle if using `short`).

---

## Engine-swap profiling (decision 2)

Stockfish is **~60× faster at strong depth-20** and **~28,000× faster at a 100k-node budget**
than Lc0-CPU (0.026–11.65 s vs ~738 s/tree). Given the ~3-GPU / CPU-rich reality, an SF-based
tree-gen would be dramatically cheaper on CPU. **But** SF is α-β, not NN-MCTS: it does **not**
produce the value-network WDL signal the GNN child-WDL supervision consumes, and `build_tree` is
Lc0-only (a provider is a real build). **Recommendation: do NOT swap now** — the libcublas fix
already unblocks the CPU lane at acceptable cost (~1 day for 50K). Hold SF as the contingency if
the CPU lane is later constrained; the speed gap (and that SF cost is dominated by depth, not
nodes) is logged for that decision.

## Artifacts

- Configs / FENs: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/smoke_u1/`
- Logs: `lmcos/slurm/logs/u1smoke-{gpu,cpu2}_*.{out,err}`; SF: `smoke_u1/sf_probe.out`
- Job IDs: GPU 9278289 · CPU(fixed) 9278353 · CPU(gate-fail) 9278287
