# Tree-generation engineering — timing, speedups, and the batched-gen NO-GO

**Ref:** `R-TREEGEN` · [Index](README.md) · Thread: LMCOS · Plan: [unify.md](../unify.md) ·
Companion: [`labnotebook.md`](../labnotebook.md)

Engineering record for generating the LMCOS search-tree dataset: pinning per-tree cost across
engines, the two output-preserving quick wins that were investigated, and the parity-gated
in-process batched evaluator that was built, benchmarked, and **abandoned**. This is an
**engineering** report (procedure / decisions / logs), not a scientific-template report.

**Bottom line.** Faithful lc0-UCI `build_tree` at ≈16.87 s/tree (A100, budget 96). One quick win
shipped (the repetition-scan guard, ~1.5× → ~11 s/tree); eval pooling and the batched in-process
evaluator were both built, measured, and rejected (slower / below the ≥5× gate). The 50K/150K runs
use the **unmodified faithful pipeline** with throughput bought via parallelism + the QoS fix.

---

## Part A — Three-engine timing smoke (was R-U1)

Pins the per-tree generation cost, clears the CPU-lane gate, and profiles Stockfish for the
engine-swap decision — prerequisites for the committed 50K reunification run.

### Summary

| | |
|---|---|
| **Description** | Steady-state tree-gen cost on 100 warm human FENs (budget 96, multipv 8, max_depth 4 — *timing-smoke config only; faithful ysagiv production regime is **max_depth 10***), three engine/node configs: Lc0-GPU (A100), Lc0-CPU/blas (pure-CPU node), Stockfish (CPU reference). |
| **Rationale** | The 50K plan is CPU-led (≈3 GPUs effective); must confirm the cost number and that lc0 even runs on CPU-only nodes. |
| **Finding** | **CPU gate: BLOCKED then FIXED.** Lc0-GPU ≈ **16.6 s/tree**; Lc0-CPU ≈ **~738 s/tree** (4 cores) once unblocked; Stockfish ≈ 0.45 s/pos (d12) … 3 s/pos (d16). **50K is feasible CPU-led in ~1 day.** |

### The CPU-lane gate: blocked, then fixed (key operational finding)

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

### Per-tree cost (the pinned numbers)

| Engine / node | Backend | Cores/GPU | s/tree | Source |
|---|---|---|---|---|
| **Lc0-GPU** (A100 80GB) | cuda | 1 GPU | **16.87** (steady mean, trees 5–83) | 9278289 |
| **Lc0-CPU** (pure-CPU node) | blas | 4 cores | **~738** (tree 1; warmup-inclusive) | 9278353 |
| Stockfish 14 (reference) | — | 4 cores | 0.45 (d12) · 2.93 (d16) · 11.65 (d20) · **0.026 (100k nodes)** s/pos | sf_probe |

Lc0-GPU steady-state settled at `roots_per_s ≈ 0.06` from tree ~5 onward (warmup ≈ tree 1–4).

### Feasibility recompute (committed 50K)

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

**Verdict: 50K feasible in ~1 day, CPU-led.** GPU-only would be ~3 days.

### Sharding recipe (the fix for the empty `human_trees_10k/`)

`submit_generate_dataset_shards.py` takes `--shard-size` and `--time` as **independent** flags;
the empty-dir failure was `--shard-size 500 --time 01:00:00` at ≥16 s/tree (needs 2–6 h). **Guard:
`shard_FENs × s_per_tree × 1.3 < wall`.** Split 50K by lane throughput:

- **GPU lane** (650/2357 ≈ 28% → ~14K): **3 shards** (one per GPU), `gpu-medium` (3-day wall),
  ~4,700 FENs/shard × 16.6 s ≈ **21.7 h/shard** ✅.
- **CPU lane** (~36K): array on the **`cpu` partition** (15-day wall) with the venv-lib
  `LD_LIBRARY_PATH`, ~350 tasks × ~103 FENs × 738 s ≈ **21.1 h/task** ✅ (or smaller tasks that
  cycle if using `short`).

### Engine-swap profiling (decision 2)

Stockfish is **~60× faster at strong depth-20** and **~28,000× faster at a 100k-node budget**
than Lc0-CPU (0.026–11.65 s vs ~738 s/tree). Given the ~3-GPU / CPU-rich reality, an SF-based
tree-gen would be dramatically cheaper on CPU. **But** SF is α-β, not NN-MCTS: it does **not**
produce the value-network WDL signal the GNN child-WDL supervision consumes, and `build_tree` is
Lc0-only (a provider is a real build). **Recommendation: do NOT swap now** — the libcublas fix
already unblocks the CPU lane at acceptable cost (~1 day for 50K). Hold SF as the contingency if
the CPU lane is later constrained.

### Part A artifacts

- Configs / FENs: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/smoke_u1/`
- Logs: `lmcos/slurm/logs/u1smoke-{gpu,cpu2}_*.{out,err}`; SF: `smoke_u1/sf_probe.out`
- Job IDs: GPU 9278289 · CPU(fixed) 9278353 · CPU(gate-fail) 9278287

---

## Part B — lc0-UCI speedup investigation (was R-U1-SPEED)

Investigation of whether the 16.87 s/tree cost can be cut **without changing the generated trees**.
Result: one quick win shipped (repetition guard), pooling rejected, true batching deferred.

### Where the time goes (cProfile, 2 trees, 34.9 s)

| Cost | Share | What |
|---|---|---|
| `readline` (blocked on lc0) | **65%** | ~860 separate `go nodes 1` UCI calls per tree |
| `can_claim_threefold_repetition` | **23%** | `board.outcome(claim_draw=True)` history scan, per child |
| python-chess board ops + PUCT | ~12% | `push`/`pop`/movegen, `_select_leaf_by_puct` |

**lc0 is not the bottleneck.** Native `go nodes 96` (internal batched search) = ~0.1 s; a single
`go nodes 1` = ~1 ms cached / ~10 ms on a distinct position. The cost is the *number of sequential
batch-1 calls* our Python MCTS makes, which defeats lc0's internal GPU batching.

### The generation algorithm

`teacher_targets.py` builds each tree with a **strictly sequential** PUCT loop (lines 1646–1683):

```
while num_expansions < budget and frontier_expandable:
    node_id, path = _select_leaf_by_puct(tree, edge_stats)   # uses ALL prior rollouts' stats
    if terminal or depth >= max_depth: backprop; continue
    raw_children  = provider.expand_node(node.fen, node.depth)   # <-- the engine work
    tree.add_children(node_id, raw_children)
    backpropagate_path(path, leaf_value, leaf_wdl)
```

and `expand_node` (`lc0.py`) is:

```
prior = prior_engine.analyse(fen)                 # 1 call: children + policy priors (multipv 8)
for child in prior.children:                      # C children (~8)
    if terminal(child): ...                        # board.outcome(claim_draw=True)  <-- repetition cost
    else: child.wdl = value_engine.analyse(child)  # 1 call PER CHILD  <-- the for-loop
```

### Eval pooling — built, validated, REJECTED (2026-06-05)

> Pooling **is NOT** "run many rollouts simultaneously." Parallel rollouts require *virtual loss*,
> which changes the trajectory → not bit-identical. We reject that.

Pooling parallelizes only the inner `for child in children` loop (the C independent child
value-evals within one expansion) across K lc0 *value* engines via `ThreadPoolExecutor` (GIL
released during blocking UCI `readline`).

**Concurrent-eval benchmark (one A100, pure-eval workload):**

| K engines | speedup | bit-mismatches |
|---|---|---|
| 2 | 1.99× | 0/250 |
| 4 | 2.46× | 0/250 |
| 8 | 3.07× | 0/250 |

This benchmark (and the 2.5–2.9× projected overall) was on lc0 **classic `go nodes 1`** mode
(latency-bound, parallelizable). **The real value engine runs `valuehead` mode** — a pure
GPU-compute-bound NN forward pass. The bit-identical gate (8 FENs, `value_pool_size=1` vs `=6`)
**failed on both axes:**

- **Slower, not faster:** per-tree (startup-excluded) **172 s pooled vs 133 s sequential (~30%
  slower)** — concurrent processes serialize on the GPU and only add coordination overhead. *The
  wrong engine mode was benchmarked.*
- **Not bit-identical:** pooled trees diverged structurally (node-feature deltas ~0.5, different
  edge counts).
- **lc0/CUDA itself IS deterministic:** two identical sequential runs matched on every field except
  a `NaN≠NaN` artifact in `value_gap`. Byte-identity is achievable; production is reproducible.

**Pooling was fully reverted** (`lc0.py`, `build_tree.py` to HEAD; tests pass). The bit-identical
gate stopped a slower, corrupting "optimization" from shipping.

### Why not true batching? (route (a) prototype)

- **(a) lc0 internal batching** (`go nodes N` + `VerboseMoveStats`): at `nodes ≥ 256` lc0 emits a
  scalar **`V` (= W−L)** for all 32 children in ~37 ms (vs ~141 ms sequential, ~3.8×) — lc0 *does*
  batch internally. **But it only gives scalar `V`.** The per-move `WL`/`D` (the W/D/L
  distribution) are populated only for visited children and are the **backed-up subtree average**,
  not the raw per-child NN WDL. The pipeline stores the **full normalized W/D/L triple** per child
  as the GNN child-WDL target (`edge_wdl_targets`), which **cannot be reconstructed from scalar
  `V`**. → Route (a) cannot faithfully produce the targets.
- **(b) run the net directly** (PyTorch/ONNX): gives batched WDL but requires replicating lc0's
  exact 112-plane encoding + WDL head bit-for-bit. High silent-corruption risk → logged as a future
  deep optimization (and pursued in Part C).

### Win that SHIPPED — skip the impossible threefold-repetition scan

`terminal_value_from_board` (`common.py:116`) called `board.outcome(claim_draw=True)`;
`claim_draw=True` runs `can_claim_threefold_repetition`, replaying the move stack — **23% of
tree-gen time**. A 3-fold repetition needs ≥ 9 plies, but boards here are a FEN plus ≤ `max_depth`
pushes with no prior history, so it is **impossible while shallow**. The fifty-move rule still
matters but is a cheap scalar (FEN halfmove clock).

**Fix (provably equivalent):** `board.outcome(claim_draw=False)` **plus** an explicit
`if board.halfmove_clock >= 100: draw`. **Gated on actual history length, not the `max_depth`
hyperparameter** (so it stays correct if `max_depth` is raised — deep boards fall back to the exact
check). Measured **~1.5× end-to-end** (16.87 → **11.1 s/tree**, A100, budget 96). Under the real
**max_depth=10** regime most boards are still shallow (realized depth ~6) so the fast path
dominates. Equivalent: 0 mismatches over 600+ boards incl. fifty-move boundary, deep fallback, and
a real 7-ply threefold; 22.6× faster check; tests pass.

### Part B conclusion

There is **no faithful quick win** for the valuehead pipeline beyond the repetition guard. The only
real lever is **route (b)** — extract the net and batch the child WDLs — pursued and abandoned in
Part C. Artifacts: `smoke_u1/parallel_eval_bench2.py`, `instrument_breakdown.py`, `profile_lc0.py`,
`profile_build_tree.py`.

---

## Part C — Batched in-process tree generation: NO-GO (was R-BATCHGEN)

> ### ⚠️ CORRECTION (2026-06-14, late) — the "1-ply valuehead" premise is FALSE.
> Throughout the build the ysagiv node `value` was described as a "1-ply best-child valuehead
> minimax." **That is wrong.** Verified over 55k reference nodes: `value == win − loss` exactly
> (the **raw lc0 value-head**); root WDL ≠ best-child WDL. The "1-ply" claim came from a
> startpos-only measurement (startpos is history-special-cased). The `re_baseline=False` path this
> report calls "faithful" computed a target ysagiv does **not** use; ysagiv's real target is the
> `re_baseline=True` raw head. This does **not** change the NO-GO decision (raw path was still only
> ~2×, below the ≥5× gate), but every "faithful 1-ply" / "valuehead minimax" phrase below is
> **defunct** — read "raw value-head" instead.

> ### ❌ NO-GO (2026-06-14) — investigated, built, measured, abandoned.
> The in-process batched evaluator was built and validated for correctness, then benchmarked and
> scrapped: the speedup doesn't justify the build.
>
> **Measured (A100, budget-16, extrapolated to budget-96):**
> | Mode | s/tree @16 | ~s/tree @96 | vs lc0-UCI 16.87 |
> |---|---|---|---|
> | `re_baseline=True` (raw value head) | 1.35 | ~8 | **~2×** |
> | `re_baseline=False` (faithful 1-ply valuehead) | 20.81 | ~125 | **~7× *slower*** |
>
> **Why:** the bottleneck is **Python-side 112-plane encoding (`to_input_tensor`) + board
> bookkeeping, not the GPU** (the A100 sits idle). The faithful 1-ply path encodes every node's
> *grandchildren* (~branching² per node). Even the raw path reached only ~2×, below the ≥5× gate.
>
> **Decision (hl4291):** "juice isn't worth the squeeze." Resume tree-gen on the **lc0-UCI path +
> the QoS fix** (run on `gpu-short`, not `gpu-test`) — throughput via parallelism, zero new code.
>
> **What was kept:** `cts.data.process_fens` (FEN-source consolidation). **What was removed:** the
> `batched_gen` module, its tests, and the `smoke/` scripts (git `2f30bbc`, `3a2bf7d`, reverted
> `2f30bbc`/`3a2bf7d`). The plan/record below is preserved for posterity; the code it references no
> longer exists in the tree.

### The intuition (why batching across trees works)

The controller is an **optimal-stopping agent over a growing tree**; its training example is the
**ordered sequence of partial trees** `T_0 → … → T_K` from one PUCT search. The expansion order is
the time axis of the stopping problem; the per-step root statistics (`oracle_root_q_trace`) evolve
online and **cannot be reconstructed from the final tree**. A single tree's PUCT is irreducibly
sequential, but there are **150K independent trees** — *that independence is the batch dimension*.

**Frontier-batching loop:** hold N trees in flight; each outer step = one expansion per tree.
(1) **Select** one leaf per tree (CPU, no NN). (2) **Gather** their children into one tensor.
(3) **One forward pass** returns policy/value/WDL for the batch. (4) **Scatter + advance** per tree
on CPU (expand, backprop, record trace). Continuous batching keeps the batch full. Because each
tree reads only its own backed-up stats, every trajectory is identical to a sequential run — no
virtual loss, no order change (the trap we explicitly avoid).

### Layered parity contract

| Layer | What | Bar |
|---|---|---|
| **L1 Search-logic** | PUCT/expansion/backprop/trace/oracle DP given a fixed eval oracle | **Bit-exact** vs sequential loop (load-bearing) |
| **L2 Evaluator** | our net vs lc0 value/WDL/policy | max abs diff < tol; argmax identical |
| **L3 Trajectory** | end-to-end order + best-move trace vs lc0-UCI | fraction identical ≈ 1.0 |
| **L4 Targets** | `edge_wdl_targets`, node values, `target_advantages` | within tol |

**Key move:** prove **L1 bit-exact first** by replaying a *fixed* evaluator (cached lc0 outputs or
deterministic mock) through both loops. This isolates "did we break the search" from "did we change
the numerics." A single switch `re_baseline: bool = False` on `NetEvaluator` governs the two lc0
quirks (text-rounded priors + bare-FEN history fill); `re_baseline=True` is the clean raw-head path.

### Pinned lc0 behaviors (confirmed against the binary)

1. ~~Value is a 1-ply minimax~~ — **DISPROVEN (see CORRECTION):** value is the **raw value-head**
   `win − loss`. The startpos measurement that suggested "1-ply" was a history special-case.
2. **Priors carry `PolicyTemperature` 1.359** — `P = softmax(logits / 1.359)` over legal moves
   (then 1e-4 text rounding). Reproduces lc0's reported P to ~1e-4 (d2d4 0.1806 vs 0.1804).
3. **History is `fen_only`** (lc0 `HistoryFill` default) == lczerolens
   `INPUT_CLASSICAL_112_PLANE_REPEATED`.

### Three real bugs the pinning test caught (each would have silently corrupted 150K)

1. `LczeroBoard.encode_move(move, us)` needs the side-to-move — policy was misindexed for Black.
2. The onnx2torch model **must be `.eval()`** — else BatchNorm uses per-batch statistics → a
   position's value depends on which others share the batch (non-deterministic).
3. The search threads **position specs** (`<fen> ||moves|| m1 …`); the net must replay the moves
   (`split_position_spec`), not treat the string as a bare FEN.

### JAX assessment (deferred to v2)

`mctx` is a JAX-native batched MCTS, but our trees are **ragged and dynamically grown** and our
targets/trace code is written against that form. Porting to fixed-size padded arrays is a
substantial rewrite **and a parity risk** (re-deriving the whole L1 contract). PyTorch delivers the
eval-batch speedup with far less risk and stays in the existing framework. Treat JAX/mctx as v2,
justified only if CPU-side bookkeeping becomes the bottleneck after eval is batched.

### Iteration guidance (learned the hard way 2026-06-14)

- **Default to cuda** for net work — the A100 is on the node; never run a CPU arm for a quick check.
- **The ONNX reload (~6–20s) is paid on every `NetEvaluator()`** — cache the loaded model
  (session-scoped fixture / one long-lived evaluator). Biggest test-suite speedup (a full net-test
  pass was ~8 min, almost all reloads). `pytest -n` (xdist) does **not** help — each worker reloads.
- Use tiny configs (budget 8–16, 2–4 trees) to iterate; budget-96 only for the headline number.
- Generous timeouts + flushed stdout; redirect long pytest to a file (don't `| tail`).
- The genuine game-start position is special-cased by lc0 (true empty history) → won't match
  REPEATED, but irrelevant since data is ply 15–75.

---

## Outcome and current production path

- **Generator:** faithful lc0-UCI `build_tree`, **max_depth=10**, value=raw valuehead, budget=96,
  with the repetition guard (~11 s/tree on compute nodes, realized depth ~6).
- **edge_wdl_targets** were silently dropped at one point → **FIXED** (`build_tree.py`
  `include_edge_wdl_targets=True`; verified finite / sum-to-1 on a compute-node canary).
- **Depth-10 dataset:** run **9745158** `--qos=gpu-short --array=0-149%44` → `lc0_trees/` (150,003
  trees). 50K reunification run: job `9282935`, 50×1000-FEN shards, `gpu-short %20`, `resume:true`.

*This report merges the former `R-U1` (engine timing smoke), `R-U1-SPEED` (lc0-UCI speedup), and
`R-BATCHGEN` (batched in-process tree gen) reports.*
