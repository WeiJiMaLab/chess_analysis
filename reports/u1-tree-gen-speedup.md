# R-U1-SPEED — Tree-generation speedup: eval-pooling + repetition guard

**Ref:** `R-U1-SPEED` · [Index](README.md) · Parent: [R-U1](u1-engine-timing-smoke.md) · Plan: [unify.md](../unify.md)

Investigation (per hl4291) of whether the 16.87 s/tree GPU cost can be cut **without changing
the generated trees**. Conclusion: yes — two output-preserving wins, both validated bit-identical.

## Where the time goes (cProfile, 2 trees, 34.9 s)

| Cost | Share | What |
|---|---|---|
| `readline` (blocked on lc0) | **65%** | ~860 separate `go nodes 1` UCI calls per tree |
| `can_claim_threefold_repetition` | **23%** | `board.outcome(claim_draw=True)` history scan, per child |
| python-chess board ops + PUCT | ~12% | `push`/`pop`/movegen, `_select_leaf_by_puct` |

**lc0 is not the bottleneck.** Native `go nodes 96` (internal batched search) = ~0.1 s; a single
`go nodes 1` = ~1 ms cached / ~10 ms on a distinct position. The cost is the *number of sequential
batch-1 calls* our Python MCTS makes, which defeats lc0's internal GPU batching.

## The generation algorithm (so the optimization is unambiguous)

`teacher_targets.py` builds each tree with a **strictly sequential** PUCT loop
(lines 1646–1683):

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

## What the pooling does — and what it does NOT

> **It is NOT "run many rollouts simultaneously."** Parallel rollouts would require *virtual loss*
> to diversify selection, which changes the search trajectory and the resulting tree — **not
> bit-identical** (this is the standard tree-parallel MCTS of the literature; we explicitly reject
> it here).

**Pooling parallelizes only the inner `for child in children` loop** — the **C independent child
value-evals within a single node expansion** — across a pool of K lc0 *value* engines (dispatched
by a `ThreadPoolExecutor`; Python releases the GIL during the blocking UCI `readline`, so the K
engines' GPU evals overlap).

Everything else is untouched and sequential:
- `_select_leaf_by_puct` still runs once per rollout on the **fully-updated** `edge_stats`.
- The prior call stays 1-per-expansion.
- Tree growth, backups, the per-step oracle snapshot trace — identical order, identical values.

### Why it is bit-identical

1. The **rollout sequence is unchanged** — selection still sees every prior rollout's result
   before choosing the next leaf. Pooling never overlaps two rollouts.
2. The C child evaluations are **pure functions**: each is a deterministic batch-1 NN forward pass
   `position → WDL`. Their order and concurrency cannot affect each other or the tree.
3. **Empirically validated:** evaluating 250 distinct positions across K=2/4/8 concurrent engines
   vs sequentially gave **0/250 differences** in the deterministic eval content (move + score +
   WDL); only volatile timing fields (`time`/`nps`) differed.

### Measured speedup of concurrent evaluation (one A100)

| K engines | speedup on pure eval workload | bit-mismatches |
|---|---|---|
| 2 | 1.99× | 0/250 |
| 4 | 2.46× | 0/250 |
| 8 | 3.07× | 0/250 |

The GPU overlaps the batch-1 evals (latency hiding) rather than serializing them.

**Realistic overall speedup** is less than the raw eval speedup, because only the child-eval loop
parallelizes. Measured breakdown (`instrument_breakdown.py`, 2 trees, 26.8 s):

| Component | Share of wall | Calls | Parallelizable |
|---|---|---|---|
| **child value-evals** | **80.3%** | 4,856 (**≈27 per expansion**) | **yes** (the for-loop) |
| terminal/repetition check | 11.7% | 4,854 | removed by the guard |
| prior call | 5.0% | 177 (1/expansion) | no (sequential) |
| selection/backup/other | ~3% | — | no |

Note: **~27 child-evals per expansion** (every legal child's WDL is computed — required for the
GNN child-WDL targets), so the for-loop is a full **80%** of wall. Using the *measured* GPU
concurrency (K=4 → 2.46×, K=8 → 3.07× on the eval workload, not ideal /K) plus the repetition
guard: optimized wall ≈ value/2.46 + prior + other ≈ **2.5× (K=4) to ~2.9× (K=8)** overall →
**16.87 → ~6 s/tree → 50K in ~1.2 days** at 3 GPUs.

## Why not true batching? (route (a) prototype result)

Batching all C children through one NN forward pass is GPU-optimal and would beat pooling — pooling's
~3× ceiling exists *because* it doesn't fuse the evals (the GPU overlaps C batch-1 kernels but can't
merge them). lc0's UCI has no "batch-evaluate these N positions" command, so the two real routes are:

- **(a) lc0 internal batching** — search the parent with `go nodes N` (+ `VerboseMoveStats`) and read
  each child's value from the per-move stats. **Prototyped (`explore_verbose.py`):** at `nodes≥256`
  lc0 emits a scalar **`V`** (= W−L) for **all 32 children** in **~37 ms** (vs ~141 ms sequential
  valuehead, ~3.8×) — lc0 *does* batch internally. **But it only gives the scalar `V`.** The per-move
  **`WL`/`D` (the W/D/L distribution)** are populated **only for visited children** (N≥1), and (i)
  PUCT won't visit all children, (ii) a visited child's `WL`/`D` is the **backed-up subtree average**,
  not the raw per-child NN WDL. The pipeline stores the **full normalized W/D/L triple** per child as
  the **GNN child-WDL target** (`value_features_from_wdl` → `edge_wdl_targets`), which **cannot be
  reconstructed from scalar `V`**. → **Route (a) cannot faithfully produce the targets.**
- **(b) run the net directly** (PyTorch/ONNX) — gives batched WDL, but requires replicating lc0's
  exact 112-plane input encoding + WDL head bit-for-bit (history planes, side-to-move flip,
  canonical transform). High silent-corruption risk; a multi-hour-to-day project with heavy
  validation. **Logged as a future deep optimization** (worth it for 100K+ if we push toward ~10×).

**Net (initial read): pooling looked like the best faithful speedup.** This was WRONG — see Outcome.

## Outcome (validation, 2026-06-05): pooling rejected

Implemented pooling (extra `valuehead` engines + threaded child-eval dispatch, default-off) and ran
the bit-identical gate (8 FENs, `value_pool_size=1` vs `=6`). **It failed on both axes:**

- **Slower, not faster.** Per-tree (startup-excluded) **172 s pooled vs 133 s sequential (~30% slower)**.
  The 2.5–3× from `parallel_eval_bench` was measured on lc0 **classic `go nodes 1`** (search setup =
  latency-bound, parallelizable). The real **value engine runs `valuehead` mode** — a pure NN forward
  pass that is **GPU-compute-bound**, so concurrent processes serialize on the GPU and only add
  thread/coordination overhead. *I benchmarked the wrong engine mode.*
- **Not bit-identical.** Pooled trees diverged structurally (node-feature deltas ~0.5, different edge
  counts) — separate from any speed issue.
- **lc0/CUDA itself IS deterministic:** two identical sequential runs matched on every field except
  `value_gap`, whose only "difference" was `NaN` (a NaN≠NaN comparison artifact, not nondeterminism).
  So byte-identity is achievable and the production pipeline is reproducible.

**Pooling was fully reverted** (`lc0.py`, `build_tree.py` back to HEAD; zero residue; tests pass). The
bit-identical gate did its job — it stopped a slower, corrupting "optimization" from shipping.

**Conclusion: there is no faithful quick win for the valuehead pipeline.** The only real lever is
**route (b)** — extract the net (ONNX/PyTorch) and batch the child WDLs with lc0's exact 112-plane
encoding — a separate, validated project (logged for 100K+). The 50K run uses the **unmodified,
faithful** pipeline at ~16.87 s/tree.

## Second win — skip the impossible threefold-repetition scan

`terminal_value_from_board` ([common.py:116](../lmcos/src/core/providers/common.py#L116)) calls
`board.outcome(claim_draw=True)`. `claim_draw=True` runs `can_claim_threefold_repetition`, which
replays the move stack — **23% of tree-gen time**. But boards here are built from a FEN plus
≤ `max_depth` (=4) pushes, with **no prior history**, so a 3-fold repetition (needs the same
position 3×, ≥ 9 plies) is **mathematically impossible**. The claimable **fifty-move** rule still
matters (it is encoded in the FEN's halfmove clock), but it is a cheap scalar check.

**Fix (provably equivalent here):** `board.outcome(claim_draw=False)` (keeps checkmate, stalemate,
insufficient material, fivefold, seventy-five-move) **plus** an explicit
`if board.halfmove_clock >= 100: draw`. Eliminates the history scan; identical terminal labels.

## Combined estimate & plan

- Eval pooling (K≈4–8) + repetition skip → measured **~2.5–2.9× overall** → **16.87 → ~6 s/tree**
  → **50K in ~1.2 days** at 3 GPUs (from 3.3 d), compounding for 100K and all re-runs.
- **Validation gate before use:** generate a sample of FENs with the current vs optimized provider
  and assert **byte-identical payloads** (the cheap bit-for-bit test). Only then use it for 50K.

## Artifacts
- `smoke_u1/parallel_eval_bench2.py` (K-engine eval speedup + bit-identity)
- `smoke_u1/instrument_breakdown.py` (parallelizable-fraction breakdown)
- `smoke_u1/profile_lc0.py`, `profile_build_tree.py` (raw lc0 + cProfile)
