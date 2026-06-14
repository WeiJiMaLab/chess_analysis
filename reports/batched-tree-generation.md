# Batched in-process tree generation — parity-gated speedup (R-BATCHGEN)

**Phase:** Prior to implementation · drafted 2026-06-14 · owner hl4291 (with ysagiv)
**Thread:** LMCOS · follow-on to [(R-U1-SPEED)](u1-tree-gen-speedup.md) ("route (b) is the only real lever")
**Companions:** [`unify.md`](../unify.md) · [`labnotebook.md`](../labnotebook.md)

| Field | Content |
|---|---|
| **Description** | Replace the lc0-over-UCI, batch-size-1 evaluation in tree generation with an **in-process, batched neural evaluator** that drives our **own sequential PUCT**, **frontier-batched across many independent trees**. Keep the expansion order and per-step snapshots byte-faithful to the current pipeline. |
| **Rationale** | We already run our own PUCT ([`_select_leaf_by_puct`](../lmcos/src/data/preprocess_gnn/teacher_targets.py)); lc0-the-binary is used only as a neural-net oracle, queried one position at a time over a UCI text pipe. ~80–85% of the 16.87 s/tree is overhead (UCI round-trips + batch-1 launches + idle accelerator), not lc0 compute (~1 ms/eval). Batching the eval is a 10–100× lever and the per-tree saving multiplies across 150K (and future 100K+ re-gens). |
| **Expectation** | Per-tree 16.87 s → sub-second; 150K from ~700 GPU-h → tens of GPU-h (or CPU-fleet-feasible). **Trajectory and targets identical** to the lc0-UCI baseline, gated by the parity tests in §4. |
| **Open questions** | (a) Can we reach **discrete-trajectory identity** across a reimplemented forward pass, or do we re-baseline? (b) lc0 net via ONNX or lczerolens/PyTorch? (c) Is JAX worth the parity risk now, or a v2? |

---

## 1. The intuition (why this works, and what it must not break)

### 1.1 The unit of training is a *trajectory*, not a tree

The controller is an **optimal-stopping agent over a growing tree**. Its training example is the **ordered sequence of partial trees** `T_0 → T_1 → … → T_K` produced by *one* PUCT search, where `T_t` is the tree after the t-th expansion.

- The GNN encodes each snapshot `T_t → z_t`.
- The MC head maps `(z_t, budget) → A_t = Q_continue − Q_halt`.
- Supervision comes from **backward DP over the trajectory**: `V_halt(t) = h_t` (value of the best root move at step t), `V_continue(t) = −cost + V*(t+1)` ([`compute_budgeted_oracle`](../lmcos/src/data/preprocess_mc/oracle.py)).

So **the expansion order is the time axis of the stopping problem, and the snapshots are the controller's state sequence.** The GNN child-WDL targets need only the *final* tree; the controller targets need the *whole trajectory*. Lose the order and the meta-control dataset ceases to exist. The per-step root statistics (`oracle_root_q_trace`) evolve as the search proceeds and **cannot be reconstructed from the final tree** — they must be recorded online.

### 1.2 The tension, and the dimension we parallelize

- The **neural net** is throughput-bound: it wants hundreds–thousands of positions per forward pass to amortize launch/transfer overhead and saturate matmul units. At batch 1 it is ~85% idle overhead.
- A **single tree's PUCT** is irreducibly sequential: the selection at step `t+1` depends on the backprop from step `t`.

You cannot speed up one tree by batching. But you have **150K independent trees**, and a position's evaluation is the same regardless of what else is in the batch. **That independence is the batch dimension.**

### 1.3 The loop (frontier batching across trees)

Hold `N` trees in flight (say N≈1,000). Repeat until all reach the 96-expansion budget:

1. **Select** (per tree, CPU, no NN): run each tree's PUCT descent to the one leaf it wants next. → N leaves.
2. **Gather**: collect those leaves' children into one tensor (~N parents for policy + ~30·N children for value).
3. **One forward pass**: the in-process net returns policy/priors + value + WDL for the whole batch.
4. **Scatter + advance** (per tree, CPU): create children with priors, attach WDL, backprop up *that tree's* path, **record that tree's per-step oracle trace**, increment its step counter.

Because the budget is fixed (96 for all), trees run in **lockstep**; *continuous batching* (drop a fresh root into a slot the instant a tree finishes) keeps the batch full.

```
                 ┌─ tree 1 ─ select leaf ─┐
   one outer     ├─ tree 2 ─ select leaf ─┤   gather
   step (= one   ├─   ...                 ├─► ~31·N positions ─► ONE forward pass
   expansion     └─ tree N ─ select leaf ─┘                            │
   per tree)                                                            ▼
                 scatter back ◄── priors / value / WDL ◄────────────────┘
                 (expand · backprop · record trace, per tree, on CPU)
```

**Division of labor:** CPU does all bookkeeping (selection, expansion, backprop, trace recording) — per-tree, small, parallel across trees. The accelerator does only evaluation, always fed a full batch. This is the classic "search on CPU, evaluate on accelerator" split; the difference is **we own the loop**, so we control snapshots and order.

### 1.4 Why order is preserved exactly — and the trap we avoid

Batching changes only the **grouping** of independent evaluations, never their **values** or **ordering**. Each tree's selection reads only its own backed-up stats, so every trajectory is identical to a purely sequential run.

The trap: you *could* batch *within* one tree (leaf-parallel MCTS with **virtual loss**), but virtual loss is an approximation that perturbs selection → different order, different snapshots. **That is exactly the within-tree step-batching the project forbids.** We sidestep it by taking one leaf per tree across many trees: no collisions, no virtual loss, no order change.

---

## 2. What "exact parity" means (the layered contract)

Bit-identical output from a *different* inference backend is generally infeasible (different kernels/reduction orders). And the current pipeline has two quirks that perfect parity would force us to copy: **priors are parsed from UCI text rounded to 0.01%** ([`parse_no_search_analysis`](../lmcos/src/core/providers/parsers.py)), and positions are evaluated **history-free** (bare `position fen`, so lc0 fills its 7 history planes by its own convention). The in-process net gives **full-precision** priors and lets us choose the history convention.

So we separate concerns into four layers, with different bars:

| Layer | What | Bar | Achievable? |
|---|---|---|---|
| **L1 Search-logic** | PUCT selection, expansion, backprop, prior-normalization, trace recording, oracle DP, teacher targets — **given a fixed evaluation oracle** | **Bit-exact** vs current sequential loop | **Yes** (pure code, same inputs) — this is the load-bearing test |
| **L2 Evaluator** | Our net forward(position) vs lc0's value/WDL/policy | Max abs diff < tol (≈1e-5 fp32; looser fp16); policy argmax identical | Yes, to high precision |
| **L3 Trajectory** | End-to-end expansion order + best-move trace vs lc0-UCI baseline | Fraction of trees identical ≈ 1.0; divergences only at near-tie PUCT decisions and shown benign | Mostly; a small fraction may fork |
| **L4 Targets** | `edge_wdl_targets`, node values, `target_advantages` on trajectory-identical trees | Within tol | Yes |

**The key move:** prove **L1 bit-exact first**, by replaying a *fixed* evaluation oracle (cached lc0 outputs, or a deterministic mock) through **both** the old sequential loop and the new batched loop and asserting byte-identical artifacts. This isolates "did we break the search/order logic" from "did we change the NN numerics." If we change everything at once, nothing matches and we learn nothing.

**Decision (resolved 2026-06-14, hl4291): (a) replicate the quirks by default; bit-for-bit replication is the priority.** A single switch `re_baseline: bool = False` on `NetEvaluator` governs both quirks: when `False` (default) priors are rounded to the verbose-move-stats text resolution (`quantize_prior_to_uci_text_resolution`) and history uses lc0's bare-FEN fill (`encoding.history_fill_for` → `LC0_BARE_FEN`); when `True` the net keeps full-precision priors + the canonical history fill (`REPEAT`) and we re-run the A0 oracle-direction checks. Get the speedup proven against the existing dataset first, then flip `re_baseline=True` later.

**Honest scope of "bit-for-bit" (important):** literal every-float identity with lc0-CUDA is *not* attainable from a different inference backend — lc0's own backends (cuda / cuda-fp16 / blas / eigen) don't even agree bit-for-bit with each other. So "bit-for-bit" splits in two: **(i) search-logic parity given the same evals — already proven byte-exact (L1, T-replay);** and **(ii) the in-process net reproducing lc0's evals.** With `re_baseline=False`, prior quantization to the 1e-4 grid *absorbs* sub-1e-4 backend noise (so priors can match exactly), but value/WDL are full-precision and will differ in low bits — so the realistic target is **identical discrete trajectory** (expansion order + best-move trace) with values within tol, divergences confined to near-tie PUCT decisions. The only way to get literal value-bit identity is to keep lc0 as the evaluator, which forfeits the speedup — so (ii) is a "tight-enough, measured" bar, not literal identity.

---

## 3. Procedure

| Step | Status |
|---|---|
| **Phase 0 — Evaluator abstraction** | |
| 0.1 Define `Evaluator.evaluate(positions) -> (priors, value, wdl)` interface (batched) | ✅ `cts.data.batched_gen.evaluator` |
| 0.2 Wrap existing lc0-UCI provider behind the interface (baseline) | ✅ `Lc0UciEvaluator` + `evaluator_to_provider` |
| 0.3 Add `CachedEvaluator` (records lc0 outputs keyed by FEN; replays deterministically) and a `MockEvaluator` (deterministic synthetic outputs) | ✅ |
| **Phase 1 — Batched search loop (L1, bit-exact)** | |
| 1.1 Refactor PUCT into the frontier-batched, eval-agnostic loop (§1.3); preserve child UCI ordering, tie-breaking, sign conventions, RNG seeding | ✅ `search.generate_trees_batched` (reuses legacy helpers) |
| 1.2 Online per-tree oracle-trace recording inside the batched loop | ✅ |
| 1.3 **Replay test:** old sequential vs new batched under the *same* fixed evaluator → byte-identical `PretrainExample` (tree, trace, targets) | ✅ **T-replay passes** (41 batched_gen tests green, full suite 293✅/1 skip) |
| **Phase 2 — In-process net evaluator (L2, numeric)** | |
| 2.1 Export/load the lc0 net (`leela2onnx` → ONNX, or lczerolens → PyTorch); pick fp32 | ⬜ |
| 2.2 Replicate lc0's 112-plane input encoding (incl. the history-fill convention chosen in §2) | ⬜ |
| 2.3 **Numeric parity test:** forward vs lc0 valuehead/policy on a held-out FEN set; report max/mean abs diff, policy-argmax agreement | ⬜ |
| **Phase 3 — End-to-end parity + speedup (L3/L4)** | |
| 3.1 Compose batched loop + in-process net; run on a held-out FEN set vs lc0-UCI baseline | ⬜ |
| 3.2 Report % trajectory-identical; characterize any divergences (near-tie?); target parity on identical trees | ⬜ |
| 3.3 Throughput + GPU-util sweep over batch size; pin per-tree cost; **go/no-go** | ⬜ |
| **Phase 4 — Scale-up** | |
| 4.1 Wire slurm config (one output dir, global-index filenames, `resume`); CPU and/or GPU lane | ⬜ |
| 4.2 Generate 150K → `trees_unfiltered/`; completeness + identity guards (`count==len(FENs)`, `root_position_spec==FEN`) | ⬜ |
| 4.3 Re-baseline: re-run A0 oracle-direction invariants on the new trees | ⬜ |

---

## 4. Tests we need (committed to `tests/`)

| ID | Test | Bar |
|---|---|---|
| T-enc | 112-plane encoding: known positions; side-to-move flip; castling/en-passant/repetition planes; **bare-FEN history fill** matches the chosen convention | exact |
| T-fwd | In-process forward == lc0 value/WDL/policy on held-out FENs | max abs diff < tol; argmax identical |
| T-sel | Batched PUCT selection == `_select_leaf_by_puct` given fixed tree+evals, **including tie-break** (earliest child on equal score) and canonical UCI child order | exact |
| T-prior | `normalize_prior_scores` + budget truncation + child ordering identical | exact |
| T-back | Backprop value/WDL accumulation + per-ply sign flips identical | exact |
| T-trace | Per-step `oracle_root_q_trace` / `oracle_best_move_trace` / visit trace recorded at identical step indices | exact |
| T-tgt | `edge_wdl_targets` (visit-weighted, perspective-flipped), node values, `target_advantages` | within tol |
| **T-replay** | **Sequential vs batched under one `CachedEvaluator` → identical `PretrainExample`** (the load-bearing L1 test) | **byte-exact** |
| T-seed | Same `--seed` → identical trees (budget-bucket RNG reproduced) | exact |
| T-batch-inv | Result for a tree is independent of batch size / which other trees share the batch | exact |
| T-nan | No NaN/inf; shapes/dtypes on batched tensors | — |
| T-existing | Repo invariants still pass (`test_oracle_stop_step_features`, `test_board_tree_features`, `test_value_gap`, `test_converged_expansions`) | pass |

---

## 5. Smoke tests (operational, before scale)

| ID | Smoke | What it proves |
|---|---|---|
| S-1tree | Generate **1 tree** new vs old; diff tree, trace, targets | end-to-end wiring + L1/L3 on a single case |
| S-Nlock | **N=8** trees in lockstep; assert each == its sequential counterpart | batching preserves per-tree order |
| S-thru | **100 trees** at batch sizes {1, 32, 256, 1024}; trees/s + `nvidia-smi` util | the 10–100× claim; pick N |
| S-mem | **N=1,000** live trees; peak RSS | memory budget for continuous batching |
| S-backend | ONNX-CPU vs ONNX-GPU vs lc0; same backend twice | numeric drift across backends; determinism |
| S-parity1k | **1,000** held-out FENs end-to-end; % trajectory-identical; divergence characterization | L3 go/no-go evidence |
| S-cpu | Batched-CPU throughput on a `short` node (fp32 ONNX-RT, K cores) | whether the CPU fleet (1,400 cores) is the cheaper lane |

---

## 6. Libraries / stack

**Recommended (lowest parity risk): PyTorch + lczerolens.**
- `lczerolens` loads Leela nets into PyTorch **and ships the lc0-faithful board encoding** (`LczeroBoard`), so the 112-plane encoding and the net come from one lc0-aligned source — far fewer encoding-parity bugs than hand-rolling planes.
- `torch` is already a project dependency (the GNN/controller). One framework end-to-end.
- `python-chess` (already used) for move generation / FEN handling within our PUCT.

**Alternative: ONNX Runtime + lc0 `leela2onnx`.**
- `leela2onnx` is a built-in lc0 subcommand (no new lib); `--onnx2pytorch` produces a PyTorch-friendly graph. Add `onnx`, `onnxruntime-gpu`.
- Pro: `onnxruntime` runs the same graph on CPU and GPU (helps the CPU-fleet lane). Con: we must reproduce the 112-plane encoding ourselves (parity risk) unless we borrow lczerolens's encoder.

**Net weights:** reuse the current `t1-256x10-distilled-swa-2432500.pb.gz` (a 256×10 ResNet-style net; supported by lczerolens and leela2onnx) so L2 compares against the *same* network we use today.

New deps to add: `lczerolens` (+ its torch deps) **or** `onnx` + `onnxruntime-gpu`. Nothing else mandatory.

---

## 7. JAX — can we, and would it help?

**Can we?** Yes. JAX/XLA gives `vmap` batching, JIT fusion, and one code path across CPU/GPU/TPU; [`mctx`](https://github.com/google-deepmind/mctx) is a JAX-native *batched* MCTS that already operates over a batch of trees.

**How it would help:**
- The batched **eval** is trivially `vmap`/JIT in JAX — but PyTorch already batches the eval, which is where ~all the speedup lives.
- The real differentiator is vectorizing the **search bookkeeping** (selection/backprop across N trees) as array ops. JAX would make that fast — *if* you represent trees as **fixed-size padded arrays with masking** (mctx's approach: pre-allocate `max_nodes`, no dynamic growth).

**Why not for v1:**
- Our trees are **ragged and dynamically grown**, and our targets/trace code is written against that ragged form. Porting to fixed-size padded JAX arrays is a substantial rewrite **and a parity risk** — padding/masking must be proven not to change any selection or target (re-deriving the whole L1 contract in a new representation).
- The speedup is in the eval batch; PyTorch delivers it with **far less parity risk** and keeps us in the framework the rest of the stack already uses.
- mctx returns the final tree + summary stats; extracting our **ordered per-step snapshots + oracle trace + exact existing targets** would mean instrumenting its functional internals anyway.

**Recommendation:** **PyTorch for v1** (batched eval, ragged search on CPU). Treat JAX/mctx as **v2**, justified *only if* — after eval is batched — the **CPU-side bookkeeping** becomes the new bottleneck. At that point the ordered options are: vectorize the bookkeeping in numpy/torch over the N trees; then numba/Cython on the inner loop; then a JAX rewrite to fixed-size trees if TPUs or full-XLA fusion are worth the parity re-validation. (Note: numba is useless for the *eval* — that's matmul under BLAS — but it is a legitimate lever for the *bookkeeping* loop, post-eval-batching.)

---

## 8. Go / no-go and performance targets

- **Go (L1):** T-replay is byte-exact. Without this the refactor is unsafe at any speed.
- **Go (L2):** forward-vs-lc0 max abs diff < tol with identical policy argmax on the held-out set.
- **Go (L3):** S-parity1k shows trajectory identity ≈ 1.0, or all divergences traced to near-tie PUCT decisions with targets matching within tol on identical trees (and a conscious §2 re-baseline decision).
- **Speedup target:** S-thru shows ≥10× trees/s vs lc0-UCI at a saturating batch; pin per-tree cost (target sub-second). If <~5×, **abort the build** and instead buy throughput with the QoS fix (gpu-short) + CPU lane on the existing lc0-UCI path.

## 9. Risks & contingencies

| Risk | Mitigation |
|---|---|
| Cross-backend numeric drift forks trajectories (L3 < 1.0) | Match precision (fp32), use lczerolens encoding, tight tol; if residual, adopt new canonical (§2b) + re-baseline (4.3) |
| **Text-rounded priors / history-fill** quirks make end-to-end identity impossible | Decide §2 (a) replicate quirks vs (b) re-baseline — recommend (b) |
| Tie-break / child-ordering mismatch silently forks order | T-sel covers exact tie-break + canonical UCI order |
| Memory blow-up at large N | Cap N; continuous batching; trees are tiny (~hundreds of B/node) so N can be thousands |
| Build cost not repaid | §8 abort gate (<5×) → fall back to parallelism-only on lc0-UCI |
| Two-framework drift (if JAX) | Stay single-framework (torch) for v1 |

---

*Next action on green-light: Phase 0 + the T-replay test (L1) — the single load-bearing gate. Mirror status into `labnotebook.md` and the procedure checklist above.*
