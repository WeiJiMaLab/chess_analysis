# Frozen per-step tree values: the fix, current design, and state

Full narrative/debugging history (what was tried, what was wrong and why, dated
progress log) lives in [labnotebook.md](labnotebook.md), entries `2026-07-10` and
`2026-07-11`. This document describes only the current, working design and its
validated state — it is not a running log and should not accumulate one; append
new work to `labnotebook.md` instead.

## The bug this fixed

The meta-controller (MC) decides, at each search step t, whether to keep
expanding the tree or halt, using a per-node encoding `<value, wdl_win,
wdl_draw, wdl_loss, wdl_var>` for every node visible in the tree at step t. That
encoding is supposed to reflect what the search currently believes about each
node. It didn't: node features were tensorized **once**, from the complete,
fully-searched tree, and every step t just sliced a growing prefix of that one
static tensor — the *set* of visible nodes grew correctly, but the *value*
attached to any node was frozen at packing time, never updated by
backpropagation as later search revised it. Proven on real data: root children
with a real 0.3–0.6 backed-up-value swing across a trajectory had bit-identical
packed encodings at the start and end of that swing.

## Current design

**`packhistory_trees`** (`src/cts/data/preprocess_mc/pack.py`, stage name
`mc_pack`) — for every raw tree, replays MCTS backpropagation over its already-
recorded expansion order (`_replay_backprop_history`) to recover the true
per-step backed-up value/WDL for every node, without re-running the engine: a
raw tree already stores final structure, each node's static leaf eval, and the
real expansion order, so replay is pure bookkeeping. Packs two things per tree:
(a) base/structural features (unchanged from the old format) and (b) a sparse
per-edge update log:

```
step_index:   int32[M]     # 0-indexed position in the FULL expansion order
                            # (expansion_parent_ids), NOT re-based to root_rank
node_id:      int32[M]     # child/edge id; parent = parent_index[node_id]
visit_count:  int32[M]     # cumulative visit_count for this edge as of this update
q_value:      float32[M]   # cumulative mean q_value as of this update
wdl:          float32[M,3] # cumulative mean win/draw/loss as of this update
```

Sorted by `(node_id, step_index)`, with CSR `node_update_ptr: int32[N+1]` for
O(log M) forward-fill lookup ("value as of step t"). Packed field names:
`update_log_step_index`, `update_log_node_id`, `update_log_visit_count`,
`update_log_q_value`, `update_log_wdl`, `node_update_ptr`, plus shard-level CSR
pointers `trajectory_update_log_ptr`, `trajectory_node_update_ptr_ptr`.
Root/creation order is ascending node id, never CSR/`children_index` (UCI-string)
order. M ≈ 200–300 per tree.

**`packhistory_GNNpretrain`** (`src/cts/data/preprocess_gnn/pack_history.py`,
stage name `gnn_pack_history`, config class `PackHistoryGNNPretrainConfig`) —
samples one step `n` per tree, uniform over `[1, search_budget − lookahead_k]`,
deterministically seeded (`sha256(source_path|draw_index|seed)`, mirrors
`deterministic_starting_budgets`). Builds `T_n` (real structure + real per-step-n
values, forward-filled from the update log) and supervises **every edge** of
`T_n` (full, unfiltered — see Known issues, fixed) with its child's k-steps-ahead
WDL, weighted by Δvisits (update-log entries in the window between step n and
step n+k). Reuses `ChildWdlModel`/`ChildWdlHead` (`src/cts/models/gnn.py`)
**unmodified** — no new model head. Step-index-to-update-log conversion goes
through `_full_scale_step` (converts the trimmed/`root_rank`-relative step `n`
to the full scale the update log uses).

**`train_encoder`** — unchanged code (`build_tree.py --stage encoder`), trained
on `packhistory_GNNpretrain`'s output instead of one-example-per-complete-tree
data. Checkpoint at `${gnnpack_history_dir}/tiny_encoder.pt`.

**`packhistory_MCmaterialize`** (`src/cts/data/preprocess_mc/materialize.py` +
`src/cts/train/controller_train.py::ControllerEpisodeDataset`, stage name
`materialize`) — `ControllerEpisodeDataset.__getitem__` does a per-step
forward-fill lookup into the update log (zero-initialized, not cloned from the
final-value baseline — see Known issues, fixed) instead of slicing one static
tensor, so the encoder is fed `T_t` populated with real step-t values, not
step-0 values repeated at every t. Runs the (now correctly-trained) encoder to
produce `z_root` per snapshot.

**`train_readout_pg`** — unchanged code. Consumes `z_root` as an opaque vector;
never needed to change.

## Directory / config layout

Everything above runs under a **new, parallel config** —
`config_ysagiv_xaba20k_history.yaml`, `config_ysagiv_xaba100k_minply15_maxply75_history.yaml`
— with every stage's output dir repointed at `_history`-suffixed paths
(`pack_history_dir`, `gnnpack_history_dir`, `materialize_history_dir`) that
mirror the old `mc_packed/`/`packed/`/`materialized/` 1:1. The old dirs and
their configs are untouched and kept indefinitely as a permanent regression
baseline. `split/` is shared (deterministic, same seed either way).

## Parameters

- **`lookahead_k = 12`** (~1/8 of the 96-step budget) — v1 default, not tuned.
- **`snapshots_per_tree = 1`** — one `n` sampled per tree, compute parity with
  the old one-example-per-tree cost. Raising this is a real lever, not yet
  exercised.

Both live in the `gnn_pack_history:` config section and are confirmed set in
both real configs currently in use.

## Validated results

- **Mechanism**: replayed root-child Q vs. the tree's own stored
  `oracle_root_q_trace` — mean error 3.3e-5, 0.16% of (tree, step, root-child)
  comparisons exceed 1e-3 on `human_trees` (residual traced to unrecoverable
  terminal-leaf-revisit backprop mass, not a bug).
- **Fix takes effect**: diff test vs. the old frozen-value `mc_packed/` output —
  structure and root oracle values byte-identical, per-step node features now
  genuinely differ at every step, confirmed at full production scale.
- **Pipeline runs end-to-end**: `xaba20k` and `xaba100k_minply15_maxply75` both
  completed all 5 stages cleanly (`packhistory_trees` →
  `packhistory_GNNpretrain` → `train_encoder` → `packhistory_MCmaterialize` →
  `train_readout_pg`).
- **The fix is a real improvement, not just mechanically correct**: matched-
  procedure comparison (same PG training loop, same cost regimes) — `z_t` beats
  a simple root-action-gap stopping heuristic in **10/10 cost regimes tested,
  all statistically significant** (paired bootstrap CI excludes 0).
  R-decodability (can the feature predict the value of continuing?): `z_t`
  R²=0.619 vs. action-gap R²=0.289. Frontier regret: `z_t`=0.0247 vs.
  action-gap=0.0307 (non-overlapping CIs). One open, partial finding:
  `z_t → action-gap` recoverability is real but lossy (R²=0.58–0.68, not ~1.0)
  — a 32-dim embedding compressing a whole tree plausibly can't preserve every
  derived scalar losslessly; not further explained.

## Test coverage (current, all passing against real data)

- `src/cts/tests/test_packhistory_trees.py` (7) — replay-vs-oracle accuracy,
  packed action-gap matches oracle at every step, diff vs. old `mc_packed/`
  (structure/oracle byte-identical, features differ).
- `src/cts/tests/test_no_future_leakage.py` (5) — no value before a node's own
  first update, `wdl_var` tracks the current step (not frozen), unvisited
  leaves are all-zero, root is always zero, same invariant re-checked in
  `pack_history.py`'s own builder independently of `controller_train.py`.
- `src/cts/tests/test_canary_future_leakage.py` (7) — adversarial
  poisoning/canary suite for the same invariant.
- `src/cts/tests/test_canary_pack_history_structural.py` (9) — edge-set
  correctness, Δvisits window, never-visited-filter topology-coupling
  regression, `root_rank` fix regression.
- `src/cts/tests/test_pack_history_fix_parity.py` (5) — cross-pipeline parity
  between `ControllerEpisodeDataset` and `pack_history.py`, loss-neutrality of
  previously-filtered edges.
- `src/cts/tests/test_canary_packhistory_trees_replay.py` (5) — replay
  determinism/poisoning canaries.
- `src/cts/tests/test_child_wdl_disambiguation.py` — sibling-prediction
  disambiguation (1 passing, 1 `xfail` documenting the still-open cross-sibling
  leakage below).

Real-data tests currently point at `xaba100k_minply15_maxply75` (has both old
and new baselines on disk; `xaba20k`'s `pack_history/` was purged and is not
being regenerated). Run explicitly — most of `src/cts/tests/` isn't covered by
`pytest.ini`'s `testpaths`.

## Known issues

Fixed, kept here only as a pointer for anyone reading old code/PRs:
- Never-visited-leaf filter (a compute-saving optimization) silently changed
  `TreeEncoder`'s attention topology for kept sibling edges — removed, full
  unfiltered edge set now always flows through.
- `root_rank` step-scale mismatch between structural and value queries in
  `pack_history.py` — fixed via `_full_scale_step`; was dormant on all real
  data seen (`root_rank == 0` on every real tree checked, structurally forced
  for a fresh single-root search), not a live corruption.
- Future-leakage: unvisited nodes/edges read their final end-of-search value
  instead of zero — fixed (zero-init, not clone-from-final-value), in both
  `wdl_var` and the general case.

Still open:
- **Cross-sibling leakage in `ChildWdlHead`**: perturbing one sibling's subtree
  measurably moves its untouched sibling's prediction by nearly the same
  amount (ratio ≈0.98–1.03), reproducible across training budgets. Real signal
  exists (siblings' predictions do differ meaningfully), but per-child
  localization does not hold cleanly. Plausibly a pre-existing property of
  `ChildWdlHead`'s architecture (shared parent aggregate, slot-gated only at
  the final concat), not proven `T_n`-specific — **deferred by explicit user
  call**, not re-verified against a complete-tree fixture. Revisit if
  `train_readout_pg`'s halt decisions ever look sensitive to it.
- **Versioning landmine**: `preprocess_mc/pack.py` serves both the old
  `mc_pack` stage and the new `packhistory_trees` stage from one module,
  unconditionally emitting the new shard format. The frozen `mc_packed/`
  regression baseline is frozen only because nobody has re-run `mc_pack` since
  the rewrite — a stray re-run would silently overwrite it with no error. No
  guard implemented.
- **Coverage skew**: late-born nodes (small `search_budget − n` window) get
  fewer valid `(n, n+k)` pairs. Monitor via `packhistory_GNNpretrain`'s
  diagnostics if late-tree calibration looks off.
- `xaba100k_minply15_maxply75`'s `z_t`-vs-action-gap head-to-head (the Wave 3
  sub-investigation done for `xaba20k`) hasn't been run yet — deliberate
  follow-up, not run unattended alongside the main chain.
- Raising `snapshots_per_tree` above 1, or smarter/sparser `n`-sampling than
  uniform-random, is a real lever not yet exercised.
