# Code cleanup — dead code sweep + today's-session bug sweep (2026-07-08)

Run while Agent 1 (ysagiv), Agent 2 (our_trees_continued), and Agent 3 (cp_regen) were
live in this same shared working directory. Per the task's safety constraint, nothing
matching `ysagiv*`, `cp_regen*`, `*e2e_controller*`, `*controller_train*`,
`*pg_controller*`, `evaluate_our_trees_continued.py`, `regime_select.py`, or anything
touched in the last ~3 hours (`find <path> -mmin -180`) was edited — only reported.
`pytest src/` was run before and after every change (baseline: 3 failed / 141 passed / 3
skipped; after Part 1's removal: same 3 failed / 135 passed / 3 skipped — the 6-test drop
is exactly the removed `test_stats_regression_pretrain.py`, nothing else changed). No
whole-repo git commands were used (`git status --short <path>` / `git diff --cached --
<path>` only).

## Part 1 — Dead code removed

Two source files + their test belong to the **S2/S3 track**, which `plan.md` line 152
explicitly documents as **"Dropped, superseded, not revisited"** (not merely "closed" —
S2/S3 never produced a documented result; `slurm_cleanup.md` independently confirms the
S2/S3 `.slurm` scripts were already archived as dead for the same reason, and that the S3
eval job "never appears in sacct" — i.e. it was written but never even run).

Verified via `grep -rln` across every `*.py`/`*.slurm`/`*.md` in the repo that each file's
only referrers were itself and already-archived `slurm/archive/*.slurm` scripts:

| Removed | Why dead |
|---|---|
| `src/cts/train/stats_regression_pretrain.py` | S2 encoder-pretraining script; only referenced by `slurm/archive/diagnosis_s2*.slurm` (archived, non-live) and itself |
| `src/analysis/tests/test_stats_regression_pretrain.py` | Tests for the above (6 tests); removed together since testing a module that no longer exists |
| `src/analysis/evaluate_s3_stats_z3.py` | S3 eval script; only referenced by `slurm/archive/diagnosis_s3_stats_materialize_and_eval.slurm` (archived; that job **never actually ran** per `slurm_cleanup.md`'s `sacct` audit — this script produced zero real results) and itself |

Stale `.pyc` files for the two source modules were also deleted from `__pycache__/`.

**Before → after**: `git status --short` for these three paths went from `??` (untracked)
to gone; `pytest src/` confirmed no other test imports them (135 passed after vs. 141
before, exactly accounting for the 6 removed tests, same 3 pre-existing failures — see
Part 2).

### False positives ruled out (NOT removed)

A naive "0 grep references" pass flagged several more files as apparently dead; each was
individually re-checked and is actually live or intentionally kept:

- **`src/cts/models/tree_mha.py`** — my first regex-based reference count showed 0 hits.
  Verified by hand: it IS imported (`from .tree_mha import TreeAttMsgLayer` in
  `src/cts/models/gnn.py`) — the regex just didn't match the relative-import form. Real,
  load-bearing production code (the encoder's attention layer). Left untouched.
- **`src/analysis/tree_diagnostics.py`** (T2, closed) — looked orphaned by name, but is
  actually imported by three *live* files: `ysagiv_sanity.py` (Agent 1),
  `cp_regen_sanity.py` (Agent 3), and `t1_render_trees_svg.py`. It's shared
  infrastructure (dup-FEN/illegal-move/linear-chain checks, tree pruning-for-display),
  not a one-off T2 script. Left untouched.
- **`src/analysis/relabel_replay.py`** (CP, closed) — imported by `cp_recalibration.py`
  (which was modified within the last 3 hours, i.e. plausibly still in active use by
  Agent 3's cp_regen work per its own docstring cross-reference) and by two test files.
  Not dead. Left untouched.
- **`src/cts/analysis/stats_ablation.py`** (S1, closed-with-results) — after removing
  `stats_regression_pretrain.py`, its only remaining "importer" is a docstring/comment
  mention in `evaluate_our_trees_continued.py` (not a real import). Genuinely
  import-orphaned now, but S1 is a **closed-with-a-documented-result** track (regret
  0.1138, confirmed via `sig_significance.md`) — kept for reproducibility, see
  "Left alone" below.

## Part 1 — Left alone (orphaned but ambiguous / provenance value)

These are import-orphaned (nothing else calls them) but were **not** removed, because
each is the one-off script that produced a result already written up and cited in
`plan.md`/a companion `.md` report — deleting them would make that already-published
number unreproducible, which the task explicitly says to avoid when ambiguous:

| File | Track | Why kept |
|---|---|---|
| `src/analysis/t1_render_trees_svg.py` | T1 (closed) | Produces the 5 SVG trees `plan.md`'s T1 section embeds; also **modified within the last 3 hours** (`g_cleanup.md`'s item 3), so doubly out of scope to touch |
| `src/analysis/quiescence_pruning_sweep.py` | T3 (closed) | Produces `t3_quiescence_sweep.png`/`t3_pruning_sweep.png`, both embedded in `plan.md`; its only "reference" elsewhere is a stray docstring cross-mention in `test_cp_recalibration.py`, not a real import |
| `src/cts/analysis/stats_ablation.py` | S1 (closed) | Produces `s1_stats_ablation.csv`, the number `sig_significance.md`'s SIG-S result is built on |
| `src/cts/analysis/zt_stats_decodability.py` | E2 (closed) | Produces `e2_zt_stats_decodability.csv`, cited directly in `plan.md`'s E2 section; only reference besides itself is the already-archived `slurm/archive/diagnosis_e2_zt_stats_decodability.slurm` |
| `src/analysis/regime_select.py` | REGIME (in-progress writeup) | Matches the task's explicit exclusion list verbatim — not touched regardless of dead-code status |

### Real duplication found, NOT consolidated (both owners are live agents)

`src/analysis/cp_regen_sanity.py` (Agent 3) and `src/analysis/ysagiv_sanity.py` (Agent 1)
each independently reimplement an almost byte-identical `run_t1`-style sanity routine —
same percentile-episode selection, same bottom-up `subtree_size` accumulation loop
(`for node_id in range(len(parent_index) - 1, 0, -1): subtree_size[parent_index[node_id]]
+= subtree_size[node_id]`), same root-dominance/dup-FEN/illegal-move check wiring, same
absolute-path-for-graphviz comment. This looks like both were copied from a common
ancestor (`t1_render_trees_svg.py`) rather than factored into one shared helper — a good
consolidation candidate. **Not touched**: both files are live-agent-owned (`cp_regen*` /
`ysagiv*`), so this is reported for the orchestrating session to relay, not fixed here.

## Part 2 — Bugs found in today's code

### 1. [Most important — currently FAILING, unresolved] BeFS rewrite doesn't fix the tunnel-vision bug it targets

`src/cts/data/preprocess_gnn/teacher_targets.py` was substantially rewritten today: a new
`TreeSearch` base class + `PUCTSearch`/`BeFSSearch` subclasses, a first-play-urgency (FPU)
fix for PUCT (`EdgeStats.fpu_value`, unvisited children now fall back to their own
sign-flipped static value instead of a fake `Q=0`), and a genuine best-first-minimax
rewrite of BeFS (`_befs_backup_from_node`, replacing the old "greedy descent on frozen
static values" with real backed-up minimax that gets re-examined after every expansion).

**The BeFS half of this rewrite does not achieve its own stated goal.** Its own new test,
`src/analysis/tests/test_befs_puct_rewrite.py::test_befs_depth_extension_on_real_fens`
(a real-Stockfish, `@pytest.mark.slow` test, not a mock), asserts that at least 2 of a
root's children get expanded past ply 1 within a 96-node budget — explicitly labeled
`"this is the tunnel-vision regression the rewrite is meant to fix"`. It currently
**fails on both its real-position parametrizations**: only **1 of 20** root children
(starting position) and **1 of 36** (a castled-middlegame FEN) ever get expanded past ply
1. This reproduces verbatim in a clean `pytest src/` run (see failure log below) — it is
not an artifact of anything I changed.

```
FAILED test_befs_depth_extension_on_real_fens[rnbqkbnr/.../RNBQKBNR w KQkq - 0 1]
FAILED test_befs_depth_extension_on_real_fens[r2q1rk1/.../R1BQ1RK1 w - - 0 8]
  AssertionError: only 1 of 36 root children ever got expanded past ply 1 (...);
  this is the tunnel-vision regression the rewrite is meant to fix
```

**Mitigating context, not a false alarm**: BeFS is not reachable through the live
pipeline right now — `config_minply15_maxply75.yaml`'s `variants.befs` block was deleted
today ("BeFS demoted to labnotebook 2026-07-07... `variants.befs` removed 2026-07-08"),
and the default `treegen.selection` is `puct`. So this bug cannot currently corrupt any
production tree generation. But it IS a real, currently-red test in the shared suite, and
the PUCT-side FPU fix in the *same* rewrite (which the review below found no fault with)
is live-default code that shares the base `TreeSearch.generate` harness with the broken
BeFS path — worth flagging prominently rather than letting a red test linger unexplained.
**I did not attempt a fix**: this is deep search-algorithm code (negamax sign-convention
bookkeeping across `_befs_backup_from_node`/`_seed_befs_child_edge_stats`/
`_select_leaf_by_befs`) that a currently-running live job (`cp-regen-gen-trees`,
job `10842167`) depends on the surrounding file for, and getting this wrong under time
pressure risks a worse, silent regression. Relay to whichever agent/session owns
`teacher_targets.py`'s rewrite.

### 2. [Currently FAILING, unresolved] v6 pretrain-format change drops `cp`/`mate`, breaking an explicit round-trip test

Same file, a separate change: `RAW_PRETRAIN_FORMAT` was bumped `v5 → v6`, and the new
`_RAW_PRETRAIN_EXCLUDED_FEATURES = frozenset({"cp", "mate"})` now drops those two columns
from every persisted tree (justified in the diff's own comment as "verified unused by
every consumer — gnn_pack, mc_pack, tree_loader.py").

That justification wasn't cross-checked against `src/analysis/tests/test_cp_provenance.py`
(also modified today), whose `TestWriterAndPackCompat.test_payload_round_trip_preserves_cp_features`
explicitly asserts `cp` and `mate` **do** survive a payload round-trip. It now fails:

```
FAILED test_cp_provenance.py::TestWriterAndPackCompat::test_payload_round_trip_preserves_cp_features
  AssertionError: payload lost feature cp
  assert 'cp' in ['value', 'wdl_win', 'wdl_draw', 'wdl_loss', 'wdl_var', 'prior', ...]
```

This is a genuine, currently-red inconsistency between two files edited in the same
session: either the v6 exclusion is wrong (something *does* still need `cp`/`mate` on
disk — plausibly CP-provenance/debugging use, given the test's name) and needs to keep
those columns, or the test is stale and needs updating to match the deliberate v6
decision. Given the test file is literally named `test_cp_provenance.py` and this appears
tied to Agent 3's CP-regen work, **flagging for Agent 3 / the orchestrating session** to
resolve one way or the other — not fixed here (both files are recent/live-adjacent).

### 3. Reviewed, no bug found (listed for completeness)

- **`src/cts/core/providers/{parsers,stockfish}.py`** — new `apply_tanh_cp_feature`
  (CP-regen's `tanh(cp_order/T)` value option). Correctly no-ops when
  `tanh_cp_temperature is None` (zero-cost for every existing config), applied uniformly
  on both the terminal and non-terminal feature paths in `stockfish.py`. Deliberately
  duplicates the one-line `tanh` formula instead of importing
  `analysis.relabel_replay.tanh_cp_value`, with an explicit, correct architectural
  justification (`cts` must not depend on the `analysis` layer built on top of it) — not
  an accidental duplication.
- **`src/cts/data/preprocess_mc/materialize.py`** — new `resume: bool = False` config
  field, default now clears a worker's shard directory (fixing a real prior incident
  described in its own comment: a rerun with regenerated `packed_data` silently resumed
  onto a stale prior run's leftover shards). `resume=True` path adds an mtime staleness
  check against `packed_data`. Logic checked out; no issue found.
- **`src/analysis/evaluate.py`, `src/cts/analysis/zt_probe.py`** — both today fixed a
  real train/eval **leakage bug**: the fit/eval split was previously by raw episode
  index, but one source tree can back multiple episodes (different truncation depths)
  that share an identical `z_t` at a common step — splitting by episode index could put
  some of a tree's episodes in fit and others in eval. Both files now split by
  `trajectory_key` (grouping episodes by source tree first). This is a good, correct fix
  — flagged here only as a positive finding, not a bug.
- **`src/analysis/utils/helpers.py`** — `_DEFAULT_CONFIG` repointed from
  `config_allply.yaml` (deleted today, per `git status`) to
  `config_minply15_maxply75.yaml`. Necessary, correct.
- **`src/cts/models/readout.py`** — `FractionStop` (readout tier) removed, docstrings
  renumbered 5→4 tiers consistently. `grep -rn "FractionStop"` across the whole repo
  confirms zero remaining references anywhere — clean removal, no dangling breakage.
- **`src/cts/data/preprocess_mc/filter_argmax.py`, `filter_xaba.py`** — both read
  correctly (`step > argmax_threshold` matches the documented `argmax > 2` criterion;
  `filter_xaba` excludes exactly the `"X*AB*A"` category). No bugs found. (Not edited
  regardless — both are shared with Agent 1's live `ysagiv_argmax_filter.slurm`/
  `ysagiv_xaba_filter.slurm` jobs even though their filenames don't match the `ysagiv*`
  glob.)
- **`src/cts/train/e2e_controller_train.py`'s `run_epoch`** — the task brief mentioned
  this session already found and fixed one docstring/behavior mismatch here (the
  promised `(mean loss, per-batch loss trace)` return value). Verified: the fix is
  already in place (`batch_loss_trace` is now collected and returned; `_write_within_epoch_trace`/
  `_plot_within_epoch` consume it). No further mismatch found in this file.
- **CI methodology audit**: every `bootstrap_ci(...)` call site in today's new/changed
  files (`evaluate.py`, `evaluate_e2e_z3.py`, `evaluate_sig_s.py`,
  `evaluate_our_trees_continued.py`, `cp_recalibration.py`, `ysagiv_sig.py`) goes through
  `cts.stats.bootstrap_ci` with `n_boot=2000`, per repo convention — no normal-theory CI
  found in the controller/regret domain. `board.py` does use `1.96 * SEM` in several new
  panel functions (`checks_diversity_curves` etc.), but this is a **pre-existing,
  separate convention** for large-sample descriptive RT-vs-feature plots (already used
  by `checks_material_band_curves` before today), not a new violation of the
  regret-CI convention — not flagged as a bug.

## Ambiguous / left alone without action

- **`pytest.ini`'s `testpaths = src/analysis/tests`** — this excludes `src/cts/tests/`
  (3 files, including 2 new today) from a bare `pytest`/`pytest .` invocation with no
  path argument. Confirmed via `git diff pytest.ini` this is **pre-existing** (only the
  `slow` marker registration was added today; `testpaths` itself predates this session),
  so it's out of scope for a "today's bugs" review, but worth a note: anyone running bare
  `pytest` (no explicit `src/` argument) silently skips `src/cts/tests/` entirely. Always
  invoke as `pytest src/` to get full coverage (which is what this cleanup pass and the
  task's own instructions specified).
- **Config `config_minply15_maxply75.yaml`, `human_analysis.trees_default`/`split.source_root`
  hardcoded-duplicate-of-`treegen.output_dir` pattern** — already identified and explicitly
  left alone by Agent 3 per `plan.md`'s own entry ("bigger, riskier change... open if
  wanted later"). Not re-litigated here.

## Part 3 — BeFS removed entirely (follow-up session, same day)

Per direct user instruction, the BeFS search rule flagged as broken in Part 2, bug #1
above (confirmed, unfixed tunnel-vision bug — the rewrite meant to fix it never did, and
BeFS was already unreachable from the live pipeline: `config_minply15_maxply75.yaml`'s
only `treegen.selection` value is `puct`, and its `variants.befs` block was deleted the
same day) was deleted rather than kept unreachable. This was a deprecation, not a repair.

Removed (in an earlier half of this same session, verified complete here):
- `src/cts/data/preprocess_gnn/teacher_targets.py`: `_befs_open_map`, `_select_leaf_by_befs`,
  `_seed_befs_child_edge_stats`, `_befs_backup_from_node`, `class BeFSSearch`. The
  `selection` field now only validates `"puct"` (`__post_init__` raises otherwise); the
  generation dispatcher always uses `PUCTSearch` directly.
- `src/analysis/tests/test_befs_puct_rewrite.py` — deleted (existed solely to test the
  now-deleted, never-fixed rewrite; this is exactly the test whose failure is documented
  as bug #1 above).

Finished in this follow-up pass:
- `src/analysis/tests/test_cp_provenance.py`: removed `class TestBefsSelection` (7 tests)
  and its two dedicated helpers `_befs_config`/`_generate` (confirmed via grep they were
  used nowhere else). `_befs_config` was renamed `_puct_config` and fixed to build a valid
  `selection="puct"` `TeacherSearchConfig` — it was still needed by `_scripted_example`
  (feeds `TestWriterAndPackCompat`), which had started raising `ValueError: unknown
  selection 'befs'` once the upstream `teacher_targets.py` validation landed. Also removed
  `TestEncoderSchemaFrozen.test_befs_suffix_stamped_into_search_config_id`, which
  constructed a `BuildTreeConfig(selection="befs")` specifically to test the now-impossible
  `"_befs"` search-config-id suffixing behavior. Module docstring updated to drop stale
  BeFS claims.
- `src/cts/data/build_tree.py`: `BuildTreeConfig.selection` was still `Literal["puct",
  "befs"]` — untouched by the teacher_targets.py-only removal, so any caller passing
  `selection="befs"` would construct fine but crash deep inside `_build_quality_config`'s
  `TeacherSearchConfig(...)` call with the same `ValueError`. Narrowed to `Literal["puct"]`
  and updated the two docstring comments that referenced `"befs"` (one now notes the
  `search_config_id` suffix branch is a harmless dead no-op rather than removing it).
- Repo-wide grep confirmed no other references to `BeFSSearch`, `_select_leaf_by_befs`,
  `_seed_befs_child_edge_stats`, `_befs_backup_from_node`, `_befs_open_map`, `_befs_config`,
  or `selection="befs"`/`selection=="befs"`/`selection: "befs"` in any `.py`/`.yaml`/`.slurm`
  file, except historical/descriptive comments that don't call removed code and were left
  alone as out of scope:
  - `slurm/pipeline/aux_trace_filter.slurm` ("switchback count from the BeFS greedy
    best-move trace") — a STUB (`exit 0`, no real logic) describing a generic
    best-move-trace-reading concept, not a call into `BeFSSearch`.
  - `src/analysis/tests/_cp_fixtures.py` (docstring mentions `_select_leaf_by_befs`,
    plus a live `replay_befs_expansion_order` helper) — only consumer is
    `test_cp_provenance.py::TestWriterAndPackCompat::test_replay_oracle_matches_on_scripted_tree`,
    which is already `@pytest.mark.skip`'d with its own note that the helper reimplements
    the *old* frozen-static BeFS rule and needs a rewrite before reuse. Left alone — not
    part of the scoped removal and already inert.
  - `src/cts/data/build_tree.py`, `src/cts/data/preprocess_gnn/teacher_targets.py`,
    `src/analysis/tests/test_engine.py`, `src/analysis/utils/tree_loader.py`,
    `slurm/pipeline/engine_analysis_pwin.slurm`, `slurm/pipeline/pipeline.yaml`,
    `config_minply15_maxply75.yaml` — assorted comments naming `befs1cp_md36` (a dataset/
    config name for trees already generated under the old BeFS path) or narrating the
    BeFS deprecation history; none call removed code.

**Before → after test counts**: immediately before this pass (i.e. with the
teacher_targets.py-only half of the removal already applied but `test_cp_provenance.py`/
`build_tree.py` not yet caught up), `pytest src/` showed **10 failed / 115 passed / 3
skipped** — all 10 failures were `ValueError: unknown selection 'befs'` (7 in
`TestBefsSelection`, 1 in the stale suffix test, 2 in `TestWriterAndPackCompat` via
`_scripted_example`). After this pass: **1 failed / 116 passed / 3 skipped** (4 subtests
also pass). The one remaining failure is
`TestWriterAndPackCompat::test_payload_round_trip_preserves_cp_features`, unchanged and
unrelated — it's exactly Part 2, bug #2 above (the v6 pretrain-format change drops
`cp`/`mate` columns, which this test explicitly asserts survive); still flagged for
whoever owns that decision, not fixed here. Bug #1 above (the BeFS tunnel-vision test) is
now gone from the suite entirely, since its test file was deleted along with `BeFSSearch`
itself — accounting for the one BeFS-related failure out of the original 3-failure
baseline that no longer applies.
