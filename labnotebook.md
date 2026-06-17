# Lab notebook — chess_analysis (CMC)

## Convention

Entries are **reverse-chronological** (newest first). Each day: one **summary line**, then:

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|

- **Description** — what was run, changed, or submitted (**Human** or **LMCOS** tag in bold where helpful).
- **Rationale** — why; what we expected.
- **Status / finding** — lead with one status emoji, then the outcome.
- **Reference** — link to the full report (e.g. [(R-A1)](reports/analysis-1-human-oracle.md)).

**Status:** ⬜ not started · ⏳ pending · ✅ done / pass · ❌ fail

**Workspace:** `human_analytics/` (DuckDB, RT/VOC figures) · `lmcos/` (`cts` pipeline, oracle analyses) · DB: `/scratch/gpfs/GRIFFITHS/hl4291/personal.db`

Index of all reports (including **open procedure steps**): [reports/README.md](reports/README.md).  
Pre-migration notebooks: [(R-ARCH-HUMAN)](reports/archive-human-analytics-notebook-legacy.md), [(R-ARCH-LMCOS)](reports/archive-lmcos-notebook-legacy.md).

---

## ⚙️ Running state (as of 2026-06-14) {#running-state}

> **150K depth-10 tree-gen LIVE — job `9745158`, gpu-short `--array=0-149%44` → `lc0_trees/`. THE faithful ysagiv replication.**

> ⚠️ **CORRECTIONS (2026-06-14, late) — earlier entries below were WRONG; flagged ~~struck~~/defunct:**
> 1. **ysagiv `max_depth` = 10, NOT 4.** Verified over **all 39,668** reference trees (global max depth 10, hard cap; depth-10 nodes unexpanded) + independent n=1000 (84% of trees go deeper than 4). My earlier "max_depth=4" was an **n=1 generalization error** (the code *default* is 4; one shallow tree maxed at 4).
> 2. **Node `value` = the RAW value-head, NOT a "1-ply best-child backup".** Verified `value==win−loss` over 55k nodes; root WDL ≠ best-child WDL. My earlier "1-ply" claim was wrong (it came from a startpos-only measurement; startpos is history-special-cased).
> 3. **The depth-4 (`9694864`) and depth-30 (`9717196`) runs are DEFUNCT and DELETED** — they used the wrong depth AND silently lacked `edge_wdl_targets` (v5 made it opt-in; the worker never opted in), so they had no child-WDL supervision. `build_tree.py` now passes `include_edge_wdl_targets=True`; the depth-10 run has valid edge targets (verified: finite, sum-to-1).

- **depth-10 replication** → `/scratch/gpfs/GRIFFITHS/hl4291/lc0_trees/` (`{index:06d}_root_{index}.pt`). Job
  **9745158**, 150×1000, `resume: true`. Config `lc0_trees_150k.yaml` (budget 96, **max_depth 10**, multipv 8,
  lc0 cuda, **`edge_wdl_targets` on**). Realized depth ~6 (matches ysagiv); ~11 s/tree on compute nodes → ETA ~10 h.
- **Verified ysagiv reference spec (4-subagent + own measurement):** `max_depth=10`; budget = **96 expansions**
  (the "96"/oracle-trace length, not total N ≈3700); node tuple `(value, wdl_win, wdl_draw, wdl_loss, wdl_var, prior)`
  with `value`=raw valuehead (win−loss), `wdl_var`=(w+l)−value²; `node_targets`=visit-weighted child-Q;
  `edge_wdl_targets`=per-edge WDL parent-flipped. Encoder pretrain = child-WDL CE (needs edge_wdl_targets);
  controller = frozen-encoder fitted-Q, BudgetedOracleConfig λ=18.537/p=2.8/maintenance-off/5 buckets×2.
- **Submit (the QoS fix):** `sbatch --qos=gpu-short --array=0-149%44 --export=ALL,CONFIG=slurm/configs/1_preprocess_data/lc0_trees_150k.yaml,SHARD_SIZE=1000,BASE_START=0,LANE_END=150000 slurm/1_preprocess_data/generate_dataset_shard_gpu_array.slurm`.
  **Do NOT pass `--partition`** (cluster rejects it; select via QOS). `gpu-test` = old stall (3-job cap); `gpu-short` = 44 GPUs/user, no group cap.
- **Progress check:** `find /scratch/gpfs/GRIFFITHS/hl4291/lc0_trees -maxdepth 1 -name '*.pt' | wc -l` (of 150000);
  `squeue -j 9745158 -h -o "%T" | sort | uniq -c`. (Use `find`, NOT `ls *.pt` — the glob is slow / overflows at 150K files.)
- **Inputs:** sample `…/lc0_trees/fens_sample_150k.txt` (150K, seed 43) + `…/lc0_trees/manifest.parquet`
  (`index, fen, fen_4field`), from the ground-truth pool `…/lmcos/fens.txt` via `cts.data.process_fens sample`.
  RT join key = the 4-field FEN against `processed_moves_nonzero`.
- **Superseded:** the old `tmp/human_trees_50k` (5,228 DB-sampled trees) was **deleted**; this run starts clean
  from the fens.txt sample. The in-process batched speedup was scrapped (~2×, encoding-bound —
  [R-BATCHGEN](reports/batched-tree-generation.md) NO-GO); generator is the faithful lc0-UCI `build_tree` with
  the ~1.5× repetition-scan fix.
- **Next once trees land:** U1.2 OSS↔RT join (`analysis/human_oracle_comparison.py`, subset-tolerant);
  then U1.3 refit; re-run [R-U3](reports/analysis-u3-baselines.md) on the new controller.

---

## 2026-06-17 {#2026-06-17}

Presentations folder restructured into a multi-deck Slidev project (mirrors `monkey_4iar/presentations`).

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Presentations — multi-deck restructure** | Page through focused, per-topic decks didactically instead of reading one long `.md`; share one component/CSS/figure base across decks; match the sibling `monkey_4iar/presentations` architecture | ✅ Flattened the single `presentations/lmcos-overview/` deck into `presentations/src/<deck>.md`. Each deck is a flat entry file; all decks share `shared/slidev-addon-base/` (addon: the 9 chess Vue diagram components + `global-bottom.vue` page-number footer + `vite.config.ts` that widens `server.fs.allow` for the figures dir), one root `style.css`, and one `public/figures` symlink. The Slidev project root is `src/`, which holds symlinks `shared → ../shared`, `style.css → ../style.css`, `public → ../public` so `@/shared/...`, `css: ./style.css`, and `/figures/...` resolve identically for every deck. | [README](presentations/README.md) |
| **Presentations — `main` deck migrated** | Preserve existing content + the 9 custom components | ✅ `lmcos-overview/slides.md` → `src/main.md` (only frontmatter change: added `addons: ["@/shared/slidev-addon-base"]`; kept `theme: default`, `css: ./style.css`, `math: katex`, `mdc: true`). Components moved into the addon's `components/` so they still auto-register on-slide. Custom `layouts/` deliberately NOT added — would override theme-default's `default` layout and risk altering main.md rendering. | [main.md](presentations/src/main.md) |
| **Presentations — per-deck launcher** | Run any deck by name | ✅ `scripts/run-deck.js` (adapted from `monkey_4iar`): maps `<deck>` → `src/<deck>.md`, discovers `src/*.md`, forwards extra Slidev flags, defaults to `main`, `dev` opens the browser. Root `package.json` scripts: `dev`/`build`/`export` → `node scripts/run-deck.js …`. Commands: `npm run dev`, `npm run dev modelrecovery`, `npm run build modelcomparison`, `npm run dev gazeanalysis -- --port 3030`. | [run-deck.js](presentations/scripts/run-deck.js) |
| **Presentations — skeleton decks** | Structure ready for content (no invented results) | ✅ `src/modelrecovery.md`, `src/modelcomparison.md`, `src/gazeanalysis.md` — frontmatter + title slide + a single "TODO — content pending" slide each. To be filled later. | — |
| **Presentations — deploy + cleanup** | Keep deploy infra; drop superseded per-deck files | ✅ `.npmrc`/`netlify.toml`/`vercel.json` moved to the presentations root (build the default `main` deck). Old `lmcos-overview/` `package.json`/`package-lock.json`/`vite.config.ts`/`README.md`/`.gitignore`/`.venv` removed (superseded by root equivalents). `.gitignore` keeps `node_modules/`, `dist/`, `components.d.ts` out of git. | — |

**Could not verify in-env:** no `node`/`npm` on this host, so `npm install` and `npm run dev -- main` were not executed. Wiring verified by close comparison to `monkey_4iar/presentations` (same `run-deck.js` arg-handling, same `@/shared/slidev-addon-base` addon mechanism, same `src/`-symlink resolution) and by checking that every `src/` symlink resolves and `run-deck.js` parses (brace-balanced). Pre-existing data note: 3 figures referenced by main.md (`log_movetime_histogram.png`, `correlation_matrix.png`, `oracle_stop_step_vs_human_rt.png`) are absent from `human_analytics/figures` — that predates this migration (figures regenerated by analysis scripts) and is unrelated to the restructure.

---

## 2026-06-16 {#2026-06-16}

human_analytics salvage + dashboard overhaul (skeptical audit → fixes → figure regen).

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Human — skeptical R² audit** | Verify the move-time/VOC methodology is clean | ✅ All findings reproduced on the 1M-row `pos_with_engine_eval`. Headline: **there is no R²/OLS layer** — every result was a bivariate Pearson r; true variance-explained is tiny (VOC R²≈0.009, MQ≈0.014, branching≈0.038 is the strongest single predictor). | — |
| **Human — entropy-VoI REMOVED** | A4 stopping-depth analysis was statistically degenerate | ✅ `compute_stopping_depth` softmaxed win-probs (~0.05 spread) at hardcoded β=1 → near-uniform → d* pinned to 1 (NaN correlations); "Approach A vs B" were byte-identical. Deleted code, tests, report, slides, figures; flagged the A4 day-rows defunct. | — |
| **Human — node-target subsystem REMOVED (lmcos)** | policy_drift was broken (softmax of raw visits → scale-sensitive; root-only) | ✅ Removed the whole auxiliary node-target path (policy_drift + value_gap + NodeTargets head/trainer/packing); unused in any live config, separate from the ysagiv child-WDL parity objective. −1450 lines; suite 229 passed. | — |
| **Human — MQ↔logRT is a confound, not "thinking hurts"** | Plot showed longer RT → lower MQ within each ply phase | ✅ r=−0.116; **partial r controlling toptwo+branching+ply+eval = −0.130 (does NOT attenuate)**; logRT adds ~1.5% MQ variance the engine features miss. Interpretation: RT is a better *latent-difficulty* sensor than engine summary stats — selection, not causation. Motivates the lc0-tree/VOC difficulty measures. | — |
| **Human — dashboard overhaul** | Make the surviving plots honest + readable | ✅ Dashboards reduced to a fixed **1×2: Quantile bins \| Quantile bins (by ply tertile)**; raw-trend + scatter panels (and the 100k-row scatter sample SQL) removed. voc_mq reports honest **R²** + MQ↔logRT across SQL ply tertiles (no causal claims). | — |
| **Human — VOC/toptwo zero-inflated binning** | ~⅔ of VOC is 0; plain ntile wastes bins on the mass | ✅ Added `Analyzer(zero_inflated, zero_threshold)`: lump `abs(x)≤thr` into one leftmost point, ntile the rest. thr=0.05 for VOC/toptwo → clean **positive** VOC↔logRT and **negative** toptwo↔logRT trends now visible per phase. | — |
| **Human — figures regenerated** | Reflect current code | ✅ All 10 Analyzer dashboards + histograms regenerated. `voc_budget` (engine-heavy scatter/corr) left as-is. Suite 29 passed. | — |
| **Human — OSS↔RT from lc0 trees** | Real per-tree value-of-computation vs Stockfish node-budget proxy | ✅ `tree_values_analysis.py`: per-tree budgeted-DP `optimal_stop_step` (OSS) joined to RT by 4-field FEN. 10k trees → r(OSS,logRT)=+0.12. OSS is **not stored** in raw trees (cost-model-dependent; derived from the oracle trace) → parallelized + slurm/tree_values.slurm. | — |
| **Human — plot-set lock-in + purge** | Settle the kept analyses; remove the rest | ✅ **FULL** (kept): ply×pInstant, log(MT) hist+normal-QQ, ply/clock/own-non-pawn-material/branching ×MT, MQ×MT. **SUBSET (lc0-tree, generated, not yet run at scale)**: OSS / VOC / Action Gap ×MT — all three now derived **from the lc0 trees** (`tree_values_analysis.py`), VOC = final_Q(best)−final_Q(shallow-best), Action Gap = top-2 final-root-Q gap. **Removed**: VOC_budget (Stockfish node-budget proxy) + script, clock_opp, total non-pawn material, all histograms except log-MT, the Spearman corr matrix. Renamed voc_mq_analysis→mq_analysis, ysagiv_oss_rt→tree_values_analysis. Docs + analysis.sh updated. | — |

**Salvage verdict:** descriptive RT dashboards + the engine VOC/MQ *computation* are sound; the missing piece is a real multivariate R² layer (logRT ~ branching+ply+clock+VOC, incremental variance) — the next analysis to build.

---

## 2026-06-14 {#2026-06-14}

FEN handling consolidated; batched in-process tree-gen built & validated against lc0 (R-BATCHGEN).

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **`process_fens` consolidation** — single source for the FEN pool (`build-pool`) + per-run sampling (`sample`); removed `sample_fens`/`validate_fens`/`export_human_fens`/`export_unique_fens` | One source of truth; fens.txt is ground truth | ✅ Committed `2f30bbc`. 4 scripts → `cts.data.process_fens` | — |
| **R-BATCHGEN plan** — batched in-process tree-gen, parity-gated; layered parity contract; JAX analysis | Route-b from R-U1-SPEED: 10–100× + native tree exposure | ✅ Report written; the only way to both speed up *and* own the tree/snapshots | [(R-BATCHGEN)](reports/batched-tree-generation.md) |
| **L1 — batched search loop** (`generate_trees_batched`, frontier-batched PUCT reusing legacy helpers) | Speed via batching across independent trees; per-tree order preserved | ✅ **Byte-exact to `build_pretrain_example`** under a fixed evaluator (T-replay). The load-bearing gate. | [(R-BATCHGEN §3)](reports/batched-tree-generation.md) |
| **Phase 2 — in-process net** (`NetEvaluator`, lczerolens + `leela2onnx` ONNX) | Replace batch-1 lc0-UCI oracle | ✅ Committed `3a2bf7d`. `re_baseline=False` reproduces lc0 **exactly** on midgame FENs | [(R-BATCHGEN §10)](reports/batched-tree-generation.md) |
| **3 lc0 quirks pinned** — ~~value=1-ply best-child valuehead minimax~~ (⚠️ **DISPROVEN — value is the RAW value-head**, see CORRECTIONS); priors at PolicyTemperature 1.359; history `fen_only`=lczerolens REPEATED | Bit-for-bit replication (user: replicate first, flip `re_baseline=True` later) | ✅ priors/history confirmed vs binary (priors d2d4 0.1806 vs 0.1804); ❌ the "1-ply" value pin was a startpos-only artifact (`value==win−loss` over 55k nodes ⇒ raw head) | [(R-BATCHGEN §2)](reports/batched-tree-generation.md) |
| **3 real bugs caught by the pinning test** — `encode_move(move, us)`; onnx2torch needs `.eval()` (BatchNorm batch-dependence); position specs (`\|\|moves\|\|`) need `split_position_spec` | Each would have silently corrupted the 150K dataset | ✅ All fixed; pinning test green | [(R-BATCHGEN §10)](reports/batched-tree-generation.md) |
| **Speedup go/no-go** — A100 bench (budget 16, extrapolated to 96) | The §8 ≥5× gate before scaling | ❌ **NO-GO.** raw `re_baseline=True` ~8 s/tree (**~2×**); faithful 1-ply ~125 s/tree (**~7× slower**). Bottleneck = Python 112-plane encoding (`to_input_tensor`), not the GPU. Below the 5× gate → not worth the build. | [(R-BATCHGEN)](reports/batched-tree-generation.md) |
| **Scrap + cleanup** (hl4291: "juice isn't worth the squeeze") | Abort route-b; keep the record | ✅ Removed `batched_gen` module + tests + `smoke/`; **kept `process_fens`** (independent FEN-source win) + R-BATCHGEN as a documented NO-GO. 252 tests collect. | — |
| **lc0-UCI quick win: repetition-scan guard** — `terminal_value_from_board` skips `can_claim_threefold_repetition` when `len(move_stack) < 7`, else exact `claim_draw=True` | R-U1-SPEED flagged the per-child 3-fold probe as 23% of wall but never applied it; a claimable 3-fold needs ≥7 plies, so it's impossible while shallow. **Gated on actual history length, not the `max_depth` hyperparameter** (hl4291) → stays correct if max_depth is ever raised (deep boards fall back to the exact check). | ✅ **~1.5× end-to-end** (16.87 → **11.1 s/tree**, A100, budget 96). Under the real **max_depth=10** regime most boards are still shallow (realized depth ~6) so the fast path dominates; boards that reach ≥7 plies correctly fall back to the exact threefold check (guard is gated on history length, not max_depth). Equivalent: 0 mismatches over 600+ boards incl. fifty-move boundary, deep fallback branch, and a real 7-ply threefold; 22.6× faster check; tests pass. Remaining cost = irreducible per-child valuehead NN eval (no faithful quick win, route-b scrapped). | [(R-U1-SPEED)](reports/u1-tree-gen-speedup.md) |
| ~~**150K tree-gen LAUNCHED** — job 9694864 (depth-4)~~ | — | ❌ **DEFUNCT — deleted.** Wrong regime (depth-4, not ysagiv's 10) AND no `edge_wdl_targets` (silent v5 opt-out → no child-WDL supervision). Superseded by job 9745158. | — |
| ~~**ysagiv alignment "verified"** — claimed max_depth=4 + value=1-ply~~ | — | ❌ **WRONG — corrected** (see [running-state CORRECTIONS](#running-state)). ysagiv max_depth=**10** (all 39,668 trees), value=**raw** value-head (not 1-ply). Error was n=1 generalization. | — |
| ~~**depth-30 variant LAUNCHED** — job 9717196~~ | — | ❌ **DEFUNCT — cancelled/deleted.** No `edge_wdl_targets`; depth-30 also overshoots ysagiv's cap of 10. | — |
| **ysagiv spec re-verified + depth-10 replication LAUNCHED** — 4 skeptical subagents + own measurement (after the n=1 errors) | Pin the reference rigorously and generate the correct set | ✅ **max_depth=10** (all 39,668 trees + n=1000), **value=raw** valuehead, budget=96 expansions, node tuple as documented; **`edge_wdl_targets` were silently dropped → FIXED** (`build_tree.py` `include_edge_wdl_targets=True`; verified finite/sum-1 on a compute-node canary). Depth-10 run **9745158** `--qos=gpu-short --array=0-149%44` → `lc0_trees/`, ~11 s/tree (compute node), realized depth ~6. | [(R-U1-SPEED)](reports/u1-tree-gen-speedup.md) |

**Next:** depth-10 run (9745158) is THE replication — let it finish (`resume:true` fills gaps); skeptical repo-wide code/doc audit + ysagiv same-root parity in flight → **implement the PUCT-stability filter** (`trees_unfiltered`→`trees_filtered`, still unbuilt) → train GNN+MC (child-WDL encoder needs the now-restored `edge_wdl_targets`) → U1.2 OSS↔RT join. Route-(b) speedup parked.

---

## 2026-06-10 {#2026-06-10}

Repo audit + scratch cleanup; QoS diagnosis; full FEN export.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Repo audit** — jordan vs main; code cleanliness; script inventory | Pre-PR hygiene check | ✅ 60 commits ahead of main; no orphan scripts (lmcos/analysis is a documented toolkit); two untracked helper modules (`_data.py`, `_plots.py`) found mid-refactor → committed | — |
| **Commit `16aefca`** — consolidate `_data.py` / `_plots.py`; converged_expansions figures; multi-GPU slurm lane | Five analysis scripts had broken imports on a clean checkout (helpers untracked) | ✅ Working tree clean; `jordan` PR-ready | — |
| **QoS diagnosis** — why 50K stalled at 5,228/50,000 | "~3 effective GPUs" logged 2026-06-05 was a misdiagnosis | ✅ Root cause: all gen jobs ran on `gpu-test` (MaxJobsPU=3) not `gpu-short` (MaxJobsPU=44). `griffith` has no GrpTRES GPU cap. Fix: switch QoS on next restart. | — |
| **Scratch cleanup** — removed legacy.db, eval_results, old metacontrol data, bench artifacts | ~10 GB freed | ✅ Scratch now 18 GB (personal.db) + 1.3 GB (tmp) | — |
| **Full FEN export** — `SELECT DISTINCT fen FROM processed_moves_nonzero` | Canonical FEN pool for tree-gen and future sampling | ✅ **110,505,438 unique FENs**, 5.4 GB → `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens.txt`; script at `lmcos/fens/export_unique_fens.py` | — |

---

## 2026-06-05 (execution) {#2026-06-05-exec}

Sprint action #1 — three-engine timing smoke — run; CPU lane unblocked; U3 baseline tests added.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **U1.0 smoke** — Lc0-GPU / Lc0-CPU / Stockfish on 100 warm human FENs (budget 96) | Pin per-tree cost; decide engine before 50K | ✅ Lc0-GPU **16.87 s/tree** (A100, steady); Lc0-CPU **~738 s/tree** (4-core, pure-CPU node); SF 0.45–11.65 s/pos (d12–20), 0.026 s @100k nodes | [(R-U1)](reports/u1-engine-timing-smoke.md) |
| **CPU-lane gate** — lc0-blas on a pure-CPU node | The 50K plan is CPU-led | ⚠️→✅ **Blocked then FIXED:** CUDA-linked lc0 needs `libcublas.so.12`; absent on CPU nodes → crash. Fix: `LD_LIBRARY_PATH` → venv `nvidia/*/lib` (no rebuild). Lane viable. | [(R-U1)](reports/u1-engine-timing-smoke.md) |
| **Feasibility** — 50K both lanes | Confirm timing | ✅ Combined ≈ 2,350 trees/hr → **50K in ~21 h**. Decision: run Lc0 50K, CPU-led + 3 GPUs. | [unify.md §7](unify.md) |
| **U3 baselines** — gain-depth-only rule + `test_budgeted_baselines.py` | Build the independent comparison thread while data generates | ✅ 12 data-independent tests pass; geometric baseline **dropped** (silly); semi-smart variants brainstormed | [unify.md §U3](unify.md) |
| **U1.1 launch** — 50K human FENs (seed 43); two-lane gen, in-repo configs + array scripts | Reunify datasets; CPU-led + 3 GPUs | 🚀 Export ✅ (50K + manifest). GPU lane `[0,14000)` cuda (array 0-13%3, 1000/shard); CPU lane `[14000,50000)` blas on `cpu` partition w/ libcublas fix (array 0-899%350, 40/shard). One output dir keyed by global index. **Canary:** GPU ✅ 10/10; CPU launched clean on `no_gpu` node → **auto-fire** full run on pass (~21 h). | [unify.md §U1.1](unify.md) |
| **CPU lane dropped** — node-speed variance (738→1338 s/tree on a slow node) risks fixed-wall timeouts | Safer to stay GPU-only | ⚠️ Pivot to **GPU-only**; CPU revisited later | [unify.md §U1.1](unify.md) |
| **Tree-gen speedup investigation** — profile + 3 routes + pooling impl/validation | Cut ~16.87 s/tree before 50K spend | ❌ **No faithful quick win.** lc0 ~1 ms/eval; 80% = `valuehead` child-eval loop (compute-bound). Pooling **slower** (172 vs 133 s/tree) **and** non-identical → reverted. Route (a) batched search gives only scalar V (not the WDL triple). Route (b) net-reimpl = only real lever (logged). lc0/CUDA **deterministic**. | [(R-U1-SPEED)](reports/u1-tree-gen-speedup.md) |
| **U1.1 fire** — faithful 50K, GPU-only | Generate the reunified trees | 🚀 Job `9282935`, 50×1000-FEN shards, `gpu-short %20`, `resume:true`; coexists with an unrelated `prod_full` job (untouched). ~3 d at low concurrency, faster if GPUs free. | [(R-U1)](reports/u1-engine-timing-smoke.md) |
| **U2 wiring + lc0 smoke** — `--nodes-deep/--nodes-shallow` for lc0 node-budget VOC | Replace Stockfish with lc0 (study standard) | ✅ Wired (lc0 default **96/1**; SF depth path unchanged). Smoke 30 pos: **~20 pos/s/GPU**, all VOC/toptwo/mq non-null + correct signs; **`nodes_shallow=1` works** (a_shallow = policy move). lc0's *native* `go nodes 96` makes U2 fast → **Option A** (search, coarse-shard); route-b not needed for U2. | [unify.md §9d](unify.md) |
| **U3 model comparison** — controller vs baselines on existing diagnostics (no GPU/new trees) | Quantify controller lift; parallel thread | ✅ On 30,630 episodes (`subtree_weighting_root_budget`): **controller avg-regret +0.026**; exact-stop **63%**. | [(R-U3)](reports/analysis-u3-baselines.md) |
| **U3 semi-smart baselines** — value-plateau + fixed-fraction-of-budget (+4 tests, 16 total) | Stronger non-trivial anchors | ✅ **Fixed-fraction (0.25 budget) is best baseline** (+0.069, beats gain-depth +0.148 & value-based rules) → stop is budget-structure-driven. **Value-plateau under-performs gain-depth** (+0.241; smoothing delays the early stops the oracle wants). Controller still wins (~2.6×). | [(R-U3)](reports/analysis-u3-baselines.md) |
| **U3 cost-vs-value decomposition + slides** — why fixed-fraction competes | hl4291 hypothesis: budget over-weighted; regret weak | ✅ Oracle stops **bimodal**: ~49% value-driven (cost≈0), ~28% **cost-forced** (budget depleted). Stops **early** (median 3.8% of budget; corr 0.41). Regret weak among good rules → **exact-stop** discriminates (0.63 vs 0.19 vs 0.10). Levers: `time_lambda`, budget dist. 2 figures added to deck. | [(R-U3)](reports/analysis-u3-baselines.md) |

---

## 2026-06-05 (planning) {#2026-06-05-plan}

Drafted the sprint North Star — **The Great Reunification** — and restructured the deck.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Plan** — `unify.md` North Star: state verification + U1/U2/U3 sub-plans, tests, contingencies, feasibility, open questions | Single reference for the reunification sprint | ✅ Verified repo matches the brief except the 3 open problems; `human_trees_10k/` empty (shard-sizing, not cost). 5 decisions for hl4291 in §7. | [unify.md](unify.md) |
| **Timing smoke** — 20-FEN tree-gen benchmark (budget 96), A100 | Pin per-tree cost before 10K/50K/100K spend | ✅ GPU **~16–41 s/tree** (0.024–0.062 roots/s); CPU/blas **~14 min/tree**. 50K < 1 day; 100K ≈ 1.5–2.4 days. **Feasible.** | [unify.md §5](unify.md) |
| **Slides** — prepend reunification deck; move prior framing + A0–A4 + architecture to appendix | Lead with the reunification narrative | ✅ 14 new slides; appendix preserved | [slides.md](presentations/lmcos-overview/slides.md) |
| **Decisions locked** (hl4291) | Resolve the 5 sprint questions | ✅ Tier **50K**; engine-swap profiling folded into a **3-engine smoke** (Stockfish / Lc0-CPU / Lc0-GPU) = sprint action #1; **depth 96 kept** (reduce epochs before depth); OSS budget = canonical `BudgetedOracleConfig()` (5 buckets 1–120, 2/bucket, **maintenance off**, λ=18.537/p=2.8, trees→96). | [unify.md §7](unify.md) |
| **Resource reality** (hl4291) | Correct GPU concurrency | ⚠️ Effective **~3 GPUs** (not the QOS 20 cap), but **CPU-rich** (~1,400 cores/`short`). Strategy flips to **CPU-led tree-gen**: 50K ≈ **1.1–1.4 d** combined; GPU-only would be ~5–8 d. **Gate:** lc0-blas must launch on a pure-CPU node (`libcublas` dynamic-load) — smoke confirms; else build a Stockfish provider. | [unify.md §5](unify.md) |

---

## 2026-06-05 {#2026-06-05}

A0a corrected and rerun on all 39,668 trees; definitions, features, and code audited.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **LMCOS A0a** — corrected `oracle_stop_step` + features; all 39,668 trees; scatter plots | Two bugs fixed; rerun at full scale (missed half the shards with old glob) | ✅ **4/4 RT directions** correct. gain_depth r=**+0.233** ✓; toptwo r=**−0.290** ✓; branching/material r≈0.01 (tiny, correct dir). r(oss, ce)=**0.525**. Branching/material near-zero expected: oracle driven by value landscape (toptwo/gain_depth), not structural complexity. 72/72 tests. | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
| **LMCOS A0b** — minimal MLP (4 features → MSE → target_advantage) | Establish minimal-MC anchor with correct oracle labels | ✅ val sign acc **54.6%** (↓ from invalid 86.4%); exact stop **5.2%**; r(pred,oracle) **+0.291**. Sign acc oscillates during training (loss ↓ but zero-crossings unstable) — features lack discriminative boundary. GNN encoder confirmed essential. | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
| **LMCOS code audit** — pack-path oracle labels; shared `board_tree_features.py`; removed analysis-layer oracle wrappers | Single source of truth: `pack.py` + `oracle.py` | ✅ Done | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
| **LMCOS A1 smoke** — 497 clean trees (1K job cut at 1h wall); corrected bugs from A0a in `human_oracle_comparison.py`; generated 10K FENs; submitted 20-shard 10K job | Smoke confirms direction before 10K spend | ✅ r(oss, log RT) = **+0.091** ✓; stale-tree guard added; 10K jobs 9266775–9266794 running | [(R-A1)](reports/analysis-1-human-oracle.md) |

**Two bugs corrected in original A0a (2026-06-03 results no longer valid):**  
(1) halt_rewards used `oracle_root_q_trace[s, best_idx[s]]` (evolving MCTS Q-estimates) instead of `oracle_root_q_trace[-1][best_idx[s]]` (teacher's fixed Q-values = `oracle_final_root_q_values`).  
(2) gain_depth used a mixed-regime formula: final Q for the numerator, evolving Q at first nonzero step for the denominator. Corrected to native Lc0 VOC = `Q_final[best_idx[-1]] − Q_final[best_idx[1]]`.

---

## 2026-06-04 {#2026-06-04}

Notebook + reports restructure; A1 1K trees done; ~~A4 entropy VoI at scale~~ (**removed 2026-06-16 — see below**); A2 tiny-GNN pretrain started.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Docs restructure** — root `labnotebook.md` + `reports/`; drop `proposed_next_steps.md` | Single chronology + procedure checklists for open work | ✅ Open steps live in active `R-*` reports | [reports/README.md](reports/README.md) |
| **LMCOS A1** — human FEN tree smoke (644 FENs, budget 96) | Same-position oracle vs RT | ✅ **644/644** `.pt`; comparison script + 10K + plots still open | [(R-A1)](reports/analysis-1-human-oracle.md) |
| **LMCOS A2** — tree-stats + tiny GNN scratch (Config D) | Skip 1-day pretrain if small encoder suffices | ⏳ Config D YAML open; child-WDL pretrain job **9215254** started | [(R-A2)](reports/analysis-2-minimal-model.md) |
| ~~**Human A4** — entropy VoI stopping (SF multidepth, 10K CPU job)~~ | — | ❌ **REMOVED 2026-06-16 (statistically degenerate).** Skeptical audit showed `compute_stopping_depth` softmaxes win-probs (~0.05 spread) at hardcoded β=1 → near-uniform → d* pins to 1 (mean d* collapses 2.46→1.00 as θ rises; r(d*,logRT) null/NaN), and "Approach A vs B" were byte-identical. Code, tests, report, slides, and 10K outputs deleted. | — |
| **LMCOS A3** — SF ELO 2000 oracle | Only if A1 Lc0 oracle mismatches humans | ⬜ On hold until A1 matched-position r | [(R-A3)](reports/analysis-3-weaker-engine.md) |
| **Repo cleanup** — scratch + orphan tests | Free disk | ✅ Paths logged | [(R-CLEANUP-0604)](reports/archive-2026-06-04-cleanup.md) |

---

## 2026-06-03 {#2026-06-03}

Analysis 0 initial run; theoretical framing for stopping proxies. **A0a results superseded by 2026-06-05 corrections.**

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **LMCOS A0a** — initial run, 5K trees, budget 43 | Directional match before human-FEN spend | ❌ Results invalid — two bugs in halt_rewards and gain_depth formula. See 2026-06-05. | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
| **LMCOS A0b** — minimal MLP vs GNN+MC sign accuracy | Test GNN necessity | ❌ Invalid (wrong halt_rewards); correct re-run ⬜ **todo** | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
| **Human theory** — stopping proxies, chasing tails, E[ΔUC] | Claims A/B/C | ✅ Documented | [(R-THEORY)](reports/human-theory-stopping.md) |

---

## 2026-06-02 {#2026-06-02}

Human pipeline cleanup, VOC/MQ engine stack, 100K Stockfish eval.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Human structural cleanup** — tests, hist bin fix | Co-locate tests; clean `personal.db` | ✅ Figures regenerated | [(R-VOC-MQ)](reports/human-voc-mq.md) |
| **Human VOC/MQ code** — unified eval pipeline | Russek-style metrics | ✅ 9 unit tests pass | [(R-VOC-MQ)](reports/human-voc-mq.md) |
| **Human 100K eval** — SF depth 5/1 | Scale VOC–RT | ✅ r(log RT, VOC)=**+0.097** | [(R-VOC-100K)](reports/human-voc-mq-100k.md) |

---

## 2026-05-29 {#2026-05-29}

LMCOS repo layout + stage-4 controller ablation harness.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **LMCOS layout** — `cts.*`, `slurm/<stage>/` | Stage-owned paths | ✅ `submit_configs.sh` | [(R-LMCOS-STAGE4)](reports/lmcos-stage4-ablation.md) |

---

## Prior work (before 2026-06-02) {#prior}

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Human dataset** — Lichess 10+0, DuckDB | RT / engine foundation | ✅ 135M nonzero-RT moves | [(R-HUMAN-DATA)](reports/reference-human-dataset.md) |
| **Human move-time dashboards** | Baseline RT | ✅ Done | [(R-MOVETIME-PRIOR)](reports/reference-move-time-prior.md) |
| **Human follow-ups** — E[ΔUC], Russek filters, full VOC loop | Deferred during A0–A4 | ⬜ See backlog | [(R-HUMAN-BACKLOG)](reports/reference-human-backlog.md) |
| **LMCOS history** — Apr–May 2026 | Encoder / topology arc | ✅ Archived | [(R-ARCH-LMCOS)](reports/archive-lmcos-notebook-legacy.md) |
