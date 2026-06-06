# The Great Reunification — sprint North Star

**Status:** planning · drafted 2026-06-05 · owner hl4291 (with ysagiv)
**Scope:** unify the human behavioral dataset and the lmcos normative pipeline onto a single
set of positions, establish the normative-vs-human upper bound, and open the model-comparison
thread. **This document is the reference for the sprint; sub-plans below are the work items.**

Companion docs: chronology in [`labnotebook.md`](labnotebook.md); stable write-ups in
[`reports/`](reports/README.md); deck in
[`presentations/lmcos-overview/slides.md`](presentations/lmcos-overview/slides.md).

---

## 0. State verification (does the repo reflect the story?)

Checked the description in the sprint brief against the repo on 2026-06-05. **It matches, with
three caveats that are exactly the open problems.**

| Claim in the brief | Reality in repo | Verdict |
|---|---|---|
| Model = GNN (GRU cell, up/down sweeps, child-attention) + ChildWDLEncoder head + MC MLP readout | `src/models/{gnn,mc,tree_mha}.py`; child-WDL targets in `preprocess_gnn/teacher_targets.py`; `train/{gnn_pretrain,controller_train}.py` | ✅ present |
| Trained on ~40K filtered FENs, tree snapshots for GNN nodes + MC advantages | A0a ran on **39,668** `filtered_shard_*` trees; oracle DP in `preprocess_mc/oracle.py`; packing in `preprocess_gnn/pack.py` | ✅ present |
| Engine = Lc0; weighted child-WDL loss; BCE sign penalty (λ); PUCT filter of trivial FENs | `core/providers/lc0.py`; `controller_packed_lambda5_*` shards exist; PUCT stability filter | ✅ present |
| Human = ~2M games, Stockfish, 0-parameter; Ply / Branching / OwnPieces / Gain / ActionGap | 1.97M games, `processed_moves_nonzero`; SF depth-5 eval; dashboards in `human_analytics/` | ✅ present |
| Findings: branching is a big, partly-orthogonal RT driver; gain (Russek) replicates | A0a + movetime dashboards + A4 all consistent | ✅ present |
| **Datasets are unified on the same positions** | **No.** `human_trees_10k/` is **empty**; only a 497-tree A1 smoke exists (r(OSS, logRT)=+0.091) | ❌ **open — U1** |
| **Gain in human data uses Lc0** | **No.** Human gain/VOC uses **Stockfish depth 5** | ❌ **open — U2** |
| **Model is compared against alternatives** | **No.** One model only; `tree_stats_baseline.py` is the only partial baseline | ❌ **open — U3** |

**Why `human_trees_10k/` is empty (root cause, now understood):** tree generation costs
**tens of seconds per position** on an A100 (see §5). The 10K job was sharded as
500 FENs × 1 h walls; at ≥16 s/tree a 500-FEN shard needs **2.2–5.7 h**, so every shard hit
the wall before writing. The failure was **shard sizing, not a fundamental cost** — see §5 for
the corrected sizing. A 20-FEN timing smoke (`bench_trees_gpu/`, `bench_trees_cpu/`,
`bench_configs/`) was already run on 2026-06-05 and gives the numbers this plan is built on.

---

## 1. The North Star (the argument, stated precisely)

> We began with a simple question: **how should a meta-controller decide when to stop thinking
> in chess?** The unit of decision is a search tree. **Input = tree; output = continue / halt.**

**Background.** In earlier work (Russek et al.) the *value of computation* — `gain =
V_deep(a_deep) − V_deep(a_shallow)` — tracks human reaction time. The reunification asks whether
a *normatively optimal* stopping rule, and a *learned* controller that approximates it, also
track human RT **on the very same positions**.

### 1A. The model — greatest possible world, then the constraints

**Ideal.** A GNN + readout (MC) head, trained by self-play to win under a cost that grows with
tree size; the inner planning loop is Lc0's MCTS+PUCT; RL teaches it the **optimal stopping step
(OSS)**.

**Constraints we hit, and the solution to each:**

| Constraint | Solution |
|---|---|
| GNN root encoding is a sparse bottleneck — `z_root` may learn nothing | **Auxiliary supervision:** `ChildWDLEncoder` reads `GNN(x_root)=z_root`, then predicts each child's search-encoded WDL from `(z_root, child_pos)`. Forces `z_root` to be informative. Needs edge data: input `(parent, child)` → target search-encoded child WDL. |
| Reward is a sparse signal for meta-control | **Dense target:** once `z_root` is good, train MC to map `(z_root, budget β)` → **advantage A**. Advantage is dense and per-step. |
| Advantage needs the whole tree's history | **Tree snapshots + DP.** For tree `T_t`: `V_halt = V_deep(a_t)`; `V_continue = V_{t+1} − cost(t+1, β)`; `V_t = max(V_halt, V_continue)`. Solved by backward DP. Snapshots `T_0…T_max` come free while growing the tree with MCTS+PUCT — so GNN data (nodes at `T_max`) and MC data (snapshots `T_0…max`) are **co-located in one artifact.** |
| Most leaves are unexpanded → child-WDL target is trivial | **Weighted loss** by subtree size; trivial children contribute ~0. |
| Advantage matters most near the decision boundary | **BCE penalty on the *sign* of advantage**, weight λ. |
| Many FENs need no further thought | **Filter them out** (PUCT stability filter). |

**Current model pipeline (built, working):** data = ~40K filtered FENs with `T_0…T_max`
snapshots → GNN child-WDL-supervised + MC advantage-supervised (MSE + λ·BCE); engine = Lc0;
metrics = child-WDL CE curves, MC advantage MSE/BCE curves, greedy-stop (first `A<0`) vs DP-OSS.
It learns well across a wide variety of positions.

### 1B. The data — greatest possible world, then the constraints

**Ideal.** Take the GNN trained *only* on self-play and show its OSS is strongly correlated with
human thinking time (`OSS ↔ logRT`). Humans, the model, and the normative target all align →
evidence for **resource rationality** and a compelling model application.

**Constraints we hit, and the solution to each:**

| Constraint | Solution |
|---|---|
| Time-control / skill confounds | One control, **60+0 (10-minute)**, no berserk, a fixed Elo tranche (≥2000) |
| Don't want conclusions to hinge on the model | **Zero-parameter analyses first** — show humans behave sensibly without any model |

**Zero-parameter findings:** Ply, Branching, Own pieces, Gain (Russek), and (negative) action
gap all track RT — with **Branching** playing a large role that Russek et al. do **not** explain
and that is largely **orthogonal** to gain.

**Current human pipeline (built):** ~2M games filtered to 60+0/≥2000; engine = Stockfish
(convenience); 0-parameter; metrics = Ply / Branching / OwnPieces / Gain / ActionGap. Solid,
with surprising results we want validated.

---

## 2. The problems (what this sprint fixes)

1. **URGENT — data mismatch (generation).** Human filtering and model-training FENs are not the
   same set. We must run the model's data-generation pipeline (grow trees, save snapshots,
   compute targets) **on the FENs from the filtered human data**, then compare the *target* OSS
   (no model training) with human RT. This is the **upper bound** on how well any normatively
   trained model can describe human behavior. **Most important deliverable.** → **U1**
2. **Engine mismatch in human gain.** Human gain uses Stockfish; Lc0 is the study standard.
   Replicate Russek's gain↔RT with Lc0. → **U2**
3. **IMPORTANT — model comparison.** We have one model and no baselines. Profile alternatives
   and lesions on the existing shards; this can run **concurrently**. → **U3**

---

## 3. Sub-analysis plans (the work, in parts)

Each part lists **goal · data · method · tests · contingencies · independence**. Reports get
`R-U*` IDs under [`reports/`](reports/README.md); chronology in the lab notebook.

### U1 — Reunify the datasets (URGENT, the critical path)

The spine of the sprint. Four steps; U1.0 is essentially done.

#### U1.0 — Timing smoke (first pass DONE; clean three-engine re-run is sprint action #1)

- **Goal:** pin per-FEN tree-generation cost to one number, and decide the engine stack, before
  the 50K spend.
- **First pass (done):** 20 benchmark FENs, budget 96, multipv 8, max_depth 4, A100
  (`bench_configs/{gpu_cuda,cpu_blas}.yaml`, `slurm/1_preprocess_data/benchmark_tree_gen.slurm`).
  - **Lc0-GPU (cuda):** 20 trees in 322 s (slurm log) vs 814 s (time-wrapper run) →
    **≈ 16–41 s/tree** (0.024–0.062 roots/s). Spread = warmup vs steady state; not yet pinned.
  - **Lc0-CPU (blas, same host):** **1 tree in 833 s ≈ 14 min/tree** (~20–50× slower). The ysagiv
    lc0 binary is CUDA-linked and needs `libcublas` even for `blas` (two `bench-cpu` runs failed
    on a missing `libcublas.so.12`; the slurm wrapper now injects `LD_LIBRARY_PATH`).
- **Clean re-run (sprint action #1 — decisions 2+5 combined):** a **three-engine** steady-state
  smoke on **≥100 warm FENs** (exclude warmup), profiling per-tree cost for:
  1. **Lc0-GPU** (production path) — pins the 16-vs-41 s number for the §5 tier math.
  2. **Lc0-CPU** (blas) — quantifies the CPU-overflow lane.
  3. **Stockfish (CPU)** — doubles as the **engine-swap cost/benefit profiling** the brief
     requires before any all-SF migration (decision 2). SF is CPU-only, so it does not contend
     with Lc0-GPU and can run concurrently.
- **Deliverable:** a one-table report (`R-U1`) of s/tree × engine, plus the §5 tier estimates
  recomputed from the pinned Lc0-GPU number, plus a go/no-go on whether SF is competitive enough
  to be worth the migration discussion.
- **Verdict vs the brief's "a few minutes" smoke target:** a 20-FEN job is ~5–14 min (fine); the
  binding signal is per-tree cost. A 100-FEN Lc0-GPU smoke ≈ 27–68 min — acceptable for a
  one-time pin.
- **Speed contingencies (in priority order, per decision 3):** (a) CPU overflow in parallel with
  GPU (§5); (b) **reduce controller epochs** before touching depth; (c) only then depth 96→64;
  (d) Stockfish swap (requires the §U1.0 profiling above + sign-off).

#### U1.1 — Generate oracle trees on human FENs at scale

- **Goal:** trees + snapshots + DP targets for the **committed 50K** human-FEN set (extensible to
  100K).
- **Data in:** human FENs exported by `analysis/export_human_fens.py` from
  `processed_moves_nonzero` (60+0, ≥2000, ply 15–75, opp clock ≥ 60 s). 10K already exported:
  `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_10k.txt` (+ `_manifest.parquet`). For 50K,
  re-export with the same filter and a fixed seed.
- **Data out:** `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_50k/` (`.pt` per FEN,
  `resume: true`).
- **Method:** `cts.data.build_tree` (`generate-dataset`), **Lc0-GPU, depth/budget 96** (decision 3,
  regime-faithful). Packing uses the **canonical `BudgetedOracleConfig()` defaults + bucket
  sampling** of §7a so episodes are format-identical to `controller_packed_*_nomaint_no_xaba`.
  **Re-shard per §5** (the fix for the empty-dir failure): `gpu-medium` QOS, ~20 fat shards
  (~2.5K FENs/shard for 50K), walls = `ceil(FENs_per_shard × pinned_s_per_tree)` × 1.3.
- **Tests:**
  - Schema/identity guard: `root_position_spec == manifest full_fen` per tree (already enforced
    at load in `human_oracle_comparison.py`; reuse to reject stale trees).
  - Completeness: `count(*.pt) == len(FENs)` per tier; `resume` re-fills gaps.
  - Smoke parity: oracle invariants from A0 (`halt_rewards[i] == Q_final[best_idx[i]]`,
    `R_halt ≤ R_continue`, `OSS ≤ converged_expansions`) must hold on a 1K sample.
- **Contingencies:** depth 96→64 (≈⅓ faster, §5); CPU overflow shards under `short` QOS;
  Stockfish engine (requires §U2 CP→WDL path; needs user sign-off, see §7).
- **Independence:** gated only by U1.0 (done) and a clean per-tree number. Blocks U1.2/U1.3.

#### U1.2 — Normative upper bound: target-OSS ↔ human RT (most important result)

- **Goal:** correlate the **DP target OSS** (no model training) with human `log(RT)` on matched
  positions — the ceiling on what a trained controller can explain.
- **Data in:** U1.1 trees + `personal.db` (`log(RT)`, clock, branching, material, gain, action
  gap) joined on FEN.
- **Method:** `analysis/human_oracle_comparison.py` (already corrected post-A0a, already uses
  `BudgetedOracleConfig()` defaults) → `compute_budgeted_oracle()` → `oracle_stop_step`.
  **Headline at full budget 96** (faithful to A0a/A1), with a **per-bucket robustness sweep**
  (§7a) as a secondary check. Pearson/Spearman vs `log(RT)`; partial correlations controlling for
  ply/branching; plots A (OSS vs logRT), B (OSS vs board features vs human-r), C (predicted-stop
  vs OSS once U1.3 exists).
- **Tests:** direction `r(OSS, logRT) > 0`; significance threshold by n (`|r| > ~3/√n`);
  cross-check that the A1 497-tree smoke (+0.091) is recovered as a subset; stability across the
  three tiers (sign and rough magnitude should not flip).
- **Contingencies:** if `r ≈ 0`, that is itself the headline (and triggers **A3/SF2000**,
  `reports/analysis-3-weaker-engine.md`, the strength-mismatch test). If `r` is strong, the
  one-paper resource-rationality story is on.
- **Independence:** depends on U1.1. Can complete and be reported **before** U1.3.

#### U1.3 — Refit the model (GNN + MC) on the human FENs

- **Goal:** retrain on the unified data; report child-WDL CE and MC advantage MSE/BCE; greedy
  stop vs DP-OSS; eventually predicted-stop ↔ human RT (Tier C).
- **Data in:** U1.1 packed episodes (co-located GNN nodes + MC snapshots).
- **Method:** `train/gnn_pretrain.py` then `train/controller_train.py` (or joint
  `unfreeze_encoder`). MC: **≤ ~1000 steps (~1 epoch)** per the brief.
- **Fitting smoke (profile first, scale never-yet):** tiny GNN (Config D — `d_embed=16`,
  `d_message=16`, `n_heads=1`, `hidden_dim=32`, 1 layer), representation 16–32 not 128, on
  10–20 shards; **plot loss curves live** and eyeball the decrease. Target: GNN converges
  "good enough" in ≤ ~2 h GPU; MC ≤ ~1 h. Speed over perfection for the first loop.
- **Tests:** `tests/test_minimal_model.py`-style shape/param-count/no-NaN/reproducibility checks;
  child-WDL CE below the trivial-baseline (predict-prior) loss; MC sign-accuracy beats the A0b
  minimal-MLP anchor (54.6%) and approaches the packed GNN+MC baseline (90.1%).
- **Contingencies:** if tiny GNN underfits, step Config D→C→B (widen gradually); if pretrain is
  the bottleneck, keep the two-stage materialized `z_root` cache; if MC won't separate, raise λ.
- **Independence:** depends on U1.1; **parallel to U1.2** (different consumers of the same trees).

### U2 — Lc0 gain in the human data (replace Stockfish)

- **Goal:** replicate Russek gain↔RT with **Lc0** so the human VOC matches the engine used
  everywhere else.
- **Data in:** human FENs (same export as U1.1) → Lc0 evaluations →
  `lc0_evaluations` in `personal.db` → join to `processed_moves` via
  `build_selected_moves_with_engine.py`.
- **Method:** reuse the engine-eval harness
  (`human_analytics/slurm/scripts/script_engine_eval.py`, `engine_eval.sh`) with an Lc0 worker;
  compute `gain = V_deep(a_deep) − V_deep(a_shallow)` in WDL space; recompute the RT correlations
  and the correlation matrix.
- **Tests:** Lc0 gain reproduces the **sign** of the SF gain↔RT result (currently SF r≈+0.10);
  Lc0-vs-SF gain agree in rank on a shared sample (Spearman > 0, sanity); branching stays a large
  orthogonal driver under Lc0 gain too.
- **Contingencies:** if Lc0's GPU dependence makes whole-dataset eval impractical, this is the
  trigger for the **engine-stack migration decision** (Stockfish everywhere) — which the brief
  says needs **explicit user approval + a written cost/benefit + a profiling run** (see §7).
- **Independence:** **fully independent** of U1/U3; can run on CPU/GPU in parallel.

### U3 — Model comparison / baseline suite (IMPORTANT, parallel thread)

- **Goal:** situate our GNN+MC controller against simple and lesioned stopping rules. Can run
  **now**, on the **existing ysagiv shards**, and be migrated elsewhere later.
- **Data in:** existing packed controller shards
  (`/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_*`).
- **Baselines to implement (eval-only, no training where possible):**
  1. GNN → **MLP over tree statistics** (the A2 `tree_stats_baseline.py` line)
  2. **Gain-depth-only** dynamic stopping
  3. **Always stop**
  4. **Never stop**
  5. **Geometric-probability** stopping
  6. **Tree-size-sensitive** stopping
- **Metrics:** OSS, `Pr(halt)` curves, regret vs DP, calibration — reuse `analysis/_budgeted/*`
  (`baselines.py`, `calibration.py`, `regret.py`, `report.py`).
- **Tests:** sanity bounds (always-stop OSS=0; never-stop OSS=budget); every baseline ≤ DP-oracle
  on its own objective; our controller beats all six on regret-vs-DP.
- **Contingencies:** if baselines rival the GNN, that reshapes the model story (and feeds back
  into U1.3's "is the GNN necessary?" question from A0b).
- **Independence:** **fully independent** of U1/U2 — different data, different output dir.

---

## 4. Dependency graph (what is parallel)

```
            U1.0 timing smoke  ──►  U1.1 generate trees ──►  U1.2 OSS↔RT upper bound  (CRITICAL)
              (DONE)                  (on human FENs)    └►  U1.3 model refit (smoke→fit)
                                                                │
  U2 Lc0 gain in human data  ───────── independent ───────────┤  (parallel, CPU/GPU)
  U3 baseline suite          ───────── independent ───────────┘  (parallel, existing shards)
```

- **Critical path:** U1.0 → U1.1 → U1.2. Everything else is parallelizable.
- U1.3 ∥ U1.2 (share U1.1 trees). U2 and U3 share **no** dependency with U1 and can start today.

---

## 5. Feasibility and timing

**Per-tree cost (A100):** ~16–41 s GPU (pin to one number, §U1.0), ~14 min CPU/blas.
Throughput (use a conservative 41 s and a central 25 s):

| Tier | Trees | GPU-hours @25 s (central) | GPU-hours @41 s (cons.) | Wall on 20 GPUs (gpu-medium) |
|---|---|---|---|---|
| acceptable | 10K | 69 | 114 | 3.5 h / 5.7 h |
| good | 50K | 347 | 568 | **17 h / 28 h** |
| ideal | 100K | 694 | 1,137 | **35 h / 57 h** |

**Committed tier: 50K** (decision 1) — 17–28 h on 20 GPUs, < 1 day even pessimistically. Pipeline
stays parameterized to extend to 100K (~1.5–2.4 days) if the pinned per-tree number is favorable.
The brief's "couple of days, max GPU QoS" is **feasible**.

**QOS available** (`gpu-medium`: ≤20 GPUs, 3-day wall; `gpu-short`: ≤44 GPUs, 1-day wall):
- Primary: `gpu-medium`, **~20 fat shards** (5K FENs/shard for 100K), walls sized to the per-tree
  number with ≥30% margin. **This is the fix for the empty-dir failure** — never size a shard so
  `FENs × s_per_tree` exceeds its wall.
- Burst: `gpu-short` (44 GPUs) for a 10K/50K push under a 1-day wall.

**CPU overflow (the brief's "max out CPU in parallel"):** at ~14 min/tree on 4 cores, the `short`
QOS (cpu≤1400 → ~350 four-core jobs) yields ~**1,500 trees/hr**, comparable to 20 GPUs. Worth it
as additive throughput, but it monopolizes the CPU allocation and inherits lc0's `libcublas`
coupling — treat as overflow, not primary.

**Depth: keep 96 (decision 3).** If a speed trade is forced, **reduce controller epochs first**;
treat depth 96→64 (~⅓ faster) as a last resort only, since it breaks regime-match with training.

**Model-fitting timing:** historically hours–days for the 128-dim GNN; the **tiny-GNN smoke**
(§U1.3) targets ≤2 h GPU to convergence "good enough," MC ≤1 h / ≤1000 steps. Profile by plotting
loss curves as it runs; do **not** scale to production until the end-to-end loop is fast.

---

## 6. Data inventory (have vs need) and where it goes

**Have:**
- Human FENs: `…/tmp/human_fens_10k.txt` (+ `_manifest.parquet`), `…/human_fens_1k.*`.
- `personal.db` (18 GB): `processed_moves_nonzero`, engine evals, RT/clock/branching.
- Existing model shards: `…/ysagiv/chess/CTS/data/controller_packed_*`, `generated_trees_*`.
- Timing smoke: `…/tmp/bench_trees_{gpu,cpu}/`, `bench_configs/`, `bench_gpu.log`.
- SF multidepth caches: `…/tmp/sf_multidepth_traces_{1k,10k}.pkl`.

**Need (and where to save):**
- 50K/100K human FEN exports → `…/tmp/human_fens_{50k,100k}.txt` (+ manifests).
- Trees → `…/tmp/human_trees_{10k,50k,100k}/` (`resume: true`).
- U1.2 join tables + figures → `lmcos/analysis/figures/` (+ `R-U1` report).
- U1.3 packed episodes / checkpoints → stage-3/4 scratch dirs per `R-LMCOS-OVERVIEW`.
- U2 Lc0 evals → `lc0_evaluations` in `personal.db`.
- U3 baseline metrics → a new `lmcos/analysis/figures/baselines/` (+ `R-U3`).

---

## 7. Decisions (resolved 2026-06-05 by hl4291)

1. **Tier = 50K ("good").** Pipeline parameterized to extend to 100K if the clean smoke lands
   near the fast end. 50K finishes < 1 day even pessimistically.
2. **Engine-swap profiling: schedule now, in parallel, folded into the per-tree smoke (5).**
   Stockfish is CPU-only and does not compete with Lc0 for GPU, so it is a free comparison.
   The smoke profiles **three engines side-by-side: Stockfish (CPU), Lc0-CPU, Lc0-GPU** — this
   doubles as the engine-stack cost/benefit profiling the brief requires before any swap. For a
   smoke (~20–100 FENs) this is not prohibitive.
3. **Depth: keep regime-matching faithfully — depth 96.** Fall back to 64 only if forced, and
   **prefer reducing epochs over reducing depth** if a speed trade is needed. The first reunified
   run uses the same depth-96 regime as model training.
4. **OSS budget: match lmcos faithfully.** That means the canonical
   `BudgetedOracleConfig()` defaults, **not** a flat single budget — see §7a. No clock-as-budget.
5. **Pin the per-tree number: yes — first sprint action,** via the three-engine smoke in (2).
   One clean steady-state run (≥100 warm FENs, exclude warmup) settles the 16-vs-41 s spread.

### 7a. The faithful OSS budget (what "match lmcos" means, precisely)

The lmcos OSS is **not** a flat budget = 96. From `preprocess_mc/oracle.py`, the canonical
`BudgetedOracleConfig()` used by `pack.py` (and already the default in
`human_oracle_comparison.py` via `_CONFIG = BudgetedOracleConfig()`) is:

- **Tree growth:** trees grown to **96 expansions** (`search_budget = min_nodes = max_nodes = 96`).
- **Budget buckets** (`DEFAULT_BUDGET_BUCKETS`, span 1–120): scramble 1–3 · medium-small 4–10 ·
  medium-large 11–25 · large 26–60 · very-large 61–120, with **`samples_per_bucket = 2`** starting
  budgets drawn per tree. The model trains across this **distribution** of starting budgets.
- **Cost:** `time_lambda = 18.537`, `time_p = 2.8`, `time_tau = 2.5`, `time_delta = 1`,
  **`maintenance_scale = 0.0` (maintenance OFF — the `nomaint` in the shard names)**.

**Implication for the plan:**
- **U1.1/U1.3 packing** must use these exact defaults + bucket sampling so the human-FEN packed
  episodes are **format-identical** to `controller_packed_*_nomaint_no_xaba`.
- **U1.2 headline correlation** uses OSS at **full budget 96** (the script's current `--budget 96`
  default, faithful to A0a/A1), with a **per-bucket robustness sweep** as a secondary check.

---

## 8. Test catalogue (one place)

| ID | Test | Where |
|---|---|---|
| T-gen-1 | `root_position_spec == manifest full_fen` (no stale trees) | `human_oracle_comparison.py` (load guard) |
| T-gen-2 | `count(*.pt) == len(FENs)` per tier; resume fills gaps | shell check |
| T-orac | A0 oracle invariants on 1K sample | `tests/test_oracle_stop_step_features.py` etc. |
| T-u12-dir | `r(OSS, logRT) > 0`, significant at n | U1.2 report |
| T-u12-sub | 497-tree A1 smoke recovered as a subset | U1.2 |
| T-u13-fit | child-WDL CE < trivial-prior; MC sign-acc > 54.6% → ~90% | U1.3 |
| T-u13-shape | shapes/params/no-NaN/reproducibility | `test_minimal_model.py` |
| T-u2 | Lc0 gain reproduces sign of SF gain↔RT; Lc0~SF rank Spearman>0 | U2 |
| T-u3 | always-stop OSS=0; never-stop OSS=budget; ours < all six on regret | U3 |

---

*End of North Star. Edit here first; mirror status into `labnotebook.md` and the `R-U*` reports.*
