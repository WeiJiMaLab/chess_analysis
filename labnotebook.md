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
| **U3 model comparison** — controller vs baselines on existing diagnostics (no GPU/new trees) | Quantify controller lift; parallel thread | ✅ On 30,630 episodes (`subtree_weighting_root_budget`): **controller avg-regret +0.026** vs best baseline (gain-depth ≤0.05) **+0.148** (5.6×), always-stop +0.52, never-stop +1.37; exact-stop **63%**. Gain-depth is best simple rule. | [(R-U3)](reports/analysis-u3-baselines.md) |

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

Notebook + reports restructure; A1 1K trees done; A4 entropy VoI at scale; A2 tiny-GNN pretrain started.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **Docs restructure** — root `labnotebook.md` + `reports/`; drop `proposed_next_steps.md` | Single chronology + procedure checklists for open work | ✅ Open steps live in active `R-*` reports | [reports/README.md](reports/README.md) |
| **LMCOS A1** — human FEN tree smoke (644 FENs, budget 96) | Same-position oracle vs RT | ✅ **644/644** `.pt`; comparison script + 10K + plots still open | [(R-A1)](reports/analysis-1-human-oracle.md) |
| **LMCOS A2** — tree-stats + tiny GNN scratch (Config D) | Skip 1-day pretrain if small encoder suffices | ⏳ Config D YAML open; child-WDL pretrain job **9215254** started | [(R-A2)](reports/analysis-2-minimal-model.md) |
| **Human A4** — entropy VoI stopping (SF multidepth, 10K CPU job) | Information-theoretic Rule B vs RT | ✅ **6,494** traces; r(d*, log RT) ≈ **0.01** at θ=0.001 — weak RT alignment | [(R-A4)](reports/analysis-4-entropy-voi.md) |
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
