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

## 2026-06-05 {#2026-06-05}

A0a corrected and rerun on all 39,668 trees; definitions, features, and code audited.

| Description | Rationale | Status / finding | Reference |
|---|---|---|---|
| **LMCOS A0a** — corrected `oracle_stop_step` + features; all 39,668 trees; scatter plots | Two bugs fixed; rerun at full scale (missed half the shards with old glob) | ✅ **4/4 RT directions** correct. gain_depth r=**+0.233** ✓; toptwo r=**−0.290** ✓; branching/material r≈0.01 (tiny, correct dir). r(oss, ce)=**0.525**. Branching/material near-zero expected: oracle driven by value landscape (toptwo/gain_depth), not structural complexity. 72/72 tests. | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
| **LMCOS A0b** — halt_rewards fix + optimal_stop eval | Establish minimal-MC anchor with correct oracle labels | ⏳ Code fixed 2026-06-05; correct re-run ⬜ **todo** (2026-06-03 run invalid) | [(R-A0)](reports/analysis-0-oracle-baseline.md) |
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
