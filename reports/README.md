# Reports index

Stable analysis and reference docs for **chess_analysis** (CMC). The [lab notebook](../labnotebook.md) is the chronological log (**Description | Rationale | Status / finding | Reference**).

Each report has:

1. **Summary table** (phase-specific fields — no procedure here)
2. **Procedure** — checklist: `| Step | Status |` with **✅ done** / **⬜ incomplete** / **⏳ in progress**
3. **Notes** (optional) — design tables, commands, static reference

### Analysis roadmap

| Analysis | Report | Phase | Status |
|----------|--------|-------|--------|
| **0** Oracle direction + minimal MC | [(R-A0)](analysis-0-oracle-baseline.md) | After | ✅ done |
| **1** Human FEN trees + oracle vs RT | [(R-A1)](analysis-1-human-oracle.md) | Active | ⏳ 1K trees done; extraction + 10K open |
| **2** Minimal model / skip pretrain | [(R-A2)](analysis-2-minimal-model.md) | Active | ⏳ tree-stats + Config D open |
| **3** SF2000 weaker engine | [(R-A3)](analysis-3-weaker-engine.md) | Prior | ⬜ on hold (needs A1) |
| **4** Entropy VoI stopping | [(R-A4)](analysis-4-entropy-voi.md) | After | ✅ done (weak RT r at scale) |

### Summary table by phase

| Phase | Fields |
|-------|--------|
| **Prior to implementation** | Description · Rationale · Expectation · Open questions / notes |
| **During active implementation** | Description · Rationale · Expectation · Open questions / notes |
| **After implementation** | Description · Rationale · Expectation · Finding |

**Cite:** `[(R-A1)](analysis-1-human-oracle.md)` from this folder; `[(R-A1)](reports/analysis-1-human-oracle.md)` from the notebook.

| Report | Phase | Thread |
|--------|-------|--------|
| [(R-HUMAN-DATA)](reference-human-dataset.md) | After | Human |
| [(R-MOVETIME-PRIOR)](reference-move-time-prior.md) | After | Human |
| [(R-VOC-MQ)](human-voc-mq.md) | After | Human |
| [(R-VOC-100K)](human-voc-mq-100k.md) | After | Human |
| [(R-HUMAN-BACKLOG)](reference-human-backlog.md) | Active | Human |
| [(R-THEORY)](human-theory-stopping.md) | After | Human |
| [(R-A0)](analysis-0-oracle-baseline.md) | After | LMCOS |
| [(R-A1)](analysis-1-human-oracle.md) | Active | LMCOS |
| [(R-A2)](analysis-2-minimal-model.md) | Active | LMCOS |
| [(R-A3)](analysis-3-weaker-engine.md) | Prior | LMCOS |
| [(R-A4)](analysis-4-entropy-voi.md) | After | Human |
| [(R-LMCOS-OVERVIEW)](lmcos-pipeline-overview.md) | After | LMCOS |
| [(R-LMCOS-STAGE4)](lmcos-stage4-ablation.md) | After | LMCOS |
| [(R-VOC-MECH)](voc-mechanism.md) | Active | LMCOS |
| [(R-CLEANUP-0604)](archive-2026-06-04-cleanup.md) | After | Repo |
| [(R-ARCH-HUMAN)](archive-human-analytics-notebook-legacy.md) | Archive | Human |
| [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md) | Archive | LMCOS |
