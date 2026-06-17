# Reports index

Stable analysis and reference docs for **chess_analysis** (CMC). The [lab notebook](../labnotebook.md) is the chronological log (**Description | Rationale | Status / finding | Reference**).

Reports come in two formats:

- **Scientific-inquiry reports** follow the **scientific template**: *Overview → Results (plot-heavy,
  figures first) → Methods → Appendix (Logs)*. Figures are embedded from the repo-root `figures/`
  (human analytics) or `lmcos/analysis/figures/` (LMCOS), with `figures/archive/` for historical
  snapshots. These are the merged inquiry reports below.
- **Reference / framework / engineering / archive reports** keep their original format (summary
  table · `| Step | Status |` procedure · notes).

> **Didactic mini-paper convention (target).** A scientific report is the **source of truth; the Slidev
> decks in `presentations/` are built off it**, so it should read like a Reader's-Digest walkthrough anyone
> can follow: **plain language** (jargon/equations in Methods), the body as **one investigation
> top-to-bottom**, **every section header a question** (not a finding), and **each section/subsection
> closing with a callout** — `> **◆ Modeling choice.** …` or `> **◆ Modeling result.** …` — that carries the
> takeaway. See `monkey_4iar`'s R-GAZE / R-MODELCMP / R-RECOVERY for the worked pattern; the chess reports
> below predate it and are migrated as they're revised.

### Analysis roadmap (merged thematic set)

| Inquiry | Report | Format | Thread | Status |
|---------|--------|--------|--------|--------|
| Oracle stop step vs human RT (Tier A/B + SF-2000) | [(R-ORACLE-RT)](oracle-stop-vs-human-rt.md) | Scientific | LMCOS | ⏳ Tier A done; 10K + SF-2000 open |
| Minimal meta-controller + budgeted baselines | [(R-MINMODEL)](minimal-model-and-baselines.md) | Scientific | LMCOS | ⏳ baselines done; Config D open |
| Human VOC / MQ vs move time | [(R-VOC-MQ)](human-voc-mq.md) | Scientific | Human | ✅ pipeline + 100K done |
| GNN encoder pretraining (child-WDL) | [(R-PRETRAIN)](gnn-pretrain.md) | Scientific | LMCOS | ✅ smoke passed; full run gated |
| Tree-generation engineering (timing, speedups, batched-gen NO-GO) | [(R-TREEGEN)](tree-generation-engineering.md) | Engineering | LMCOS | ✅ faithful path shipped |

**Cite:** `[(R-ORACLE-RT)](oracle-stop-vs-human-rt.md)` from this folder;
`[(R-ORACLE-RT)](reports/oracle-stop-vs-human-rt.md)` from the notebook.

### Full report list

| Report | Format | Thread |
|--------|--------|--------|
| [(R-HUMAN-DATA)](reference-human-dataset.md) | Reference | Human |
| [(R-MOVETIME-PRIOR)](reference-move-time-prior.md) | Reference | Human |
| [(R-HUMAN-BACKLOG)](reference-human-backlog.md) | Reference | Human |
| [(R-VOC-MQ)](human-voc-mq.md) | Scientific | Human |
| [(R-THEORY)](human-theory-stopping.md) | Framework | Human |
| [(R-ORACLE-RT)](oracle-stop-vs-human-rt.md) | Scientific | LMCOS |
| [(R-MINMODEL)](minimal-model-and-baselines.md) | Scientific | LMCOS |
| [(R-PRETRAIN)](gnn-pretrain.md) | Scientific | LMCOS |
| [(R-TREEGEN)](tree-generation-engineering.md) | Engineering | LMCOS |
| [(R-LMCOS-OVERVIEW)](lmcos-pipeline-overview.md) | Reference | LMCOS |
| [(R-LMCOS-STAGE4)](lmcos-stage4-ablation.md) | Reference | LMCOS |
| [(R-CLEANUP-0604)](archive-2026-06-04-cleanup.md) | Archive | Repo |
| [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md) | Archive | LMCOS |
