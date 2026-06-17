# Reports index

Stable analysis and reference docs for **chess_analysis** (CMC). The [lab notebook](../labnotebook.md) is the chronological log (**Description | Rationale | Status / finding | Reference**).

Reports come in two formats:

- **Scientific-inquiry reports** follow the **scientific template**: *Overview → Results (plot-heavy,
  figures first) → Methods → Appendix (Logs)*. Figures are embedded from the repo-root `figures/`
  (human analytics) or `lmcos/analysis/figures/` (LMCOS), with `figures/archive/` for historical
  snapshots.
- **Reference / engineering / archive reports** keep their original format (summary
  table · `| Step | Status |` procedure · notes).

> **Didactic mini-paper convention (target).** A scientific report is the **source of truth; the Slidev
> decks in `presentations/` are built off it**, so it should read like a Reader's-Digest walkthrough anyone
> can follow: **plain language** (jargon/equations in Methods), the body as **one investigation
> top-to-bottom**, and **every section header a question** (not a finding, and not numbered). Each
> section/subsection closes with a typed callout that carries the takeaway:
> - `> **Decision:** …` — a modeling/methodological choice.
> - `> **Result:** …` — an empirical finding (e.g. "think time is log-normal").
> - `> **Clarification:** …` — resolves a likely misreading (e.g. a confound, not a sign bug).
>
> See `monkey_4iar`'s R-GAZE / R-MODELCMP / R-RECOVERY for the worked pattern.

> **Cross-references live here, not in the reports.** Individual reports avoid citing each other; this
> index is the single place that maps how they relate. Reports are primarily about **scientific questions**.

## Reports

| Inquiry | Report | Format | Thread | Status |
|---------|--------|--------|--------|--------|
| Human move time — what board features predict it (distribution, per-feature dashboards, board correlations) | [(R-MOVETIME-BOARD)](movetime_board.md) | Scientific | Human | ✅ done |
| Human move time — does a normative lc0 model match it? (VOC / MQ / OSS / action gap, oracle-stop tiers, lc0 correlations) | [(R-MOVETIME-MODEL)](movetime_model.md) | Scientific | Human | ✅ 100K done; SF-2000 + residualized MQ open |
| Minimal meta-controller + budgeted baselines | [(R-MINMODEL)](minimal-model-and-baselines.md) | Scientific | LMCOS | ⏳ baselines done; Config D open |
| GNN encoder pretraining (child-WDL) | [(R-PRETRAIN)](gnn-pretrain.md) | Scientific | LMCOS | ✅ smoke passed; full run gated |
| Tree-generation engineering (timing, speedups, batched-gen NO-GO) | [(R-TREEGEN)](tree-generation-engineering.md) | Engineering | LMCOS | ✅ faithful path shipped |
| Human Lichess dataset (tables, filters) | [(R-HUMAN-DATA)](reference-human-dataset.md) | Reference | Human | ✅ stable |
| 2026-06-04 cleanup | [(R-CLEANUP-0604)](archive-2026-06-04-cleanup.md) | Archive | Repo | — |
| Legacy lmcos notebook (Apr–May 2026) | [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md) | Archive | LMCOS | — |

## How the threads relate

- **Human thread.** [(R-HUMAN-DATA)](reference-human-dataset.md) is the dataset all human analyses sit on.
  The move-time inquiry is in two parts: [(R-MOVETIME-BOARD)](movetime_board.md) — what board features
  predict think time (subsumes the former move-time-prior baselines) — and
  [(R-MOVETIME-MODEL)](movetime_model.md) — whether the lc0 model matches it (subsumes the VOC/MQ work,
  the oracle-stop-vs-RT comparison, and the stopping-theory validation tiers).
- **LMCOS thread.** [(R-TREEGEN)](tree-generation-engineering.md) generates the search trees →
  [(R-PRETRAIN)](gnn-pretrain.md) pretrains the encoder → [(R-MINMODEL)](minimal-model-and-baselines.md)
  fits and benchmarks the meta-controller. The oracle-stop comparison in
  [(R-MOVETIME-MODEL)](movetime_model.md) reuses these trees. Full pipeline/layout and historical
  (Apr–May 2026) experiments: [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md) and `lmcos/slurm/README.md`.

**Cite:** `[(R-MOVETIME-MODEL)](movetime_model.md)` from this folder; `[(R-MOVETIME-MODEL)](reports/movetime_model.md)` from the notebook.
