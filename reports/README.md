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
| Human move time — does a normative lc0 model match it? (Gain / MQ / GSS / action gap, oracle-stop tiers, lc0 correlations) | [(R-MOVETIME-MODEL)](movetime_model.md) | Scientific | Human | ✅ done; SF-2000 + residualized MQ open |
| Branching & resource-rational deliberation (why decision width drives RT; mechanisms + predictions) | [(R-BRANCH)](branching.md) | Scientific (draft) | Human | 📝 proposal; P1–P5 open |
| Data reference — human Lichess dataset + lc0 tree generation | [(R-DATA)](reference-data.md) | Reference | Data | ✅ stable |

Older lmcos work (Apr–May 2026; GNN-pretrain, meta-controller, tree-gen engineering) lives in the
lab notebook's [§ Legacy section](../labnotebook.md#legacy) (the former archive, merged in).

## How the threads relate

The repo centres on the **human move-time inquiry**, all sitting on [(R-DATA)](reference-data.md)
(the human Lichess dataset + the lc0 search-tree dataset):

- [(R-MOVETIME-BOARD)](movetime_board.md) — what board features predict think time (branching dominates).
- [(R-MOVETIME-MODEL)](movetime_model.md) — whether lc0-search quantities (Gain / MQ / GSS / action gap)
  and the normative oracle track human RT.
- [(R-BRANCH)](branching.md) — a resource-rational account of the branching effect, with tree mechanisms
  and predictions (draft).

The **LMCOS model-training thread** (GNN encoder pretraining, the meta-controller and its budgeted
baselines) is no longer tracked as standalone reports — its record lives in the lab notebook's
[§ Legacy section](../labnotebook.md#legacy), the `lmcos/` code, and `lmcos/slurm/README.md`.

**Cite:** `[(R-MOVETIME-MODEL)](movetime_model.md)` from this folder; `[(R-MOVETIME-MODEL)](reports/movetime_model.md)` from the notebook.
