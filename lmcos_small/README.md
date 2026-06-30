# `lmcos_small/` — chess meta-reasoning: human RT analysis + Stockfish halt-model pipeline

This directory is the merge of the former **`human_analytics/`** and **`lmcos_tiny/`**
trees into one place that shares a **single config** (`configs/core.yaml`). It holds the two
halves of the small-scale chess meta-reasoning work:

- **`human/`** — the human-RT-vs-feature analysis. Board features + engine signals
  (Gain / MQ / GSS / action-gap / H(π)) computed from Stockfish search trees, regressed against
  human response time. Entry points: `human/board.py`, `human/engine.py`; library in
  `human/utils/`; tests in `human/tests/`. See **[`human/README.md`](human/README.md)**.
- **`src/`, `pipeline/`, `analysis/`, `configs/`** — the tree-generation + deliberation
  pipeline (the `cts` package): FENs → Stockfish search trees → a 4-policy *halt-model*
  comparison on a Stockfish strength ladder (Elo 1800/2000/2200). See
  **[`README_pipeline.md`](README_pipeline.md)**.

## Layout

```
lmcos_small/
├── README.md            this file (orientation)
├── README_pipeline.md   the cts pipeline handbook (former lmcos_tiny/README.md)
├── pyproject.toml       installable `cts` package (src-layout)
├── env.sh               `source env.sh` → PYTHONPATH=src (this fork wins over the editable install)
├── configs/
│   ├── core.yaml        SINGLE unified config (pipeline `globals`+stages AND `human_analysis:`)
│   └── render_stage.py  shell helper to QUERY resolved pipeline values (--get)
├── src/cts/             the `cts` package (tree gen, encoder, MC oracle, readout, eval)
├── pipeline/            cts orchestration, one slurm file per pipeline STAGE (helpers/ aside)
├── analysis/            cts analysis scripts (VOC / OSS / pruning / RT figures) + analysis/slurm/
├── human/               human-RT analysis (board.py, engine.py, utils/, tests/)
└── slurm/
    ├── logs/            cts pipeline job logs
    └── human/           human-analysis slurm (analysis.sh, preprocess.sh, tree_values.slurm, scripts/)
```

## The single config (`configs/core.yaml`)

One file now feeds **both** loaders:

- **Pipeline (`cts`)** reads the sectioned, `${...}`-interpolated stages
  (`globals`, `treegen`, `split`, `gnn_pack`, `mc_pack`, `encoder`, `materialize`, `train`,
  `eval`, …) via `cts._config.load_config(... --stage NAME)`. Unchanged from before, except
  `globals.figures_dir` now points at `figures/normative`.
- **Human analysis** reads a new flat top-level **`human_analysis:`** section
  (`trees_default`, `cache_default`, `key_default`, `selected_db_default`, `table_*`,
  `response_time_histogram_bins`, `qq_plot_quantile_probes`) via
  `human/utils/helpers.py` → `CONFIG`. The loader slices the `human_analysis` section out of
  `core.yaml`, so `CONFIG["trees_default"]` etc. behave exactly as before.

## Related docs

- **`../README.md`** — workspace-wide overview.
- **`../labnotebook.md`** — chronological log (historical entries predate this merge and still
  reference `human_analytics/` / `lmcos_tiny/` as a record).
- **`../reports/`** — stable `R-*` write-ups.
