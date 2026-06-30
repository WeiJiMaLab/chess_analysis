# `/` — chess meta-reasoning: human RT analysis + Stockfish halt-model pipeline

This directory is the merge of the former **`human_analytics/`** and **`lmcos_tiny/`**
trees into one place that shares a **single config** (`configs/core.yaml`). It holds the two
halves of the small-scale chess meta-reasoning work:

- **`src/analysis/`** — the human-RT-vs-feature analysis. Board features + engine signals
  (Gain / MQ / GSS / action-gap / H(π)) computed from Stockfish search trees, regressed against
  human response time. Entry points: `src/analysis/board.py`, `src/analysis/engine.py`; library in
  `src/analysis/utils/`; tests in `src/analysis/tests/`. See **[`src/analysis/README.md`](src/analysis/README.md)**.
- **`src/`, `pipeline/`, `analysis/`, `configs/`** — the tree-generation + deliberation
  pipeline (the `cts` package): FENs → Stockfish search trees → a 4-policy *halt-model*
  comparison on a Stockfish strength ladder (Elo 1800/2000/2200). See
  **[`README_pipeline.md`](README_pipeline.md)**.

## Layout

```
/
├── README.md            this file (orientation)
├── README_pipeline.md   the cts pipeline handbook (former lmcos_tiny/README.md)
├── pyproject.toml       installable `cts` package (src-layout)
├── env.sh               `source env.sh` → PYTHONPATH=src:src/analysis (this fork wins over the editable install)
├── configs/
│   ├── core.yaml        SINGLE unified config (pipeline `globals`+stages AND `human_analysis:`)
│   └── render_stage.py  shell helper to QUERY resolved pipeline values (--get)
├── src/
│   ├── cts/             the `cts` package (tree gen, encoder, MC oracle, readout, eval, stats)
│   └── analysis/        cts analysis scripts + human-RT analysis (make_rt_figures.py, board.py, engine.py, utils/, preprocess.py, tests/)
└── slurm/               EVERY batch script, one tree (logs unified under slurm/logs/)
    ├── pipeline/        cts stage jobs 1a–4 (+ helpers/setup_env.sh, submit_all.sh)
    └── analysis/        cts analysis + human jobs (voc / oss / prune / figures, board.sh, preprocess.sh, tree_values.slurm)
    └── logs/            all job logs
```

> **Shared library, not duplication.** Both halves import the `cts` package: stats helpers
> (`spearman` / `partial_spearman` / `bootstrap_ci`) live once in `cts.stats`, and the human OSS
> readout reuses `cts.data.preprocess_mc`'s budgeted-oracle trajectory. `src/analysis/preprocess.py` holds
> the DB-table producers next to the `src/analysis/utils/` library they use (no sys.path shims).

## The single config (`configs/core.yaml`)

One file now feeds **both** loaders:

- **Pipeline (`cts`)** reads the sectioned, `${...}`-interpolated stages
  (`globals`, `treegen`, `split`, `gnn_pack`, `mc_pack`, `encoder`, `materialize`, `train`,
  `eval`, …) via `cts._config.load_config(... --stage NAME)`. Unchanged from before, except
  `globals.figures_dir` now points at `figures/normative`.
- **Human analysis** reads a new flat top-level **`human_analysis:`** section
  (`trees_default`, `cache_default`, `key_default`, `selected_db_default`, `table_*`,
  `response_time_histogram_bins`, `qq_plot_quantile_probes`) via
  `src/analysis/utils/helpers.py` → `CONFIG`. The loader slices the `human_analysis` section out of
  `core.yaml`, so `CONFIG["trees_default"]` etc. behave exactly as before.

## Related docs

- **`../README.md`** — workspace-wide overview.
- **`../labnotebook.md`** — chronological log (historical entries predate this merge and still
  reference `human_analytics/` / `lmcos_tiny/` as a record).
- **`../reports/`** — stable `R-*` write-ups.
