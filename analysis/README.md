# Chess Thinking Dynamics — `analysis/` handbook

**Audience:** coding agents, research assistants, and human contributors working in `chess_analysis/analysis/`.

This file is the **single** orientation doc for this tree: layout, commands, conventions, and enough scientific context to interpret what the code is doing.

> [!NOTE]
> **Environment:** activate the project venv from the repo root: `source .venv/bin/activate`.

---

## 1. Quick reference — commands

All paths are relative to the **`chess_analysis/`** repo root (parent of `analysis/`).

| Task | Command |
| :--- | :--- |
| **Activate env** | `source .venv/bin/activate` |
| **Moves ETL (games → shards → merge → `process_moves`)** | `bash analysis/slurm/preprocess.sh` (or `preprocess.py get_games` / `shard` / `merge` / `process_moves` separately) |
| **Regenerate standard figures** | `bash analysis/slurm/analysis.sh` |
| **Move-time histograms** | `python analysis/move_time_summary.py` |
| **Move-time dashboards** (clock, branching, material, ply) | `python analysis/movetime_analysis.py` (optional: `--only clock pieces_exc self_pieces_exc ply …`) |
| **Ply vs instant-move probability** | `python analysis/ply_premove.py` |
| **Engine eval (cluster)** | `bash analysis/slurm/engine_eval.sh` (or `--merge-only` when parquets exist) |
| **Merge eval shards only (legacy)** | `python analysis/slurm/scripts/script_merge_evals.py --engine stockfish` |
| **Slidev deck (LMCOS overview)** | `cd analysis/presentations/lmcos-overview && npm install && npm run dev` (symlink `public/figures` per that README) |

Standard dashboards (`movetime_analysis`, `move_time_summary`, `ply_premove`) are wired from **`bash analysis/slurm/analysis.sh`** (see repo-root paths there).

**Outputs:** analysis scripts write figures under **`analysis/figures/`**.

**Search-tree / lmcos visuals:** Graphviz rendering of tensorized trees lives in **`lmcos/demos/`** (e.g. `helper_tensorization.py`, `helper_gnn.py`), not as a standalone script under `analysis/`.

---

## 2. Repository layout

```text
chess_analysis/
├── .venv/
├── data/                         # Optional staging (parquets, scratch outputs)
├── README.md                     # Workspace / lmcos overview
└── analysis/
    ├── README.md                 # This file
    ├── figures/                  # Matplotlib outputs from dashboards and exploratory scripts
    ├── exploratory/              # Ad hoc analyses (heatmaps, smoke tests, quantify_early_ply); PYTHONPATH=analysis
    ├── presentations/             # Slidev deck (`lmcos-overview/`) + shared SVG assets
    ├── utils/                    # Library: Analyzer, plots, helpers, selected_db (table names)
    ├── slurm/
    │   ├── scripts/              # Pipeline Python CLIs (+ _bootstrap.py)
    │   ├── *.sh, *.sbatch       # Orchestration (calls scripts/ with repo-root paths)
    │   └── logs/                 # `ld-moves_*.out/.err`, optional `preprocess_driver.log`, `eval_*`
    ├── movetime_analysis.py      # DuckDB move-time dashboards (see DEFAULT_ANALYSES)
    ├── move_time_summary.py
    ├── ply_premove.py
    └── ...
```

### Where to put new code

| Location | Put here |
| :--- | :--- |
| **`analysis/slurm/scripts/`** | New **ETL / engine / join** entry points. Start with `ensure_src()` from `_bootstrap.py` so `import utils` works when run as `python analysis/slurm/scripts/...` from repo root. |
| **`analysis/`** (top-level `.py`) | New **dashboards, reports, thin CLIs** that read `personal.db` (see `utils/selected_db.py`) and write figures. |
| **`analysis/utils/`** | **Reusable** plotting, SQL aggregation patterns, `Analyzer`/`Variable`—**not** one-shot pipeline drivers. |
| **`analysis/exploratory/`** | Experiments and one-off plots; follow existing `sys.path` patterns. |
| **`analysis/presentations/`** | Slidev decks (`lmcos-overview/`) and presentation assets only—not Python pipeline code. |

---

## 3. Scientific context (why this code exists)

**Goal:** study how humans **allocate time** in chess as a function of **clock pressure** (budget) and **position complexity** (demand).

| Theme | Takeaway |
| :--- | :--- |
| **Heavy tails** | Most moves are fast; long thinks dominate variance. Analyses use **$\log T$** (with `EPSILON` in `utils.helpers`) unless there is a strong reason not to. |
| **Clock** | More remaining clock associates with longer thinks; signal is clearest when **ply** and player heterogeneity are accounted for. |
| **VOC (value of computation)** | Engine-defined gain from deep vs shallow search relates to think time; often discussed vs **ply** “arc” (midgame peak ~40–50). |
| **Scale** | Core DuckDB pipelines target on the order of **~10⁸ moves**; always prefer **SQL-side** aggregation and sampling. |

For publication-style figures, `utils.analysis.Analyzer.save_dashboard` defaults to **2×2**: global raw trend and quantile bins on the top row, the same pair **by `ply_tertiles`** (global ply tertiles from `processed_moves` / `processed_moves_nonzero`; overlaid) on the bottom. Pass **`include_quantile_heatmap=True`** (and **`quantile_heatmap_row='move_ply'`** on construction) to also write a **standalone** quantile×quantile heatmap PNG (default path: same stem as the dashboard plus `_quantile_heatmap` before the extension; override with ``heatmap_output_path``). Use **`layout="1x3"`** for raw | quantile | scatter, or **`layout="1x2"`** for raw | quantile only (see `movetime_analysis.ply_movetime`).

---

## 4. Data stack and pipeline (for implementers)

### Databases

- **`core` / `lichess.db`:** read-only Lichess mirror (multi‑TB).
- **`personal.db`:** project workspace; **`games`**, **`moves`**, **`processed_moves`**, **`processed_moves_nonzero`**, engine eval tables, etc. Default path: **`utils.selected_db.SELECTED_DB_DEFAULT`** (cluster scratch).

### Human moves ETL

1. **`preprocess.py get_games`** → table **`games`** (same role as legacy `selected_games`).
2. **`preprocess.py shard`** (Slurm array) → `selected_moves_*.parquet` under **`staging_dir`**; filters match legacy shard+berserk+grant policy.
3. **`preprocess.py merge`** → **`moves`** (from parquets); **`preprocess.py process_moves`** → **`processed_moves`** (features + `fen` + `ply_tertiles`) and **`processed_moves_nonzero`** (`move_time > 0`). **`preprocess.sh`** runs both after shards.

**Orchestration:** `bash analysis/slurm/preprocess.sh`:

- Clears **`staging_dir`** and prior **`ld-moves_*`** logs in **`analysis/slurm/logs/`** (does not remove **`eval_*`**).
- **`get_games`** (login-node DuckDB settings) → **`games`**.
- Submits a **Slurm array** of **`preprocess.py shard`** tasks; prints periodic **`squeue`** status (`PREPROCESS_SHARD_POLL_SEC`, default 30s).
- **`merge`** then **`process_moves`** on the login node, using **`DUCKDB_MERGE_THREADS`** / **`DUCKDB_MERGE_MEMORY_LIMIT`** (separate from shard task memory).

**Long / fragile SSH sessions:** run the driver detached so merge continues after disconnect, e.g.  
`nohup bash analysis/slurm/preprocess.sh >> analysis/slurm/logs/preprocess_driver.log 2>&1 &`

Typical filters: see **`preprocess.py` `main()` `config`** (date window, initial clock, increment, min Elo).

### `preprocess.py` — DuckDB + staging layout

| Step | Argument | Same directory holds |
| :--- | :--- | :--- |
| **`get_games`** | **`work_dir`** | DuckDB `temp_directory` while building **`games`** |
| **`preprocess_game_shard`** | **`staging_dir`** | DuckDB `temp_directory` **and** `selected_moves_<partition>_<segment>.parquet` |
| **`merge_game_shards`** | **`staging_dir`** | DuckDB `temp_directory` **and** glob `selected_moves_*.parquet` → **`moves`** |
| **`run_process_moves`** | **`work_dir`** | Opens **`personal.db`**; spill under **`work_dir`**; builds **`processed_moves`** / **`processed_moves_nonzero`** (`process_moves` in `preprocess.py`) |

**Thread / memory:** `preprocess.sh`: **`get_games`** uses `DUCKDB_THREADS` / `DUCKDB_MEMORY_LIMIT` (defaults 40 / 64GB); **`merge`** and **`process_moves`** use **`DUCKDB_MERGE_THREADS`** / **`DUCKDB_MERGE_MEMORY_LIMIT`** (defaults **64** / **200GB**). Shards use Slurm CPUs and `DUCKDB_SHARD_MEM`. Array jobs set `PREPROCESS_TOTAL_SHARDS` to match the Slurm task count.

**Hygiene:** clear **`staging_dir`** before a new shard run (`preprocess.sh` does this); stale `selected_moves_*.parquet` would pollute the merge glob.

### Engine evaluation

- **Worker:** `analysis/slurm/scripts/script_engine_eval.py` → shard parquets.
- **Eval parquet merge:** `script_engine_eval.py` **`merge`** (or legacy `script_merge_evals.py`) → `{stockfish,lc0}_evaluations` in `personal.db`.
- **Join to moves:** `build_selected_moves_with_engine.py` joins **`processed_moves`** to `{stockfish,lc0}_evaluations` on **`fen`** → **`selected_moves_with_engine`**.

Details: **`bash analysis/slurm/engine_eval.sh`**, **`engine_eval_shard.sbatch`**, logs under **`analysis/slurm/logs/`**.

### Pipeline vs analysis — ordered workflow

1. `bash analysis/slurm/preprocess.sh` (or equivalent `preprocess.py` steps).
2. **Optional:** `bash analysis/slurm/engine_eval.sh` → `python analysis/slurm/scripts/build_selected_moves_with_engine.py`
3. **Figures:** `bash analysis/slurm/analysis.sh` or individual `analysis/*.py` tools in §1.

---

## 5. Conventions for contributors and agents

### Design

- **Readability over cleverness:** prefer two explicit functions to one overloaded CLI.
- **Minimal CLIs:** stable, few flags; document defaults in `--help`.
- **`preprocess.py`:** one directory per step for DuckDB spill **and** artifacts (`work_dir` / `staging_dir` above); do not add parallel “alternate tmpdir” tunnels via `**kwargs`.
- **Names:** descriptive columns and variables (`log_clock_ply_residual`), not `x_adj`.
- **Paths:** `os.path.join` + anchor to `__file__`, or use `_bootstrap.src_root()` / `project_root()` in **`slurm/scripts/`**.

### Structure

- Separate **load / compute / plot** in analysis code.
- All runnable entry logic under **`if __name__ == "__main__":`**.
- **Bivariate DuckDB plots:** **`Analyzer`** + **`Variable`** (`utils.analysis`); keep aggregations in **SQL** when possible; sample large pulls before plotting.
- **Matplotlib “poster” figures:** call **`apply_poster_style()`** (`utils.helpers`); no top/right spines; use project font sizes.
- **Graphviz / non-matplotlib:** no `apply_poster_style()`; anchor output paths under `analysis/figures/` or keep visuals inside **`lmcos/demos/`** helpers.

### Statistics

- **Log space** for move time and clock in standard analyses.
- **Fixed effects / de-meaning:** subtract group means (e.g. by ply or player) when exploring confounding.

---

## 6. Quality control (do not regress)

### Legacy vs new ETL (smoke)

- **Script:** `analysis/slurm/scripts/tests/compare_legacy_new_pipeline_smoke.py` — builds **two** isolated DuckDB trees under a scratch root (default **`/scratch/gpfs/GRIFFITHS/hl4291/tmp/pipeline_smoke`**): **legacy** mirrors deleted `preprocess_data.py` (shard = neg-time filter only; berserk + grant on merged `moves`; then `_selected_moves` / `_nonzero_T`), **new** uses current `preprocess.py` (shard applies neg + berserk + grant; `merge` → `process_moves`).
- **What to expect:** `moves` **row counts** usually **differ** (legacy keeps bad games until feature SQL; new drops them earlier), but **`COUNT(DISTINCT gid)` on the positive-time feature table** should **match** for the same `[start, end)` window when `lichess.db` / parquets are unchanged.
- **Example:** `PYTHONPATH=analysis python3 analysis/slurm/scripts/tests/compare_legacy_new_pipeline_smoke.py --clean --start-date 2023-10-01 --end-date 2023-10-05`
- **Automated (optional):** `RUN_PIPELINE_COMPARE=1 PYTHONPATH=analysis python3 -m unittest discover -s analysis/slurm/scripts/tests -p 'test_pipeline_compare_smoke.py' -v` (slow; short window).

- **Negative move times:** exclude affected games when building analysis tables (pipeline enforces this for core paths).
- **Berserk:** dedicated detection; do not mix berserk games into clock analyses without an explicit policy.
- **Grant more time (GMT):** windowed detection on zero-increment games; tables like `grant_more_time_games` feed joins.

---

## 7. Search-tree visuals (`lmcos`)

There is **no** standalone `analysis/performance/visualize_tree_expansion.py` in this repo. Tutorial notebooks under **`lmcos/demos/`** import helpers such as **`helper_tensorization.py`** and **`helper_gnn.py`**, which build **Graphviz** `Digraph`s from packed examples / `SearchTree` (`lmcos/tree.py`).

**Typical deps:** `torch`, Python package `graphviz`, system `dot`, `chess`.

---

## 8. Engine throughput notes (Depth 5 calibration)

| Engine | ~pos/s | Note |
| :--- | :--- | :--- |
| Stockfish, hash not cleared | ~800 | Noisy / unreliable at low depth. |
| Stockfish, clear hash | ~75 | **Deterministic**; production default. |
| lc0 v0.32.1 (A450 example) | ~30 | Slower U CI loop. |

---

## 9. Related docs

- **`chess_analysis/README.md`** — workspace-wide context (behavior + **`lmcos`** meta-controller thread).
- **`lmcos/LAB_NOTEBOOK.md`** — dated experiments for the neural search-control line.
