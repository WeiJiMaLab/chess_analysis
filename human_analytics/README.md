# Chess Thinking Dynamics — `human_analytics/` handbook

**Audience:** coding agents, research assistants, and human contributors working in `chess_analysis/human_analytics/`.

This file is the **single** orientation doc for this tree: layout, commands, conventions, and enough scientific context to interpret what the code is doing.

> [!NOTE]
> **Environment:** activate the project venv from the repo root: `source .venv/bin/activate`.

---

## 1. Quick reference — commands

All paths are relative to the **`chess_analysis/`** repo root (parent of `human_analytics/`).

| Task | Command |
| :--- | :--- |
| **Activate env** | `source .venv/bin/activate` |
| **Moves ETL (games → shards → merge → `process_moves`)** | `bash human_analytics/slurm/preprocess.sh` (or `preprocess.py get_games` / `shard` / `merge` / `process_moves` separately) |
| **Regenerate standard figures** | `bash human_analytics/slurm/analysis.sh` |
| **log(MT) histogram + normal QQ** | `python human_analytics/move_time_summary.py` |
| **Move-time dashboards** (clock, branching, own non-pawn material, ply) | `python human_analytics/movetime_analysis.py` (optional: `--only clock npossiblemoves self_pieces_exc ply`) |
| **Ply vs instant-move probability** | `python human_analytics/ply_premove.py` |
| **Tree-derived OSS / VOC / Action Gap / MQ vs RT** (lc0-tree subset) | `sbatch human_analytics/slurm/tree_values.slurm` (`tree_values_analysis.py`; not part of the full-dataset pipeline) |
| **Engine eval (positions)** | `python human_analytics/slurm/scripts/build_pos_with_engine_eval.py eval --engine stockfish` |
| **Build selected-moves-with-engine join** | `python human_analytics/slurm/scripts/build_selected_moves_with_engine.py` |
| **Slidev deck (LMCOS overview)** | `cd human_analytics/presentations/lmcos-overview && npm install && npm run dev` (symlink `public/figures` per that README) |

The **full-dataset** plots (`move_time_summary`, `movetime_analysis`, `ply_premove`) are wired from **`bash human_analytics/slurm/analysis.sh`**. The **generated values** (OSS / VOC / Action Gap / **MQ**, derived from the lc0 search trees) are computed on the **subset of positions that have a tree** via `tree_values_analysis.py` and run separately on the cluster. **MQ moved from FULL to SUBSET**: it is now the Lc0 definition — the post-search root-value loss of the human's played move, `final_Q(played) − final_Q(best) ≤ 0` — not the former Stockfish `pos_with_engine_eval.mq` (`e_win_taken − e_win_best`), which has been retired.

**Outputs:** analysis scripts write figures under the **repo-root `figures/`** directory (single source of truth; the `presentations/public/figures` symlink points here). Older / intermediate snapshots live under **`figures/archive/`**.

---

## 2. Repository layout

```text
chess_analysis/
├── .venv/
├── data/                         # Optional staging (parquets, scratch outputs)
├── figures/                      # Matplotlib outputs from dashboards (single source of truth)
│   └── archive/                  # Older / intermediate figure snapshots (e.g. u3_baselines/)
├── README.md                     # Workspace / lmcos overview
└── human_analytics/
    ├── README.md                 # This file
    ├── exploratory/              # Ad hoc analyses (heatmaps, smoke tests, quantify_early_ply); PYTHONPATH=human_analytics
    ├── presentations/             # Slidev deck (`lmcos-overview/`) + shared SVG assets
    ├── utils/                    # Library: Analyzer, plots, helpers, selected_db (table names)
    ├── slurm/
    │   ├── scripts/              # Pipeline Python CLIs (+ _bootstrap.py)
    │   ├── *.sh, *.sbatch       # Orchestration (calls scripts/ with repo-root paths)
    │   └── logs/                 # `ld-moves_*.out/.err`, optional `preprocess_driver.log`, `eval_*`
    ├── movetime_analysis.py      # FULL: clock / branching / own-material / ply vs MT dashboards
    ├── move_time_summary.py      # FULL: log(MT) histogram + normal QQ
    ├── ply_premove.py            # FULL: ply vs instant-move probability
    ├── engine_analysis.py        # engine VOC/MQ primitives (library)
    ├── tree_values_analysis.py   # SUBSET: OSS / VOC / Action Gap / MQ from lc0 trees vs RT (cluster)
    └── ...
```

### Where to put new code

| Location | Put here |
| :--- | :--- |
| **`human_analytics/slurm/scripts/`** | New **ETL / engine / join** entry points. Start with `ensure_src()` from `_bootstrap.py` so `import utils` works when run as `python human_analytics/slurm/scripts/...` from repo root. |
| **`human_analytics/`** (top-level `.py`) | New **dashboards, reports, thin CLIs** that read `personal.db` (see `utils/selected_db.py`) and write figures. |
| **`human_analytics/utils/`** | **Reusable** plotting, SQL aggregation patterns, `Analyzer`/`Variable`—**not** one-shot pipeline drivers. |
| **`human_analytics/exploratory/`** | Experiments and one-off plots; follow existing `sys.path` patterns. |
| **`human_analytics/presentations/`** | Slidev decks (`lmcos-overview/`) and presentation assets only—not Python pipeline code. |

---

## 3. Scientific context (why this code exists)

**Goal:** study how humans **allocate time** in chess as a function of **clock pressure** (budget) and **position complexity** (demand).

| Theme | Takeaway |
| :--- | :--- |
| **Heavy tails** | Most moves are fast; long thinks dominate variance. Analyses use **$\log T$** (with `EPSILON` in `utils.helpers`) unless there is a strong reason not to. |
| **Clock** | More remaining clock associates with longer thinks; signal is clearest when **ply** and player heterogeneity are accounted for. |
| **VOC (value of computation)** | Engine-defined gain from deep vs shallow search relates to think time; often discussed vs **ply** “arc” (midgame peak ~40–50). |
| **Scale** | Core DuckDB pipelines target on the order of **~10⁸ moves**; always prefer **SQL-side** aggregation and sampling. |

For publication-style figures, `utils.analysis.Analyzer.save_dashboard` produces a fixed **1×2**: **quantile bins** (global, left) and **quantile bins by ply tertile** (right). Raw-trend and scatter panels were removed — quantile binning is the canonical view. Ply tertiles are settled **a priori** from the whole-dataset `move_ply` distribution (`quantile_disc` at 1/3, 2/3 over `ply_tertile_source`, default `processed_moves_nonzero` → cuts ≈ 27, 56) and applied as fixed boundaries, so the segmentation is identical across plots. A log-`move_time` y-axis keeps log spacing but labels ticks in seconds (`exp`). For near-zero-inflated x (VOC / Action Gap / MQ — the last has a large mass at exactly 0 where the human played the engine-best move), pass `zero_inflated=True, zero_threshold=0.05` to lump the near-zero mass (`abs(x) ≤ threshold`, so MQ ∈ [−0.05, 0]) into one point and quantile-bin the rest. Optional `include_quantile_heatmap=True` (+ `quantile_heatmap_row='move_ply'`) writes a standalone quantile×quantile heatmap.

---

## 4. Data stack and pipeline (for implementers)

### Databases

- **`core` / `lichess.db`:** read-only Lichess mirror (multi‑TB).
- **`personal.db`:** project workspace; **`games`**, **`moves`**, **`processed_moves`**, **`processed_moves_nonzero`**, engine eval tables, etc. Default path: **`utils.selected_db.SELECTED_DB_DEFAULT`** (cluster scratch).

### Human moves ETL

1. **`preprocess.py get_games`** → table **`games`** (same role as legacy `selected_games`).
2. **`preprocess.py shard`** (Slurm array) → `selected_moves_*.parquet` under **`staging_dir`**; filters match legacy shard+berserk+grant policy.
3. **`preprocess.py merge`** → **`moves`** (from parquets); **`preprocess.py process_moves`** → **`processed_moves`** (features + `fen` + `ply_tertiles`) and **`processed_moves_nonzero`** (`move_time > 0`). **`preprocess.sh`** runs both after shards.

**Orchestration:** `bash human_analytics/slurm/preprocess.sh`:

- Clears **`staging_dir`** and prior **`ld-moves_*`** logs in **`human_analytics/slurm/logs/`** (does not remove **`eval_*`**).
- **`get_games`** (login-node DuckDB settings) → **`games`**.
- Submits a **Slurm array** of **`preprocess.py shard`** tasks; prints periodic **`squeue`** status (`PREPROCESS_SHARD_POLL_SEC`, default 30s).
- **`merge`** then **`process_moves`** on the login node, using **`DUCKDB_MERGE_THREADS`** / **`DUCKDB_MERGE_MEMORY_LIMIT`** (separate from shard task memory).

**Long / fragile SSH sessions:** run the driver detached so merge continues after disconnect, e.g.  
`nohup bash human_analytics/slurm/preprocess.sh >> human_analytics/slurm/logs/preprocess_driver.log 2>&1 &`

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

- **Worker / eval:** `human_analytics/slurm/scripts/build_pos_with_engine_eval.py eval` → `{stockfish,lc0}_evaluations` in `personal.db`.
- **Join to moves:** `build_selected_moves_with_engine.py` joins **`processed_moves`** to `{stockfish,lc0}_evaluations` on **`fen`** → **`selected_moves_with_engine`**.

> *(The `engine_eval.sh` / `engine_eval_shard.sbatch` wrappers and `script_engine_eval.py` / `script_merge_evals.py` were removed; `build_pos_with_engine_eval.py` is the current entry point.)*

### Pipeline vs analysis — ordered workflow

1. `bash human_analytics/slurm/preprocess.sh` (or equivalent `preprocess.py` steps).
2. **Optional:** `python human_analytics/slurm/scripts/build_pos_with_engine_eval.py eval` → `python human_analytics/slurm/scripts/build_selected_moves_with_engine.py`
3. **Figures:** `bash human_analytics/slurm/analysis.sh` or individual `human_analytics/*.py` tools in §1.

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
- **Graphviz / non-matplotlib:** no `apply_poster_style()`; anchor output paths under the repo-root `figures/` directory.

### Statistics

- **Log space** for move time and clock in standard analyses.
- **Fixed effects / de-meaning:** subtract group means (e.g. by ply or player) when exploring confounding.

---

## 6. Quality control (do not regress)

- **Negative move times:** exclude affected games when building analysis tables (pipeline enforces this for core paths).
- **Berserk:** dedicated detection; do not mix berserk games into clock analyses without an explicit policy.
- **Grant more time (GMT):** windowed detection on zero-increment games; tables like `grant_more_time_games` feed joins.

---

## 7. Search-tree visuals (`lmcos`)

Tree tensorization and encoding live under **`lmcos/src/`** (`cts.core`, `cts.models`). There is no standalone visualization script in this repo.

---

## 8. Engine throughput notes (Depth 5 calibration)

| Engine | ~pos/s | Note |
| :--- | :--- | :--- |
| Stockfish, hash not cleared | ~800 | Noisy / unreliable at low depth. |
| Stockfish, clear hash | ~75 | **Deterministic**; production default. |
| lc0 v0.32.1 (A450 example) | ~30 | Slower U CI loop. |

---

## 9. Related docs

- **`../README.md`** — workspace-wide CMC overview (behavior + `lmcos` meta-controller).
- **`labnotebook.md`** (repo root) — chronological log; **`reports/`** — stable `R-*` write-ups for human + LMCOS analyses.
