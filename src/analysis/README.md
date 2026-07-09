# Chess Thinking Dynamics — `src/analysis/` handbook

**Audience:** coding agents, research assistants, and human contributors working in `chess_analysis/src/analysis/`.

This file is the **single** orientation doc for this tree: layout, commands, conventions, and enough scientific context to interpret what the code is doing.

> [!NOTE]
> **Environment:** activate the project venv from the repo root: `source .venv/bin/activate`.

---

## 1. Quick reference — commands

All paths are relative to the **`chess_analysis/`** repo root (parent of `src/analysis/`).

| Task | Command |
| :--- | :--- |
| **Activate env** | `source .venv/bin/activate` |
| **Moves ETL (games → shards → merge → `process_moves`)** | `bash slurm/analysis/preprocess.sh` (or `preprocess.py get_games` / `shard` / `merge` / `process_moves` separately) |
| **Regenerate standard figures** | `bash slurm/analysis/board.sh` |
| **log(MT) histogram + normal QQ** | `python src/analysis/move_time_summary.py` |
| **Response-time dashboards** (clock, branching, own non-pawn material, ply) | `python src/analysis/movetime_analysis.py` (optional: `--only clock legal_moves own_material ply`) |
| **Ply vs instant-move probability** | `python src/analysis/ply_premove.py` |
| **Tree-derived GSS / VOC / Action Gap / MQ vs RT** (lc0-tree subset) | `sbatch slurm/analysis/tree_values.slurm` (`tree_values_analysis.py`; not part of the full-dataset pipeline) |
| **Slidev deck (LMCOS overview)** | `cd src/analysis/presentations/lmcos-overview && npm install && npm run dev` (symlink `public/figures` per that README) |

The **full-dataset** plots (`move_time_summary`, `movetime_analysis`, `ply_premove`) are wired from **`bash slurm/analysis/board.sh`**. The **generated values** (GSS / VOC / Action Gap / **MQ**, derived from the lc0 search trees) are computed on the **subset of positions that have a tree** via `tree_values_analysis.py` and run separately on the cluster. **MQ moved from FULL to SUBSET**: it is now the Lc0 definition — the post-search root-value loss of the human's played move, `final_Q(played) − final_Q(best) ≤ 0` — not the former Stockfish `pos_with_engine_eval.mq` (`e_win_taken − e_win_best`), which has been retired.

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
└── src/analysis/
    ├── README.md                 # This file
    ├── exploratory/              # Ad hoc analyses (heatmaps, smoke tests, quantify_early_ply); PYTHONPATH=src
    ├── presentations/             # Slidev deck (`lmcos-overview/`) + shared SVG assets
    ├── board.py                  # Act 1: board-feature regressors vs RT (`--all`)
    ├── engine.py                 # Act 2: engine GSS/VOC/gap/MQ vs RT (from SF trees)
    ├── utils/                    # Library: Analyzer, plots, helpers, selected_db, tree_loader (all cts-native)
    ├── preprocess.py             # the moves ETL (processed_moves[_nonzero])
    └── tests/                    # board/engine unit tests

# human-analysis SLURM lives under the unified tree, not here:
#   slurm/analysis/{preprocess.sh, board.sh, tree_values.slurm, logs/}
```

### Where to put new code

| Location | Put here |
| :--- | :--- |
| **`src/analysis/`** | New **ETL / engine / join** entry points. Run with `PYTHONPATH=src` so `import analysis.utils` resolves (no sys.path shim); the batch launcher goes in `slurm/analysis/`. |
| **`src/analysis/`** (top-level `.py`) | New **dashboards, reports, thin CLIs** that read `personal.db` (see `utils/selected_db.py`) and write figures. |
| **`src/analysis/utils/`** | **Reusable** plotting, SQL aggregation patterns, `Analyzer`/`Variable`—**not** one-shot pipeline drivers. |
| **`src/analysis/exploratory/`** | Experiments and one-off plots; follow existing `sys.path` patterns. |
| **`src/analysis/presentations/`** | Slidev decks (`lmcos-overview/`) and presentation assets only—not Python pipeline code. |

---

## 3. Scientific context (why this code exists)

**Goal:** study how humans **allocate time** in chess as a function of **clock pressure** (budget) and **position complexity** (demand).

| Theme | Takeaway |
| :--- | :--- |
| **Heavy tails** | Most moves are fast; long thinks dominate variance. Analyses use **$\log T$** (with `EPSILON` in `utils.helpers`) unless there is a strong reason not to. |
| **Clock** | More remaining clock associates with longer thinks; signal is clearest when **ply** and player heterogeneity are accounted for. |
| **VOC (value of computation)** | Engine-defined gain from deep vs shallow search relates to response time; often discussed vs **ply** “arc” (midgame peak ~40–50). |
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

**Orchestration:** `bash slurm/analysis/preprocess.sh`:

- Clears **`staging_dir`** and prior **`ld-moves_*`** logs in **`slurm/analysis/logs/`** (does not remove **`eval_*`**).
- **`get_games`** (login-node DuckDB settings) → **`games`**.
- Submits a **Slurm array** of **`preprocess.py shard`** tasks; prints periodic **`squeue`** status (`PREPROCESS_SHARD_POLL_SEC`, default 30s).
- **`merge`** then **`process_moves`** on the login node, using **`DUCKDB_MERGE_THREADS`** / **`DUCKDB_MERGE_MEMORY_LIMIT`** (separate from shard task memory).

**Long / fragile SSH sessions:** run the driver detached so merge continues after disconnect, e.g.  
`nohup bash slurm/analysis/preprocess.sh >> slurm/analysis/logs/preprocess_driver.log 2>&1 &`

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

### Engine signals are cts-native (no live engine)

All engine-derived quantities (GSS / Gain / Action Gap / **MQ** / OSS / frac-good) are read off the
**Stockfish search trees** via `utils.tree_loader`, using the cts `"value"` feature
(**value = p_win − p_loss**) and the oracle Q-values — the same convention as the `cts` pipeline.
There is **no live-UCI evaluation** on the human side: the former `engine_eval.py` (a
`p_win + 0.5·p_draw` win-prob evaluator) and its retired DB-table producers
(`build_pos_with_engine_eval`, `build_selected_moves_with_engine` → `*_evaluations` /
`pos_with_engine_eval` / `selected_moves_with_engine`, all unread) were removed.

### Pipeline vs analysis — ordered workflow

1. `bash slurm/analysis/preprocess.sh` (or equivalent `preprocess.py` steps) → `processed_moves[_nonzero]`.
2. **Figures:** `bash slurm/analysis/board.sh` or individual `src/analysis/*.py` tools in §1.

---

## 5. Conventions for contributors and agents

### Design

- **Readability over cleverness:** prefer two explicit functions to one overloaded CLI.
- **Minimal CLIs:** stable, few flags; document defaults in `--help`.
- **`preprocess.py`:** one directory per step for DuckDB spill **and** artifacts (`work_dir` / `staging_dir` above); do not add parallel “alternate tmpdir” tunnels via `**kwargs`.
- **Names:** descriptive columns and variables (`log_clock_ply_residual`), not `x_adj`.
- **Paths:** `os.path.join` + anchor to `__file__`; for `preprocess.py` CLIs rely on `PYTHONPATH=src` (set by the launcher) rather than a sys.path shim.

### Structure

- Separate **load / compute / plot** in analysis code.
- All runnable entry logic under **`if __name__ == "__main__":`**.
- **Bivariate DuckDB plots:** **`Analyzer`** + **`Variable`** (`utils.analysis`); keep aggregations in **SQL** when possible; sample large pulls before plotting.
- **Matplotlib “poster” figures:** call **`apply_poster_style()`** (`utils.helpers`); no top/right spines; use project font sizes.
- **Graphviz / non-matplotlib:** no `apply_poster_style()`; anchor output paths under the repo-root `figures/` directory.

### Statistics

- **Log space** for response time and clock in standard analyses.
- **Fixed effects / de-meaning:** subtract group means (e.g. by ply or player) when exploring confounding.

---

## 6. Quality control (do not regress)

- **Negative response times:** exclude affected games when building analysis tables (pipeline enforces this for core paths).
- **Berserk:** dedicated detection; do not mix berserk games into clock analyses without an explicit policy.
- **Grant more time (GMT):** windowed detection on zero-increment games; tables like `grant_more_time_games` feed joins.

---

## 7. Search-tree visuals (`lmcos`)

Tree tensorization and encoding live under **`src/cts/`** (`cts.core`, `cts.models`) — the former
standalone `lmcos/` tree was merged into this repo (see repo-root `README.md`); `lmcos/src/` no
longer exists as a path. There is no standalone visualization script in this repo. The active
metacontroller (`z_t` encoder + readout) research question that `cts/` exists to answer is tracked
in the repo-root **[`hypotheses.md`](../../hypotheses.md)** and **[`plan.md`](../../plan.md)**, not
here — this file's scope is the human-RT analysis pipeline (`board.py`/`engine.py`) only.

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
