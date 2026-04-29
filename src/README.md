# Chess Thinking Dynamics — `src/` handbook

**Audience:** coding agents, research assistants, and human contributors working in `chess_analysis/src/`.

This file is the **single** orientation doc for this tree: layout, commands, conventions, and enough scientific context to interpret what the code is doing.

> [!NOTE]
> **Environment:** activate the project venv from the repo root: `source .venv/bin/activate`.

---

## 1. Quick reference — commands

All paths are relative to the **`chess_analysis/`** repo root (parent of `src/`).

| Task | Command |
| :--- | :--- |
| **Activate env** | `source .venv/bin/activate` |
| **Select games** | `python src/slurm/scripts/preprocess_data.py select_games` |
| **Extract / merge / preprocess moves** | `bash src/slurm/_preprocess.sh` |
| **Regenerate standard figures** | `bash src/slurm/script_analysis.sh` |
| **Move-time histograms** | `python src/move_time_summary.py` |
| **Move-time dashboards** (clock, branching, ply) | `python src/movetime_analysis.py` (optional: `--only clock ply …`) |
| **Ply vs instant-move probability** | `python src/ply_premove.py` |
| **Engine eval (cluster)** | `bash src/slurm/engine_eval.sh` (or `--merge-only` when parquets exist) |
| **Build `selected_moves_with_engine`** | `python src/slurm/scripts/build_selected_moves_with_engine.py` (after eval tables exist) |
| **VOC / parquet feature pipeline** | `python src/slurm/scripts/script_process_data.py` |
| **Merge eval shards only (legacy)** | `python src/slurm/scripts/script_merge_evals.py --engine stockfish` |
| **Search-tree expansion (Graphviz)** | `python src/performance/visualize_tree_expansion.py` (needs `torch`, `graphviz`, `chess`) |

**Outputs:** analysis scripts write figures under **`src/figures/`**.

---

## 2. Repository layout

```text
chess_analysis/
├── .venv/
├── data/                         # Staging parquets, processed outputs (e.g. VOC)
├── README.md                     # Workspace / lmcos overview
└── src/
    ├── README.md                 # This file
    ├── figures/                  # Matplotlib posters, exploratory, performance SVGs
    ├── performance/              # e.g. visualize_tree_expansion.py (lmcos-adjacent)
    ├── exploratory/              # Ad hoc analyses; add src/ to path, then `import utils`
    ├── utils/                    # Library: Analyzer, plots, helpers, features (no pipeline CLIs)
    ├── slurm/
    │   ├── scripts/              # Pipeline Python CLIs (+ _bootstrap.py)
    │   ├── *.sh, *.sbatch       # Orchestration (calls scripts/ with repo-root paths)
    │   └── logs/
    ├── movetime_analysis.py
    ├── move_time_summary.py
    ├── ply_premove.py
    └── ...
```

### Where to put new code

| Location | Put here |
| :--- | :--- |
| **`src/slurm/scripts/`** | New **ETL / engine / join** entry points. Start with `ensure_src()` from `_bootstrap.py` so `import utils` works when run as `python src/slurm/scripts/...` from repo root. |
| **`src/`** (top-level `.py`) | New **dashboards, reports, thin CLIs** that read `personal.db` and write figures. |
| **`src/utils/`** | **Reusable** plotting, SQL aggregation patterns, `Analyzer`/`Variable`, feature helpers—**not** one-shot pipeline drivers. |
| **`src/exploratory/`** | Experiments and one-off plots; follow existing `sys.path` patterns. |

---

## 3. Scientific context (why this code exists)

**Goal:** study how humans **allocate time** in chess as a function of **clock pressure** (budget) and **position complexity** (demand).

| Theme | Takeaway |
| :--- | :--- |
| **Heavy tails** | Most moves are fast; long thinks dominate variance. Analyses use **$\log T$** (with `EPSILON` in `utils.helpers`) unless there is a strong reason not to. |
| **Clock** | More remaining clock associates with longer thinks; signal is clearest when **ply** and player heterogeneity are accounted for. |
| **VOC (value of computation)** | Engine-defined gain from deep vs shallow search relates to think time; often discussed vs **ply** “arc” (midgame peak ~40–50). |
| **Scale** | Core DuckDB pipelines target on the order of **~10⁸ moves**; always prefer **SQL-side** aggregation and sampling. |

For publication-style figures, dashboards use a **1×3** layout (raw trend, quantile bins, scatter) from `utils.analysis.Analyzer`, or **1×2** for ply-stage views (see `movetime_analysis.ply_movetime`).

---

## 4. Data stack and pipeline (for implementers)

### Databases

- **`core` / `lichess.db`:** read-only Lichess mirror (multi‑TB).
- **`personal.db`:** project workspace; `selected_games`, move tables, engine eval tables, etc. Path is often cluster-specific; scripts default to a scratch path—check each CLI.

### Sharded extraction

1. **`preprocess_data.py select_games`** → `selected_games` in `personal.db`.
2. **`bash src/slurm/_preprocess.sh`** → Slurm **`preprocess_shard.sbatch`** (`process_shard`), then local **merge / berserk / preprocess** into analysis-ready tables.
3. **QC:** exclude games with **any negative** `move_time` where policy requires clean clocks.

Typical filters for `selected_games` (verify in `preprocess_data.py` defaults): Nov–Dec 2023 window, **10+0**, both Elos **≥ 2000**.

### Engine evaluation

- **Worker:** `src/slurm/scripts/script_engine_eval.py` → shard parquets.
- **Merge:** same module **`merge`** subcommand (or legacy `script_merge_evals.py`) → `{stockfish,lc0}_evaluations` in `personal.db`.
- **Join to moves:** `build_selected_moves_with_engine.py` → **`selected_moves_with_engine`** (not created by merge alone).

Details: **`bash src/slurm/engine_eval.sh`**, **`engine_eval_shard.sbatch`**, logs under **`src/slurm/logs/`**.

### Pipeline vs analysis — ordered workflow

1. `python src/slurm/scripts/preprocess_data.py select_games`
2. `bash src/slurm/_preprocess.sh`
3. **Optional:** `python src/slurm/scripts/script_process_data.py` (parquet VOC path; not required for core DuckDB dashboards)
4. **Optional:** `bash src/slurm/engine_eval.sh` → `python src/slurm/scripts/build_selected_moves_with_engine.py`
5. **Figures:** `bash src/slurm/script_analysis.sh` or individual `src/*.py` tools in §1.

---

## 5. Conventions for contributors and agents

### Design

- **Readability over cleverness:** prefer two explicit functions to one overloaded CLI.
- **Minimal CLIs:** stable, few flags; document defaults in `--help`.
- **Names:** descriptive columns and variables (`log_clock_ply_residual`), not `x_adj`.
- **Paths:** `os.path.join` + anchor to `__file__`, or use `_bootstrap.src_root()` / `project_root()` in **`slurm/scripts/`**.

### Structure

- Separate **load / compute / plot** in analysis code.
- All runnable entry logic under **`if __name__ == "__main__":`**.
- **Bivariate DuckDB plots:** **`Analyzer`** + **`Variable`** (`utils.analysis`); keep aggregations in **SQL** when possible; sample large pulls before plotting.
- **Matplotlib “poster” figures:** call **`apply_poster_style()`** (`utils.helpers`); no top/right spines; use project font sizes.
- **Graphviz / non-matplotlib:** no `apply_poster_style()`; still anchor output paths; prefer `src/figures/performance/` for tree exports.

### Statistics

- **Log space** for move time and clock in standard analyses.
- **Fixed effects / de-meaning:** subtract group means (e.g. by ply or player) when exploring confounding; mirror **`src/exploratory/phase_segmented_*`** patterns if relevant.

---

## 6. Quality control (do not regress)

- **Negative move times:** exclude affected games when building analysis tables (pipeline enforces this for core paths).
- **Berserk:** dedicated detection; do not mix berserk games into clock analyses without an explicit policy.
- **Grant more time (GMT):** windowed detection on zero-increment games; tables like `grant_more_time_games` feed joins.

---

## 7. Search-tree expansion (`lmcos`-adjacent)

**Script:** `src/performance/visualize_tree_expansion.py`  
Loads a **single** `.pt` (legacy `PretrainExample` or **`cts_raw_pretrain_example_v1`** dict), rebuilds `SearchTree` (`lmcos/tree.py`), writes **Graphviz** `.svg`/`.gv` under a chosen output dir (default under `src/figures/performance/`).

**Deps:** `torch`, Python `graphviz`, system `dot`, `python-chess`. Example: `pip install torch graphviz chess`.

```bash
python src/performance/visualize_tree_expansion.py --pt-path /path/to/example.pt --name myrun --steps 5
```

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
