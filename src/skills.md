# Skills: Chess Thinking Dynamics Analysis

This document serves as a comprehensive guide for developers and researchers working in this repository. It covers the project's structure, coding standards, data pipeline, and visual philosophy.

---

## 1. Project Overview & Philosophy
The goal of this project is to analyze human resource allocation (time) in chess, focusing on **Clock Pressure** (budget) and **Value of Computation** (demand).

### Core Design Principles
*   **Readability First**: Simple, explicit code is preferred over complex abstractions.
*   **Minimalist Interfaces**: Scripts should do one thing well with stable, predictable CLI interfaces.
*   **Path Portability**: All paths **must** be constructed using `os.path.join()` and anchored via `os.path.abspath(__file__)` to ensure functionality across local and cluster environments.
*   **Linear Logic**: Avoid "clever" or overly generalized frameworks. Follow the patterns established in `src/utils/`.

---

## 2. Repository Structure

```text
.
├── .venv/                  # Virtual environment (Python 3.10+)
├── README.md               # Main project overview (repo root)
├── data/                   # Staging parquet shards and processed data
└── src/
    ├── README.md           # Technical analysis overview
    ├── skills.md              # This guide
    ├── figures/            # Output for matplotlib/posters, exploratory, performance SVGs, etc.
    │   ├── exploratory/
    │   └── performance/    # e.g. Graphviz output from performance/visualize_tree_expansion.py
    ├── performance/      # Meta-controller / lmcos-adjacent tooling (tree expansion viz, future benchmarks)
    ├── exploratory/        # Ad hoc analysis scripts
    ├── slurm/              # Batch scripts (preprocess, engine_eval_shard.sbatch, engine_eval.sh, …)
    ├── utils/              # Core library: helpers, plotting, preprocessing
    ├── clock_movetime.py   # Analysis: Clock effect
    ├── ply_movetime.py     # Analysis: Mid-game arc
    └── ...                 # Other specific analysis scripts
```

---

## 3. Environment & Dependencies
*   **Virtual Environment**: Always activate the `.venv` in the root: `source .venv/bin/activate`.
*   **DuckDB**: The primary engine for data manipulation. 
    *   `core` (`lichess.db`): Read-only, multi-terabyte database.
    *   `personal.db`: Local "write-back" database for game subsets and intermediate results.
*   **PyTorch & Graphviz** (as needed): Scripts under `src/performance/` that load `.pt` tree checkpoints (e.g. `visualize_tree_expansion.py`) require `torch`, the Python `graphviz` package, and a system `graphviz` install (`dot` on `PATH`), plus `python-chess` for Unicode board text in node labels. Install as needed, e.g. `pip install torch graphviz` and `pip install chess` (see `requirements-jupyter.txt` for `chess`).

---

## 4. The Data Pipeline
The analysis follows a strict sequential pipeline to handle 100M+ moves:

1.  **Selection**: `python src/utils/preprocess_data.py select_games`
    *   Filters games by date, Elo, and time control from `core`.
2.  **Extraction**: `bash src/slurm/script_preprocess.sh`
    *   Launches Slurm arrays to extract moves into shards.
    *   Merges shards into `personal.db`.
    *   Runs **Berserk Detection** and **Engine FEN construction**.
3.  **VOC Generation**: `python src/script_process_data.py`
    *   Calculates complexity features (Value of Computation) using Stockfish.
4.  **Analysis**: `bash src/slurm/script_analysis.sh`
    *   Executes individual analysis scripts (Clock, Ply, VOC, etc.).

---

## 5. Coding Standards

### Structural Requirements
*   **Strict Modularity**: Scripts must separate **Data Loading**, **Modeling/Analysis**, and **Plotting** into distinct functions.
*   **Main Block**: All execution logic must be inside `if __name__ == "__main__":`.
*   **Explicit Naming**: Use descriptive names (e.g., `log_clock_ply_controlled`) instead of shorthand.

### Statistical & Visual Standards
*   **Log-Space**: Operate in $\log$ space for move times ($\log T$) and clock times to normalize heavy tails. Use `EPSILON = 1e-6` from `utils.helpers`.
*   **Fixed Effects**: Use the "de-meaning" pattern (subtracting group means) to control for player profiles or game stages.
*   **Poster Style**: Every plot **must** call `apply_poster_style()` from `src.utils.helpers`.
    *   No top/right borders.
    *   Specific font sizes for publication-ready "Posters."
*   **Analyzer Framework**: For bivariate analysis, use `utils.analysis.Analyzer`. 
    *   **SQL-Native**: Binning and aggregations must be done in SQL where possible for performance.
    *   **Memory Efficiency**: Always sample large datasets (e.g., `USING SAMPLE`) at the SQL level before pulling into DataFrames for visualization.
    *   **Variable Class**: Use the `Variable` dataclass to define metrics and transformations.
*   **Standard Layouts**:
    *   **1x3 Dashboard**: Raw Trend, Quantile Bins, and Scatterplot.
    *   **1x2 Side-by-Side**: Used for game stage dynamics (Ply).
*   **Non-matplotlib diagrams**: Graph-based exports (e.g. **Graphviz** `.svg` from `src/performance/visualize_tree_expansion.py`) do not use `apply_poster_style()`; keep labels readable (monospace helps for aligned Unicode boards) and write outputs under `src/figures/performance/` using path anchoring as usual.

---

## 6. Common Commands

| Task | Command |
| :--- | :--- |
| **Activate Env** | `source .venv/bin/activate` |
| **Select Games** | `python src/utils/preprocess_data.py select_games` |
| **Run Full Pipeline** | `bash src/slurm/script_preprocess.sh` |
| **Run All Analysis** | `bash src/slurm/script_analysis.sh` |
| **Engine eval: shards + merge + `selected_moves_with_engine`** | `bash src/slurm/engine_eval.sh` (or `bash src/slurm/engine_eval.sh --merge-only` after parquets exist) |
| **Plot Clock Effect** | `python src/clock_movetime.py` |
| **Plot Ply Effect** | `python src/ply_movetime.py` |
| **Tree expansion (lmcos `.pt`, Graphviz + Unicode boards)** | `python src/performance/visualize_tree_expansion.py` (see `src/README.md`, section 4) |

---

## 7. Quality Control
*   **Negative Times**: Always exclude games with *any* negative move times (lag compensation artifacts).
*   **Berserk Games**: Flag and handle Berserk games separately using range-based clock detection (`[-5.0, +0.1]` tolerance on Move 2).
*   **Grant More Time**: Detect GMT anomalies using window functions (`lag()`), ensuring logic is restricted to zero-increment games.
