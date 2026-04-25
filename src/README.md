# Chess Thinking Dynamics: Analysis Overview

Welcome! This directory contains the analysis pipeline and results for our study on **Chess Thinking Dynamics**. 

> [!NOTE]
> **Environment**: A native Python virtual environment is available in the project root (`.venv`). Always ensure it is activated before running scripts: `source .venv/bin/activate`.

Our goal is to understand how human players allocate their most precious resource—**time**—based on external pressure (the clock) and internal demand (the complexity of the position).

---

## 1. The Behavioral Findings

We treat chess as a "natural laboratory" for studying resource allocation. Here is what we've discovered so far:

### The Thinking Distribution
Thinking time is extremely heavy-tailed. Most moves are fast, but "long thinks" can span minutes. We use **natural log-transformations ($\log T$)** to normalize this variance, allowing us to interpret residuals as percentage deviations from the baseline.

### Clock Pressure: The "Budget" Effect
How does your remaining clock time affect how long you think? 
- **The Finding**: We find a robust correlation between remaining clock and think time.
- **The Insight**: If you have more time on your clock, you spend more time on the current move.
- **The Paradox**: This relationship is only visible once we control for both the game stage (Ply) and the individual player's speed profile. Without these controls, the signal is masked by opening theory and player-level speed differences.

### Value of Computation (VOC): The "Demand" Effect
Beyond the clock, the primary driver of thinking time is **complexity**. We quantify this as the "Value of Computation"—the gain in win probability discovered by a deep engine search (Depth 14) vs. a shallow heuristic (Depth 1).
- **The Finding**: Move time scales with the square root of prospective gain.
- **Stability**: Unlike the clock effect, the VOC effect is remarkably stable across all stages of the game.

### The Mid-game "Arc"
Thinking time isn't constant. It follows a characteristic quadratic arc, peaking around ply 40-50 (the height of the mid-game) before tapering off as the board simplifies in the endgame. Our current analysis validates this across a massive dataset of **98.5 million moves** from 1.3 million games.

---

## 2. Our Visual Framework: The "Poster-Style"

To maintain consistency across metrics, we analyze every variable using a standardized **1x3 Horizontal Dashboard** (defined in `src/utils/analysis.py`) or a **Side-by-Side Analysis** for game stage dynamics. These layouts are designed for publication-quality "Poster" presentation:

| Panel | Name | Purpose |
| :--- | :--- | :--- |
| **Panel 1** | Raw Trend | Shows the average think time at each value of the metric. |
| **Panel 2** | Quantile Bins | Aggregates data into 20 equal-sized bins to show the robust trend. |
| **Panel 3** | Scatterplot | Density view of raw data, sampled to 100k points for performance. |

For game stage analysis (`ply_movetime.py`), we use a side-by-side view comparing raw ply trends with quantile-binned perspectives, utilizing shaded 95% Confidence Interval regions for visual clarity.

---

## 3. Data Infrastructure & Pipeline

Handling millions of moves requires more than just a simple script. We use a high-performance parallelized pipeline:

### The Database Stack
- **Primary Database (`core`)**: A multi-terabyte DuckDB instance (`lichess.db`) containing the full history of the Lichess Open Database.
- **Personal Database (`personal.db`)**: A localized DuckDB instance used for "write-back" operations, storing specific game subsets and intermediate analysis results.
- **Engine-Only FEN**: We store a 4-field FEN (Board, Turn, Castling, En Passant) optimized for chess engines, excluding the move counters to minimize storage.
- **On-the-fly Transformations**: To maintain data integrity and flexibility, we store raw metrics (e.g., `player_clock_time`) and compute transformations like $\log(T)$ or $\log(\text{Clock})$ on-the-fly using the `Analyzer` framework's SQL-native engine.
- **SQL-Native Performance**: To process nearly 100 million moves in seconds, we utilize SQL-level window functions (`ntile`) and high-speed aggregations directly in DuckDB within the `Analyzer` core.

### Sharded Extraction (Slurm)
To handle "larger-than-memory" move extraction, we use a Slurm-based sharding process:
1. **Select Games**: `python src/utils/preprocess_data.py select_games` builds `selected_games` from `core.games`.
2. **Pipeline Launch**: `bash src/slurm/script_preprocess.sh` manages the full extraction cycle:
   - Submits a Slurm array (`preprocess_shard.sbatch` using `process_shard`) to extract moves.
   - Once shards are ready, it locally runs **merge**, **berserk detection**, and **preprocessing** into final analysis tables.
3. **Quality Control**: We automatically **exclude entire games** that contain any negative move times (artifacts of lag compensation or manual clock additions) to ensure the integrity of our behavioral models.

Current `selected_games` filter (used by `select_games`):
- Date window: `utc_datetime >= '2023-11-01'` and `< '2023-12-31'` (Nov-Dec 2023).
- Time control: `initial_clock = 600` and `clock_increment = 0` (10+0).
- Strength floor: `white_elo >= 2000` and `black_elo >= 2000`.
- Projection: `SELECT gid, utc_datetime`.

### Engine Evaluation (Stockfish & lc0)
We generate high-quality move evaluations by distributing search tasks across the cluster:
- **Unified Pipeline**: `src/script_engine_eval.py` supports both Stockfish and `lc0`.
- **Determinism**: We ensure reproducible results by clearing the Stockfish hash table before every position search.
- **Scale**: The pipeline is designed to evaluate over **116 million unique positions**, utilizing Slurm arrays to process 1,000,000 positions per node in parallel.

---

## 4. How to Run the Analysis

1. **Select Games**: `python src/utils/preprocess_data.py select_games` (Builds `selected_games` from `core.games` using Nov-Dec 2023, 10+0, both Elo ≥ 2000).
2. **Extraction & Preprocess**: `bash src/slurm/script_preprocess.sh` (Pulls moves for selected games, merges, identifies berserkers, and precalculates features).
3. **Engine Evaluation**: 
   - `sbatch src/slurm/engine_eval_array.sbatch` (Runs Stockfish evaluation at Depth 5).
   - Use `--export=ALL,ENGINE=lc0` for `lc0` evaluations.
4. **Analysis**: 
   - `bash src/slurm/script_analysis.sh` (Runs all core analysis scripts below).
   - `python src/move_time_summary.py` (SQL-binned move-time / log move-time histograms).
   - `python src/clock_movetime.py` (Player clock vs. move time).
   - `python src/npossiblemoves_movetime.py` (Number of legal moves vs. raw move time).
   - `python src/ply_movetime.py` (Ply stage vs. log move time).
   - `python src/ply_premove.py` (Ply stage vs. probability of "instant moves").
   - `python src/voc_movetime.py` (VOC vs. think time).

Results and figures are saved to `src/figures/`.

---

## 5. Multi-Engine Benchmarks (Depth 5)

We compared Stockfish 14 and `lc0 v0.32.1` to optimize our evaluation strategy:

| Engine | Throughput (pos/s) | Notes |
| :--- | :--- | :--- |
| **Stockfish (No Clear Hash)** | **~800** | Highly inconsistent at low depth. |
| **Stockfish (Clear Hash)** | **~75** | **100% Deterministic**. Used for production. |
| **lc0 v0.32.1 (A100)** | **~30** | Deterministic but slower for sequential UCI. |

---

## 6. Guide for Contributors

This repository follows a strict "Readability First" philosophy. If you are adding new analysis scripts or modifying the pipeline, you are expected to adhere to these design principles:

### Design Philosophy
- **Readability over Flexibility**: We prefer simple, explicit code over complex abstractions. Do not use overly generic "Swiss-army knife" functions; it is better to have two clear, slightly redundant functions than one "clever" function with ten optional flags.
- **Informative Naming**: Model names, variables, and columns must be descriptive. For example, use `log_clock_ply_controlled` instead of `x_adj`.
- **Minimalist Interfaces**: Avoid adding dozens of CLI arguments. A script should do one thing well with a stable, predictable interface.

### Structural Requirements
- **Strict Modularity**: Every script must separate **Data Loading**, **Modeling/Analysis**, and **Plotting** into distinct, well-scoped functions. 
- **Path Portability (Non-Negotiable)**: All paths must be constructed using `os.path.join()` and anchored to the script's location via `os.path.abspath(__file__)`. This ensures that scripts remain functional whether run locally or in a Slurm batch environment.
- **Main Block Execution**: All execution logic must reside within `if __name__ == "__main__":` blocks.

- **The `Analyzer` Framework**: For bivariate analyses, use the `Analyzer` and `Variable` classes in `utils.analysis`. This ensures that SQL aggregations, quantile binning, and plotting remain consistent across all metrics.
- **Standardized Aesthetics**: All plots **must** call `apply_poster_style()` from `utils.helpers`. We use a specific visual language (no top/right borders, specific font sizes) to ensure figures are "poster-ready."
- **Import Robustness**: Scripts intended for direct execution from `src/utils/` must use the hybrid import pattern (`try...except ImportError`) to handle both modular and script-level execution.
- **Memory Efficiency**: The `Analyzer` framework automatically samples large datasets for visualization (100k points for scatterplots), preventing OOM errors on compute nodes while maintaining statistical integrity.
- **Fixed-Effect Pipeline**: When controlling for game stage or player speed, use the "de-meaning" pattern (subtracting the group mean) demonstrated in `fe_clocktime_movetime.py`.
- **Log-Space Normalization**: Always operate in $\log$ space for move times and clock times unless there is a specific theoretical reason to do otherwise. We use a standardized `EPSILON = 1e-6` from `utils.helpers` for all log transforms.
