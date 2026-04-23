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
- **The Finding**: We find a robust "Elasticity of Thinking" $\approx 0.54$. 
- **What it means**: If you have 10% more time on your clock, you tend to spend about 5.4% more time on the current move.
- **The Paradox**: This relationship is only visible once we control for both the game stage (Ply) and the individual player's speed profile. Without these controls, the signal is masked by opening theory and player-level speed differences.

### Value of Computation (VOC): The "Demand" Effect
Beyond the clock, the primary driver of thinking time is **complexity**. We quantify this as the "Value of Computation"—the gain in win probability discovered by a deep engine search (Depth 14) vs. a shallow heuristic (Depth 1).
- **The Finding**: Move time scales with the square root of prospective gain.
- **Stability**: Unlike the clock effect, the VOC effect is remarkably stable across all stages of the game.

### The Mid-game "Arc"
Thinking time isn't constant. It follows a characteristic quadratic arc, peaking around ply 40-50 (the height of the mid-game) before tapering off as the board simplifies in the endgame. Our current analysis validates this across a massive dataset of **98.5 million moves** from 1.3 million games.

---

## 2. Our Visual Framework: The "Poster-Style"

To maintain consistency across metrics, we analyze every variable using a standardized **2x2 Quad-View** (defined in `src/utils/plots.py`) or a **Side-by-Side Analysis** for game stage dynamics. These layouts are designed for publication-quality "Poster" presentation:

| Quadrant / Panel | Name | Purpose |
| :--- | :--- | :--- |
| **Top-Left** | Hexbin Density | Shows the raw correlation and where the data is "clumped." |
| **Top-Right** | Binned Trend (Raw) | Shows how $\log T$ changes across quantiles of the metric. |
| **Bottom-Left** | Ply-Stability Plot | Shows the regression slope ($\beta$) at different stages of the game. |
| **Bottom-Right** | Binned Trend (Rank) | Standardizes the x-axis to a 0-1 rank for easier comparison between different metrics. |

For game stage analysis (`ply_movetime.py`), we use a side-by-side view comparing raw ply trends with quantile-binned perspectives, utilizing shaded 95% Confidence Interval regions for visual clarity.

---

## 3. Data Infrastructure & Pipeline

Handling millions of moves requires more than just a simple script. We use a high-performance parallelized pipeline:

### The Database Stack
- **Primary Database (`core`)**: A multi-terabyte DuckDB instance (`lichess.db`) containing the full history of the Lichess Open Database.
- **Personal Database (`personal.db`)**: A localized DuckDB instance used for "write-back" operations, storing specific game subsets and intermediate analysis results.
- **SQL Pre-calculation**: To process nearly 100 million moves in seconds, we utilize SQL-level window functions (`ntile`) and aggregations directly in DuckDB to generate "stat-tables" before plotting.

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

Smoke-test status: this query's count matches `selected_games` exactly (`1,309,489` games).

### VOC Generation
Computational complexity metrics are generated by distributing Stockfish 14 evaluations across a compute cluster (`voc_compute.sbatch`), computing the delta between shallow and deep searches for each position.

---

## 4. How to Run the Analysis

1. **Select Games**: `python src/utils/preprocess_data.py select_games` (Builds `selected_games` from `core.games` using Nov-Dec 2023, 10+0, both Elo ≥ 2000).
2. **Extraction & Preprocess**: `bash src/slurm/script_preprocess.sh` (Pulls moves for selected games, merges, identifies berserkers, and precalculates features).
3. **VOC Generation**: `python src/script_process_data.py` (Calculates VOC and complexity features using Stockfish).
4. **Analysis**: 
   - `python src/move_time_summary.py` (SQL-binned move-time / log move-time histograms → `figures/move_time_summary/`).
   - `python src/clock_movetime.py` (Player or opponent clock vs. move time, 2×2 panel → `figures/clock_movetime/`; use `--opp` for opponent clock, `--include_zeroT` as needed; outputs `combined.png`, `combined_opp.png`, `combined_include_zeroT.png`, `combined_include_zeroT_opp.png`).
   - `python src/npossiblemoves_movetime.py` (Number of legal moves vs. raw move time, 2×2 panel → `figures/npossiblemoves_movetime/`; use `--include_zeroT` as needed; outputs `combined.png`, `combined_include_zeroT.png`).
   - `python src/ply_movetime.py` (Ply stage vs. log move time → `figures/ply_movetime/`; use `--include_zeroT` as needed).
   - `python src/voc_movetime.py` (VOC vs. think time).

Results and figures are saved to `src/figures/`.

---

## 5. Guide for Contributors

This repository follows a strict "Readability First" philosophy. If you are adding new analysis scripts or modifying the pipeline, you are expected to adhere to these design principles:

### Design Philosophy
- **Readability over Flexibility**: We prefer simple, explicit code over complex abstractions. Do not use overly generic "Swiss-army knife" functions; it is better to have two clear, slightly redundant functions than one "clever" function with ten optional flags.
- **Informative Naming**: Model names, variables, and columns must be descriptive. For example, use `log_clock_ply_controlled` instead of `x_adj`.
- **Minimalist Interfaces**: Avoid adding dozens of CLI arguments. A script should do one thing well with a stable, predictable interface.

### Structural Requirements
- **Strict Modularity**: Every script must separate **Data Loading**, **Modeling/Analysis**, and **Plotting** into distinct, well-scoped functions. 
- **Path Portability (Non-Negotiable)**: All paths must be constructed using `os.path.join()` and anchored to the script's location via `os.path.abspath(__file__)`. This ensures that scripts remain functional whether run locally or in a Slurm batch environment.
- **Main Block Execution**: All execution logic must reside within `if __name__ == "__main__":` blocks.

### Statistical & Visual Standards
- **Standardized Aesthetics**: All plots **must** call `apply_poster_style()` from `utils.helpers`. We use a specific visual language (no top/right borders, specific font sizes) to ensure figures are "poster-ready."
- **Fixed-Effect Pipeline**: When controlling for game stage or player speed, use the "de-meaning" pattern (subtracting the group mean) demonstrated in `fe_clocktime_movetime.py`.
- **Log-Space Normalization**: Always operate in $\log$ space for move times and clock times unless there is a specific theoretical reason to do otherwise.

When generating new code for this directory, do not suggest "highly flexible" or "generalized" frameworks. Instead, provide linear, readable, and modular scripts that follow the existing patterns in `src/utils/`. Prioritize code that can be understood at a glance.
