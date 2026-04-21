# Chess Thinking Dynamics: Analysis Overview

This document summarizes the Value of Computation (VOC) analysis pipeline, standardized metric definitions, and current analytical results. All analyses utilize a "Poster-Style" visual framework to maintain consistency across different metrics.

---

## 1. Experimental Pipeline

The analysis is built on a high-performance parallelized pipeline that processes chess games to compute deep engine metrics using Stockfish 14.

-   **Data Source**: `moves_500.parquet` (36,728 total rows).
-   **Processing**: Parallelized across CPU cores using the `multiprocessing.Pool` initializer pattern to reuse engine instances.
-   **Core Metrics**: Merged into [processed_moves_500.parquet](file:///home/hl4291/chess_analysis/data/processed/processed_moves_500.parquet).

---

## 2. Standardized Analysis (2x2 Quad-Views)

To ensure comparability, every metric is analyzed using a standardized 2x2 grid defined in `src/utils/plots.py`. This layout isolates specific aspects of the relationship:

| Quadrant | Analysis Type | Purpose |
| :--- | :--- | :--- |
| **[0, 0]** | Hexbin Density | Visualizes the raw correlation and data density. |
| **[0, 1]** | Binned Trend (Raw) | Shows mean $\log T$ across quantiles of the raw x-variable. |
| **[1, 0]** | Ply-Stability Plot | Shows the regression slope ($\beta$) across game stages (Move Ply). |
| **[1, 1]** | Binned Trend (Rank) | Shows mean $\log T$ across normalized quantile ranks (0 to 1). |

---

## 3. Analysis Phases

### A. Temporal Context (Phase 1)
Before analyzing complexity, we quantify the impact of "environmental" factors: clock pressure and game stage.

1.  **Clock-Time Pressure**: Relationship between remaining clock and move time ($\log T$).
    -   **Standard Analysis**: [Poster](file:///home/hl4291/chess_analysis/src/figures/clocktime_movetime/clock_standard_analysis.png)
    -   **Ply-Controlled**: Removing the effect of game stage. [Poster](file:///home/hl4291/chess_analysis/src/figures/clocktime_movetime/clock_ply_controlled.png)
    -   **Double-Controlled**: Removing both stage and individual player speed. [Poster](file:///home/hl4291/chess_analysis/src/figures/clocktime_movetime/clock_double_controlled.png)

2.  **Move Ply Impact**: We find a characteristic quadratic "arc" in thinking time, peaking in the mid-game (around ply 40-50).
    -   **Quadratic Fit**: [Poster](file:///home/hl4291/chess_analysis/src/figures/ply_movetime/ply_impact_comparison.png)

### B. Computational Complexity (Phase 2)
We evaluate three metrics to quantify the "difficulty" of a position and its impact on human response time.

-   **VOC (Single Sample)**: Evaluation difference between a shallow (d1) and deep (d14) search. [Poster](file:///home/hl4291/chess_analysis/src/figures/voc_movetime/standard_voc_single_sample.png)
-   **VOC (Consideration Set)**: Entropy-like measure of value-of-computation across the top-5 candidates. [Poster](file:///home/hl4291/chess_analysis/src/figures/voc_movetime/standard_voc_consideration_set.png)
-   **Best-Two Difference**: Evaluation gap between the top 2 moves. Lower values indicate higher "competition" between options. [Poster](file:///home/hl4291/chess_analysis/src/figures/voc_movetime/standard_best_two_diff.png)

---

## 4. Statistical Validation

A **Linear Mixed Effects Model** ([clocktime_movetime.py](file:///home/hl4291/chess_analysis/src/clocktime_movetime.py)) is used to formally quantify effects while controlling for player variance.

**Key Findings**:
-   **Clock Coefficient**: $\sim 0.56$ ($p < 0.001$), indicating robust scaling of thinking time with budget.
-   **Move Ply**: Significant positive linear and negative quadratic terms ($p < 0.001$), confirming the mid-game arc observed in Phase 1.
