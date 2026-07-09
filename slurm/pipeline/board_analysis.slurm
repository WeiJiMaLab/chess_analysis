#!/bin/bash
# Board step (--plot): the four RT analyses over filtered_moves ⋈ board_features.
# Depends on board_merge (built board_features from the featurize array), which
# depends on filter_and_sample (filtered_moves). Reads only the DB; no trees/model.
#   sbatch --dependency=afterok:<board_merge> slurm/pipeline/board_analysis.slurm
# Figures land under human_analysis.figures_dir/board.
#SBATCH --job-name=board-analysis
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 64G total: correlation_matrix ranks 11 columns (RANK() OVER unpartitioned ORDER BY,
# twice) over only a 1M-row SAMPLE of the windowed table (not the full ~89M rows —
# see correlation_matrix()'s `USING SAMPLE 1000000 ROWS`), so the working set is far
# smaller than the stale comment this replaced implied. A completed run (job
# 10866194, 2026-07-09) measured MaxRSS=~39G; 64G leaves real headroom under that for
# pandas/matplotlib/OS without requesting 2x more than any observed run has needed.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"

source "${TINY}/slurm/helpers/setup_env.sh"
export DUCKDB_MEMORY_LIMIT=48GB

echo "board analysis (--plot) on $(hostname) config=$CONFIG at $(date)"
python -m analysis.board --plot
echo "Finished at $(date)"
