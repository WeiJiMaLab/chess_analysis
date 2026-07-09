#!/bin/bash
# DIAGNOSTIC ONLY (plan.md S1) -- NOT a pipeline stage. Runs the stats-ablation probe
# (src/cts/analysis/stats_ablation.py) on the real, full (~5.6K-episode) validation split.
# Submitted to SLURM (not the shared login-node shell) per this session's convention --
# also fixes the unconstrained-BLAS-threading slowdown (torch.set_num_threads(4) in-script;
# a 400-episode smoke test on the login node took 29s after the fix, vs 28+ min hung before it).
#
#   sbatch slurm/diagnosis_s1_stats_ablation.slurm
#SBATCH --job-name=diag-s1-stats-ablation
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

echo "diag_s1_stats_ablation host=$(hostname)"; date

python -m cts.analysis.stats_ablation --config "$CONFIG" \
    --out-dir "$TINY/outputs/figures/minply15_maxply75/diagnosis"

echo "DIAG-S1 DONE"; date
