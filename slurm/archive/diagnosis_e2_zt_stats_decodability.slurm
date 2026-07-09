#!/bin/bash
# DIAGNOSTIC ONLY (plan.md E2) -- NOT a pipeline stage. Runs the already-built
# src/cts/analysis/zt_stats_decodability.py probe on the real (full, ~5.6K-episode)
# validation split. Submitted to SLURM (rather than run on the login node) because a
# direct interactive run OOM'd against the per-session cgroup cap (~302G) -- the login
# node/session is shared across the 3 parallel diagnosis subagents tonight.
#
#   sbatch slurm/diagnosis_e2_zt_stats_decodability.slurm
#SBATCH --job-name=diag-e2-zt-stats
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

echo "diag_e2_zt_stats_decodability host=$(hostname)"; date

python -m cts.analysis.zt_stats_decodability --config "$CONFIG" \
    --out-dir "$TINY/outputs/figures/minply15_maxply75/diagnosis"

echo "DIAG-E2 DONE"; date
