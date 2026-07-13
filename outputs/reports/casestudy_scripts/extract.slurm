#!/bin/bash
# One-off: extract Action-Gap vs Meta-Controller divergent-episode candidates for casestudy.md
# (see extract.py, same directory). Same fit cost as one analysis.evaluate frontier fit
# (eval.slurm's comment: frontier+decodability ~24-30min on its own node-speed variance) -- well
# under 1h, so routed to qos=test per della-qos-time-routing.
#SBATCH --job-name=casestudy-extract
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"
source "${TINY}/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "casestudy-extract on $(hostname) at $(date)"
python outputs/reports/casestudy_scripts/extract.py
echo "Finished at $(date)"
