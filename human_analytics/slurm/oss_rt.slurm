#!/bin/bash
#SBATCH --job-name=oss-rt
#SBATCH --output=/home/hl4291/chess_analysis/human_analytics/slurm/logs/oss_rt_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/human_analytics/slurm/logs/oss_rt_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=02:00:00

# OSS (oracle stop step) from the ysagiv lc0 trees vs human RT, at scale.
# OSS is derived per-tree from the stored oracle trace (budgeted DP oracle);
# the cost is loading individual .pt trees, so we parallelize across CPUs.
#   N_TREES (default 100000), N_WORKERS (default = cpus-per-task)

set -euo pipefail
PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"
mkdir -p human_analytics/slurm/logs

source /home/hl4291/venv/bin/activate

N_TREES="${N_TREES:-100000}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-40}}"

echo "OSS↔RT: n_trees=${N_TREES} n_workers=${N_WORKERS} on $(hostname) at $(date)"
PYTHONPATH=human_analytics python3 human_analytics/ysagiv_oss_rt.py \
    --n-trees "${N_TREES}" \
    --n-workers "${N_WORKERS}" \
    --output human_analytics/figures/oss_vs_rt.png
echo "Finished at $(date)"
