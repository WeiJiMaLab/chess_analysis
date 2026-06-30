#!/bin/bash
#SBATCH --job-name=tree-values
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/human/logs/tree_values_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_small/slurm/human/logs/tree_values_%j.err
# CPU partition by design: this job runs NO Leela/lc0 inference. The lc0 evals are
# already baked into the pre-generated .pt trees (value heads + oracle traces); here we
# only read them (torch.load on CPU), compute stats (greedy stopping step, VOC/MQ/action
# gap), join in DuckDB, and plot. A GPU would sit idle. (The GPU work — tree generation
# — lives in the upstream lc0 tree-gen pipeline.) Modest footprint so it backfills fast.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Tree-derived "generated values" (GSS / VOC / Action Gap / MQ) from the canonical
# human_trees vs human RT. Per-tree values are read from the stored tree (MQ also joins
# the human's played move) and cached to parquet, so re-plots need no trees/cluster.
#   N_TREES (default 100000), N_WORKERS (default = cpus-per-task)

set -euo pipefail
PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"
mkdir -p lmcos_small/slurm/human/logs

source /home/hl4291/venv/bin/activate

N_TREES="${N_TREES:-100000}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-40}}"

echo "tree values: n_trees=${N_TREES} n_workers=${N_WORKERS} on $(hostname) at $(date)"
# REFRESH forces a clean recompute of the per-tree parquet (default off). Needed
# whenever tree_loader gains columns (e.g. greedy_frac_good / OSS) the cache predates.
REFRESH_FLAG="${REFRESH:+--refresh}"
PYTHONPATH=lmcos_small/src:lmcos_small/human python3 lmcos_small/human/engine.py \
    --mode tree_values \
    --n-trees "${N_TREES}" \
    --n-workers "${N_WORKERS}" \
    ${REFRESH_FLAG}
echo "Finished at $(date)"
