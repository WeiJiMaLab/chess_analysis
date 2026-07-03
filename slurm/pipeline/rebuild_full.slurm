#!/bin/bash
# One-off full-scale rebuild of the processed-move features into a SIDE-BY-SIDE
# candidate table `full_rebuild` (+ `full_rebuild_nonzero`), WITHOUT touching the
# official processed_moves / processed_moves_nonzero. Exercises the in-process
# captures/checks featurize over every distinct FEN (~145M rows / tens of millions
# of distinct FENs) at real scale. Opens personal.db read_write — no other process
# may hold it. Verify math vs the official tables, then promote if it checks out.
#   sbatch slurm/pipeline/rebuild_full.slurm
#SBATCH --job-name=rebuild-full
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
# Small footprint: DuckDB is capped at 40 GB and spills the rest to scratch, so this
# backfills onto any node instead of waiting behind Priority for a big allocation.
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK:-16}" DUCKDB_MEMORY_LIMIT=40GB
echo "rebuild_full host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} config=$CONFIG"; date
python -m analysis.preprocess process_moves --table full_rebuild
echo "REBUILD_FULL DONE"; date
