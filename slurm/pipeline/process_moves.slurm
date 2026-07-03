#!/bin/bash
# Rebuild processed_moves / processed_moves_nonzero from the `moves` table with the
# board-feature columns (material/in_check/prev_move_was_capture in SQL, plus the
# python-chess n_captures_avail / n_checks_avail featurize step). Reads paths from
# the config `preprocess:` section (personal_db unified with analysis). This ONLY
# runs process_moves — not the full get_games/shard/merge orchestrator. Requires
# no other process to hold personal.db open (it opens read_write).
#   sbatch slurm/pipeline/process_moves.slurm
#SBATCH --job-name=process-moves
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # exports CONFIG (defaults to the active run)

# DuckDB uses these for the 145M-row SQL rebuild; the captures/checks featurize
# uses the 32 cores via a python-chess worker pool (os.cpu_count()).
export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export DUCKDB_MEMORY_LIMIT=100GB

echo "process_moves host=$(hostname) config=$CONFIG cpus=${SLURM_CPUS_PER_TASK}"; date
python -m analysis.preprocess process_moves --config "$CONFIG"
echo "PROCESS_MOVES DONE"; date
