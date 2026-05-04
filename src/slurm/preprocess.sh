#!/usr/bin/env bash
#
# Orchestrates preprocess.py: get_games → Slurm shard array → merge (moves + processed_moves / processed_moves_nonzero).
#
#   bash src/slurm/preprocess.sh
#
# Staging path must match preprocess.py config["staging_dir"].
# Array size: tasks 0..ARRAY_END → export PREPROCESS_TOTAL_SHARDS=$((ARRAY_END + 1)) inside the job.

set -euo pipefail

# --- paths & layout ---
PROJECT_DIR="/home/hl4291/chess_analysis"
LOG_DIR="${PROJECT_DIR}/src/slurm/logs"
# Keep in sync with preprocess.py main() config["staging_dir"].
STAGING_DIR="/scratch/gpfs/GRIFFITHS/hl4291/tmp/ld_moves_shard"

# --- Slurm array (0 through ARRAY_END) ---
ARRAY_END=10
TOTAL_SHARDS=$((ARRAY_END + 1))

# --- per-task Slurm resources ---
SLURM_CPUS_PER_TASK=20
SLURM_MEM="8G"
SLURM_TIME_LIMIT="01:15:00"

# --- DuckDB: login node vs array worker ---
DUCKDB_LOGIN_THREADS="${DUCKDB_THREADS:-40}"
DUCKDB_LOGIN_MEM="${DUCKDB_MEMORY_LIMIT:-64GB}"
# Keep near Slurm --mem for shard tasks (DuckDB spill cap).
DUCKDB_SHARD_MEM="${DUCKDB_SHARD_MEM:-8GB}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"
# Remove prior preprocess array logs only (same dir holds e.g. eval_* from engine_eval).
shopt -s nullglob
rm -f "${LOG_DIR}"/ld-moves_*.out "${LOG_DIR}"/ld-moves_*.err
shopt -u nullglob

activate_venv() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
  else
    echo "Warning: .venv not found; using PATH python." >&2
  fi
}

activate_venv

# --- fresh staging (parquet shards); not done inside get_games ---
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# --- 1. games table ---
export DUCKDB_THREADS="${DUCKDB_LOGIN_THREADS}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_LOGIN_MEM}"
python3 src/slurm/scripts/preprocess.py get_games

# --- 2. shard (one Slurm array) ---
job_script="$(mktemp "${TMPDIR:-/tmp}/preprocess-shard.XXXXXX.sbatch")"
trap 'rm -f "${job_script}"' EXIT

cat >"${job_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=ld-moves
#SBATCH --output=${LOG_DIR}/ld-moves_%A_%a.out
#SBATCH --error=${LOG_DIR}/ld-moves_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME_LIMIT}
#SBATCH --array=0-${ARRAY_END}

set -euo pipefail
cd "${PROJECT_DIR}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "Warning: .venv not found; using PATH python." >&2
fi

echo "task \${SLURM_ARRAY_TASK_ID:-?} / ${TOTAL_SHARDS} on \$(hostname) at \$(date -Is)"

export DUCKDB_THREADS="\${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK}}"
export DUCKDB_MEMORY_LIMIT="\${DUCKDB_MEMORY_LIMIT:-${DUCKDB_SHARD_MEM}}"
# Must match array task count (here ${TOTAL_SHARDS}).
export PREPROCESS_TOTAL_SHARDS=${TOTAL_SHARDS}

python3 src/slurm/scripts/preprocess.py shard
EOF

sbatch --wait "${job_script}"

# --- 3. merge shards → moves, then rebuild processed_moves / processed_moves_nonzero ---
export DUCKDB_THREADS="${DUCKDB_LOGIN_THREADS}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_LOGIN_MEM}"
python3 src/slurm/scripts/preprocess.py merge

echo "Done at $(date -Is)"
