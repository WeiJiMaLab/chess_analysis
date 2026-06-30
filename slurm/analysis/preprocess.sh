#!/usr/bin/env bash
#
# Orchestrates preprocess.py: get_games → Slurm shard array → merge (parquet → moves) → process_moves (processed_*).
#
#   bash slurm/analysis/preprocess.sh
#
# Optional: PREPROCESS_SHARD_POLL_SEC=15  Seconds between shard-array status lines (default 30).
# Optional: DUCKDB_MERGE_THREADS / DUCKDB_MERGE_MEMORY_LIMIT  For merge + process_moves on the login node (default 64 threads, 200GB).
# Staging path must match preprocess.py config["staging_dir"].
# Array size: tasks 0..ARRAY_END → export PREPROCESS_TOTAL_SHARDS=$((ARRAY_END + 1)) inside the job.

set -euo pipefail

# --- paths & layout ---
PROJECT_DIR="/home/hl4291/chess_analysis"
LOG_DIR="${PROJECT_DIR}/slurm/analysis/logs"
# Keep in sync with preprocess.py main() config["staging_dir"].
STAGING_DIR="/scratch/gpfs/GRIFFITHS/hl4291/tmp/ld_moves_shard"

# --- Slurm array (0 through ARRAY_END) ---
ARRAY_END=10
TOTAL_SHARDS=$((ARRAY_END + 1))

# --- per-task Slurm resources ---
SLURM_CPUS_PER_TASK=20
SLURM_MEM="8G"
SLURM_TIME_LIMIT="01:15:00"

# --- DuckDB: login node vs array worker vs merge ---
DUCKDB_LOGIN_THREADS="${DUCKDB_THREADS:-40}"
DUCKDB_LOGIN_MEM="${DUCKDB_MEMORY_LIMIT:-64GB}"
# Keep near Slurm --mem for shard tasks (DuckDB spill cap).
DUCKDB_SHARD_MEM="${DUCKDB_SHARD_MEM:-8GB}"
# Merge scans all parquets; process_moves rebuilds feature tables — both use heavy DuckDB settings by default.
DUCKDB_MERGE_THREADS="${DUCKDB_MERGE_THREADS:-64}"
DUCKDB_MERGE_MEM="${DUCKDB_MERGE_MEMORY_LIMIT:-200GB}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"
# Remove prior preprocess array logs only (same dir holds e.g. eval_* from engine_eval).
shopt -s nullglob
rm -f "${LOG_DIR}"/ld-moves_*.out "${LOG_DIR}"/ld-moves_*.err
shopt -u nullglob

export ELO=2000
source slurm/helpers/setup_env.sh
export PYTHONPATH="${PYTHONPATH}:src/analysis"

# --- fresh staging (parquet shards); not done inside get_games ---
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# --- 1. games table ---
export DUCKDB_THREADS="${DUCKDB_LOGIN_THREADS}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_LOGIN_MEM}"
python3 src/analysis/preprocess.py get_games

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
export ELO=2000
source slurm/helpers/setup_env.sh
export PYTHONPATH="\${PYTHONPATH}:src/analysis"

echo "task \${SLURM_ARRAY_TASK_ID:-?} / ${TOTAL_SHARDS} on \$(hostname) at \$(date -Is)"

export DUCKDB_THREADS="\${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK}}"
export DUCKDB_MEMORY_LIMIT="\${DUCKDB_MEMORY_LIMIT:-${DUCKDB_SHARD_MEM}}"
# Must match array task count (here ${TOTAL_SHARDS}).
export PREPROCESS_TOTAL_SHARDS=${TOTAL_SHARDS}

python3 src/analysis/preprocess.py shard
EOF

# Submit array without --wait so we can print periodic status (squeue does not stream).
shard_poll_sec="${PREPROCESS_SHARD_POLL_SEC:-30}"
shard_job_id="$(sbatch --parsable "${job_script}")"
shard_job_id="${shard_job_id%%.*}"
if [[ -z "${shard_job_id}" ]]; then
  echo "ERROR: sbatch did not return a job id." >&2
  exit 1
fi

echo ""
echo "=== Shard array submitted: job ${shard_job_id} (${TOTAL_SHARDS} tasks), polling every ${shard_poll_sec}s ==="
while squeue -h -j "${shard_job_id}" 2>/dev/null | grep -q .; do
  echo ""
  echo "--- $(date -Is)  shard job ${shard_job_id} ---"
  squeue -j "${shard_job_id}" -o "%.18i %.9P %.2C %.8u %.8T %.10M %.6D %.20R" 2>/dev/null || true
  _r="$(squeue -h -j "${shard_job_id}" -t R 2>/dev/null | wc -l | tr -d ' ')"
  _pd="$(squeue -h -j "${shard_job_id}" -t PD 2>/dev/null | wc -l | tr -d ' ')"
  echo "  summary: ${_r} running, ${_pd} pending (batch size ${TOTAL_SHARDS})"
  sleep "${shard_poll_sec}"
done

echo ""
echo "=== Shard array left the queue (${shard_job_id}); verifying task states (sacct) ==="
sleep 2
_failed="$(sacct -j "${shard_job_id}" -n -o JobID,State,ExitCode 2>/dev/null | awk '
  $1 ~ /^[0-9]+_[0-9]+$/ {
    if ($2 != "COMPLETED") { print $0; next }
    if ($3 != "" && $3 != "0:0") { print $0 }
  }
')"
if [[ -n "${_failed}" ]]; then
  echo "ERROR: shard array has failed or non-zero-exit tasks:" >&2
  echo "${_failed}" >&2
  sacct -j "${shard_job_id}" -o JobID,JobName,State,ExitCode,Elapsed >&2
  exit 1
fi
echo "All ${TOTAL_SHARDS} shard tasks COMPLETED."

# --- 3. merge parquets → table moves ---
echo ""
echo "================================================================" >&2
echo " MERGE starting at $(date -Is)" >&2
echo " DuckDB: threads=${DUCKDB_MERGE_THREADS} memory_limit=${DUCKDB_MERGE_MEM}" >&2
echo " Loading all parquets into personal.db (often the slowest step)." >&2
echo "================================================================" >&2
echo ""
export DUCKDB_THREADS="${DUCKDB_MERGE_THREADS}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MERGE_MEM}"
python3 src/analysis/preprocess.py merge

# --- 4. process_moves: moves → processed_moves / processed_moves_nonzero ---
echo ""
echo "================================================================" >&2
echo " PROCESS_MOVES starting at $(date -Is)" >&2
echo " DuckDB: threads=${DUCKDB_MERGE_THREADS} memory_limit=${DUCKDB_MERGE_MEM}" >&2
echo "================================================================" >&2
echo ""
export DUCKDB_THREADS="${DUCKDB_MERGE_THREADS}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MERGE_MEM}"
python3 src/analysis/preprocess.py process_moves

echo "Done at $(date -Is)"
