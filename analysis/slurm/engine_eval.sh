#!/usr/bin/env bash
# Run the full **distributed engine eval** pipeline:
#   1) Slurm array over shards (one parquet per task)  →  engine_eval_shard.sbatch
#   2) Merge all shard parquets into personal.db and join to move rows → script_engine_eval.py merge
#
# Wait: `sbatch --wait` until the array finishes (same pattern as preprocess.sh).
# Logs: analysis/slurm/logs/eval_*.out (see engine_eval_shard.sbatch)
#
# Usage (same PROJECT_DIR convention as preprocess.sh):
#   bash /home/hl4291/chess_analysis/analysis/slurm/engine_eval.sh
#   ENGINE=lc0 bash analysis/slurm/engine_eval.sh
#   bash analysis/slurm/engine_eval.sh --merge-only
#   bash analysis/slurm/engine_eval.sh --no-wait   # submit only; then --merge-only when the array is done

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

mkdir -p analysis/slurm/logs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "Warning: .venv not found; using PATH python."
fi

ENGINE="${ENGINE:-stockfish}"
LIMIT="${LIMIT:-1000000}"
DB="${DB:-/scratch/gpfs/GRIFFITHS/hl4291/personal.db}"
INPUT_DIR="${INPUT_DIR:-/scratch/gpfs/GRIFFITHS/hl4291/tmp/eval_results}"
MERGE_ONLY=0
NO_WAIT=0

usage() {
  cat <<EOF
Usage: $0 [options]

  --engine stockfish|lc0   (default: stockfish, or env ENGINE)
  --db PATH                 personal.db (default: GRIFFITHS scratch)
  --input-dir PATH         parent of <engine>/*.parquet (default: .../tmp/eval_results)
  --limit N                 per-shard cap passed to script_engine_eval (default: 1000000)
  --merge-only              skip Slurm; only merge parquets and build selected_moves_with_engine
  --no-wait                 submit the shard array only; exit immediately (re-run with --merge-only when done)
  -h, --help

Default wait: sbatch --wait (same pattern as preprocess.sh for shard arrays).

Env: ENGINE, LIMIT, DB, INPUT_DIR (same meaning as options where applicable).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --db)
      DB="$2"
      shift 2
      ;;
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --merge-only)
      MERGE_ONLY=1
      shift
      ;;
    --no-wait)
      NO_WAIT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export ENGINE
export LIMIT

run_merge_and_join() {
  echo "Merge into ${DB} (${ENGINE}_evaluations) and build selected_moves_with_engine on $(hostname) at $(date)"
  python3 analysis/slurm/scripts/script_engine_eval.py merge --engine "${ENGINE}" --db "${DB}" --input_dir "${INPUT_DIR}"
  echo "Finished engine eval pipeline at $(date)"
}

if [[ "${MERGE_ONLY}" -eq 1 ]]; then
  echo "Merge-only (no Slurm) on $(hostname) at $(date)"
  run_merge_and_join
  exit 0
fi

if ! command -v sbatch &>/dev/null; then
  echo "Error: sbatch not found. Run on a Slurm head node, or use --merge-only if shards are already on disk." >&2
  exit 1
fi

if [[ "${NO_WAIT}" -eq 1 ]]; then
  echo "Submitting engine eval array on $(hostname) at $(date)"
  JID=$(
    sbatch --parsable --export=ALL,ENGINE="${ENGINE}",LIMIT="${LIMIT}" \
      "${PROJECT_DIR}/analysis/slurm/engine_eval_shard.sbatch"
  )
  echo "Submitted Slurm array job: ${JID} (logs: analysis/slurm/logs/eval_*.out)"
  echo "Not waiting (--no-wait). When the job completes, run:"
  echo "  bash analysis/slurm/engine_eval.sh --merge-only --engine ${ENGINE} --db ${DB} --input-dir ${INPUT_DIR}"
  exit 0
fi

echo "Submitting engine eval array; blocking until all shard tasks complete (sbatch --wait) on $(hostname) at $(date)"
set +e
JID=$(
  sbatch --parsable --wait --export=ALL,ENGINE="${ENGINE}",LIMIT="${LIMIT}" \
    "${PROJECT_DIR}/analysis/slurm/engine_eval_shard.sbatch"
)
_wc=$?
set -e
if [[ "${_wc}" -ne 0 ]]; then
  echo "sbatch --wait failed (exit ${_wc}); not running merge. Check analysis/slurm/logs/ and re-run with --merge-only when parquets are ready." >&2
  exit "${_wc}"
fi
echo "Array job ${JID} finished at $(date). Logs: analysis/slurm/logs/eval_*.out"

run_merge_and_join
