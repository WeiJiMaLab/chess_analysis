#!/usr/bin/env bash
# Submit (or run locally) the three hl4291 controller smoke configs in parallel,
# then plot val loss + regret comparison when all training jobs succeed.
#
# Usage:
#   cd /home/hl4291/chess_analysis/lmcos
#   ./slurm/4_supervised_controller/submit_smoke_controller_parallel.sh
#   ./slurm/4_supervised_controller/submit_smoke_controller_parallel.sh --local

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"
LOCAL=0
PLOT_ONLY=0

for arg in "$@"; do
  case "${arg}" in
    --local) LOCAL=1 ;;
    --plot-only) PLOT_ONLY=1 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 1 ;;
  esac
done

CONFIGS=(
  legacy_root_budget
  subtree_weighting_root_budget
  subtree_weighting_root
)

plot_smoke_comparison() {
  local log_root="${PROJECT_DIR}/slurm/logs/4_supervised_controller"
  cd "${PROJECT_DIR}"
  export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
  python3 -m cts.train.controller_train plot-metrics-compare \
    "${log_root}/legacy_root_budget/metrics.yaml" \
    "${log_root}/subtree_weighting_root_budget/metrics.yaml" \
    "${log_root}/subtree_weighting_root/metrics.yaml" \
    -o "${log_root}/smoke_comparison.png" \
    --title "Controller smoke comparison"
  echo "Wrote ${log_root}/smoke_comparison.png"
}

if [[ "${PLOT_ONLY}" -eq 1 ]]; then
  cd "${PROJECT_DIR}"
  set +u
  module --force purge >/dev/null 2>&1 || true
  module load anaconda3/2023.3
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
  fi
  set -u
  plot_smoke_comparison
  exit 0
fi

cd "${PROJECT_DIR}"
export PROJECT_DIR VENV_DIR

for name in "${CONFIGS[@]}"; do
  mkdir -p "slurm/logs/4_supervised_controller/${name}"
done

submit_one() {
  local name="$1"
  export CONFIG="${PROJECT_DIR}/slurm/configs/4_supervised_controller/${name}.yaml"
  if [[ "${LOCAL}" -eq 1 ]]; then
    echo "[local] starting ${name}"
    bash slurm/4_supervised_controller/controller_train_supervised.slurm
  else
    sbatch \
      --job-name="cts-smoke-${name}" \
      --export=ALL,PROJECT_DIR,CONFIG,VENV_DIR \
      slurm/4_supervised_controller/controller_train_supervised.slurm
  fi
}

submit_plot() {
  local deps="$1"
  if [[ "${LOCAL}" -eq 1 ]]; then
    echo "[local] plotting smoke comparison"
    plot_smoke_comparison
  else
    sbatch \
      --job-name="cts-smoke-plot-comparison" \
      --output="${PROJECT_DIR}/slurm/logs/%x_%j.out" \
      --error="${PROJECT_DIR}/slurm/logs/%x_%j.err" \
      --cpus-per-task=1 \
      --mem=4G \
      --time=00:10:00 \
      --dependency="afterok:${deps}" \
      --export=ALL,PROJECT_DIR,VENV_DIR \
      --wrap="cd ${PROJECT_DIR} && bash slurm/4_supervised_controller/submit_smoke_controller_parallel.sh --plot-only"
  fi
}

pids=()
job_ids=()
for name in "${CONFIGS[@]}"; do
  if [[ "${LOCAL}" -eq 1 ]]; then
    submit_one "${name}" &
    pids+=("$!")
  else
    job_id="$(submit_one "${name}" | awk '{print $NF}')"
    job_ids+=("${job_id}")
    echo "Submitted ${name} (job ${job_id})"
  fi
done

if [[ "${LOCAL}" -eq 1 ]]; then
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -eq 0 ]]; then
    submit_plot ""
  else
    echo "Skipping comparison plot: one or more training jobs failed." >&2
    exit 1
  fi
else
  deps="$(IFS=:; echo "${job_ids[*]}")"
  plot_job_id="$(submit_plot "${deps}" | awk '{print $NF}')"
  echo "Submitted comparison plot (job ${plot_job_id}, afterok:${deps})"
fi

echo "Submitted ${#CONFIGS[@]} smoke controller jobs (VENV_DIR=${VENV_DIR})."
