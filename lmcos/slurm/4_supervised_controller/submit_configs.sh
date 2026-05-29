#!/usr/bin/env bash
# Submit (or run locally) every YAML in slurm/configs/4_supervised_controller/,
# then plot val loss + regret comparison when all training jobs succeed.
#
# Usage:
#   cd /home/hl4291/chess_analysis/lmcos
#   ./slurm/4_supervised_controller/submit_configs.sh
#   ./slurm/4_supervised_controller/submit_configs.sh --local
#
# Slurm stdout/stderr → slurm/logs/
# Training artifacts → slurm/outputs/4_supervised_controller/ (flat: <run>.yaml, <run>.png, comparison.png)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"
STAGE=4_supervised_controller
CONFIG_DIR="${PROJECT_DIR}/slurm/configs/${STAGE}"
OUTPUT_DIR="${PROJECT_DIR}/slurm/outputs/${STAGE}"
LOCAL=0
PLOT_ONLY=0

CONFIG_PATHS=()
CONFIG_NAMES=()
METRICS_PATHS=()

for arg in "$@"; do
  case "${arg}" in
    --local) LOCAL=1 ;;
    --plot-only) PLOT_ONLY=1 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 1 ;;
  esac
done

resolve_path() {
  local path="$1"
  if [[ "${path}" != /* ]]; then
    path="${PROJECT_DIR}/${path}"
  fi
  printf '%s' "${path}"
}

discover_configs() {
  CONFIG_PATHS=()
  CONFIG_NAMES=()
  METRICS_PATHS=()

  mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/slurm/logs"

  shopt -s nullglob
  local config
  for config in "${CONFIG_DIR}"/*.yaml; do
    local name
    name="$(basename "${config}" .yaml)"
    local metrics_path
    metrics_path="$(grep -E '^metrics_path:' "${config}" | head -1 | sed -E 's/^metrics_path:[[:space:]]*//' || true)"
    if [[ -z "${metrics_path}" ]]; then
      metrics_path="${OUTPUT_DIR}/${name}.yaml"
    else
      metrics_path="$(resolve_path "${metrics_path}")"
    fi
    CONFIG_PATHS+=("${config}")
    CONFIG_NAMES+=("${name}")
    METRICS_PATHS+=("${metrics_path}")
  done
  shopt -u nullglob

  if [[ "${#CONFIG_PATHS[@]}" -eq 0 ]]; then
    echo "No configs found in ${CONFIG_DIR}" >&2
    exit 1
  fi
}

plot_comparison() {
  if [[ "${#METRICS_PATHS[@]}" -lt 2 ]]; then
    echo "Skipping comparison plot: need at least 2 configs (found ${#METRICS_PATHS[@]})." >&2
    return 0
  fi

  cd "${PROJECT_DIR}"
  export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
  python3 -m cts.train.controller_train plot-metrics-compare \
    "${METRICS_PATHS[@]}" \
    -o "${OUTPUT_DIR}/comparison.png" \
    --title "Controller comparison"
  echo "Wrote ${OUTPUT_DIR}/comparison.png"
}

if [[ "${PLOT_ONLY}" -eq 1 ]]; then
  discover_configs
  cd "${PROJECT_DIR}"
  set +u
  module --force purge >/dev/null 2>&1 || true
  module load anaconda3/2023.3
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
  fi
  set -u
  plot_comparison
  exit 0
fi

cd "${PROJECT_DIR}"
export PROJECT_DIR VENV_DIR
discover_configs

submit_one() {
  local config="$1"
  local name="$2"
  export CONFIG="${config}"
  if [[ "${LOCAL}" -eq 1 ]]; then
    echo "[local] starting ${name}"
    bash slurm/4_supervised_controller/controller_train_supervised.slurm
  else
    sbatch \
      --job-name="cts-ctrl-${name}" \
      --export=ALL,PROJECT_DIR,CONFIG,VENV_DIR \
      slurm/4_supervised_controller/controller_train_supervised.slurm
  fi
}

submit_plot() {
  local deps="$1"
  if [[ "${#CONFIG_PATHS[@]}" -lt 2 ]]; then
    echo "Skipping comparison plot: need at least 2 configs (found ${#CONFIG_PATHS[@]})."
    return 0
  fi
  if [[ "${LOCAL}" -eq 1 ]]; then
    echo "[local] plotting comparison"
    plot_comparison
  else
    sbatch \
      --job-name="cts-ctrl-plot-comparison" \
      --output="${PROJECT_DIR}/slurm/logs/%x_%j.out" \
      --error="${PROJECT_DIR}/slurm/logs/%x_%j.err" \
      --cpus-per-task=1 \
      --mem=4G \
      --time=00:10:00 \
      --dependency="afterok:${deps}" \
      --export=ALL,PROJECT_DIR,VENV_DIR \
      --wrap="cd ${PROJECT_DIR} && bash slurm/4_supervised_controller/submit_configs.sh --plot-only"
  fi
}

pids=()
job_ids=()
for i in "${!CONFIG_PATHS[@]}"; do
  if [[ "${LOCAL}" -eq 1 ]]; then
    submit_one "${CONFIG_PATHS[$i]}" "${CONFIG_NAMES[$i]}" &
    pids+=("$!")
  else
    job_id="$(submit_one "${CONFIG_PATHS[$i]}" "${CONFIG_NAMES[$i]}" | awk '{print $NF}')"
    job_ids+=("${job_id}")
    echo "Submitted ${CONFIG_NAMES[$i]} (job ${job_id})"
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
  if [[ "${#job_ids[@]}" -ge 2 ]]; then
    deps="$(IFS=:; echo "${job_ids[*]}")"
    plot_job_id="$(submit_plot "${deps}" | awk '{print $NF}')"
    echo "Submitted comparison plot (job ${plot_job_id}, afterok:${deps})"
  else
    submit_plot ""
  fi
fi

echo "Submitted ${#CONFIG_PATHS[@]} controller jobs from ${CONFIG_DIR} (VENV_DIR=${VENV_DIR})."
