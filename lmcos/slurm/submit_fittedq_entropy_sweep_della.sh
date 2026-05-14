#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"
PACKED_ROOT="${PACKED_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_oracle96_trace_nomaint_no_xaba_entropy}"
PACKED_TRAIN_DATA="${PACKED_TRAIN_DATA:-${PACKED_ROOT}/train_manifest.json}"
PACKED_VALIDATION_DATA="${PACKED_VALIDATION_DATA:-${PACKED_ROOT}/validation_manifest.json}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1.pt}"
SWEEP_ROOT="${SWEEP_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/sweeps/fittedq_entropy_sign_sweep}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-slurm/train_fitted_q_controller_della.slurm}"
SWEEP_NAME="${SWEEP_NAME:-$(basename "${SWEEP_ROOT}")}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/${SWEEP_NAME}}"
CONFIG_ROOT="${CONFIG_ROOT:-${SWEEP_ROOT}/configs}"

SIGN_LOSS_WEIGHTS="${SIGN_LOSS_WEIGHTS:-0.1 0.25 0.5}"
LEARNING_RATES="${LEARNING_RATES:-1e-4 3e-4 1e-3}"
Q_HIDDEN_VALUES="${Q_HIDDEN_VALUES:-256 512}"

SEED="${SEED:-0}"
Q_HIDDEN_LAYERS="${Q_HIDDEN_LAYERS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAINTENANCE_SCALE="${MAINTENANCE_SCALE:-0.0}"
GREEDY_EVAL_INTERVAL="${GREEDY_EVAL_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"

CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-64G}"
PRIMER_TIME="${PRIMER_TIME:-04:00:00}"
RERUN_TIME="${RERUN_TIME:-03:00:00}"
CONSTRAINT="${CONSTRAINT:-nomig}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SWEEP_ROOT}"
mkdir -p "${LOG_ROOT}"
mkdir -p "${CONFIG_ROOT}"

sanitize() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

submit() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "$*"
    return 0
  fi
  "$@"
}

write_config() {
  local config_path="$1"
  local sign_loss_weight="$2"
  local learning_rate="$3"
  local hidden_dim="$4"
  local checkpoint_path="$5"
  local diagnostics_path="$6"

  cat > "${config_path}" <<EOF
packed_train_data: ${PACKED_TRAIN_DATA}
packed_validation_data: ${PACKED_VALIDATION_DATA}
encoder_checkpoint: ${ENCODER_CHECKPOINT}
output_checkpoint: ${checkpoint_path}
output_diagnostics: ${diagnostics_path}
seed: ${SEED}
hidden_dim: ${hidden_dim}
hidden_layers: ${Q_HIDDEN_LAYERS}
batch_size: ${BATCH_SIZE}
episode_batch_size: ${EPISODE_BATCH_SIZE}
epochs: ${EPOCHS}
learning_rate: ${learning_rate}
weight_decay: ${WEIGHT_DECAY}
sign_loss_weight: ${sign_loss_weight}
maintenance_scale: ${MAINTENANCE_SCALE}
greedy_eval_interval: ${GREEDY_EVAL_INTERVAL}
log_interval: ${LOG_INTERVAL}
EOF
}

submit_train_job() {
  local sign_loss_weight="$1"
  local learning_rate="$2"
  local hidden_dim="$3"
  local time_limit="$4"
  local dependency="${5:-}"

  local suffix="sign$(sanitize "${sign_loss_weight}")_lr$(sanitize "${learning_rate}")_h${hidden_dim}"
  local checkpoint_path="${SWEEP_ROOT}/checkpoints/fittedq_${suffix}.pt"
  local diagnostics_path="${SWEEP_ROOT}/analysis/fittedq_${suffix}_diagnostics.jsonl"
  local log_path="${LOG_ROOT}/fittedq_${suffix}_%j.out"
  local config_path="${CONFIG_ROOT}/fittedq_${suffix}.yaml"

  write_config "${config_path}" "${sign_loss_weight}" "${learning_rate}" "${hidden_dim}" "${checkpoint_path}" "${diagnostics_path}"

  local -a cmd=(
    sbatch
    --parsable
    --constraint="${CONSTRAINT}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${time_limit}"
    --output="${log_path}"
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${config_path}"
  )

  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency="afterok:${dependency}")
  fi
  cmd+=("${TRAIN_SCRIPT}")
  submit "${cmd[@]}"
}

primer_job_id="$(
  submit_train_job "0.25" "3e-4" "256" "${PRIMER_TIME}"
)"

for sign_loss_weight in ${SIGN_LOSS_WEIGHTS}; do
  for learning_rate in ${LEARNING_RATES}; do
    for hidden_dim in ${Q_HIDDEN_VALUES}; do
      if [[ "${sign_loss_weight}" == "0.25" && "${learning_rate}" == "3e-4" && "${hidden_dim}" == "256" ]]; then
        continue
      fi
      submit_train_job "${sign_loss_weight}" "${learning_rate}" "${hidden_dim}" "${RERUN_TIME}" "${primer_job_id}"
    done
  done
done
