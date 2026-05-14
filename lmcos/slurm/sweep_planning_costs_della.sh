#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split}"
SWEEP_ROOT="${SWEEP_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/planning_cost_sweeps}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain.pt}"
CONFIG_ROOT="${CONFIG_ROOT:-${SWEEP_ROOT}/configs}"

PACK_SCRIPT="${PACK_SCRIPT:-slurm/pack_controller_episodes_della.slurm}"
FILTER_SCRIPT="${FILTER_SCRIPT:-slurm/filter_packed_episodes_della.slurm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-slurm/train_fitted_q_controller_della.slurm}"

COST_KINDS="${COST_KINDS:-linear power}"
CONTINUE_COST_VALUES="${CONTINUE_COST_VALUES:-0.001 0.003 0.01}"
POWER_EXPONENT_VALUES="${POWER_EXPONENT_VALUES:-1.25 1.5 2.0}"

FILTER_PACKED="${FILTER_PACKED:-1}"
MIN_HALT_REWARD_RANGE="${MIN_HALT_REWARD_RANGE:-0.10}"
MAX_PER_STOP_STEP="${MAX_PER_STOP_STEP:-500}"
FILTER_SHARD_SIZE="${FILTER_SHARD_SIZE:-512}"

SEED="${SEED:-0}"
PACK_SHARD_SIZE="${PACK_SHARD_SIZE:-500}"
PACK_NUM_WORKERS="${PACK_NUM_WORKERS:-4}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
TRAIN_EPISODE_BATCH_SIZE="${TRAIN_EPISODE_BATCH_SIZE:-8}"
DRY_RUN="${DRY_RUN:-1}"

mkdir -p "${SWEEP_ROOT}"
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

write_pack_config() {
  local config_path="$1"
  local packed_root="$2"
  local continue_cost="$3"
  local kind="$4"
  local exponent="$5"

  cat > "${config_path}" <<EOF
split_root: ${SPLIT_ROOT}
output_root: ${packed_root}
shard_size: ${PACK_SHARD_SIZE}
num_workers: ${PACK_NUM_WORKERS}
continue_cost: ${continue_cost}
planning_cost_kind: ${kind}
planning_cost_exponent: ${exponent}
EOF
}

write_filter_config() {
  local config_path="$1"
  local packed_root="$2"
  local filtered_root="$3"

  cat > "${config_path}" <<EOF
train_manifest: ${packed_root}/train_manifest.json
validation_manifest: ${packed_root}/validation_manifest.json
output_dir: ${filtered_root}
min_halt_reward_range: ${MIN_HALT_REWARD_RANGE}
max_per_stop_step: ${MAX_PER_STOP_STEP}
shard_size: ${FILTER_SHARD_SIZE}
seed: ${SEED}
EOF
}

write_train_config() {
  local config_path="$1"
  local data_root="$2"
  local checkpoint_path="$3"
  local diagnostics_path="$4"
  local continue_cost="$5"
  local kind="$6"
  local exponent="$7"

  cat > "${config_path}" <<EOF
packed_train_data: ${data_root}/train_manifest.json
packed_validation_data: ${data_root}/validation_manifest.json
encoder_checkpoint: ${ENCODER_CHECKPOINT}
output_checkpoint: ${checkpoint_path}
output_diagnostics: ${diagnostics_path}
continue_cost: ${continue_cost}
planning_cost_kind: ${kind}
planning_cost_exponent: ${exponent}
epochs: ${TRAIN_EPOCHS}
batch_size: ${TRAIN_BATCH_SIZE}
episode_batch_size: ${TRAIN_EPISODE_BATCH_SIZE}
seed: ${SEED}
EOF
}

for kind in ${COST_KINDS}; do
  exponents="1.0"
  if [[ "${kind}" == "power" ]]; then
    exponents="${POWER_EXPONENT_VALUES}"
  fi

  for continue_cost in ${CONTINUE_COST_VALUES}; do
    for exponent in ${exponents}; do
      suffix="kind_${kind}_c$(sanitize "${continue_cost}")"
      if [[ "${kind}" == "power" ]]; then
        suffix="${suffix}_e$(sanitize "${exponent}")"
      fi

      config_root="${SWEEP_ROOT}/${suffix}"
      packed_root="${config_root}/packed"
      filtered_root="${config_root}/filtered"
      checkpoint_path="${config_root}/checkpoints/fittedq_${suffix}.pt"
      diagnostics_path="${config_root}/analysis/fittedq_${suffix}_diagnostics.jsonl"

      pack_config="${CONFIG_ROOT}/pack_${suffix}.yaml"
      filter_config="${CONFIG_ROOT}/filter_${suffix}.yaml"
      train_config="${CONFIG_ROOT}/train_${suffix}.yaml"

      echo "config=${suffix}"

      write_pack_config "${pack_config}" "${packed_root}" "${continue_cost}" "${kind}" "${exponent}"

      pack_cmd=(
        sbatch
        --parsable
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${pack_config}"
        "${PACK_SCRIPT}"
      )

      if [[ "${FILTER_PACKED}" == "1" ]]; then
        write_filter_config "${filter_config}" "${packed_root}" "${filtered_root}"
        write_train_config "${train_config}" "${filtered_root}" "${checkpoint_path}" "${diagnostics_path}" "${continue_cost}" "${kind}" "${exponent}"
      else
        write_train_config "${train_config}" "${packed_root}" "${checkpoint_path}" "${diagnostics_path}" "${continue_cost}" "${kind}" "${exponent}"
      fi

      if [[ "${DRY_RUN}" == "1" ]]; then
        submit "${pack_cmd[@]}"
        if [[ "${FILTER_PACKED}" == "1" ]]; then
          submit sbatch --parsable --dependency=afterok:'<pack_job_id>' \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${filter_config}" \
            "${FILTER_SCRIPT}"
          submit sbatch --parsable --dependency=afterok:'<filter_job_id>' \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${train_config}" \
            "${TRAIN_SCRIPT}"
        else
          submit sbatch --parsable --dependency=afterok:'<pack_job_id>' \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${train_config}" \
            "${TRAIN_SCRIPT}"
        fi
        continue
      fi

      pack_job_id="$(submit "${pack_cmd[@]}")"

      if [[ "${FILTER_PACKED}" == "1" ]]; then
        filter_job_id="$(
          submit sbatch --parsable --dependency=afterok:${pack_job_id} \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${filter_config}" \
            "${FILTER_SCRIPT}"
        )"
        submit sbatch --parsable --dependency=afterok:${filter_job_id} \
          --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${train_config}" \
          "${TRAIN_SCRIPT}"
      else
        submit sbatch --parsable --dependency=afterok:${pack_job_id} \
          --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONFIG="${train_config}" \
          "${TRAIN_SCRIPT}"
      fi
    done
  done
done
