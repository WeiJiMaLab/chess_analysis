#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split}"
SWEEP_ROOT="${SWEEP_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/planning_cost_sweeps}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain.pt}"

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

      echo "config=${suffix}"

      pack_cmd=(
        sbatch
        --parsable
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SPLIT_ROOT="${SPLIT_ROOT}",OUTPUT_ROOT="${packed_root}",SHARD_SIZE="${PACK_SHARD_SIZE}",NUM_WORKERS="${PACK_NUM_WORKERS}",CONTINUE_COST="${continue_cost}",PLANNING_COST_KIND="${kind}",PLANNING_COST_EXPONENT="${exponent}"
        "${PACK_SCRIPT}"
      )

      if [[ "${DRY_RUN}" == "1" ]]; then
        submit "${pack_cmd[@]}"
        if [[ "${FILTER_PACKED}" == "1" ]]; then
          submit sbatch --parsable --dependency=afterok:<pack_job_id> \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",TRAIN_MANIFEST="${packed_root}/train_manifest.json",VALIDATION_MANIFEST="${packed_root}/validation_manifest.json",OUTPUT_DIR="${filtered_root}",MIN_HALT_REWARD_RANGE="${MIN_HALT_REWARD_RANGE}",MAX_PER_STOP_STEP="${MAX_PER_STOP_STEP}",SHARD_SIZE="${FILTER_SHARD_SIZE}",SEED="${SEED}" \
            "${FILTER_SCRIPT}"
          submit sbatch --parsable --dependency=afterok:<filter_job_id> \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PACKED_TRAIN_DATA="${filtered_root}/train_manifest.json",PACKED_VALIDATION_DATA="${filtered_root}/validation_manifest.json",ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT}",OUTPUT_CHECKPOINT="${checkpoint_path}",OUTPUT_DIAGNOSTICS="${diagnostics_path}",CONTINUE_COST="${continue_cost}",PLANNING_COST_KIND="${kind}",PLANNING_COST_EXPONENT="${exponent}",EPOCHS="${TRAIN_EPOCHS}",BATCH_SIZE="${TRAIN_BATCH_SIZE}",EPISODE_BATCH_SIZE="${TRAIN_EPISODE_BATCH_SIZE}",SEED="${SEED}" \
            "${TRAIN_SCRIPT}"
        else
          submit sbatch --parsable --dependency=afterok:<pack_job_id> \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PACKED_TRAIN_DATA="${packed_root}/train_manifest.json",PACKED_VALIDATION_DATA="${packed_root}/validation_manifest.json",ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT}",OUTPUT_CHECKPOINT="${checkpoint_path}",OUTPUT_DIAGNOSTICS="${diagnostics_path}",CONTINUE_COST="${continue_cost}",PLANNING_COST_KIND="${kind}",PLANNING_COST_EXPONENT="${exponent}",EPOCHS="${TRAIN_EPOCHS}",BATCH_SIZE="${TRAIN_BATCH_SIZE}",EPISODE_BATCH_SIZE="${TRAIN_EPISODE_BATCH_SIZE}",SEED="${SEED}" \
            "${TRAIN_SCRIPT}"
        fi
        continue
      fi

      pack_job_id="$(submit "${pack_cmd[@]}")"

      if [[ "${FILTER_PACKED}" == "1" ]]; then
        filter_job_id="$(
          submit sbatch --parsable --dependency=afterok:${pack_job_id} \
            --export=ALL,PROJECT_DIR="${PROJECT_DIR}",TRAIN_MANIFEST="${packed_root}/train_manifest.json",VALIDATION_MANIFEST="${packed_root}/validation_manifest.json",OUTPUT_DIR="${filtered_root}",MIN_HALT_REWARD_RANGE="${MIN_HALT_REWARD_RANGE}",MAX_PER_STOP_STEP="${MAX_PER_STOP_STEP}",SHARD_SIZE="${FILTER_SHARD_SIZE}",SEED="${SEED}" \
            "${FILTER_SCRIPT}"
        )"
        submit sbatch --parsable --dependency=afterok:${filter_job_id} \
          --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PACKED_TRAIN_DATA="${filtered_root}/train_manifest.json",PACKED_VALIDATION_DATA="${filtered_root}/validation_manifest.json",ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT}",OUTPUT_CHECKPOINT="${checkpoint_path}",OUTPUT_DIAGNOSTICS="${diagnostics_path}",CONTINUE_COST="${continue_cost}",PLANNING_COST_KIND="${kind}",PLANNING_COST_EXPONENT="${exponent}",EPOCHS="${TRAIN_EPOCHS}",BATCH_SIZE="${TRAIN_BATCH_SIZE}",EPISODE_BATCH_SIZE="${TRAIN_EPISODE_BATCH_SIZE}",SEED="${SEED}" \
          "${TRAIN_SCRIPT}"
      else
        submit sbatch --parsable --dependency=afterok:${pack_job_id} \
          --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PACKED_TRAIN_DATA="${packed_root}/train_manifest.json",PACKED_VALIDATION_DATA="${packed_root}/validation_manifest.json",ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT}",OUTPUT_CHECKPOINT="${checkpoint_path}",OUTPUT_DIAGNOSTICS="${diagnostics_path}",CONTINUE_COST="${continue_cost}",PLANNING_COST_KIND="${kind}",PLANNING_COST_EXPONENT="${exponent}",EPOCHS="${TRAIN_EPOCHS}",BATCH_SIZE="${TRAIN_BATCH_SIZE}",EPISODE_BATCH_SIZE="${TRAIN_EPISODE_BATCH_SIZE}",SEED="${SEED}" \
          "${TRAIN_SCRIPT}"
      fi
    done
  done
done
