#!/bin/bash
# Submit the cost-sweep pipeline: reoracle (CPU, ~5min) -> train (GPU, ~5min).
# No repacking or rematerializing — oracle labels are recomputed from existing
# packed halt_rewards; z_t features are reused from existing materialized cache.

set -euo pipefail

LMCOS=/home/hl4291/chess_analysis/lmcos
SCRATCH=/scratch/gpfs/GRIFFITHS/hl4291
CONFIGS="$LMCOS/slurm/cost_sweep/configs"
SLURM="$LMCOS/slurm/cost_sweep"

mkdir -p "$SCRATCH/cost_sweep/logs"
mkdir -p "$LMCOS/slurm/cost_sweep/outputs"

# ── Baseline (no-norm, existing materialized cache, no reoracle needed) ───────
mkdir -p "$SCRATCH/cost_sweep/baseline_nonorm"
JID_BASE=$(sbatch --parsable \
  --export=SWEEP_NAME=baseline_nonorm,TRAIN_CONFIG="$CONFIGS/baseline_nonorm_train.yaml" \
  "$SLURM/train.slurm")
echo "baseline_nonorm train: $JID_BASE"

# ── Helper: reoracle -> train ─────────────────────────────────────────────────
submit_condition() {
    local NAME="$1"
    local REORACLE_CFG="$2"
    local TRAIN_CFG="$3"

    JID_RO=$(sbatch --parsable \
      --export=SWEEP_NAME="$NAME",REORACLE_CONFIG="$REORACLE_CFG" \
      "$SLURM/reoracle.slurm")
    echo "$NAME reoracle: $JID_RO"

    JID_TRAIN=$(sbatch --parsable \
      --dependency=afterok:"$JID_RO" \
      --export=SWEEP_NAME="$NAME",TRAIN_CONFIG="$TRAIN_CFG" \
      "$SLURM/train.slurm")
    echo "$NAME train: $JID_TRAIN"
}

submit_condition lambda_1.0 \
  "$CONFIGS/lambda_1.0_reoracle.yaml" \
  "$CONFIGS/lambda_1.0_train.yaml"

submit_condition lambda_5.0 \
  "$CONFIGS/lambda_5.0_reoracle.yaml" \
  "$CONFIGS/lambda_5.0_train.yaml"

submit_condition linear_0.003 \
  "$CONFIGS/linear_0.003_reoracle.yaml" \
  "$CONFIGS/linear_0.003_train.yaml"

echo ""
echo "Jobs submitted. Monitor: squeue -u $USER"
echo "Logs: $SCRATCH/cost_sweep/logs/"
