#!/bin/bash
# DIAGNOSTIC ONLY (plan.md E1) — NOT a pipeline stage. Longer-epoch rerun of
# pretrain-child-wdl-encoder (same config/data/architecture as
# slurm/pipeline/train_encoder.slurm) to check whether the loss plateau seen
# in the real 6-epoch run (train 0.753->0.525, val 0.607->0.525, flat from
# epoch 2) is a real ceiling or just undertraining.
#
# Writes to a SEPARATE scratch checkpoint path -- does NOT touch or overwrite
# the real production checkpoint ($MAT/../packed/tiny_encoder.pt) that
# downstream stages (materialize, train, eval) depend on.
#
#   sbatch slurm/diagnosis_e1_encoder_longtrain.slurm
#SBATCH --job-name=diag-e1-encoder
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

E1_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/e1_encoder
E1_CKPT="$E1_DIR/tiny_encoder_ep30.pt"
mkdir -p "$E1_DIR"

echo "diag_e1_encoder host=$(hostname)"; nvidia-smi -L || true; date
echo "E1_CKPT=$E1_CKPT (scratch path, real checkpoint ${ENC} untouched)"

python -m cts.data.build_tree --config "$CONFIG" --stage encoder \
    --override epochs=30 \
    --override output_checkpoint="$E1_CKPT" \
    --override checkpoint_dir="$E1_DIR"

echo "DIAG-E1 ENCODER DONE"; date
