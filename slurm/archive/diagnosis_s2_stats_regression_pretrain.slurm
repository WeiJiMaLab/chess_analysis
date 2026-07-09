#!/bin/bash
# plan.md S2 -- train a NEW TreeEncoder from scratch with a regression head predicting the
# S1-confirmed stat vector {n_nodes, height, width, current_value} directly (MSE), replacing
# the child-WDL pretraining objective E1/E2 diagnosed as a real ceiling that also loses basic
# tree-structure information (see diagnosis.md "E3 -- Synthesis"). Same TreeEncoder backbone/
# architecture as production (encoder.* config), same real mc_packed episode data Z2 uses.
#
# Writes to a SEPARATE scratch checkpoint path -- does NOT touch the production encoder
# checkpoint (${packed_dir}/tiny_encoder.pt).
#
# Timing per Z2's real measured GPU throughput (slurm/z2_e2e_controller_train.slurm's
# comment): ~0.65-0.7s/batch at episode_batch_size=8 -> full epoch (train 22338/8=2792
# batches + val 5599/8=700 batches) * 0.7s ~= 41 min. S2's per-batch cost should be <=
# Z2's (plain MSE forward+backward, no per-step oracle-return python loop), so 1 epoch
# comfortably fits a 58-min gpu-test budget with room for env setup/checkpointing/plotting.
#
#   sbatch slurm/diagnosis_s2_stats_regression_pretrain.slurm
#SBATCH --job-name=diag-s2-stats-encoder
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

S2_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/s2_stats_encoder
S2_CKPT="$S2_DIR/s2_stats_encoder.pt"
mkdir -p "$S2_DIR"

echo "diag_s2_stats_regression_pretrain host=$(hostname)"; nvidia-smi -L || true; date
echo "S2_CKPT=$S2_CKPT (scratch path, production encoder checkpoint ${ENC} untouched)"

python -m cts.train.stats_regression_pretrain --config "$CONFIG" \
    --output-checkpoint "$S2_CKPT" \
    --epochs 1 \
    --episode-batch-size 8 \
    --device cuda \
    --plot-out "$TINY/outputs/figures/minply15_maxply75/diagnosis"

echo "DIAG-S2 DONE"; date
