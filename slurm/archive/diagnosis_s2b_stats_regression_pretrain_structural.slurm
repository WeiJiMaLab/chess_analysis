#!/bin/bash
# plan.md S2 -- THE REAL S2 test (corrected 2026-07-08): train a NEW TreeEncoder from scratch
# with a regression head predicting ONLY the structural stat vector {n_nodes, height, width}
# (current_value EXCLUDED -- it's a value-backup result, not a structural stat, and is
# architecturally unavailable to z_t-Controller; see plan.md's S1 methodological correction).
#
# Companion to (NOT a replacement for) diagnosis_s2_stats_regression_pretrain.slurm, which
# trains the 4-target (incl. current_value) SIDE EXPERIMENT -- both checkpoints get compared
# in S3, clearly labeled.
#
# Writes to a SEPARATE scratch checkpoint path -- does NOT touch the production encoder
# checkpoint (${packed_dir}/tiny_encoder.pt) or the side-experiment's s2_stats_encoder dir.
#
# Same timing budget as the 4-target run (diagnosis_s2_stats_regression_pretrain.slurm):
# ~0.6s/batch measured on real GPU data (see that script's job 10833402 log), full epoch
# (train 2792 batches + val 700 batches) * ~0.6s ~= 35 min, comfortably inside 58 min.
#
#   sbatch slurm/diagnosis_s2b_stats_regression_pretrain_structural.slurm
#SBATCH --job-name=diag-s2b-structural-encoder
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

S2B_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/s2b_structural_encoder
S2B_CKPT="$S2B_DIR/s2b_structural_encoder.pt"
mkdir -p "$S2B_DIR"

echo "diag_s2b_structural_encoder host=$(hostname)"; nvidia-smi -L || true; date
echo "S2B_CKPT=$S2B_CKPT (scratch path, production encoder checkpoint ${ENC} untouched)"

python -m cts.train.stats_regression_pretrain --config "$CONFIG" \
    --output-checkpoint "$S2B_CKPT" \
    --stats n_nodes,height,width \
    --epochs 1 \
    --episode-batch-size 8 \
    --device cuda \
    --plot-out "$TINY/outputs/figures/minply15_maxply75/diagnosis"

echo "DIAG-S2B STRUCTURAL DONE"; date
