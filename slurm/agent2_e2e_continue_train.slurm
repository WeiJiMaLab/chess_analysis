#!/bin/bash
# plan.md Phase 2, Agent 2, sub-line A -- continue the e2e (encoder-unfrozen) Z2 controller
# past its first epoch (job 10831440, checkpoint at
# /scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller/e2e_controller.pt).
# SIG-Z (sig_significance.md) already confirmed this 1-epoch checkpoint significantly beats the
# original FROZEN z_t but not yet SingleHalt* (paired CI [-0.0004,+0.0090], straddles 0, barely) --
# more epochs is the next lever.
#
# Warm-starts from the Z2 checkpoint's FULL model_state_dict (encoder+head) via the new
# resume_checkpoint hook (controller_train.py, _build_model_and_optimizer) and continues for
# epochs=3 MORE epochs, saving a checkpoint (+ encoder-only sidecar) every epoch (already
# e2e_controller_train.py's default behavior) so each epoch can be paired-significance-evaluated
# separately, not just the final one.
#
# Z0's measured real throughput (~40 min/epoch, I/O-inclusive) makes 3 epochs ~2h -- over the
# 1h della QOS threshold, so this routes to gpu-short (never --partition=gpu directly), unlike
# the original 1-epoch Z2 run which fit in gpu-test.
#
# Writes to a SEPARATE scratch dir -- does NOT touch the Z2 checkpoint or the real production
# controller checkpoint.
#
#   sbatch slurm/agent2_e2e_continue_train.slurm
#SBATCH --job-name=agent2-e2e-continue
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh" gpu

Z2_CKPT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller/e2e_controller.pt
OUT_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_e2e_continued
mkdir -p "$OUT_DIR"
OUT_CKPT="$OUT_DIR/e2e_controller_continued.pt"

echo "agent2_e2e_continue host=$(hostname) resume_from=$Z2_CKPT"; nvidia-smi -L || true; date
test -f "$Z2_CKPT" || { echo "ERROR: Z2 checkpoint missing: $Z2_CKPT" >&2; exit 1; }

python -u -m cts.train.e2e_controller_train --config "$CONFIG" --stage train \
    --set unfreeze_encoder=true \
    --set resume_checkpoint="$Z2_CKPT" \
    --set output_checkpoint="$OUT_CKPT" \
    --set epochs=3 \
    --set pg_max_episodes=15000

echo "AGENT2 E2E CONTINUE DONE -> $OUT_CKPT"; date
