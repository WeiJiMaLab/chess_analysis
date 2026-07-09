#!/bin/bash
# plan.md Phase 2, Agent 2, sub-line A -- REVISED per coordinator instruction: gpu-short is
# capped at 44 GPUs cluster-wide and 3 parallel Phase-2 tracks are all competing for it right
# now. The lever is wall-clock, not resource size (cpu/mem/gpu request is already identical to
# the pipeline's own train_encoder.slurm) -- a single epoch (~40-50min) stays under Della's 1h
# threshold and routes to the much-less-contended gpu-test QOS instead of gpu-short.
#
# One epoch per submission, CHAINED: each run resumes from the checkpoint the PREVIOUS run in
# the chain produced (via RESUME_CKPT/OUT_STEM env vars), so 3 submissions of this script give
# the same net 3-epoch continuation the original single 2.5h job would have, just in
# gpu-test-sized increments that clear the queue faster.
#
# Usage:
#   RESUME_CKPT=<path> OUT_STEM=e2e_step1 sbatch slurm/agent2_e2e_continue_train_1ep.slurm
#SBATCH --job-name=agent2-e2e-continue-1ep
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh" gpu

OUT_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_e2e_continued
mkdir -p "$OUT_DIR"
RESUME_CKPT="${RESUME_CKPT:?must set RESUME_CKPT}"
OUT_STEM="${OUT_STEM:?must set OUT_STEM}"
OUT_CKPT="$OUT_DIR/${OUT_STEM}.pt"

echo "agent2_e2e_continue_1ep host=$(hostname) resume_from=$RESUME_CKPT -> $OUT_CKPT"; nvidia-smi -L || true; date
test -f "$RESUME_CKPT" || { echo "ERROR: resume checkpoint missing: $RESUME_CKPT" >&2; exit 1; }

python -u -m cts.train.e2e_controller_train --config "$CONFIG" --stage train \
    --set unfreeze_encoder=true \
    --set resume_checkpoint="$RESUME_CKPT" \
    --set output_checkpoint="$OUT_CKPT" \
    --set epochs=1 \
    --set pg_max_episodes=15000

echo "AGENT2 E2E CONTINUE 1EP DONE -> $OUT_CKPT"; date
