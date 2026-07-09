#!/bin/bash
# plan.md Phase 2, Agent 6 -- does e2e (joint encoder+head) training improve on Agent 1's
# already-strong FROZEN-encoder z_t result on the ysagiv xaba-filtered population
# (ysagiv_trees.md: z_t significantly beats SingleHalt* AND Stats at lambda=0.0005, m=0)?
# our_trees_continued.md (Agent 2) found e2e training is the ONLY lever anywhere in this
# investigation that beat SingleHalt* even once, on our OWN corpus -- nobody has tried it on
# the ysagiv xaba population yet.
#
# Reuses Agent 1's already-packed mc_packed data (config_ysagiv_xaba_e2e.yaml's
# train.packed_{train,validation}_data point straight at Agent 1's READ-ONLY ysagiv_xaba/
# mc_packed dir -- never repacked here) and INITIALIZES from Agent 1's frozen, trained
# tiny_encoder.pt (train.encoder_checkpoint) -- same "warm start the base encoder, then
# unfreeze + train jointly against the real PG expected-regret objective" recipe as
# our_trees_continued.md sub-line A / cts.train.e2e_controller_train.
#
# One epoch per submission, CHAINABLE via RESUME_CKPT/OUT_STEM (same pattern as
# slurm/agent2_e2e_continue_train_1ep.slurm) -- keeps each job under Della's 1h threshold,
# routed to the far-less-contended gpu-test QOS (gpu-short was badly contended when Agent 2
# hit it; routing here from the start per direct instruction).
#
# Usage:
#   sbatch slurm/agent6_ysagiv_xaba_e2e_train_1ep.slurm                      # epoch 1, no resume
#   RESUME_CKPT=<epoch1 ckpt> OUT_STEM=e2e_epoch2 \
#     sbatch --dependency=afterok:<epoch1 jobid> slurm/agent6_ysagiv_xaba_e2e_train_1ep.slurm
#SBATCH --job-name=agent6-ysagiv-xaba-e2e-train-1ep
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_ysagiv_xaba_e2e.yaml"
source "$TINY/slurm/helpers/setup_env.sh" gpu

CKPT_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_e2e/e2e_checkpoints
mkdir -p "$CKPT_DIR"
OUT_STEM="${OUT_STEM:-e2e_epoch1}"
OUT_CKPT="$CKPT_DIR/${OUT_STEM}.pt"
RESUME_CKPT="${RESUME_CKPT:-}"

echo "agent6_ysagiv_xaba_e2e_train_1ep host=$(hostname) out=$OUT_CKPT resume=${RESUME_CKPT:-<none, fresh from tiny_encoder.pt>}"
nvidia-smi -L || true; date

SET_ARGS=(--set unfreeze_encoder=true --set output_checkpoint="$OUT_CKPT" --set epochs=1 --set pg_max_episodes=15000)
if [ -n "$RESUME_CKPT" ]; then
    test -f "$RESUME_CKPT" || { echo "ERROR: resume checkpoint missing: $RESUME_CKPT" >&2; exit 1; }
    SET_ARGS+=(--set resume_checkpoint="$RESUME_CKPT")
fi

python -u -m cts.train.e2e_controller_train --config "$CONFIG" --stage train "${SET_ARGS[@]}"

echo "AGENT6 YSAGIV XABA E2E TRAIN 1EP DONE -> $OUT_CKPT (encoder sidecar: ${OUT_CKPT%.pt}_encoder.pt)"; date
