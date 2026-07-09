#!/bin/bash
# plan.md Phase 2, Agent 6 -- materialize z_t on Agent 1's ysagiv xaba validation set using
# ONE e2e-trained encoder checkpoint (produced by agent6_ysagiv_xaba_e2e_train_1ep.slurm), then
# run the SAME paired-bootstrap significance eval (analysis.ysagiv_sig) Agent 1/4 use, at the
# SAME 3 REGIME-confirmed regimes (lambda=0.01/m=0, lambda=0.001/m=0.001, lambda=0.0005/m=0) so
# results are directly comparable to ysagiv_trees.md's frozen-encoder baseline table.
#
# Writes to THIS agent's own isolated scratch (materialized cache) and figures dir
# (outputs/figures/ysagiv/xaba_e2e/) -- never touches Agent 1's materialized/ dir or
# outputs/figures/ysagiv/xaba/.
#
# Usage:
#   ENC_CKPT=<...>_encoder.pt EPOCH_TAG=epoch1 \
#     sbatch --dependency=afterok:<train_jobid> slurm/agent6_ysagiv_xaba_e2e_materialize_eval.slurm
#SBATCH --job-name=agent6-ysagiv-xaba-e2e-mat-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_ysagiv_xaba_e2e.yaml"
source "$TINY/slurm/helpers/setup_env.sh" gpu
export OMP_NUM_THREADS=4

ENC_CKPT="${ENC_CKPT:?must set ENC_CKPT (the *_encoder.pt sidecar from the e2e train job)}"
EPOCH_TAG="${EPOCH_TAG:?must set EPOCH_TAG (e.g. epoch1)}"

RUN_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_e2e
WORK="$RUN_DIR/materialized_work_${EPOCH_TAG}"
CACHE="$RUN_DIR/materialized/validation_cache_${EPOCH_TAG}.pt"
OUT_DIR="$TINY/outputs/figures/ysagiv/xaba_e2e"
mkdir -p "$OUT_DIR" "$(dirname "$CACHE")"

echo "agent6_ysagiv_xaba_e2e_materialize_eval host=$(hostname) enc=$ENC_CKPT tag=$EPOCH_TAG"
nvidia-smi -L || true; date
test -f "$ENC_CKPT" || { echo "ERROR: encoder checkpoint missing: $ENC_CKPT" >&2; exit 1; }

echo "=== materialize ($EPOCH_TAG) ==="
python -u -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=materialize --set device=cuda \
    --set num_workers=1 --set worker_index=0 \
    --set output_dir="$WORK" \
    --set packed_data="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$ENC_CKPT" \
    --set loader_workers=0 --set resume=false

echo "=== merge ($EPOCH_TAG) ==="
python -u -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=merge --set num_workers=1 \
    --set output_dir="$WORK" \
    --set final_cache="$CACHE" \
    --set manifest_path="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$ENC_CKPT"

echo "=== paired-significance eval (analysis.ysagiv_sig) at 3 REGIME-confirmed regimes, $EPOCH_TAG ==="
for SPEC in "0.01:0.0:lambda0p01_m0" "0.001:0.001:lambda0p001_m0p001" "0.0005:0.0:lambda0p0005_m0"; do
    LAM="${SPEC%%:*}"; REST="${SPEC#*:}"; MSCALE="${REST%%:*}"; TAG="${REST#*:}"
    echo "--- ysagiv_sig @ time_lambda=$LAM maintenance_scale=$MSCALE tag=${EPOCH_TAG}_${TAG} ---"
    python -u -m analysis.ysagiv_sig \
        --packed-root "$MCP" --cache "$CACHE" \
        --out-dir "$OUT_DIR" --tag "${EPOCH_TAG}_${TAG}" \
        --d-embed 32 --time-mode linear --time-lambda "$LAM" \
        --maintenance-scale "$MSCALE" --maintenance-exponent 1.0 \
        --max-episodes 15000
done

echo "AGENT6 YSAGIV XABA E2E MATERIALIZE+EVAL DONE ($EPOCH_TAG) -> $OUT_DIR"; date
