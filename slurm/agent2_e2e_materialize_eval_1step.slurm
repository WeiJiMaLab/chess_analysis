#!/bin/bash
# plan.md Phase 2, Agent 2, sub-line A -- materialize + paired-eval for ONE chained-training
# step's encoder checkpoint (see agent2_e2e_continue_train_1ep.slurm). Routed to gpu-test
# (under Della's 1h threshold, same rationale as the chained trainer -- Z3's own precedent
# budgeted 40min for materialize+merge+eval combined on the full validation split).
#
# Usage:
#   STEM=e2e_step1 EPOCH=2 sbatch slurm/agent2_e2e_materialize_eval_1step.slurm
#SBATCH --job-name=agent2-e2e-mat-eval-1step
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh" gpu
export OMP_NUM_THREADS=4

STEM="${STEM:?must set STEM (e.g. e2e_step1)}"
EPOCH="${EPOCH:?must set EPOCH (overall epoch label, e.g. 2)}"
E2E_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_e2e_continued
OUT_DIR=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_e2e_continued
mkdir -p "$OUT_DIR"

ENC="$E2E_DIR/${STEM}_encoder.pt"
WORK="$E2E_DIR/materialized_validation_work_${STEM}"
CACHE="$E2E_DIR/materialized_validation_cache_${STEM}.pt"

echo "agent2_e2e_materialize_eval_1step host=$(hostname) stem=$STEM epoch=$EPOCH"; nvidia-smi -L || true; date
test -f "$ENC" || { echo "ERROR: encoder checkpoint missing: $ENC" >&2; exit 1; }

echo "=== materialize ==="
python -u -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=materialize --set device=cuda \
    --set num_workers=1 --set worker_index=0 \
    --set output_dir="$WORK" \
    --set packed_data="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$ENC" \
    --set loader_workers=0 --set resume=false

echo "=== merge ==="
python -u -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=merge --set num_workers=1 \
    --set output_dir="$WORK" \
    --set final_cache="$CACHE" \
    --set manifest_path="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$ENC"

echo "=== paired-significance eval, 3 regimes ==="
python -u -m analysis.evaluate_our_trees_continued \
    --mode e2e \
    --packed-root "$MCP" \
    --original-cache "$MAT/validation_cache.pt" \
    --checkpoints "$CACHE" \
    --epochs "$EPOCH" \
    --regimes "0.01:0.0,0.01:0.001,0.001:0.0005" \
    --d-embed 32 \
    --out-dir "$OUT_DIR"

echo "AGENT2 E2E MATERIALIZE+EVAL 1STEP DONE ($STEM, epoch=$EPOCH) -> $OUT_DIR"; date
