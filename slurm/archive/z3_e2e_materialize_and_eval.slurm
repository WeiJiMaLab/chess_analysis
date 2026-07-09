#!/bin/bash
# plan.md Z3 -- materialize z_t from the Z2 end-to-end (encoder-unfrozen) checkpoint on the
# FULL validation split, then run the frontier/decodability comparison against
# SingleHalt*/Stats-Controller/original-(frozen-encoder)-z_t.
#
# Depends on slurm/z2_e2e_controller_train.slurm having produced
# $Z2_DIR/e2e_controller_encoder.pt (the encoder-only checkpoint format
# cts.data.preprocess_mc.materialize.materialize_worker consumes -- see
# e2e_controller_train.py's _save_checkpoint). Submit with:
#   sbatch --dependency=afterok:<z2_jobid> slurm/z3_e2e_materialize_and_eval.slurm
#
# Does NOT touch the production materialized cache or controller checkpoint -- writes to the
# SAME z2_e2e_controller scratch dir the training job used, plus a diagnosis figures subdir.
#
#SBATCH --job-name=z3-e2e-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
# Materializing z_t for just the validation split (5599 episodes, single worker, inference-only --
# no backward pass) should be much faster than a training epoch; 40 min leaves generous headroom
# for the (cheap, seconds-scale) readout re-fits and plotting the z3 comparison script does on top.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

Z2_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller
E2E_ENC="$Z2_DIR/e2e_controller_encoder.pt"
E2E_CACHE_WORK="$Z2_DIR/materialized_validation_work"
E2E_CACHE="$Z2_DIR/materialized_validation_cache.pt"
Z3_OUT=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/diagnosis/z3_e2e_comparison
mkdir -p "$Z3_OUT"

echo "z3_e2e_materialize_and_eval host=$(hostname)"; nvidia-smi -L || true; date
test -f "$E2E_ENC" || { echo "ERROR: e2e encoder checkpoint missing: $E2E_ENC (did Z2 finish?)" >&2; exit 1; }

echo "=== materialize e2e z_t on validation split (single worker, full 5599 episodes) ==="
python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=materialize --set device=cuda \
    --set num_workers=1 --set worker_index=0 \
    --set output_dir="$E2E_CACHE_WORK" \
    --set packed_data="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$E2E_ENC" \
    --set loader_workers=0 --set resume=false

echo "=== merge into final cache ==="
python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=merge --set num_workers=1 \
    --set output_dir="$E2E_CACHE_WORK" \
    --set final_cache="$E2E_CACHE" \
    --set manifest_path="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$E2E_ENC"

echo "=== Z3 comparison: e2e z_t vs SingleHalt*/Stats-Controller/original z_t ==="
python -m analysis.evaluate_e2e_z3 \
    --packed-root "$MCP" \
    --e2e-cache "$E2E_CACHE" \
    --original-cache "$MAT/validation_cache.pt" \
    --baseline-fig-dir /home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative \
    --out-dir "$Z3_OUT" \
    --d-embed 32 \
    --maintenance-scales 0.0,0.05,0.1

echo "Z3 E2E EVAL DONE"; date
