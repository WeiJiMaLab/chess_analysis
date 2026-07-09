#!/bin/bash
# plan.md S3 -- materialize z_t from the S2 stats-predicting-encoder checkpoint on the FULL
# validation split, then run the frontier/decodability comparison against
# SingleHalt*/Stats-Controller/original-(child-WDL frozen-encoder)-z_t.
#
# Depends on slurm/diagnosis_s2_stats_regression_pretrain.slurm having produced
# $S2_DIR/s2_stats_encoder.pt (the encoder-only checkpoint format
# cts.data.preprocess_mc.materialize.materialize_worker consumes -- see
# stats_regression_pretrain.py's save_encoder_checkpoint call). Submit with:
#   sbatch --dependency=afterok:<s2_jobid> slurm/diagnosis_s3_stats_materialize_and_eval.slurm
#
# Does NOT touch the production materialized cache or controller checkpoint -- writes to the
# SAME s2_stats_encoder scratch dir the training job used, plus a diagnosis figures subdir.
#
#SBATCH --job-name=diag-s3-stats-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
# Materializing z_t for just the validation split (5599 episodes, single worker, inference-only
# -- no backward pass) should be much faster than a training epoch; 40 min leaves generous
# headroom for the (cheap, seconds-scale) readout re-fits and plotting the S3 comparison does.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

S2_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/s2_stats_encoder
S2_ENC="$S2_DIR/s2_stats_encoder.pt"
S2_CACHE_WORK="$S2_DIR/materialized_validation_work"
S2_CACHE="$S2_DIR/materialized_validation_cache.pt"
S3_OUT=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/diagnosis/s3_stats_comparison
mkdir -p "$S3_OUT"

echo "diag_s3_stats_materialize_and_eval host=$(hostname)"; nvidia-smi -L || true; date
test -f "$S2_ENC" || { echo "ERROR: S2 encoder checkpoint missing: $S2_ENC (did S2 finish?)" >&2; exit 1; }

echo "=== materialize S2 stats-predicting z_t on validation split (single worker, full 5599 episodes) ==="
python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=materialize --set device=cuda \
    --set num_workers=1 --set worker_index=0 \
    --set output_dir="$S2_CACHE_WORK" \
    --set packed_data="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$S2_ENC" \
    --set loader_workers=0 --set resume=false

echo "=== merge into final cache ==="
python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=merge --set num_workers=1 \
    --set output_dir="$S2_CACHE_WORK" \
    --set final_cache="$S2_CACHE" \
    --set manifest_path="$MCP/validation_manifest.json" \
    --set encoder_checkpoint="$S2_ENC"

echo "=== S3 comparison: S2 stats-predicting z_t vs SingleHalt*/Stats-Controller/original z_t ==="
python -m analysis.evaluate_s3_stats_z3 \
    --packed-root "$MCP" \
    --s2-cache "$S2_CACHE" \
    --original-cache "$MAT/validation_cache.pt" \
    --baseline-fig-dir /home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative \
    --out-dir "$S3_OUT" \
    --d-embed 32 \
    --maintenance-scales 0.0,0.05,0.1

echo "DIAG-S3 STATS EVAL DONE"; date
