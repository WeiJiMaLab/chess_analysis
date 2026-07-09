#!/bin/bash
# Probe X (plan.md "External expert input" -- xaba filter on our own corpus). Materializes ONLY
# the validation split (single worker -- the base run's validation split is small, ~5.7K trees /
# 12 shards, so no array is needed) of the freshly xaba-filtered mc_pack output into a z_t cache,
# reusing the base run's ALREADY-TRAINED frozen encoder (config_probe_x_xaba.yaml's
# materialize.encoder_checkpoint points at it directly -- no retrain). Only "validation" is
# materialized because analysis.evaluate's assessment framework (_load_assessment_data) only ever
# reads the "validation" split, internally re-splitting it 70/30 into fit/eval -- "train" is unused
# by any of this investigation's paired-significance scripts.
#   sbatch --dependency=afterok:<mcpack_only_jobid> slurm/probe_x_xaba_materialize.slurm
#SBATCH --job-name=probe-x-xaba-materialize
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_probe_x_xaba.yaml"
source "$TINY/slurm/helpers/setup_env.sh"

echo "probe_x_xaba_materialize host=$(hostname)"; date
test -f "$ENC" || { echo "ERROR: encoder missing: $ENC" >&2; exit 1; }
echo "MCP=$MCP MAT=$MAT ENC=$ENC"

echo "=== materialize validation (single worker) ==="
python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
  --set command=materialize --set device=cpu \
  --set num_workers=1 --set worker_index=0 \
  --set output_dir="$MAT/validation_work" --set packed_data="$MCP/validation_manifest.json" \
  --set loader_workers=0 --set resume=false

echo "=== merge validation -> validation_cache.pt ==="
python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
  --set command=merge --set num_workers=1 \
  --set output_dir="$MAT/validation_work" --set final_cache="$MAT/validation_cache.pt" \
  --set manifest_path="$MCP/validation_manifest.json"

test -f "$MAT/validation_cache.pt" || { echo "ERROR: cache not written: $MAT/validation_cache.pt" >&2; exit 1; }
echo "PROBE X XABA MATERIALIZE DONE -> $MAT/validation_cache.pt"; date
