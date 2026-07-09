#!/bin/bash
# Probe X, condition B materialize. Single-worker validation-split materialize + merge, same
# pattern as condition A's slurm/probe_x_xaba_materialize.slurm, pointed at condition B's config
# (random-30K-subsample + exclude_xaba-only mc_pack output). Reuses the same frozen encoder.
#   sbatch --dependency=afterok:<condb_split_pack_jobid> slurm/probe_x_xaba_condb_materialize.slurm
#SBATCH --job-name=probe-x-xaba-condb-materialize
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:55:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_probe_x_xaba_condb.yaml"
source "$TINY/slurm/helpers/setup_env.sh"

echo "probe_x_xaba_condb_materialize host=$(hostname)"; date
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
echo "PROBE X XABA CONDB MATERIALIZE DONE -> $MAT/validation_cache.pt"; date
