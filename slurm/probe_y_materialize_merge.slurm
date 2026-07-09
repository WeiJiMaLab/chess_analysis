#!/bin/bash
# Probe Y -- merge half. Stitches the probe_y_materialize.slurm array's per-worker shards
# into the final {train,validation}_cache.pt this probe's eval script reads. Mirrors
# slurm/pipeline/pack_root_merge.slurm exactly, pointed at config_probe_y_subtree.yaml.
#   sbatch --dependency=afterok:<probe_y_materialize_arrayjob> slurm/probe_y_materialize_merge.slurm
#SBATCH --job-name=probe-y-materialize-merge
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="${CONFIG:-$TINY/config_probe_y_subtree.yaml}"
source "$TINY/slurm/helpers/setup_env.sh"

MCP=$($RENDER --get globals.mc_packed_dir)
MAT=$($RENDER --get globals.materialized_dir)

echo "probe_y_materialize_merge host=$(hostname)"; date
for SP in train validation; do
  NW=$(find "$MAT/${SP}_work" -maxdepth 1 -type d -name 'worker_*' | wc -l)
  (( NW > 0 )) || { echo "ERROR: no worker_* dirs in $MAT/${SP}_work" >&2; exit 1; }
  echo "=== merge $SP ($NW workers) -> ${SP}_cache.pt ==="
  python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=merge --set num_workers="$NW" \
    --set output_dir="$MAT/${SP}_work" --set final_cache="$MAT/${SP}_cache.pt" \
    --set manifest_path="$MCP/${SP}_manifest.json"
  test -f "$MAT/${SP}_cache.pt" || { echo "ERROR: cache not written: $MAT/${SP}_cache.pt" >&2; exit 1; }
done

echo "PROBE Y MATERIALIZE MERGE DONE -> $MAT/{train,validation}_cache.pt"; date
