#!/bin/bash
# Pipeline step — pack_root merge half (CPU). Stitches the parallel materialize
# worker shards (worker_NN/) into the final per-split z_t cache. Runs afterok on
# the pack_root array; submit_all.sh wires this automatically and points
# downstream (train_readout_pg) at THIS job. The worker count is detected from
# the worker_* dirs, so it matches whatever array size pack_root used.
#   sbatch --dependency=afterok:<pack_root_arrayjob> slurm/pipeline/pack_root_merge.slurm
#SBATCH --job-name=pack-root-merge
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

echo "pack_root_merge host=$(hostname)"; date
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

echo "PACK ROOT MERGE DONE -> $MAT/{train,validation}_cache.pt"; date
