#!/bin/bash
# Phase 2c (merge) — stitch the parallel materialize worker shards into the final
# per-split z_t cache. CPU-only (reads worker_NN/manifest.pt, renumbers shards,
# writes <split>_cache.pt). Submit afterok on the 2b_pack_root worker array.
#
#   ELO=2000 NWORKERS=40 sbatch --dependency=afterok:<rootreps_array_jobid> \
#       pipeline/2c_merge_root.slurm
#
#SBATCH --job-name=tiny-rootreps-merge
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00

set -euo pipefail
: "${ELO:?ELO required (e.g. 2000)}"
: "${NWORKERS:?NWORKERS required (must match the worker array size)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh"

echo "elo=$ELO merge NWORKERS=$NWORKERS host=$(hostname)"; date

for SP in train validation; do
  echo "=== merge $SP ==="
  python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set globals.sf_elo="$ELO" --set command=merge --set num_workers="$NWORKERS" \
    --set output_dir="$MAT/${SP}_work" --set final_cache="$MAT/${SP}_cache.pt" \
    --set manifest_path="$MCP/${SP}_manifest.json"
  test -f "$MAT/${SP}_cache.pt" || { echo "ERROR: cache not written: $MAT/${SP}_cache.pt" >&2; exit 1; }
done

echo "ROOT REPS MERGE DONE elo=$ELO -> $MAT/{train,validation}_cache.pt"; date
