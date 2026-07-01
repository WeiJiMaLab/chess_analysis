#!/bin/bash
# Pipeline step — pack_root (CPU), the ARRAY half. Materializes the
# frozen-encoder root representations (z_t cache) the readout trains on. Each
# array task is one materialize worker (worker_index = SLURM_ARRAY_TASK_ID)
# processing its 1/N slice of BOTH splits into worker_NN/ subdirs;
# pack_root_merge.slurm (afterok) stitches the shards into the final cache.
# The frozen-encoder forward is inference-only -> CPU. submit_all.sh submits this
# as an array and chains the merge automatically (manifest: array + merge).
#   sbatch --array=0-39 --dependency=afterok:<train_encoder_jobid> slurm/pipeline/pack_root.slurm
#SBATCH --job-name=pack-root
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=01:05:00

set -euo pipefail
WID=${SLURM_ARRAY_TASK_ID:?must be submitted as an array (--array=0-N)}
NW=${SLURM_ARRAY_TASK_COUNT:?array count unavailable}

TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # CPU: no cudatoolkit needed

LW=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))   # DataLoader background workers

echo "pack_root worker=$WID/$NW loader_workers=$LW host=$(hostname) device=cpu"; date
test -f "$ENC" || { echo "ERROR: encoder missing: $ENC" >&2; exit 1; }

for SP in train validation; do
  echo "=== materialize $SP worker $WID/$NW (cpu) ==="
  python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=materialize --set device=cpu \
    --set num_workers="$NW" --set worker_index="$WID" \
    --set output_dir="$MAT/${SP}_work" --set packed_data="$MCP/${SP}_manifest.json" \
    --set loader_workers="$LW"
done

echo "PACK ROOT WORKER $WID/$NW DONE"; date
