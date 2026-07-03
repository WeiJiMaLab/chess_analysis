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
#SBATCH --time=06:00:00
# --time 3:00->6:00: a few materialize workers hung early in the VALIDATION split
# (train finished, second DataLoader deadlocked — the classic torch multiprocessing
# loader hang) and were killed at 3h. Re-run stragglers with LOADER_WORKERS=0 (a
# synchronous loader can't deadlock); the longer wall is headroom.

set -euo pipefail
WID=${SLURM_ARRAY_TASK_ID:?must be submitted as an array (--array=0-N)}
# num_workers defaults to the array size, but can be overridden (NWORKERS=) so a
# subset re-run of stragglers keeps the SAME slicing as the original full array.
NW=${NWORKERS:-${SLURM_ARRAY_TASK_COUNT:?array count unavailable}}

TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # CPU: no cudatoolkit needed

# DataLoader background workers. Defaults to cpus-1, but LOADER_WORKERS=0 forces
# a synchronous loader (no worker subprocesses) — set that on a straggler re-run
# to sidestep the multiprocessing-DataLoader deadlock that hung the validation split.
LW=${LOADER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}

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
