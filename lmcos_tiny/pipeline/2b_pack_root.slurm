#!/bin/bash
# Phase 2b — pack root representations (materialize), PARALLEL worker shards.
# Each array task is one materialize worker (worker_index = SLURM_ARRAY_TASK_ID)
# processing its 1/NWORKERS slice of BOTH splits into worker_NN/ subdirs. A
# separate 2c_merge_root.slurm (afterok) stitches the shards into the final cache.
# loader_workers overlaps the CPU prefix-rebuild with the GPU encode.
#
#   ELO=2000 NWORKERS=40 sbatch --array=0-39 --dependency=afterok:<encoder_jobid> \
#       pipeline/2b_pack_root.slurm
#   ELO=2000 NWORKERS=40 sbatch --dependency=afterok:<this_array_jobid> \
#       pipeline/2c_merge_root.slurm
#
#SBATCH --job-name=tiny-rootreps
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%A_%a.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${ELO:?ELO required (e.g. 2000)}"
: "${NWORKERS:?NWORKERS required (= array size, max 44)}"
WID=${SLURM_ARRAY_TASK_ID:?must be submitted as an array (--array=0-N)}

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh" gpu

LW=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))   # DataLoader background workers

echo "elo=$ELO worker=$WID/$NWORKERS loader_workers=$LW host=$(hostname)"; nvidia-smi -L || true; date
test -f "$ENC" || { echo "ERROR: encoder missing: $ENC" >&2; exit 1; }

for SP in train validation; do
  echo "=== materialize $SP worker $WID/$NWORKERS ==="
  python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set globals.sf_elo="$ELO" --set command=materialize \
    --set num_workers="$NWORKERS" --set worker_index="$WID" \
    --set output_dir="$MAT/${SP}_work" --set packed_data="$MCP/${SP}_manifest.json" \
    --set loader_workers="$LW"
done

echo "ROOT REPS WORKER $WID/$NWORKERS DONE elo=$ELO"; date
