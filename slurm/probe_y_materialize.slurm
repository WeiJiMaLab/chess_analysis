#!/bin/bash
# Probe Y (plan.md "External expert input" section) -- materialize half, the ARRAY job.
# Materializes z_t from Yotam Sagiv's subtree-weighted encoder (d_embed=128, ~4x our
# production encoder's width) against Agent 1's already-built, READ-ONLY argmax>2-filtered
# human_trees mc_packed episodes (train + validation). Mirrors slurm/pipeline/pack_root.slurm's
# array/worker pattern exactly (a local smoke test showed materialize is genuinely slow --
# ~10s/episode single-threaded on this encoder, so a single-worker job would take ~10h for
# the full 3,723-episode corpus; array parallelism is required, matching pack_root.slurm's
# own real-world calibration for a similar workload). probe_y_materialize_merge.slurm
# (afterok) stitches the shards. Writes ONLY under our own scratch
# (/scratch/gpfs/GRIFFITHS/hl4291/lmcos/probe_y_subtree_weighted/) -- never into
# ysagiv_argmax/ or /scratch/gpfs/GRIFFITHS/ysagiv/.
#   CONFIG=config_probe_y_subtree.yaml sbatch --array=0-15 slurm/probe_y_materialize.slurm
# For a straggler re-run of specific indices, set NWORKERS to the ORIGINAL array size (not
# the re-run's smaller --array count) so the dataset chunk boundaries match the already-
# completed workers' slicing exactly, e.g.:
#   NWORKERS=16 sbatch --array=1,7,11,12 slurm/probe_y_materialize.slurm
#SBATCH --job-name=probe-y-materialize
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
# --mem 6G->12G (2026-07-08, coordinator-diagnosed): 4/16 tasks (_1,_7,_11,_12) were
# OOM-killed at the original 6G ceiling -- sacct showed even the 12 COMPLETED tasks riding
# at 6286-6292MB RSS, i.e. essentially zero margin, not bad luck. 12G gives real headroom.
# --qos short->test: each task only takes 4-10min (well under 1h), and test is far less
# contended than short on this cluster (della QOS time-routing convention).

set -euo pipefail
WID=${SLURM_ARRAY_TASK_ID:?must be submitted as an array (--array=0-N)}
NW=${NWORKERS:-${SLURM_ARRAY_TASK_COUNT:?array count unavailable}}

TINY=/home/hl4291/chess_analysis
export CONFIG="${CONFIG:-$TINY/config_probe_y_subtree.yaml}"
source "$TINY/slurm/helpers/setup_env.sh"   # CPU: no cudatoolkit needed
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # avoid thread oversubscription under cgroups

MCP=$($RENDER --get globals.mc_packed_dir)
MAT=$($RENDER --get globals.materialized_dir)
ENC=$($RENDER --get materialize.encoder_checkpoint)

echo "probe_y_materialize worker=$WID/$NW host=$(hostname) MCP=$MCP MAT=$MAT ENC=$ENC"; date
test -f "$ENC" || { echo "ERROR: encoder missing: $ENC" >&2; exit 1; }

for SP in train validation; do
  echo "=== materialize $SP worker $WID/$NW (cpu) ==="
  python -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
    --set command=materialize --set device=cpu \
    --set num_workers="$NW" --set worker_index="$WID" \
    --set output_dir="$MAT/${SP}_work" --set packed_data="$MCP/${SP}_manifest.json" \
    --set loader_workers=0 --set resume=false
done

echo "PROBE Y MATERIALIZE WORKER $WID/$NW DONE"; date
