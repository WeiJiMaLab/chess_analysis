#!/bin/bash
# Phase 2 (rerun) — pack root representations ONLY (materialize), reusing the
# already-trained tiny encoders. One Elo rung per array task. Use this when the
# encoder exists but the z_t caches are missing (e.g. the overnight materialize
# crashed). Identical materialize logic to train_encoder_pack_rootreps.slurm,
# minus the encoder-training step.
#
#   sbatch --array=0-2 pipeline/2_train_encoder_pack_rootreps/pack_rootreps_rerun.slurm
#SBATCH --job-name=tiny-rootreps-rerun
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%A_%a.err
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
ELOS=(1800 2000 2200)
ELO=${ELOS[${SLURM_ARRAY_TASK_ID:-0}]}

TINY=/home/hl4291/chess_analysis/lmcos_tiny
VENV_DIR=/home/hl4291/venv
ROOT=/scratch/gpfs/GRIFFITHS/hl4291
CONFIG=$TINY/configs/elo${ELO}.yaml
WORK=$ROOT/sf_pack_configs/elo${ELO}
MCP=$ROOT/sf_mc_packed/elo${ELO}
MAT=$ROOT/sf_mc_materialized/elo${ELO}
ENC=$ROOT/sf_packed/elo${ELO}/tiny_encoder.pt
mkdir -p "$ROOT/sf_pack_logs" "$WORK" "$MAT"
RENDER="python $TINY/configs/render_stage.py $CONFIG"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
source "$VENV_DIR/bin/activate"
set -u
export PYTHONPATH="$TINY/src"; export PYTHONUNBUFFERED=1

echo "elo=$ELO host=$(hostname)"; nvidia-smi -L || true; date
test -f "$ENC" || { echo "ERROR: encoder missing: $ENC" >&2; exit 1; }

for SP in train validation; do
  WORKER_DIR=$MAT/${SP}_worker
  FINAL_CACHE=$MAT/${SP}_cache.pt
  MANIFEST=$MCP/${SP}_manifest.json
  echo "=== materialize $SP (worker) ==="
  $RENDER materialize --out "$WORK/materialize_${SP}.yaml" \
    --set command=materialize --set num_workers=1 --set worker_index=0 \
    --set output_dir="$WORKER_DIR" --set packed_data="$MANIFEST"
  python -u -m cts.data.preprocess_mc.materialize --config "$WORK/materialize_${SP}.yaml"
  echo "=== materialize $SP (merge) ==="
  $RENDER materialize --out "$WORK/materialize_${SP}_merge.yaml" \
    --set command=merge --set num_workers=1 \
    --set output_dir="$WORKER_DIR" --set final_cache="$FINAL_CACHE" --set manifest_path="$MANIFEST"
  python -u -m cts.data.preprocess_mc.materialize --config "$WORK/materialize_${SP}_merge.yaml"
  test -f "$FINAL_CACHE" || { echo "ERROR: cache not written: $FINAL_CACHE" >&2; exit 1; }
done

echo "ROOT REPS (rerun) DONE elo=$ELO"; date
