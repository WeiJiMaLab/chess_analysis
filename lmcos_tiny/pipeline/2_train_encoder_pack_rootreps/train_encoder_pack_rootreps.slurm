#!/bin/bash
# Phase 2 — train encoder + pack root representations (GPU).
#   encoder     -> cts.data.build_tree (pretrain-child-wdl-encoder)  [tiny encoder]
#   materialize -> cts.data.preprocess_mc.materialize                [frozen enc -> z_t cache]
# Both sections live in configs/elo${ELO}.yaml. The materialize section holds only
# shared params; per-split/-command paths are injected with render_stage --set.
#
#   ELO=1800 sbatch --dependency=afterok:<packtrees_jobid> \
#     pipeline/2_train_encoder_pack_rootreps/train_encoder_pack_rootreps.slurm
#SBATCH --job-name=tiny-encoder-reps
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.err
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
VENV_DIR=/home/hl4291/venv
ROOT=/scratch/gpfs/GRIFFITHS/hl4291
CONFIG=$TINY/configs/elo${ELO}.yaml
WORK=$ROOT/sf_pack_configs/elo${ELO}
MCP=$ROOT/sf_mc_packed/elo${ELO}
MAT=$ROOT/sf_mc_materialized/elo${ELO}
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

# ── train encoder (tiny, child-WDL) ───────────────────────────────────────────
echo "=== encoder ==="
$RENDER encoder --out "$WORK/encoder.yaml"
python -m cts.data.build_tree --config "$WORK/encoder.yaml"

# ── pack root reps: materialize frozen encoder -> z_t cache (per split) ────────
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

echo "ENCODER + ROOT REPS DONE elo=$ELO"; echo "  caches: $MAT/{train,validation}_cache.pt"; date
