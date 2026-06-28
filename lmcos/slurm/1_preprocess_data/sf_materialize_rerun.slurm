#!/bin/bash
# P2 Stockfish ladder — materialize-only rerun (stage 7), one Elo rung per array task.
# The overnight sf_pack_gpu run trained the encoders (stage 5, tiny_encoder.pt present)
# but crashed at stage 7 because it piped the YAML via `--config -` (stdin), which the
# config loader does not support. This script reruns ONLY stage 7 against the existing
# encoders, writing each materialize/merge config to a real file.
#
# Submit (all three rungs):
#   sbatch --array=0-2 slurm/1_preprocess_data/sf_materialize_rerun.slurm
#SBATCH --job-name=sf-materialize
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

ROOT=/scratch/gpfs/GRIFFITHS/hl4291
PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv

GNN=$ROOT/sf_packed/elo${ELO}
MCP=$ROOT/sf_mc_packed/elo${ELO}
MAT=$ROOT/sf_mc_materialized/elo${ELO}
ENC=$GNN/tiny_encoder.pt
CFG=$ROOT/sf_pack_configs/elo${ELO}
mkdir -p "$ROOT/sf_pack_logs" "$CFG" "$MAT"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
source "$VENV_DIR/bin/activate"
set -u

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1

echo "elo=$ELO host=$(hostname) start=$(date)"
nvidia-smi -L || true
test -f "$ENC" || { echo "ERROR: encoder missing: $ENC" >&2; exit 1; }

# ── stage 7: materialize (frozen encoder -> z_root cache), GPU ─────────────────
for SP in train validation; do
  WORKER_DIR=$MAT/${SP}_worker
  FINAL_CACHE=$MAT/${SP}_cache.pt
  MANIFEST=$MCP/${SP}_manifest.json
  echo "=== stage 7 materialize $SP (worker) ==="
  cat > "$CFG/materialize_${SP}.yaml" <<YAML
command: materialize
output_dir: $WORKER_DIR
num_workers: 1
worker_index: 0
packed_data: $MANIFEST
encoder_checkpoint: $ENC
device: cuda
episode_batch_size: 8
max_snapshots_per_shard: 250000
hidden_dim: 256
hidden_layers: 3
YAML
  python -u -m cts.data.preprocess_mc.materialize --config "$CFG/materialize_${SP}.yaml"
  echo "=== stage 7 materialize $SP (merge) ==="
  cat > "$CFG/materialize_${SP}_merge.yaml" <<YAML
command: merge
output_dir: $WORKER_DIR
num_workers: 1
final_cache: $FINAL_CACHE
manifest_path: $MANIFEST
encoder_checkpoint: $ENC
YAML
  python -u -m cts.data.preprocess_mc.materialize --config "$CFG/materialize_${SP}_merge.yaml"
  test -f "$FINAL_CACHE" || { echo "ERROR: cache not written: $FINAL_CACHE" >&2; exit 1; }
done

echo "MATERIALIZE DONE elo=$ELO $(date)"
echo "  caches: $MAT/{train,validation}_cache.pt"
