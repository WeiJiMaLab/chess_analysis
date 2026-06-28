#!/bin/bash
# P2 Stockfish ladder — GPU stages for one Elo rung (run after sf_pack_cpu).
#   stage 5: encoder pretrain (child-WDL)  (cts.data.build_tree pretrain-child-wdl-encoder)
#   stage 7: materialize (frozen encoder -> z_root cache, train + validation + merge)
# Tiny encoder (k=1, d_embed=32) per the P2 POC; materialize single-worker.
# NOTE: the materialize *worker* needs num_workers>=1 (num_workers:0 div-by-zeros).
#
# Submit (per rung, after sf_pack_cpu):
#   ELO=1800 sbatch --dependency=afterok:<cpu_jobid> slurm/1_preprocess_data/sf_pack_gpu.slurm
#SBATCH --job-name=sf-pack-gpu
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.err
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${ELO:?ELO env var required (e.g. 1800)}"

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

# ── stage 5: encoder pretrain (child-WDL), GPU ────────────────────────────────
cat > "$CFG/encoder.yaml" <<YAML
command: pretrain-child-wdl-encoder
train_dir: $GNN/train_manifest.json
validation_dir: $GNN/validation_manifest.json
output_checkpoint: $ENC
device: cuda
k: 1
d_embed: 32
d_message: 32
node_embed_hidden: 32
n_heads: 2
d_att: 16
decoder_hidden: 32
batch_size: 256
learning_rate: 1.0e-3
epochs: 6
loss_type: huber
huber_delta: 1.0
num_workers: 0
log_interval: 20
YAML
echo "=== stage 5 encoder ==="; python -m cts.data.build_tree --config "$CFG/encoder.yaml"
test -f "$ENC" || { echo "ERROR: encoder checkpoint not written: $ENC" >&2; exit 1; }

# ── stage 7: materialize (frozen encoder -> z_root cache), GPU ─────────────────
# NOTE: the config loader (cts._config.load_config) reads --config as a real file
# path via Path(path).read_text(); it does NOT support `--config -`/stdin. Write
# each stage's YAML to a file and pass the path (same as the encoder stage above).
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

echo "GPU PACK DONE elo=$ELO $(date)"
echo "  encoder: $ENC"
echo "  caches:  $MAT/{train,validation}_cache.pt"
