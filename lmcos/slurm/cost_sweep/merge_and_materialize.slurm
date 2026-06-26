#!/bin/bash
#SBATCH --job-name=cs-mat
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%j.err
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

: "${SWEEP_NAME:?SWEEP_NAME env var required}"

PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv
ROOT=/scratch/gpfs/GRIFFITHS/hl4291
SWEEP_DIR="$ROOT/cost_sweep/$SWEEP_NAME"
ENCODER_CKPT="$ROOT/GNN/encoder_best.pt"

module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3 >/dev/null 2>&1 || true
module load cudatoolkit/12.8 >/dev/null 2>&1 || true
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "sweep=$SWEEP_NAME host=$(hostname) start=$(date)"
nvidia-smi -L || true

# ── 1. Merge pack chunks ──────────────────────────────────────────────────────
python3 - <<'PY'
import json, glob, os, sys
from pathlib import Path

sweep_dir = Path(os.environ["SWEEP_DIR"])
mc_dir = sweep_dir / "mc"
mc_dir.mkdir(parents=True, exist_ok=True)

for split, mf in (("train", "train_manifest.json"), ("validation", "validation_manifest.json")):
    chunks = sorted(glob.glob(str(sweep_dir / "mc_chunks" / "chunk_*" / mf)))
    if not chunks:
        print(f"ERROR: no chunks found for {split}", file=sys.stderr)
        sys.exit(1)
    base = json.load(open(chunks[0]))
    entries = []
    for c in chunks:
        for e in json.load(open(c)).get("entries", []):
            e = dict(e)
            e["shard_index"] = len(entries)
            entries.append(e)
    out = {k: v for k, v in base.items() if k != "entries"}
    out["entries"] = entries
    if "total_episodes" in out:
        out["total_episodes"] = sum(e.get("num_episodes", 0) for e in entries)
    json.dump(out, open(mc_dir / mf, "w"))
    print(f"{split} -> {len(entries)} shards, {sum(e.get('num_episodes', 0) for e in entries)} episodes")
PY

echo "merge done $(date)"

# ── 2. Materialize (single worker per split) ──────────────────────────────────
MAT_DIR="$SWEEP_DIR/materialized"
mkdir -p "$MAT_DIR"

for SPLIT in train validation; do
    MANIFEST="$SWEEP_DIR/mc/${SPLIT}_manifest.json"
    WORKER_DIR="$MAT_DIR/${SPLIT}_worker"
    FINAL_CACHE="$MAT_DIR/${SPLIT}_cache.pt"

    echo "materializing $SPLIT ..."
    python -u -m cts.data.preprocess_mc.materialize --config - <<YAML
command: materialize
output_dir: $WORKER_DIR
num_workers: 0
worker_index: 0
packed_data: $MANIFEST
encoder_checkpoint: $ENCODER_CKPT
device: cuda
episode_batch_size: 8
max_snapshots_per_shard: 250000
hidden_dim: 256
hidden_layers: 3
YAML

    echo "merging $SPLIT ..."
    python -u -m cts.data.preprocess_mc.materialize --config - <<YAML
command: merge
output_dir: $WORKER_DIR
num_workers: 1
final_cache: $FINAL_CACHE
manifest_path: $MANIFEST
encoder_checkpoint: $ENCODER_CKPT
YAML

done

echo "materialize done $(date)"
