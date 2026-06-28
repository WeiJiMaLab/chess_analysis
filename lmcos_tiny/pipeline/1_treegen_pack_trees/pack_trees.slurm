#!/bin/bash
# Phase 1b — pack trees (CPU). Three sub-stages, all reading sections of
# configs/elo${ELO}.yaml via render_stage:
#   split    -> cts.data.preprocess_gnn.split   (train/val partition)
#   gnn_pack -> cts.data.preprocess_gnn.pack    (encoder training examples)
#   mc_pack  -> cts.data.preprocess_mc.pack     (budgeted-DP oracle episodes)
#
#   ELO=1800 sbatch --dependency=afterok:<gentrees_jobid> \
#     pipeline/1_treegen_pack_trees/pack_trees.slurm
#SBATCH --job-name=tiny-packtrees
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
VENV_DIR=/home/hl4291/venv
CONFIG=$TINY/configs/elo${ELO}.yaml
WORK=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_configs/elo${ELO}
mkdir -p "$TINY/slurm/logs" "$WORK"
RENDER="python $TINY/configs/render_stage.py $CONFIG"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
source "$VENV_DIR/bin/activate"
set -u
export PYTHONPATH="$TINY/src"; export PYTHONUNBUFFERED=1
NW=${SLURM_CPUS_PER_TASK:-32}

echo "elo=$ELO host=$(hostname) cpus=$NW"; date

echo "=== split ==="
$RENDER split --out "$WORK/split.yaml"
python -m cts.data.preprocess_gnn.split --config "$WORK/split.yaml"

echo "=== gnn_pack (encoder examples) ==="
$RENDER gnn_pack --out "$WORK/gnn_pack.yaml"
python -m cts.data.preprocess_gnn.pack --config "$WORK/gnn_pack.yaml"

echo "=== mc_pack (oracle episodes) ==="
$RENDER mc_pack --set num_workers=$NW --out "$WORK/mc_pack.yaml"
python -m cts.data.preprocess_mc.pack --config "$WORK/mc_pack.yaml"

echo "PACK TREES DONE elo=$ELO"; date
