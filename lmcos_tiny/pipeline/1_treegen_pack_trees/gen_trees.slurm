#!/bin/bash
# Phase 1a — tree generation (CPU array). Stockfish search trees for one rung.
# Renders the `treegen` section of configs/elo${ELO}.yaml, then drives build_tree
# over a contiguous FEN index range per array task.
#
#   ELO=1800 SHARD_SIZE=2500 BASE_START=0 LANE_END=50000 \
#     sbatch --array=0-19 pipeline/1_treegen_pack_trees/gen_trees.slurm
#SBATCH --job-name=tiny-gentrees
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"
: "${SHARD_SIZE:?SHARD_SIZE required}"; : "${BASE_START:?BASE_START required}"; : "${LANE_END:?LANE_END required}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
VENV_DIR=/home/hl4291/venv
CONFIG=$TINY/configs/elo${ELO}.yaml
WORK=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_configs/elo${ELO}
mkdir -p "$TINY/slurm/logs" "$WORK"

start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE )); (( end > LANE_END )) && end=${LANE_END}
if (( start >= LANE_END )); then echo "no-op: start=${start} >= LANE_END=${LANE_END}"; exit 0; fi

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
source "$VENV_DIR/bin/activate"
set -u
export PYTHONPATH="$TINY/src"; export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "elo=$ELO task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} host=$(hostname)"; date
python "$TINY/configs/render_stage.py" "$CONFIG" treegen --out "$WORK/treegen.yaml"
python -m cts.data.build_tree --config "$WORK/treegen.yaml" \
    --override start_index=${start} --override end_index=${end}
date
