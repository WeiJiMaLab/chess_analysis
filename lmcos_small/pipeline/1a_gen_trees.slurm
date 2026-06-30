#!/bin/bash
# Phase 1a — tree generation (CPU array). Stockfish search trees for one rung.
# Renders the `treegen` section of configs/elo${ELO}.yaml, then drives build_tree
# over a contiguous FEN index range per array task.
#
#   ELO=1800 SHARD_SIZE=2500 BASE_START=0 LANE_END=50000 \
#     sbatch --array=0-19 pipeline/1a_gen_trees.slurm
#SBATCH --job-name=tiny-gentrees
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"
: "${SHARD_SIZE:?SHARD_SIZE required}"; : "${BASE_START:?BASE_START required}"; : "${LANE_END:?LANE_END required}"

TINY=/home/hl4291/chess_analysis/lmcos_small
source "$TINY/pipeline/helpers/setup_env.sh"

start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE )); (( end > LANE_END )) && end=${LANE_END}
if (( start >= LANE_END )); then echo "no-op: start=${start} >= LANE_END=${LANE_END}"; exit 0; fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "elo=$ELO task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} host=$(hostname)"; date
# build_tree reads the `treegen` section of core.yaml directly; --set resolves
# ${sf_elo} in the paths, --override sets this task's FEN index range.
python -m cts.data.build_tree --config "$CONFIG" --stage treegen --set globals.sf_elo="$ELO" \
    --override start_index=${start} --override end_index=${end}
date
