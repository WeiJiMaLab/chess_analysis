#!/bin/bash
# Phase 1b — tree filter, PARALLEL CPU array. Each task classifies the
# trees[ARRAY_TASK_ID::NSHARDS] slice of sf_trees/elo${ELO} (PUCT-stable ∩
# monotone) and writes shards/clean_trees_shard_<id>.txt under ${filtered_dir}.
# A separate 1c_pack_trees.slurm (afterok) concatenates the shards into the
# canonical clean_trees.txt and runs split/gnn_pack/mc_pack.
#
#   ELO=2000 sbatch --array=0-99 --dependency=afterok:<gentrees_jobid> \
#       pipeline/1b_filter_trees.slurm
#
#SBATCH --job-name=tiny-filter
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"
WID=${SLURM_ARRAY_TASK_ID:?must be submitted as an array (--array=0-N)}
NSHARDS=${SLURM_ARRAY_TASK_COUNT:?array count unavailable}

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh"

echo "elo=$ELO shard=$WID/$NSHARDS host=$(hostname)"; date

echo "=== filter shard $WID/$NSHARDS (PUCT-stable ∩ monotone) ==="
python -m cts.data.filter_trees_by_trace --config "$CONFIG" --stage filter \
  --set globals.sf_elo="$ELO" --set n_workers=1 \
  --set num_shards="$NSHARDS" --set shard_index="$WID"

echo "FILTER SHARD $WID/$NSHARDS DONE elo=$ELO"; date
