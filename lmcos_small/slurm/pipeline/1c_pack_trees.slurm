#!/bin/bash
# Phase 1c — merge filter shards + pack (CPU). Runs afterok on the filter
# array. Concatenates shards/clean_trees_shard_*.txt into the canonical
# clean_trees.txt (under ${filtered_dir}), then runs the remaining sequential
# pack stages — split is fast, gnn_pack is serial in-process, mc_pack uses its
# own internal process pool:
#   split    -> cts.data.preprocess_gnn.split   (train/val partition)
#   gnn_pack -> cts.data.preprocess_gnn.pack    (encoder training examples)
#   mc_pack  -> cts.data.preprocess_mc.pack     (budgeted-DP oracle episodes)
#
#   ELO=2000 sbatch --dependency=afterok:<filter_array_jobid> \
#       slurm/pipeline/1c_pack_trees.slurm
#
#SBATCH --job-name=tiny-mergepack
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_small
source "$TINY/slurm/pipeline/helpers/setup_env.sh"
NW=${SLURM_CPUS_PER_TASK:-16}

FILTERED=$($RENDER --get globals.filtered_dir)
SHARD_DIR="$FILTERED/shards"
CLEAN="$FILTERED/clean_trees.txt"

echo "elo=$ELO host=$(hostname) cpus=$NW filtered=$FILTERED"; date

echo "=== merge filter shards -> clean_trees.txt ==="
shopt -s nullglob
SHARDS=("$SHARD_DIR"/clean_trees_shard_*.txt)
(( ${#SHARDS[@]} > 0 )) || { echo "ERROR: no shard files in $SHARD_DIR" >&2; exit 1; }
# Concatenate all shard basenames, drop blank lines, sort -> canonical list.
cat "${SHARDS[@]}" | sed '/^$/d' | sort -u > "$CLEAN"
echo "merged ${#SHARDS[@]} shards -> $(wc -l < "$CLEAN") clean trees in $CLEAN"

echo "=== split (filtered subset via include_list) ==="
python -m cts.data.preprocess_gnn.split --config "$CONFIG" --stage split --set globals.sf_elo="$ELO"

echo "=== gnn_pack (encoder examples) ==="
python -m cts.data.preprocess_gnn.pack --config "$CONFIG" --stage gnn_pack --set globals.sf_elo="$ELO"

echo "=== mc_pack (oracle episodes) ==="
python -m cts.data.preprocess_mc.pack --config "$CONFIG" --stage mc_pack --set globals.sf_elo="$ELO" --set num_workers="$NW"

echo "MERGE AND PACK DONE elo=$ELO"; date
