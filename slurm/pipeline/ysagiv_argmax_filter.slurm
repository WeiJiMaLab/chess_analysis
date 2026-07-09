#!/bin/bash
# Phase 2 Agent 1 (ysagiv human_trees) -- Filter A: "thinking helps" argmax>2 filter,
# reusing cts.data.preprocess_mc.filter_argmax unmodified. Runs on the seeded 20K-tree
# raw sample (symlinks under hl4291 scratch, NOT ysagiv's tree -- see ysagiv_trees.md).
# Output (include_list) is written under hl4291 scratch only.
#   sbatch slurm/pipeline/ysagiv_argmax_filter.slurm
#SBATCH --job-name=ysagiv-argmax-filter
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

TREES_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/raw_sample
OUT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/argmax/include_list.txt

echo "ysagiv argmax_filter trees_dir=${TREES_DIR} -> ${OUT} host=$(hostname)"; date
python -m cts.data.preprocess_mc.filter_argmax --trees-dir "$TREES_DIR" --output "$OUT"
echo "YSAGIV ARGMAX FILTER DONE"; date
