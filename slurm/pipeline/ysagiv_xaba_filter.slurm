#!/bin/bash
# Phase 2 Agent 1 (ysagiv human_trees) -- Filter B: xaba-exclusion filter (coordinator
# correction 2026-07-08), via the new cts.data.preprocess_mc.filter_xaba (mirrors
# filter_argmax.py's CLI/output shape; reuses the existing
# `_source_top_level_root_churn_category` classifier from preprocess_mc/pack.py). Runs on
# the SAME seeded 20K-tree raw sample as ysagiv_argmax_filter.slurm, for a fair head-to-head.
# Output (include_list) is written under hl4291 scratch only.
#   sbatch slurm/pipeline/ysagiv_xaba_filter.slurm
#SBATCH --job-name=ysagiv-xaba-filter
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
OUT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/xaba/include_list.txt

echo "ysagiv xaba_filter trees_dir=${TREES_DIR} -> ${OUT} host=$(hostname)"; date
python -m cts.data.preprocess_mc.filter_xaba --trees-dir "$TREES_DIR" --output "$OUT"
echo "YSAGIV XABA FILTER DONE"; date
