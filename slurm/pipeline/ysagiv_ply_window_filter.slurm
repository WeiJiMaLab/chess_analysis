#!/bin/bash
# Ply-window filter for ysagiv `human_trees` (2026-07-09): keeps only trees whose root FEN
# occurs at ply 15-75 in OUR OWN move database (analysis.ysagiv_ply_filter -- see that module's
# docstring for the FEN-format/overlap validation). Restricted to the xaba100k include list
# (`--restrict-to`) so the two filters intersect in one pass -- the combined population is what
# config_ysagiv_xaba100k_minply15_maxply75.yaml's split.include_list points at. Depends on BOTH
# ysagiv_sample.slurm (raw_sample) AND ysagiv_xaba_filter.slurm (the xaba include list) having
# already run for the 100K sample.
#   sbatch --dependency=afterok:<ysagiv_xaba_filter_jobid> slurm/pipeline/ysagiv_ply_window_filter.slurm
#SBATCH --job-name=ysagiv-ply-window-filter
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
# --mem 8G->48G (2026-07-09): OOM-killed at 8G -- `analysis.ysagiv_ply_filter` connects to
# personal.db the same way filter_and_sample.slurm does (`analysis.utils.helpers.connect`,
# DUCKDB_MEMORY_LIMIT defaulting to 40GB) to SELECT DISTINCT fen over an 89M-row table; matches
# filter_and_sample.slurm's own 48G/40GB convention for the same class of query.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK:-16}" DUCKDB_MEMORY_LIMIT=40GB

TREES_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/raw_sample
RESTRICT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/xaba/include_list.txt
OUT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/xaba_minply15_maxply75/include_list.txt

echo "ysagiv ply_window_filter trees_dir=${TREES_DIR} restrict_to=${RESTRICT} -> ${OUT} host=$(hostname)"; date
python -m analysis.ysagiv_ply_filter --trees-dir "$TREES_DIR" --restrict-to "$RESTRICT" --output "$OUT"
echo "YSAGIV PLY WINDOW FILTER DONE"; date
