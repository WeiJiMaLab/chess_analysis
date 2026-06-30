#!/bin/bash
# Stage 2d — DATA ANALYSIS (engine-derived RT correspondence, NO refitting).
#
# A parallel fork off the stage-1 trees: it needs only the trees + the human RT DB, NOT the
# GNN 2a–2c branch, and it is NOT 4_eval (no model is being evaluated — these are engine-level
# fits with zero/one free parameter). Heavy steps: the per-tree budgeted-oracle VOC signals and
# the softmax-VOC(policy) tau sweep, over the UNFILTERED population by default (VOC_FILTERED=0).
# Outputs land in sf_analysis/<SET>/{voc_signals,voc_tau_sweep}.parquet; figures are a fast
# post-step (make_rt_figures, run off the parquets). The halt-policy zoo (regret_vs_compute,
# regret_by_model) and the hindsight halter stay in 3_train/4_eval — they require training.
#
#   for S in n1md36 n100md36 n1md36_eps0.05 n1md36_eps0.1 n1md36_eps0.3; do
#     sbatch --export=ALL,VOC_SET=$S lmcos_tiny/pipeline/2d_data_analysis.slurm
#   done
#
#SBATCH --job-name=2d_data_analysis
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/2d_%x_%j_%a.out

set -euo pipefail
SET="${VOC_SET:?set VOC_SET=n1md36|n100md36|n1md36_eps0.1|...}"
export VOC_SET="$SET"
export VOC_FILTERED="${VOC_FILTERED:-0}"   # 0 = unfiltered (the full population)
cd /home/hl4291/chess_analysis
PY=/home/hl4291/venv/bin/python

echo "=== 2d_data_analysis SET=$SET FILTERED=$VOC_FILTERED ($(date)) ==="
echo "--- per-tree budgeted-oracle VOC signals ---"
$PY lmcos_tiny/analysis/compute_voc_signals.py
echo "--- softmax-VOC(policy) tau sweep ---"
$PY lmcos_tiny/analysis/compute_voc_tau_sweep.py
echo "=== 2d done: sf_analysis/$SET/{voc_signals,voc_tau_sweep}.parquet ($(date)) ==="
