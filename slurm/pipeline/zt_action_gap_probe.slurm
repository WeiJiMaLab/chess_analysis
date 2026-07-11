#!/bin/bash
# One-off probe (CPU) — does z_t alone linearly/MLP-recover action_gap (top1-top2 root Q)?
# Not a pipeline stage; ad-hoc script requested mid-investigation, kept here since every other
# SLURM invocation in this session goes through sbatch per this repo's convention (never
# backgrounded on the login node -- the two prior interactive attempts got SIGKILLed there).
#SBATCH --job-name=zt-ag-probe
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "zt_action_gap_probe host=$(hostname)"; date
python3 "$TINY/slurm/pipeline/_zt_recovers_action_gap.py"
echo "PROBE DONE"; date
