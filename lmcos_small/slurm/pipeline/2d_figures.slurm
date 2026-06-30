#!/bin/bash
# 2d figures — fast post-step: make_rt_figures off the sf_analysis/<SET> parquets.
# Run ONCE on the chosen baseline (overwrites the canonical figures/normative/*.png).
#   sbatch --dependency=afterok:<2d_job> --export=ALL,VOC_SET=n1md36 lmcos_small/slurm/pipeline/2d_figures.slurm
#SBATCH --job-name=2d_figures
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/2d_figures_%j.out
set -euo pipefail
export VOC_SET="${VOC_SET:?set VOC_SET}"
cd /home/hl4291/chess_analysis
echo "=== make_rt_figures SET=$VOC_SET ($(date)) ==="
/home/hl4291/venv/bin/python lmcos_small/src/analysis/make_rt_figures.py
echo "=== figures done ($(date)) ==="
ls -1 figures/normative/*.png 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '
