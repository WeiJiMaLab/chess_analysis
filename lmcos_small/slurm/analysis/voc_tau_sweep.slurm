#!/bin/bash
#SBATCH --job-name=voc_tau
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/voc_tau_sweep_%j.out

set -euo pipefail
cd /home/hl4291/chess_analysis
/home/hl4291/venv/bin/python lmcos_small/src/analysis/compute_voc_tau_sweep.py
