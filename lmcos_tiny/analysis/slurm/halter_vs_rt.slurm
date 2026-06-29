#!/bin/bash
#SBATCH --job-name=halter-rt
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/halter_vs_rt.%j.out
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
/home/hl4291/venv/bin/python /home/hl4291/chess_analysis/lmcos_tiny/analysis/halter_vs_rt.py
