#!/bin/bash
#SBATCH --job-name=prune_regen_an
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/prune_regen_an.%j.out
cd /home/hl4291/chess_analysis
set -euo pipefail
export PYTHONPATH=/home/hl4291/chess_analysis/lmcos_small/src${PYTHONPATH:+:$PYTHONPATH}
/home/hl4291/venv/bin/python lmcos_small/analysis/prune_regen_analyze.py
