#!/bin/bash
#SBATCH --job-name=prune_analyze
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/analysis/logs/prune_analyze.%j.out
cd /home/hl4291/chess_analysis
set -euo pipefail
export PYTHONPATH=/home/hl4291/chess_analysis/lmcos_small/src${PYTHONPATH:+:$PYTHONPATH}
/home/hl4291/venv/bin/python lmcos_small/analysis/prune_proxy.py analyze
