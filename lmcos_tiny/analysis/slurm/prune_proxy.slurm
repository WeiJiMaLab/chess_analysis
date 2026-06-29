#!/bin/bash
#SBATCH --job-name=prune_proxy
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/prune_proxy.%j.out

cd /home/hl4291/chess_analysis
set -euo pipefail
/home/hl4291/venv/bin/python lmcos_tiny/analysis/prune_proxy.py
