#!/bin/bash
#SBATCH --job-name=construal_proxy
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/analysis/logs/construal_proxy_%j.out
cd /home/hl4291/chess_analysis
set -euo pipefail
export VOC_SET="${VOC_SET:-n1md36}"
/home/hl4291/venv/bin/python lmcos_small/analysis/construal_proxy.py
echo "=== analyze ==="
/home/hl4291/venv/bin/python lmcos_small/analysis/construal_proxy.py analyze
