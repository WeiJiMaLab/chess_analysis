#!/bin/bash
#SBATCH --job-name=voc_signals
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/hl4291/chess_analysis/scratch_halt/voc_signals.%j.out

cd /home/hl4291/chess_analysis
/home/hl4291/venv/bin/python scratch_halt/compute_voc_signals.py
