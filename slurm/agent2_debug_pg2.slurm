#!/bin/bash
#SBATCH --job-name=agent2-debug-pg2
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS=4
cd "$TINY/src"

python -u - <<'PYEOF'
import numpy as np
import torch
from pathlib import Path
import os

from analysis.evaluate_our_trees_continued import evaluate_pg_checkpoint, build_pg_model, pg_checkpoint_regret
from analysis.evaluate import _load_assessment_data, _return_curves, _oracle_config, _fit_stop_controllers, _regret_at
from dataclasses import replace

MCP = os.environ["MCP"]; MAT = os.environ["MAT"]
CKPT = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt"
CONFIG_PATH = "/home/hl4291/chess_analysis/config_minply15_maxply75.yaml"

config = replace(_oracle_config(Path(MCP)), time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0)
episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(Path(MCP), f"{MAT}/validation_cache.pt", 32, 50, 0)
curves = _return_curves(episodes, config)
ev = [curves[i] for i in ev_idx]

model = build_pg_model(CONFIG_PATH, CKPT, d_embed=32, unfreeze_encoder=False)
stops = []
with torch.no_grad():
    from analysis.evaluate_our_trees_continued import _pg_features
    from cts.data.preprocess_mc.oracle import predicted_stop_from_advantages
    for i in ev_idx:
        feats = _pg_features(z_by_ep[i], 32)
        adv, _ = model.predict_from_features(feats)
        s = predicted_stop_from_advantages(adv.reshape(-1).tolist())
        stops.append(s)
stops = np.array(stops)
print("stops", stops.tolist())
print("episode lengths", [len(c) for c in ev])

ckpt_regret = _regret_at(ev, stops)
print("ckpt_regret per episode", ckpt_regret.tolist())
print("ckpt_regret mean", ckpt_regret.mean())

always_stop_regret = _regret_at(ev, np.zeros(len(ev_idx), dtype=int))
print("always_stop_regret mean", always_stop_regret.mean())
always_continue_regret = _regret_at(ev, np.array([len(c)-1 for c in ev]))
print("always_continue_regret mean", always_continue_regret.mean())

# find the worst episode
worst = int(np.argmax(ckpt_regret))
print("worst episode idx", worst, "ev_idx", ev_idx[worst], "stop", stops[worst], "curve len", len(ev[worst]))
print("curve", ev[worst].tolist())
PYEOF
