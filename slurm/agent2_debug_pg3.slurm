#!/bin/bash
#SBATCH --job-name=agent2-debug-pg3
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
from dataclasses import replace

from analysis.evaluate_our_trees_continued import build_pg_model, _pg_features
from analysis.evaluate import _load_assessment_data, _return_curves, _oracle_config, _regret_at
from cts.data.preprocess_mc.oracle import predicted_stop_from_advantages

MCP = os.environ["MCP"]; MAT = os.environ["MAT"]
CKPT = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt"
CONFIG_PATH = "/home/hl4291/chess_analysis/config_minply15_maxply75.yaml"

config = replace(_oracle_config(Path(MCP)), time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0)
episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(Path(MCP), f"{MAT}/validation_cache.pt", 32, 200, 0)
curves = _return_curves(episodes, config)
ev = [curves[i] for i in ev_idx]
print("n episodes", len(episodes), "n ev", len(ev_idx))

model = build_pg_model(CONFIG_PATH, CKPT, d_embed=32, unfreeze_encoder=False)
stops = []
with torch.no_grad():
    for i in ev_idx:
        feats = _pg_features(z_by_ep[i], 32)
        adv, _ = model.predict_from_features(feats)
        s = predicted_stop_from_advantages(adv.reshape(-1).tolist())
        stops.append(s)
stops = np.array(stops)
ckpt_regret = _regret_at(ev, stops)
print("ckpt_regret mean", ckpt_regret.mean(), "max", ckpt_regret.max(), "median", np.median(ckpt_regret))

order = np.argsort(-ckpt_regret)[:5]
for k in order:
    i = ev_idx[k]
    print("--- episode ev_idx", i, "stop", stops[k], "regret", ckpt_regret[k], "curve_len", len(ev[k]))
    print("curve max", ev[k].max(), "curve at stop", ev[k][min(stops[k], len(ev[k])-1)])
    print("tree_sizes[:5]", episodes[i]["tree_sizes"][:5], "...[-5:]", episodes[i]["tree_sizes"][-5:])
    print("halt_rewards range", min(episodes[i]["halt_rewards"]), max(episodes[i]["halt_rewards"]))
    print("z range for this ep", z_by_ep[i].min(), z_by_ep[i].max())
PYEOF
