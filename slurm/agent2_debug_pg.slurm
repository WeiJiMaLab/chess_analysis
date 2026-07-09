#!/bin/bash
#SBATCH --job-name=agent2-debug-pg
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
import torch
import numpy as np
from pathlib import Path

ckpt = torch.load("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt", map_location="cpu", weights_only=False)
print("metadata:", ckpt["metadata"])
sd = ckpt["model_state_dict"]
for k, v in sd.items():
    print(k, tuple(v.shape))

from analysis.evaluate_our_trees_continued import build_pg_model, _pg_features
from analysis.evaluate import _load_assessment_data, _load_zt_by_episode

model = build_pg_model("/home/hl4291/chess_analysis/config_minply15_maxply75.yaml",
                       "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt",
                       d_embed=32, unfreeze_encoder=False)
print("model built ok")
print("controller_inputs", model.controller_inputs)

import os
MCP = os.environ["MCP"]
MAT = os.environ["MAT"]
episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(Path(MCP), f"{MAT}/validation_cache.pt", 32, 50, 0)
print("n episodes", len(episodes), "n fit", len(fit_idx), "n ev", len(ev_idx))
i = ev_idx[0]
feats = _pg_features(z_by_ep[i], 32)
print("feats shape", feats.shape, "z range", z_by_ep[i].min(), z_by_ep[i].max())
with torch.no_grad():
    adv, sign = model.predict_from_features(feats)
print("adv", adv.flatten().tolist()[:20])
print("halt_rewards", episodes[i]["halt_rewards"][:20])
print("tree_sizes", episodes[i]["tree_sizes"][:20])
PYEOF
