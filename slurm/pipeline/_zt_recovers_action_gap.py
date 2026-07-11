import sys
sys.path.insert(0, "/home/hl4291/chess_analysis/src")

import numpy as np
from pathlib import Path

from analysis.evaluate import _load_split_episodes, _load_zt_by_episode
from cts.analysis.zt_probe import _linear_r2, _mlp_r2, _episode_split_mask

PACKED_ROOT = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/pack_history")
CACHE_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/materialize_history/validation_cache.pt"
D_EMBED = 32
MAX_EPISODES = 5000
SEED = 0

print("loading episodes (with action gaps)...", flush=True)
episodes = _load_split_episodes(PACKED_ROOT, "validation", max_episodes=MAX_EPISODES, load_action_gaps=True)
print(f"loaded {len(episodes)} episodes", flush=True)

print("loading z_t...", flush=True)
z_by_ep = _load_zt_by_episode(episodes, CACHE_PATH, D_EMBED)

step_counts = [len(ep["halt_rewards"]) for ep in episodes]
total = int(sum(step_counts))
z = np.concatenate(z_by_ep, axis=0)
ag = np.concatenate([np.asarray(ep["action_gaps"], np.float64) for ep in episodes])
steps = np.concatenate([np.arange(n, dtype=np.float64) for n in step_counts])
traj_keys = [ep["trajectory_key"] for ep in episodes]

is_tr = _episode_split_mask(step_counts, total, seed=SEED, trajectory_keys=traj_keys)
is_te = ~is_tr

print(f"total snapshots={total} train={is_tr.sum()} test={is_te.sum()}", flush=True)
print(f"action_gap stats: mean={ag.mean():.4f} std={ag.std():.4f} min={ag.min():.4f} max={ag.max():.4f}", flush=True)

# z_t alone -> action_gap
lin_r2 = _linear_r2(z[is_tr], ag[is_tr], z[is_te], ag[is_te])
mlp_r2 = _mlp_r2(z[is_tr], ag[is_tr], z[is_te], ag[is_te], seed=SEED)
print(f"[z_t -> action_gap]  linear R^2={lin_r2:.4f}  mlp R^2={mlp_r2:.4f}", flush=True)

# z_t + steps -> action_gap (steps alone carries a lot of trajectory-position signal; check
# whether z_t adds anything beyond what steps alone gives)
zs = np.column_stack([z, steps])
lin_r2_zs = _linear_r2(zs[is_tr], ag[is_tr], zs[is_te], ag[is_te])
mlp_r2_zs = _mlp_r2(zs[is_tr], ag[is_tr], zs[is_te], ag[is_te], seed=SEED)
print(f"[z_t+steps -> action_gap]  linear R^2={lin_r2_zs:.4f}  mlp R^2={mlp_r2_zs:.4f}", flush=True)

# steps alone -> action_gap (baseline to see what z_t adds beyond trajectory position)
st = steps.reshape(-1, 1)
lin_r2_s = _linear_r2(st[is_tr], ag[is_tr], st[is_te], ag[is_te])
mlp_r2_s = _mlp_r2(st[is_tr], ag[is_tr], st[is_te], ag[is_te], seed=SEED)
print(f"[steps -> action_gap]  linear R^2={lin_r2_s:.4f}  mlp R^2={mlp_r2_s:.4f}", flush=True)

# shuffle floor
rng = np.random.default_rng(SEED + 7)
z_shuf = z[rng.permutation(total)]
shuf_r2 = _linear_r2(z_shuf[is_tr], ag[is_tr], z_shuf[is_te], ag[is_te])
print(f"[shuffle floor]  linear R^2={shuf_r2:.4f}", flush=True)

print("DONE", flush=True)
