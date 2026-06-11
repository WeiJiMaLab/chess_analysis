"""Follow real searches step by step and see what triggers the halt.

The sweeps map which landscape features the stop function is sensitive to; this asks
which feature actually MOVES during a real growing search to push the advantage
across zero -- on-distribution, on real episodes.

Per episode, per step (snapshot), it records the controller advantage, the decoded
landscape (best value, top-two margin, and candidate_count = how many children sit
within ``candidate_eps`` of the best), the tree size N_t, and T_t; plus the oracle
stop step. The controller's halt step is the first step with advantage <= 0. Then it
aligns every episode at its own halt step and reports:

  - the mean trajectory of each feature in the steps around the halt;
  - the per-feature change from the step before the halt to the halt step (the
    attribution: which feature moved as the score crossed zero);
  - the controller halt step vs the oracle halt step.

Attribution caveat: the halt is deterministic given the snapshot, so "trigger" means
whichever feature's change at that step carried the score across zero -- a
decomposition of a real crossing, not a separate causal manipulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from cts.analysis.wdl_subspace_ablation import _load_controller, _load_decoder
from cts.train.controller_train import (
    ControllerEpisodeDataset,
    _build_packed_loader,
    _extract_all_episode_metadata,
)


class HaltTrajectoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller_checkpoint: str
    decoder_checkpoint: str
    manifest_path: str
    output_json: str
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 0
    candidate_eps: float = 0.15  # a child counts as a candidate if value >= best - eps
    align_window: int = 6  # steps before the halt to report in the aligned trajectory
    min_halt_for_trigger: int = 5  # only episodes that searched >= this many steps feed the trigger analysis
    decoder_hidden_dim: Optional[int] = None


def main(config: HaltTrajectoryConfig) -> None:
    device = torch.device(config.device)
    model, _ = _load_controller(config.controller_checkpoint, device)
    d_embed = int(model.encoder.d_embed)
    decoder = _load_decoder(config.decoder_checkpoint, d_embed, device, config.decoder_hidden_dim)
    enc_device = model.encoder.device
    eps = config.candidate_eps

    feats = {k: [] for k in ("advantage", "best", "margin", "candidate_count", "n_t", "t_t")}
    loader = _build_packed_loader(config.manifest_path, batch_size=config.batch_size, shuffle=False, seed=0, num_workers=config.num_workers)
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            tb = batch.tree_batch
            encoded = model.encoder(tb)
            nt = batch.tree_sizes.to(enc_device, dtype=torch.float32)
            tt = batch.time_budgets.to(enc_device, dtype=torch.float32)
            adv, _ = model.predict_from_features(torch.cat([encoded.root_states, nt[:, None], tt[:, None]], dim=-1))
            B = int(adv.shape[0])

            rc = torch.isin(tb.edge_parent, tb.root_index).nonzero(as_tuple=True)[0]
            seg = tb.tree_index[tb.edge_parent[rc]].to(device)
            wdl = torch.softmax(decoder(encoded.node_states[tb.edge_parent[rc]], model.encoder.slot_embeddings(tb.edge_slot[rc])), dim=-1)
            cval = wdl[:, 0] - wdl[:, 2]
            neg = torch.full((B,), float("-inf"), device=device)
            best = neg.clone().scatter_reduce(0, seg, cval, reduce="amax", include_self=True)
            seg_best = best[seg]
            wo = torch.where(cval >= seg_best, torch.full_like(cval, float("-inf")), cval)
            second = neg.clone().scatter_reduce(0, seg, wo, reduce="amax", include_self=True)
            margin = torch.where(torch.isfinite(best - second), best - second, torch.zeros(B, device=device))
            cand = torch.zeros(B, device=device).index_add_(0, seg, (cval >= seg_best - eps).float())

            feats["advantage"].append(adv.detach().float().cpu().numpy())
            feats["best"].append(torch.where(torch.isfinite(best), best, torch.zeros(B, device=device)).detach().float().cpu().numpy())
            feats["margin"].append(margin.detach().float().cpu().numpy())
            feats["candidate_count"].append(cand.detach().float().cpu().numpy())
            feats["n_t"].append(nt.detach().cpu().numpy())
            feats["t_t"].append(tt.detach().cpu().numpy())

    arr = {k: np.concatenate(v) for k, v in feats.items()}
    dataset = ControllerEpisodeDataset(config.manifest_path)
    meta = _extract_all_episode_metadata(dataset)
    total = sum(m.num_steps for m in meta)
    if arr["advantage"].shape[0] != total:
        raise ValueError(f"snapshots {arr['advantage'].shape[0]} != episode steps {total}")

    fnames = ["advantage", "best", "margin", "candidate_count", "n_t"]
    W = config.align_window
    # aligned trajectories: rel step in [-W, +1]; at-halt delta = feature(halt) - feature(halt-1)
    aligned = {f: {r: [] for r in range(-W, 2)} for f in fnames}
    at_halt_delta = {f: [] for f in fnames}
    ctrl_halts, oracle_halts, episode_lens = [], [], []
    n_trigger = 0
    off = 0
    for m in meta:
        n = m.num_steps
        sl = slice(off, off + n)
        adv_ep = arr["advantage"][sl]
        below = np.where(adv_ep <= 0)[0]
        halt = int(below[0]) if below.size else n - 1
        ctrl_halts.append(halt)
        oracle_halts.append(int(m.oracle_stop_step))
        episode_lens.append(n)
        if halt >= config.min_halt_for_trigger:  # only episodes that actually searched before halting
            n_trigger += 1
            for f in fnames:
                fe = arr[f][sl]
                for r in range(-W, 2):
                    idx = halt + r
                    if 0 <= idx < n:
                        aligned[f][r].append(float(fe[idx]))
                at_halt_delta[f].append(float(fe[halt] - fe[halt - 1]))
        off += n

    ctrl_halts = np.array(ctrl_halts); oracle_halts = np.array(oracle_halts)
    payload = {
        "controller_checkpoint": config.controller_checkpoint,
        "num_episodes": len(meta),
        "num_snapshots": int(total),
        "candidate_eps": eps,
        "min_halt_for_trigger": config.min_halt_for_trigger,
        "num_trigger_episodes": n_trigger,
        "halt_steps": {
            "controller_mean": float(ctrl_halts.mean()), "controller_median": float(np.median(ctrl_halts)),
            "oracle_mean": float(oracle_halts.mean()), "oracle_median": float(np.median(oracle_halts)),
            "corr_controller_oracle": float(np.corrcoef(ctrl_halts, oracle_halts)[0, 1]),
            "mean_episode_length": float(np.mean(episode_lens)),
        },
        "aligned_trajectory": {f: {str(r): (float(np.mean(v)) if v else None) for r, v in aligned[f].items()} for f in fnames},
        "at_halt_delta": {f: {"mean": (float(np.mean(at_halt_delta[f])) if at_halt_delta[f] else None),
                              "std": (float(np.std(at_halt_delta[f])) if at_halt_delta[f] else None),
                              "n": len(at_halt_delta[f])} for f in fnames},
        "note": "aligned_trajectory: mean of each feature at step (halt + r). at_halt_delta: mean change from the "
        "step before the halt to the halt step -- the feature that moves most (relative to its own scale) is the "
        "trigger. candidate_count = children within candidate_eps of the best. Attribution, not a causal test.",
    }
    print(f"[halt-traj] episodes={len(meta)} ctrl_halt_mean={ctrl_halts.mean():.2f} oracle_halt_mean={oracle_halts.mean():.2f} "
          f"corr={payload['halt_steps']['corr_controller_oracle']:.3f}  trigger_episodes(halt>={config.min_halt_for_trigger})={n_trigger}", flush=True)
    for f in fnames:
        d = payload["at_halt_delta"][f]
        if d["mean"] is not None:
            print(f"[halt-traj] at-halt delta {f:>16}: {d['mean']:+.4f} +/- {d['std']:.4f}", flush=True)
    out = Path(config.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[halt-traj] wrote {out}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(HaltTrajectoryConfig, main)
