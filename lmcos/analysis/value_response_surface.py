"""The 'how': map the controller's stop function over the value landscape.

We established (constructed-tree probe) that the subtree-weighted controller's stop
decision causally reads the value landscape, keyed on the best-move value. This maps
the actual function by sweeping the landscape on constructed trees at fixed size:

  - best sweep: vary the best-move value (margin/tail fixed) -> advantage(best) per
    budget; gives the shape and the zero-crossing (the stop threshold in best-value)
    and how it shifts with the time budget T_t;
  - margin sweep: vary the top-two gap (best fixed) -> advantage(margin);
  - candidate sweep: hold the best fixed and vary the NUMBER of other moves within
    candidate_eps of the best (one epsilon; extras placed candidate_gap < eps below
    best) -> does the count of near-best candidates move the decision?

x-axis is the DECODED value (what the decoder reads back from z), not the set value,
so the curve is the controller-as-a-function-of-the-decoded-landscape. Constructed,
off-distribution by design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from cts.analysis.synthetic_trees import _build_observation
from cts.analysis.wdl_subspace_ablation import _load_controller, _load_decoder
from cts.core.schema import tree_encoder_feature_schema
from cts.core.tensorizer import collate_tensorized_observations


class ValueResponseSurfaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller_checkpoint: str
    decoder_checkpoint: str
    output_json: str
    device: str = "cuda"
    decoder_hidden_dim: Optional[int] = None
    n_children: int = 8
    grandchildren: int = 5  # FIXED size (node count held constant across the whole surface)
    budgets: List[float] = [4.0, 12.0, 30.0, 60.0]
    best_grid: List[float] = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
    margin_grid: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
    fixed_margin: float = 0.1  # margin used during the best sweep
    fixed_best: float = 0.5  # best used during the margin sweep
    candidate_eps: float = 0.15  # the single epsilon: a child is a candidate if within this of the best
    candidate_gap: float = 0.05  # extra candidates placed this far below best (inside the eps band; < eps)
    candidate_counts: List[int] = [0, 1, 2, 3, 4, 5, 6]  # number of extra candidates to sweep
    repeats: int = 12
    tree_batch: int = 256
    seed: int = 0


def _run(specs, model, decoder, schema, device, tree_batch, grandchildren):
    """specs: list of (child_values, budget, tag). Returns advantage, decoded best/margin, tags."""
    adv_all, best_all, margin_all, tags = [], [], [], []
    rng = np.random.RandomState(0)
    enc_device = model.encoder.device
    with torch.inference_mode():
        for s in range(0, len(specs), tree_batch):
            chunk = specs[s:s + tree_batch]
            obs = [_build_observation(cv, grandchildren, rng, schema, enc_device) for cv, _, _ in chunk]
            tb = collate_tensorized_observations(obs)
            encoded = model.encoder(tb)
            budgets = torch.tensor([b for _, b, _ in chunk], dtype=torch.float32, device=enc_device)
            sizes = torch.tensor([int(o.node_features.shape[0]) for o in obs], dtype=torch.float32, device=enc_device)
            adv, _ = model.predict_from_features(torch.cat([encoded.root_states, sizes[:, None], budgets[:, None]], dim=-1))
            rc = torch.isin(tb.edge_parent, tb.root_index).nonzero(as_tuple=True)[0]
            seg = tb.tree_index[tb.edge_parent[rc]].to(device)
            wdl = torch.softmax(decoder(encoded.node_states[tb.edge_parent[rc]], model.encoder.slot_embeddings(tb.edge_slot[rc])), dim=-1)
            cval = (wdl[:, 0] - wdl[:, 2]).detach().float().cpu().numpy()
            segnp = seg.detach().cpu().numpy()
            nloc = len(chunk)
            best = np.full(nloc, np.nan); second = np.full(nloc, np.nan)
            order = np.argsort(segnp, kind="stable"); sid = segnp[order]; v = cval[order].astype(np.float64)
            uniq, st = np.unique(sid, return_index=True)
            gmax = np.maximum.reduceat(v, st)
            rowmax = np.repeat(gmax, np.diff(np.append(st, sid.size)))
            gsec = np.maximum.reduceat(np.where(v >= rowmax, -np.inf, v), st)
            best[uniq] = gmax; second[uniq] = np.where(np.isfinite(gsec), gsec, gmax)
            adv_all.append(adv.detach().float().cpu().numpy())
            best_all.append(best); margin_all.append(best - second)
            tags.extend([t for _, _, t in chunk])
    return np.concatenate(adv_all), np.concatenate(best_all), np.concatenate(margin_all), tags


def _curve(x: np.ndarray, adv: np.ndarray, n_bins: int = 10):
    """Mean advantage + stop-rate across quantile bins of x, plus the zero-crossing of mean adv."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    centers, mean_adv, stop = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        centers.append(float(0.5 * (edges[b] + edges[b + 1])))
        mean_adv.append(float(adv[m].mean())); stop.append(float((adv[m] <= 0).mean()))
    zc = float("nan")
    for i in range(len(mean_adv) - 1):
        if mean_adv[i] > 0 >= mean_adv[i + 1]:
            f = mean_adv[i] / (mean_adv[i] - mean_adv[i + 1])
            zc = float(centers[i] + f * (centers[i + 1] - centers[i])); break
    return {"x": centers, "mean_advantage": mean_adv, "stop_rate": stop, "zero_crossing": zc}


def main(config: ValueResponseSurfaceConfig) -> None:
    device = torch.device(config.device)
    model, _ = _load_controller(config.controller_checkpoint, device)
    d_embed = int(model.encoder.d_embed)
    decoder = _load_decoder(config.decoder_checkpoint, d_embed, device, config.decoder_hidden_dim)
    schema = tree_encoder_feature_schema()
    rng = np.random.RandomState(config.seed)
    nc = config.n_children

    def landscape(best, margin, n_second=1):
        vals = [best] + [best - margin] * n_second + [best - 1.0] * (nc - 1 - n_second)
        return [v + 0.02 * rng.randn() for v in vals[:nc]]

    specs = []
    for bud in config.budgets:
        for best in config.best_grid:
            for _ in range(config.repeats):
                specs.append((landscape(best, config.fixed_margin), bud, ("best", bud)))
        for mg in config.margin_grid:
            for _ in range(config.repeats):
                specs.append((landscape(config.fixed_best, mg), bud, ("margin", bud)))
        for ncand in config.candidate_counts:  # candidate sweep: number of extra moves within eps of best
            for _ in range(config.repeats * 2):
                specs.append((landscape(config.fixed_best, config.candidate_gap, n_second=ncand), bud, (f"cand{ncand}", bud)))
    print(f"[surface] {len(specs)} trees (grandchildren={config.grandchildren}, n_children={nc})", flush=True)

    adv, best, margin, tags = _run(specs, model, decoder, schema, device, config.tree_batch, config.grandchildren)
    tag_arr = np.array([f"{a}|{b:g}" for a, b in tags])

    out: Dict[str, Dict] = {"best_sweep": {}, "margin_sweep": {}, "candidate_sweep": {}}
    for bud in config.budgets:
        bm = tag_arr == f"best|{bud:g}"
        out["best_sweep"][f"T_t={bud:g}"] = _curve(best[bm], adv[bm])
        mm = tag_arr == f"margin|{bud:g}"
        out["margin_sweep"][f"T_t={bud:g}"] = _curve(margin[mm], adv[mm])
        cand = {}
        for ncand in config.candidate_counts:
            cm = tag_arr == f"cand{ncand}|{bud:g}"
            if cm.any():
                cand[str(ncand + 1)] = {"n_candidates": ncand + 1, "mean_advantage": float(adv[cm].mean()),
                                        "stop_rate": float((adv[cm] <= 0).mean())}
        out["candidate_sweep"][f"T_t={bud:g}"] = cand
        bc = out["best_sweep"][f"T_t={bud:g}"]; mc = out["margin_sweep"][f"T_t={bud:g}"]
        cvals = [v["mean_advantage"] for v in cand.values()]
        print(f"[surface] T_t={bud:g}: best->adv {bc['mean_advantage'][0]:+.2f}..{bc['mean_advantage'][-1]:+.2f} "
              f"zc(best)={bc['zero_crossing']:.2f}  margin->adv {mc['mean_advantage'][0]:+.2f}..{mc['mean_advantage'][-1]:+.2f}  "
              f"cand(1..{config.candidate_counts[-1] + 1})->adv {cvals[0]:+.2f}..{cvals[-1]:+.2f}", flush=True)

    payload = {"controller_checkpoint": config.controller_checkpoint, "n_trees": len(specs),
               "grandchildren_fixed": config.grandchildren, "candidate_eps": config.candidate_eps,
               "candidate_gap": config.candidate_gap, "surface": out,
               "note": "single epsilon = candidate_eps: a child is a candidate if within that of the best. "
               "best_sweep / margin_sweep: mean advantage vs decoded best / top-two margin per budget (with the "
               "zero-crossing stop threshold). candidate_sweep: advantage vs the NUMBER of candidates within eps "
               "of the best (extras placed candidate_gap < eps below best), best fixed -- isolates the candidate "
               "count from best-value and the top-two gap."}
    Path(config.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(config.output_json).write_text(json.dumps(payload, indent=2))
    print(f"[surface] wrote {config.output_json}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(ValueResponseSurfaceConfig, main)
