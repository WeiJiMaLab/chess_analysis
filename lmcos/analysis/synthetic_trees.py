"""Constructed-tree probe: does the controller's stop decision depend on the value
landscape when value and search size are set independently?

The natural data has value and search progress collinear, which muddied every
observational and perturbation test. Here we sidestep it: we build artificial
search trees with the value landscape and the node count chosen independently, run
them through the real encoder -> controller (the only pipeline that matters), and
read the controller's advantage. The decoder reads the same root embedding and
confirms the value landscape we built actually lands.

Node features are (value, wdl_win, wdl_draw, wdl_loss, wdl_var) -- the value IS a
node feature -- so a tree's value landscape is just the root children's value
features, and its size is just how many nodes we attach. We sweep:
  - best-move value and top-two margin (the landscape), and
  - tree size (grandchildren count) and time budget T_t,
independently, with random fillers, so decoded value is decorrelated from size by
construction. Then "does advantage track value at fixed size" is a direct read --
no conditioning needed (though the table is emitted in the step-1 format so the
matched-cell test can be run on it too).

Off-distribution is fine: we are mapping the controller's function, not estimating
a population quantity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from cts.analysis.wdl_subspace_ablation import _load_controller, _load_decoder
from cts.core.schema import tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeObservation, collate_tensorized_observations


class SyntheticTreesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller_checkpoint: str
    decoder_checkpoint: str
    output_npz: str
    output_json: str
    device: str = "cuda"
    decoder_hidden_dim: Optional[int] = None
    n_children: int = 8  # root children per tree
    best_values: List[float] = [0.0, 0.3, 0.6, 0.9]  # best root-child value (landscape)
    margins: List[float] = [0.0, 0.1, 0.3, 0.6]  # gap from best to second
    grandchildren: List[int] = [0, 2, 5, 10, 20]  # per-child subtree size -> total node count (the size knob)
    budgets: List[float] = [4.0, 12.0, 30.0, 60.0]  # T_t values to feed the controller
    repeats: int = 8  # random-filler repeats per (best, margin, grandchildren, budget) cell
    tree_batch: int = 256  # trees per encoder batch
    seed: int = 0


def _wdl_features(value: float, rng: np.random.RandomState) -> Dict[str, float]:
    """A node-feature dict realizing a scalar value (win - loss) as a W/D/L triple."""
    value = float(np.clip(value, -0.98, 0.98))
    draw = float(np.clip(0.2 + 0.05 * rng.randn(), 0.02, 0.6))
    win = (1.0 - draw) / 2.0 + value / 2.0
    loss = (1.0 - draw) / 2.0 - value / 2.0
    return {"value": value, "wdl_win": win, "wdl_draw": draw, "wdl_loss": loss,
            "wdl_var": float(np.clip(0.05 + 0.02 * rng.rand(), 0.0, 0.5))}


def _build_observation(child_values, n_grand, rng, schema, device):
    """Root + ``len(child_values)`` children + ``n_grand`` leaf grandchildren each."""
    rows, parent_index, depth = [], [], []
    edge_parent, edge_child, edge_slot = [], [], []

    def add(node_id, parent, dep, value):
        rows.append(schema.vectorize_tuple(_wdl_features(value, rng)))
        parent_index.append(parent)
        depth.append(dep)

    add(0, -1, 0, float(np.mean(child_values)))  # root
    next_id = 1
    child_ids = []
    for slot, cv in enumerate(child_values):
        cid = next_id; next_id += 1
        child_ids.append((cid, cv))
        add(cid, 0, 1, cv)
        edge_parent.append(0); edge_child.append(cid); edge_slot.append(slot)
    for cid, cv in child_ids:  # grandchildren (the size filler)
        for g in range(n_grand):
            gid = next_id; next_id += 1
            add(gid, cid, 2, cv + 0.1 * rng.randn())
            edge_parent.append(cid); edge_child.append(gid); edge_slot.append(g)

    return TensorizedTreeObservation(
        node_features=torch.tensor(rows, dtype=schema.dtype, device=device),
        parent_index=torch.tensor(parent_index, dtype=torch.long, device=device),
        edge_parent=torch.tensor(edge_parent, dtype=torch.long, device=device),
        edge_child=torch.tensor(edge_child, dtype=torch.long, device=device),
        edge_slot=torch.tensor(edge_slot, dtype=torch.long, device=device),
        depth=torch.tensor(depth, dtype=torch.long, device=device),
        root_index=0,
        feature_names=schema.feature_names,
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(config: SyntheticTreesConfig) -> None:
    device = torch.device(config.device)
    rng = np.random.RandomState(config.seed)
    model, _ = _load_controller(config.controller_checkpoint, device)
    d_embed = int(model.encoder.d_embed)
    decoder = _load_decoder(config.decoder_checkpoint, d_embed, device, config.decoder_hidden_dim)
    schema = tree_encoder_feature_schema()
    enc_device = model.encoder.device

    # Build the full design of synthetic trees (value landscape x size x budget x repeats).
    specs = []  # (child_values, n_grand, budget)
    for bv in config.best_values:
        for mg in config.margins:
            # best, second = best - margin, the rest descend below second
            rest = list(np.linspace(bv - mg - 0.05, bv - 1.0, max(config.n_children - 2, 1)))
            base = [bv, bv - mg] + rest
            base = base[: config.n_children]
            for ng in config.grandchildren:
                for bud in config.budgets:
                    for _ in range(config.repeats):
                        cv = [c + 0.03 * rng.randn() for c in base]
                        specs.append((cv, ng, bud))
    print(f"[synth] {len(specs)} trees  (n_children={config.n_children})", flush=True)

    adv_all, nt_all, tt_all = [], [], []
    cv_all, snap_all = [], []  # ragged decoded root-child values
    snap_offset = 0
    with torch.inference_mode():
        for start in range(0, len(specs), config.tree_batch):
            chunk = specs[start:start + config.tree_batch]
            obs = [_build_observation(cv, ng, rng, schema, enc_device) for cv, ng, _ in chunk]
            tb = collate_tensorized_observations(obs)
            encoded = model.encoder(tb)
            budgets = torch.tensor([b for _, _, b in chunk], dtype=torch.float32, device=enc_device)
            sizes = torch.tensor([int(o.node_features.shape[0]) for o in obs], dtype=torch.float32, device=enc_device)
            adv, _ = model.predict_from_features(torch.cat([encoded.root_states, sizes[:, None], budgets[:, None]], dim=-1))

            rc = torch.isin(tb.edge_parent, tb.root_index).nonzero(as_tuple=True)[0]
            seg = tb.tree_index[tb.edge_parent[rc]].to(device)
            slot_states = model.encoder.slot_embeddings(tb.edge_slot[rc])
            wdl = torch.softmax(decoder(encoded.node_states[tb.edge_parent[rc]], slot_states), dim=-1)
            child_value = (wdl[:, 0] - wdl[:, 2])

            b = len(chunk)
            adv_all.append(adv.detach().float().cpu().numpy())
            nt_all.append(sizes.detach().cpu().numpy())
            tt_all.append(budgets.detach().cpu().numpy())
            cv_all.append(child_value.detach().float().cpu().numpy())
            snap_all.append(snap_offset + seg.detach().cpu().numpy())
            snap_offset += b

    advantage = np.concatenate(adv_all)
    n_t = np.concatenate(nt_all)
    t_t = np.concatenate(tt_all)
    child_value = np.concatenate(cv_all)
    child_snapshot_id = np.concatenate(snap_all)
    num = advantage.shape[0]

    # Per-tree decoded best / margin (segment reduce on CPU).
    best = np.full(num, np.nan)
    second = np.full(num, np.nan)
    order = np.argsort(child_snapshot_id, kind="stable")
    sid, val = child_snapshot_id[order], child_value[order].astype(np.float64)
    uniq, st = np.unique(sid, return_index=True)
    gmax = np.maximum.reduceat(val, st)
    rowmax = np.repeat(gmax, np.diff(np.append(st, sid.size)))
    val_wo = np.where(val >= rowmax, -np.inf, val)
    gsec = np.maximum.reduceat(val_wo, st)
    best[uniq] = gmax
    second[uniq] = np.where(np.isfinite(gsec), gsec, gmax)
    margin = best - second

    out_npz = Path(config.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, advantage=advantage.astype("float32"), n_t=n_t.astype("float32"),
                        t_t=t_t.astype("float32"), child_value=child_value.astype("float32"),
                        child_snapshot_id=child_snapshot_id.astype("int64"))

    # Decorrelation check + the direct read, at fixed budget (value vs size are independent by design).
    payload = {
        "controller_checkpoint": config.controller_checkpoint,
        "num_trees": num,
        "decorrelation_check": {
            "corr_decoded_best_vs_size": _corr(best, n_t),
            "corr_decoded_margin_vs_size": _corr(margin, n_t),
            "note": "should be ~0 -- value and size were set independently",
        },
        "advantage_vs_value": {},
        "note": "advantage_vs_value: within each T_t, the correlation of advantage with the decoded best "
        "value and margin across constructed trees. Value and size are decorrelated by construction, so a "
        "nonzero correlation here is the controller using value, not a size confound.",
    }
    for bud in sorted(set(t_t.tolist())):
        m = t_t == bud
        payload["advantage_vs_value"][f"T_t={bud:g}"] = {
            "n": int(m.sum()),
            "mean_advantage": float(advantage[m].mean()),
            "stop_rate": float((advantage[m] <= 0).mean()),
            "corr_advantage_best": _corr(advantage[m], best[m]),
            "corr_advantage_margin": _corr(advantage[m], margin[m]),
            "corr_advantage_size": _corr(advantage[m], n_t[m]),
        }
    out_json = Path(config.output_json)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[synth] decorrelation corr(best,size)={payload['decorrelation_check']['corr_decoded_best_vs_size']:.3f} "
          f"(want ~0)", flush=True)
    for k, v in payload["advantage_vs_value"].items():
        print(f"[synth] {k}: corr(adv,best)={v['corr_advantage_best']:+.3f} corr(adv,margin)={v['corr_advantage_margin']:+.3f} "
              f"corr(adv,size)={v['corr_advantage_size']:+.3f}  stop={v['stop_rate']:.2f}", flush=True)
    print(f"[synth] wrote {out_npz} and {out_json}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(SyntheticTreesConfig, main)
