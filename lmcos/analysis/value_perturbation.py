"""Step 3 of the value-readout characterization: the causal cut.

Step 2 (descriptive) showed the advantage correlates with the decoded value
(best-move value / spread more than top-two margin), but the rerun "budget
heuristic" control reproduced most of that correlation -- so it's largely the
progress<->value collinearity, not proven value-reading. This step removes the
confound by intervention on the real controller.

Method (no surrogate, no retraining):
  1. Collect, over a sample of snapshots: z_root, the decoded best-move value and
     top-two margin (forward decode), N_t, T_t, and the controller's advantage.
  2. Fit linear directions in z_root: a progress direction (ridge N_t ~ z_root)
     and value directions (ridge best_value ~ z_root, margin ~ z_root). Report the
     R^2s -- R^2(N_t) is the "z_root is an N_t-proxy" null check; R^2(value) is how
     linearly the value is carried.
  3. Orthogonalize each value direction against the progress direction, so moving
     along it changes the decoded value while holding the progress-proxy fixed.
  4. Perturb: z' = z_root + delta * sigma * direction; re-run the frozen head on
     [z', N_t, T_t]; report the change in advantage and stop-rate vs delta, within
     T_t bins, for the value directions and a matched value-irrelevant control.

If the advantage shifts along the progress-orthogonalized value direction but not
the control, the decision causally reads value beyond progress. Run on the
subtree-weighted controller and on the rerun controller (the collinearity
baseline); the subtree-weighted response above the rerun response is the signal.

Linear directions are a first-order cut (the value direction is the linear z_root
axis most predictive of decoded value); delta is reported in units of the spread
of real z_root along that axis, so the perturbation stays modest / on-distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from cts.analysis.wdl_subspace_ablation import _load_controller, _load_decoder
from cts.train.controller_train import _build_packed_loader


class ValuePerturbationConfig(BaseModel):
    """CLI config for ``cts.analysis.value_perturbation``."""

    model_config = ConfigDict(extra="forbid")

    controller_checkpoint: str
    decoder_checkpoint: str
    manifest_path: str
    output_json: str
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 0
    max_snapshots: int = 200000  # cap the sample (gradients/perturb are cheap but bound memory)
    ridge_lambda: float = 10.0
    n_tt_bins: int = 5
    delta_sigmas: List[float] = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    seed: int = 0
    decoder_hidden_dim: Optional[int] = None


def _segment_best_and_margin(
    child_value: torch.Tensor, seg: torch.Tensor, num_segments: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-segment best and (best - second) of ``child_value`` keyed by ``seg`` (forward only)."""
    seg = seg.to(child_value.device)  # packed tree_batch indices live on CPU; scatter needs same device
    neg = torch.full((num_segments,), float("-inf"), device=child_value.device)
    best = neg.clone().scatter_reduce(0, seg, child_value, reduce="amax", include_self=True)
    seg_best = best[seg]
    without_best = torch.where(child_value >= seg_best, torch.full_like(child_value, float("-inf")), child_value)
    second = neg.clone().scatter_reduce(0, seg, without_best, reduce="amax", include_self=True)
    margin = best - second
    margin = torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin))  # single/all-equal -> 0
    best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
    return best, margin


def _ridge_unit_direction(x: torch.Tensor, y: torch.Tensor, lam: float) -> Tuple[torch.Tensor, float]:
    """Centered ridge of y on x; return (unit-norm coefficient direction, R^2)."""
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean()
    d = xc.shape[1]
    gram = xc.transpose(0, 1) @ xc + lam * torch.eye(d, device=x.device, dtype=x.dtype)
    w = torch.linalg.solve(gram, xc.transpose(0, 1) @ yc)
    pred = xc @ w
    ss_res = float(((yc - pred) ** 2).sum().item())
    ss_tot = float((yc ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    norm = w.norm()
    return (w / norm if norm > 0 else w), r2


def _orthonormalize(v: torch.Tensor, against: torch.Tensor) -> torch.Tensor:
    """Component of unit ``v`` orthogonal to unit ``against``, renormalized."""
    v = v - (v @ against) * against
    n = v.norm()
    return v / n if n > 0 else v


def _tt_bins(t_t: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(t_t, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.digitize(t_t, edges[1:-1]), 0, len(edges) - 2)
    return idx, edges


def main(config: ValuePerturbationConfig) -> None:
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model, _ = _load_controller(config.controller_checkpoint, device)
    d_embed = int(model.encoder.d_embed)
    decoder = _load_decoder(config.decoder_checkpoint, d_embed, device, config.decoder_hidden_dim)
    encoder_device = model.encoder.device
    loader = _build_packed_loader(config.manifest_path, batch_size=config.batch_size, shuffle=False, seed=0, num_workers=config.num_workers)

    z_chunks: List[torch.Tensor] = []
    best_chunks, margin_chunks, nt_chunks, tt_chunks, adv_chunks = [], [], [], [], []
    total = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            tb = batch.tree_batch
            encoded = model.encoder(tb)
            z = encoded.root_states  # [B, d_embed]
            nt = batch.tree_sizes.to(encoder_device, dtype=torch.float32)
            tt = batch.time_budgets.to(encoder_device, dtype=torch.float32)
            adv, _ = model.predict_from_features(torch.cat([z, nt[:, None], tt[:, None]], dim=-1))

            rc = torch.isin(tb.edge_parent, tb.root_index).nonzero(as_tuple=True)[0]
            seg = tb.tree_index[tb.edge_parent[rc]]  # [E_rc] in 0..B-1
            slot_states = model.encoder.slot_embeddings(tb.edge_slot[rc])
            wdl = torch.softmax(decoder(encoded.node_states[tb.edge_parent[rc]], slot_states), dim=-1)
            child_value = wdl[:, 0] - wdl[:, 2]  # resolved root-player value (step 1: corr +0.92 with halt_reward)
            best, margin = _segment_best_and_margin(child_value, seg, int(z.shape[0]))

            z_chunks.append(z.detach().float().cpu())
            best_chunks.append(best.detach().float().cpu())
            margin_chunks.append(margin.detach().float().cpu())
            nt_chunks.append(nt.detach().float().cpu())
            tt_chunks.append(tt.detach().float().cpu())
            adv_chunks.append(adv.detach().float().cpu())
            total += int(z.shape[0])
            if total >= config.max_snapshots:
                break

    z = torch.cat(z_chunks)[: config.max_snapshots].to(device)
    best = torch.cat(best_chunks)[: config.max_snapshots].to(device)
    margin = torch.cat(margin_chunks)[: config.max_snapshots].to(device)
    n_t = torch.cat(nt_chunks)[: config.max_snapshots].to(device)
    t_t_np = torch.cat(tt_chunks)[: config.max_snapshots].numpy()
    adv0 = torch.cat(adv_chunks)[: config.max_snapshots].to(device)
    n = int(z.shape[0])
    print(f"[perturb] sample={n} d_embed={d_embed}", flush=True)

    p_dir, r2_progress = _ridge_unit_direction(z, n_t, config.ridge_lambda)
    v_best, r2_best = _ridge_unit_direction(z, best, config.ridge_lambda)
    v_margin, r2_margin = _ridge_unit_direction(z, margin, config.ridge_lambda)
    v_best_pure = _orthonormalize(v_best, p_dir)
    v_margin_pure = _orthonormalize(v_margin, p_dir)
    rand = torch.randn(d_embed, device=device)
    rand = rand / rand.norm()
    control = _orthonormalize(_orthonormalize(rand, p_dir), v_best)  # value- and progress-orthogonal

    directions = {"value_best_pure": v_best_pure, "margin_pure": v_margin_pure, "control": control}
    print(
        f"[perturb] R2(N_t|z)={r2_progress:.3f}  R2(best|z)={r2_best:.3f}  R2(margin|z)={r2_margin:.3f}  "
        f"(R2(N_t|z) is the z_root-as-progress-proxy null)",
        flush=True,
    )

    tt_idx, tt_edges = _tt_bins(t_t_np, config.n_tt_bins)
    base_stop = (adv0 <= 0).float().cpu().numpy()
    adv0_np = adv0.cpu().numpy()
    t_t_col = torch.cat(tt_chunks)[: config.max_snapshots].to(device)  # real budgets for the head

    results: Dict[str, Dict] = {}
    with torch.inference_mode():
        for name, d in directions.items():
            sigma = float((z @ d).std().item())  # spread of real z_root along this axis
            per_delta = []
            for ds in config.delta_sigmas:
                zp = z + (ds * sigma) * d
                advp = model.predict_from_features(torch.cat([zp, n_t[:, None], t_t_col[:, None]], dim=-1))[0]
                d_adv = (advp - adv0).cpu().numpy()
                stop_p = (advp <= 0).float().cpu().numpy()
                by_bin = []
                for b in range(len(tt_edges) - 1):
                    m = tt_idx == b
                    if m.sum() < 50:
                        continue
                    by_bin.append({
                        "t_t_range": [float(tt_edges[b]), float(tt_edges[b + 1])],
                        "mean_delta_advantage": float(d_adv[m].mean()),
                        "delta_stop_rate": float(stop_p[m].mean() - base_stop[m].mean()),
                    })
                per_delta.append({"delta_sigma": ds, "by_budget_bin": by_bin,
                                  "mean_delta_advantage_all": float(d_adv.mean())})
            results[name] = {"axis_sigma": sigma, "sweep": per_delta}

    payload = {
        "controller_checkpoint": config.controller_checkpoint,
        "decoder_checkpoint": config.decoder_checkpoint,
        "sample_snapshots": n,
        "r2_progress_Nt_given_z": r2_progress,
        "r2_best_value_given_z": r2_best,
        "r2_margin_given_z": r2_margin,
        "t_t_bin_edges": [float(e) for e in tt_edges],
        "baseline_mean_advantage": float(adv0_np.mean()),
        "baseline_stop_rate": float(base_stop.mean()),
        "perturbation": results,
        "note": "delta_sigma is in units of the std of real z_root along each axis. value_best_pure "
        "and margin_pure are the value directions orthogonalized against the progress direction "
        "(so they change decoded value while holding the progress-proxy fixed); control is value- "
        "and progress-orthogonal. A value-specific reader moves under value_best_pure/margin_pure "
        "but not control; compare against the rerun controller as the collinearity baseline.",
    }
    out = Path(config.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[perturb] wrote {out}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(ValuePerturbationConfig, main)
