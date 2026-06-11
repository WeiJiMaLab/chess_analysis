"""Step 3 (redone): matched-snapshot test -- is value actually USED, holding progress fixed?

The perturbation approach extracted linear directions from a nonlinearly-encoded
GNN embedding; "decodable in principle" also isn't "used in fact". This test avoids
both issues. It uses only real snapshots and the controller's real decisions, and
controls progress by conditioning on the TRUE step count (tree size N_t, a known
per-snapshot number) and the time budget T_t -- not a decoded proxy.

Within each (N_t, T_t) cell the controller's input still varies in the decoded
value landscape. We ask whether, inside the cell, the advantage / stop decision
moves with value (best-move value, top-two margin, spread):

  - within-cell correlation of advantage with each value feature, cell-weighted;
  - a matched contrast: stop-rate for high-value vs low-value snapshots in the same
    cell (split at the cell median), cell-weighted.

Conditioning on N_t is conservative -- value and N_t are collinear, so binning on
N_t removes the value variation that rides along with progress, leaving only value
variation independent of progress. So a within-cell effect is a LOWER bound on
value use: a positive result proves the decision uses value beyond progress; a null
means the behavior reduces to progress+budget (value use, if any, is observationally
inseparable from progress). Compare the subtree-weighted controller to the rerun one.

Reads only the step-1 table (.npz); pure NumPy, CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict

from cts.analysis.value_readout_summarize import _per_snapshot_features, _quantile_bins


class ValueConditionalConfig(BaseModel):
    """CLI config for ``cts.analysis.value_conditional``."""

    model_config = ConfigDict(extra="forbid")

    input_npz: str
    output_json: str
    n_nt_bins: int = 6  # quantile bins over the true step count N_t (the progress control)
    n_tt_bins: int = 5  # quantile bins over the time budget T_t
    min_cell: int = 300  # skip (N_t, T_t) cells smaller than this


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(config: ValueConditionalConfig) -> None:
    data = np.load(config.input_npz)
    advantage = data["advantage"].astype(np.float64)
    n_t = data["n_t"].astype(np.float64)
    t_t = data["t_t"].astype(np.float64)
    stop = (advantage <= 0).astype(np.float64)
    num = advantage.shape[0]

    feats = _per_snapshot_features(data["child_value"], data["child_snapshot_id"], num)
    best, second, vmin, count = feats["best"], feats["second"], feats["min"], feats["count"]
    features = {"best": best, "margin": best - second, "spread": best - vmin}

    valid = count >= 2
    nt_idx, nt_edges = _quantile_bins(np.where(valid, n_t, np.nan), config.n_nt_bins)
    tt_idx, tt_edges = _quantile_bins(np.where(valid, t_t, np.nan), config.n_tt_bins)
    cell = nt_idx * 1000 + tt_idx  # unique cell id per (N_t bin, T_t bin)

    print(f"[conditional] snapshots={num} valid(>=2 children)={int(valid.sum())} "
          f"cells={config.n_nt_bins}x{config.n_tt_bins}", flush=True)

    out_features: Dict[str, Dict] = {}
    for fname, fval in features.items():
        cell_corr: List[float] = []
        cell_dstop: List[float] = []
        cell_n: List[int] = []
        cell_vstd: List[float] = []
        for c in np.unique(cell[valid]):
            m = valid & (cell == c)
            if m.sum() < config.min_cell:
                continue
            a = advantage[m]
            f = fval[m]
            s = stop[m]
            if np.std(f) == 0:
                continue
            cell_corr.append(_corr(a, f))
            med = np.median(f)
            hi, lo = f > med, f <= med
            if hi.sum() < 10 or lo.sum() < 10:
                cell_dstop.append(np.nan)
            else:
                cell_dstop.append(float(s[hi].mean() - s[lo].mean()))  # high-value minus low-value stop rate
            cell_vstd.append(float(np.std(f)))
            cell_n.append(int(m.sum()))
        w = np.array(cell_n, dtype=np.float64)
        cc = np.array(cell_corr)
        dd = np.array(cell_dstop)
        wsum = w.sum() if w.size else 0.0
        out_features[fname] = {
            "n_cells": int(w.size),
            "conditioned_corr_advantage": (float(np.nansum(cc * w) / w[np.isfinite(cc)].sum()) if np.isfinite(cc).any() else float("nan")),
            "matched_delta_stop_high_minus_low": (float(np.nansum(dd * w) / w[np.isfinite(dd)].sum()) if np.isfinite(dd).any() else float("nan")),
            "mean_within_cell_value_std": (float((np.array(cell_vstd) * w).sum() / wsum) if wsum else float("nan")),
            "unconditioned_corr_advantage": _corr(advantage[valid], fval[valid]),  # for contrast (step-2-style)
        }
        print(f"[conditional] {fname:>7}: conditioned corr={out_features[fname]['conditioned_corr_advantage']:+.3f}  "
              f"(unconditioned {out_features[fname]['unconditioned_corr_advantage']:+.3f})  "
              f"matched dStop={out_features[fname]['matched_delta_stop_high_minus_low']:+.3f}  "
              f"cells={out_features[fname]['n_cells']}", flush=True)

    payload = {
        "input_npz": config.input_npz,
        "num_snapshots": num,
        "valid_snapshots": int(valid.sum()),
        "n_nt_bins": config.n_nt_bins,
        "n_tt_bins": config.n_tt_bins,
        "nt_bin_edges": [float(e) for e in nt_edges],
        "tt_bin_edges": [float(e) for e in tt_edges],
        "features": out_features,
        "note": "conditioned_corr / matched_delta_stop are computed WITHIN (N_t, T_t) cells then "
        "cell-weighted, so progress (the true step count N_t) and budget are held fixed. "
        "unconditioned_corr is the whole-set value correlation for contrast. A controller that "
        "uses value keeps a within-cell effect; one that stops on step-count+clock collapses to ~0 "
        "once N_t is conditioned on. matched_delta_stop>0 means it stops more on higher-value (more "
        "decided) positions at matched progress/budget.",
    }
    out = Path(config.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[conditional] wrote {out}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(ValueConditionalConfig, main)
