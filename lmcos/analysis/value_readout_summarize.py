"""Step 2 of the value-readout characterization: look at the readout.

Reads the per-snapshot table from ``cts.analysis.value_readout_table`` and, without
fitting anything or assuming a functional form, describes how the controller's
advantage depends on the decoded value distribution. It reduces the ragged
root-child values to per-snapshot features (best, second, top-two margin, spread,
softmax entropy, child count), then -- holding the time budget T_t fixed in bins --
reports:

  - mean advantage and stop-rate (advantage <= 0) across margin bins and across
    spread bins, so a threshold (advantage crossing 0 as the best move pulls ahead)
    is just visible in the numbers;
  - the margin value where mean advantage crosses 0 within each budget bin (the
    apparent threshold, if there is one);
  - within each budget bin, the correlation of the advantage with each feature, so
    you can see which one it tracks most.

Output is a compact JSON of these grids and correlations -- small enough to read
directly. Pure NumPy, CPU-only; runs on the login node. No plotting (plot the JSON
locally if you want a picture).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict


class ValueReadoutSummarizeConfig(BaseModel):
    """CLI config for ``cts.analysis.value_readout_summarize``."""

    model_config = ConfigDict(extra="forbid")

    input_npz: str  # output of cts.analysis.value_readout_table
    output_json: str
    n_tt_bins: int = 5  # quantile bins over T_t
    n_value_bins: int = 10  # quantile bins over each value feature


def _per_snapshot_features(
    child_value: np.ndarray, child_snapshot_id: np.ndarray, num_snapshots: int
) -> Dict[str, np.ndarray]:
    """Reduce ragged per-child values to per-snapshot (best, second, min, count, entropy)."""
    best = np.full(num_snapshots, np.nan)
    second = np.full(num_snapshots, np.nan)
    vmin = np.full(num_snapshots, np.nan)
    entropy = np.full(num_snapshots, np.nan)
    count = np.zeros(num_snapshots, dtype=np.int64)
    if child_value.size == 0:
        return {"best": best, "second": second, "min": vmin, "entropy": entropy, "count": count}

    order = np.argsort(child_snapshot_id, kind="stable")
    sid = child_snapshot_id[order]
    v = child_value[order].astype(np.float64)
    uniq, start = np.unique(sid, return_index=True)
    counts = np.diff(np.append(start, sid.size))
    gid = np.repeat(np.arange(uniq.size), counts)

    gmax = np.maximum.reduceat(v, start)
    gmin = np.minimum.reduceat(v, start)
    rowmax = gmax[gid]
    v_wo_max = np.where(v >= rowmax, -np.inf, v)
    gsecond = np.maximum.reduceat(v_wo_max, start)  # -inf for single-child / all-equal groups
    # softmax-over-values entropy per group (stabilized)
    ev = np.exp(v - rowmax)
    gsum = np.add.reduceat(ev, start)
    p = ev / gsum[gid]
    gent = -np.add.reduceat(p * np.log(p + 1e-12), start)

    best[uniq] = gmax
    vmin[uniq] = gmin
    gsecond = np.where(np.isfinite(gsecond), gsecond, np.nan)
    second[uniq] = gsecond
    entropy[uniq] = gent
    count[uniq] = counts
    return {"best": best, "second": second, "min": vmin, "entropy": entropy, "count": count}


def _quantile_bins(x: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_index per row, bin_edges) from quantiles of finite x."""
    finite = x[np.isfinite(x)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)  # collapse degenerate edges
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    return idx, edges


def _zero_crossing(margin_centers: list, mean_adv: list) -> float:
    """First margin where mean advantage crosses from >0 to <=0 (linear interp); NaN if none."""
    for i in range(len(mean_adv) - 1):
        a0, a1 = mean_adv[i], mean_adv[i + 1]
        if a0 is None or a1 is None:
            continue
        if a0 > 0 >= a1:
            frac = a0 / (a0 - a1) if a0 != a1 else 0.0
            return float(margin_centers[i] + frac * (margin_centers[i + 1] - margin_centers[i]))
    return float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def main(config: ValueReadoutSummarizeConfig) -> None:
    """Load the table, reduce to per-snapshot features, emit budget-binned readout numbers."""
    data = np.load(config.input_npz)
    advantage = data["advantage"].astype(np.float64)
    t_t = data["t_t"].astype(np.float64)
    num_snapshots = advantage.shape[0]
    feats = _per_snapshot_features(data["child_value"], data["child_snapshot_id"], num_snapshots)
    best, second, vmin, entropy, count = (feats[k] for k in ("best", "second", "min", "entropy", "count"))
    margin = best - second
    spread = best - vmin

    valid = count >= 2  # margin/spread defined
    print(
        f"[summarize] snapshots={num_snapshots} with>=2 children={int(valid.sum())} "
        f"({100 * valid.mean():.1f}%)",
        flush=True,
    )

    tt_idx, tt_edges = _quantile_bins(t_t, config.n_tt_bins)
    feature_arrays = {"margin": margin, "spread": spread, "best": best, "second": second, "entropy": entropy, "count": count.astype(np.float64)}

    budget_bins = []
    for b in range(len(tt_edges) - 1):
        in_b = valid & (tt_idx == b)
        if in_b.sum() < 50:
            continue
        adv_b = advantage[in_b]
        row = {
            "t_t_range": [float(tt_edges[b]), float(tt_edges[b + 1])],
            "n": int(in_b.sum()),
            "mean_advantage": float(adv_b.mean()),
            "stop_rate": float((adv_b <= 0).mean()),
            "corr_advantage_with": {f: _corr(advantage[in_b], arr[in_b]) for f, arr in feature_arrays.items()},
        }
        # advantage / stop-rate across margin bins, and the apparent zero-crossing
        for fname in ("margin", "spread"):
            f_idx, f_edges = _quantile_bins(feature_arrays[fname][in_b], config.n_value_bins)
            centers, mean_adv, stop_rate, ns = [], [], [], []
            for k in range(len(f_edges) - 1):
                cell = f_idx == k
                centers.append(float(0.5 * (f_edges[k] + f_edges[k + 1])))
                if cell.sum() < 10:
                    mean_adv.append(None); stop_rate.append(None); ns.append(int(cell.sum())); continue
                mean_adv.append(float(adv_b[cell].mean()))
                stop_rate.append(float((adv_b[cell] <= 0).mean()))
                ns.append(int(cell.sum()))
            row[f"advantage_vs_{fname}"] = {"centers": centers, "mean_advantage": mean_adv, "stop_rate": stop_rate, "n": ns}
            if fname == "margin":
                row["margin_zero_crossing"] = _zero_crossing(centers, mean_adv)
        budget_bins.append(row)

    payload = {
        "input_npz": config.input_npz,
        "num_snapshots": num_snapshots,
        "snapshots_with_2plus_children": int(valid.sum()),
        "t_t_bin_edges": [float(e) for e in tt_edges],
        "feature_means": {f: float(np.nanmean(arr)) for f, arr in feature_arrays.items()},
        "budget_bins": budget_bins,
        "note": "Within each T_t bin: corr_advantage_with shows which value feature the advantage "
        "tracks; advantage_vs_margin/spread are budget-held-fixed curves; margin_zero_crossing is "
        "the margin where mean advantage crosses 0 (the apparent stop threshold), NaN if it doesn't.",
    }
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"[summarize] wrote {output_json}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(ValueReadoutSummarizeConfig, main)
