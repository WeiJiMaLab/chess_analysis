#!/usr/bin/env python3
"""Histogram of value_gap (from packed topology_targets column 0 = value_gap).

Example:
  python3 scripts/plot_value_gap_packed.py \
    --manifest /scratch/.../pretrain_packed_50k_topology_full/train_manifest.json \
    --out-dir analysis_outputs/value_gap_packed
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def collect_finite_value_gaps(
    manifest_path: Path,
    *,
    max_shards: int,
    max_finite_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Concatenate finite value_gap targets from shards; uniformly subsample to ``max_finite_samples`` if needed."""

    import torch

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = list(manifest["entries"])
    if max_shards > 0:
        rng_entry = random.Random(seed)
        rng_entry.shuffle(entries)
        entries = entries[:max_shards]

    finite_parts: list[np.ndarray] = []
    total_nodes = 0
    nan_nodes = 0

    for entry in entries:
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        topo = payload.get("topology_targets")
        if topo is None:
            raise ValueError(f"No topology_targets: {entry['path']}")

        node_ptr = payload["node_ptr"]
        n_take = int(entry["num_examples"])
        ns = int(node_ptr[0].item())
        ne = int(node_ptr[n_take].item())

        raw = topo[ns:ne].float()[:, 0]
        n_here = int(raw.numel())
        total_nodes += n_here
        
        is_finite = torch.isfinite(raw)
        nan_nodes += int((~is_finite).sum().item())

        finite_vals = raw[is_finite]
        if finite_vals.numel() > 0:
            finite_parts.append(finite_vals.numpy().astype(np.float64))

        del payload

    streamed_finite = int(sum(c.size for c in finite_parts)) if finite_parts else 0
    if not finite_parts:
        stacked = np.array([], dtype=np.float64)
    else:
        stacked = np.concatenate(finite_parts)
        if stacked.size > max_finite_samples:
            rng_np = np.random.default_rng(seed)
            idx = rng_np.choice(stacked.size, size=max_finite_samples, replace=False)
            stacked = stacked[idx]

    finite_nodes_seen = total_nodes - nan_nodes
    summary: dict[str, int | float] = {
        "total_nodes_scanned": total_nodes,
        "nan_nodes": nan_nodes,
        "finite_nodes_seen": finite_nodes_seen,
        "nan_fraction": nan_nodes / max(total_nodes, 1),
        "plot_sample_size": stacked.size,
        "streamed_finite_total": streamed_finite,
        "uniform_subsampled": streamed_finite > max_finite_samples,
    }
    return stacked, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot value_gap histograms from packed data.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/value_gap_packed"))
    parser.add_argument("--max-shards", type=int, default=0, help="0 = all shards in manifest")
    parser.add_argument("--max-finite-samples", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib required: pip install matplotlib"
        ) from exc

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gaps, summary = collect_finite_value_gaps(
        args.manifest,
        max_shards=args.max_shards,
        max_finite_samples=args.max_finite_samples,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))
    manifest_name = Path(args.manifest).stem
    with (out_dir / f"{manifest_name}_value_gap_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if gaps.size == 0:
        raise SystemExit("No finite value_gap in scanned range.")

    # --- Figure 1: raw gaps (heavy tail clipped for readability, e.g. p999)
    p99 = np.quantile(gaps, 0.99)
    p999 = np.quantile(gaps, 0.999)
    clip_hi = float(min(p999 * 1.1, gaps.max()))
    clipped = np.clip(gaps, 0.0, clip_hi)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    counts, bins, _ = ax1.hist(clipped, bins=args.bins, color="#2c7bb6", edgecolor="white", alpha=0.85)
    ax1.set_title(
        f"Value Gap (finite only, clipped \u2264 {clip_hi:.1f} cp; p99={p99:.1f})\n"
        f"{args.manifest.name} — sampled {gaps.size:,} finite nodes"
    )
    ax1.set_xlabel("value_gap (centipawns)")
    ax1.set_ylabel("count")
    fig1.savefig(out_dir / f"{manifest_name}_value_gap_hist_clip.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Full range log-y (same clip)
    fig1b, ax1b = plt.subplots(figsize=(10, 5))
    ax1b.hist(clipped, bins=args.bins, color="#2c7bb6", edgecolor="white", alpha=0.85)
    ax1b.set_yscale("log")
    ax1b.set_title(f"Same as above, log frequency (counts)")
    ax1b.set_xlabel("value_gap (centipawns)")
    ax1b.set_ylabel("count (log)")
    fig1b.savefig(out_dir / f"{manifest_name}_value_gap_hist_clip_logy.png", dpi=150, bbox_inches="tight")
    plt.close(fig1b)

    print(f"wrote PNGs under {out_dir}")


if __name__ == "__main__":
    main()
