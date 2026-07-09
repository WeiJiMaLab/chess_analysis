"""E2 diagnostic (plan.md): can z_t ALONE decode basic tree stats {n_nodes, height, width}?

Standalone diagnostic script — NOT a pipeline stage, does not touch/modify any pipeline
code. Reuses the exact machinery already used all session:

  * ``_linear_r2`` / ``_mlp_r2`` / ``_episode_split_mask``  (``cts.analysis.zt_probe``)
  * ``_load_split_episodes`` / ``_load_zt_by_episode``      (``analysis.evaluate``)

which is the same loading/splitting pattern behind ``analysis.evaluate``'s
``_compute_decodability_data`` (the corrected, non-nested probe: each feature set is
tested ALONE, never as a cumulative superset).

For each of ``{n_nodes, height, width}`` we report THREE held-out R^2 numbers:

  1. ``z_t`` alone (MLP probe)      — can the encoder root embedding recover the stat?
  2. trivial self-upper-bound        — the stat predicting ITSELF (R^2~=1 sanity ceiling;
                                        linear probe, since the mapping is an exact identity).
  3. shuffle floor                   — row-shuffled z_t (breaks the z<->target
                                        correspondence; linear probe, ~0).

Reuses the ALREADY-materialized real validation z_t cache (``eval.materialized_validation_cache``)
-- no regeneration, no training. Fast (probe-only).

    python -m cts.analysis.zt_stats_decodability --config config_minply15_maxply75.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.utils.helpers import load_config_section
from analysis.utils.plots import save_pdf_png
from analysis.evaluate import _load_split_episodes, _load_zt_by_episode
from cts.analysis.zt_probe import _linear_r2, _mlp_r2, _episode_split_mask


_TARGETS = ("n_nodes", "height", "width")
_EPISODE_KEY = {"n_nodes": "tree_sizes", "height": "heights", "width": "widths"}


def compute_stats_decodability(packed_root: Path, cache_path, *, d_embed: int = 32,
                                max_episodes: int | None = None, seed: int = 0) -> list[dict]:
    """Per-stat {z_t, trivial upper bound, shuffle floor} held-out R^2 rows.

    ``packed_root``/``cache_path`` are the validation packed root + materialized cache
    (same alignment-checked loaders as the rest of the eval harness — a mismatch between
    packed ``n_nodes`` and the cache's ``N_t`` column raises inside ``_load_zt_by_episode``
    rather than silently emitting garbage).
    """
    episodes = _load_split_episodes(packed_root, "validation", max_episodes=max_episodes)
    z_by_ep = _load_zt_by_episode(episodes, cache_path, d_embed)
    step_counts = [len(ep["halt_rewards"]) for ep in episodes]
    total = int(sum(step_counts))
    z = np.concatenate(z_by_ep, axis=0)

    traj_keys = [ep["trajectory_key"] for ep in episodes]
    is_tr = _episode_split_mask(step_counts, total, seed=seed, trajectory_keys=traj_keys)
    is_te = ~is_tr

    # Row-shuffled z control (break the z<->target correspondence) -> the R^2 floor.
    z_shuf = z[np.random.default_rng(seed + 7).permutation(total)]

    rows: list[dict] = []
    for target in _TARGETS:
        y = np.concatenate([np.asarray(ep[_EPISODE_KEY[target]], dtype=np.float64) for ep in episodes])
        y_train, y_test = y[is_tr], y[is_te]
        y_train_col, y_test_col = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

        zt_linear = _linear_r2(z[is_tr], y_train, z[is_te], y_test)
        zt_mlp = _mlp_r2(z[is_tr], y_train, z[is_te], y_test, seed=seed)
        trivial_r2 = _linear_r2(y_train_col, y_train, y_test_col, y_test)
        shuffle_r2 = _linear_r2(z_shuf[is_tr], y_train, z_shuf[is_te], y_test)

        rows.append({
            "stat": target,
            "zt_linear_R2": round(zt_linear, 6),
            "zt_mlp_R2": round(zt_mlp, 6),
            "trivial_upper_bound_R2": round(trivial_r2, 6),
            "shuffle_R2": round(shuffle_r2, 6),
            "n_test": int(is_te.sum()),
            "target_var": round(float(y_test.var()), 6),
            "target_min": float(y_test.min()),
            "target_max": float(y_test.max()),
        })
    return rows


def _plot(rows: list[dict], out_dir: Path) -> str:
    """Bar chart: one group per stat, 3 bars = {z_t (MLP), trivial upper bound, shuffle floor}."""
    stats = [r["stat"] for r in rows]
    series = {
        "$z_t$ (MLP)": [r["zt_mlp_R2"] for r in rows],
        "trivial upper bound\n(stat predicts itself)": [r["trivial_upper_bound_R2"] for r in rows],
        "shuffle floor": [r["shuffle_R2"] for r in rows],
    }
    x = np.arange(len(stats))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=label)
    for i, r in enumerate(rows):
        ax.text(x[i] - width, r["zt_mlp_R2"] + 0.02, f"{r['zt_mlp_R2']:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylim(min(-0.05, min(min(v) for v in series.values()) - 0.05), 1.08)
    ax.set_ylabel("held-out $R^2$")
    ax.set_title("E2: decodability of tree stats from $z_t$ alone\n(validation, real 400K-tree materialized cache)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.text(
        0.5, -0.02,
        "Plain-English finding: $z_t$ carries only a weak-to-moderate echo of tree shape (R^2=0.07-0.17, above the\n"
        "shuffle floor but far below the R^2~=1 a trivial readout of the raw stats gets) -- the encoder loses most\n"
        "of this basic structural information rather than preserving it.",
        ha="center", va="top", fontsize=8.5, style="italic",
    )
    fig.tight_layout()
    return save_pdf_png(fig, str(out_dir), "e2_zt_stats_decodability")


def run(config_path: str, limit: int | None, seed: int, out_dir: str | None) -> None:
    eval_cfg = load_config_section("eval", config_path)
    train_cfg = load_config_section("train", config_path)
    packed_root = Path(eval_cfg["packed_root"])
    val_cache_path = Path(eval_cfg["materialized_validation_cache"])
    d_embed = int(train_cfg["d_embed"])
    out = Path(out_dir) if out_dir else Path(eval_cfg["out_dir"]) / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[zt_stats_decodability] packed_root={packed_root}")
    print(f"[zt_stats_decodability] validation_cache={val_cache_path}")
    print(f"[zt_stats_decodability] d_embed={d_embed} out_dir={out} limit={limit}")

    rows = compute_stats_decodability(packed_root, val_cache_path, d_embed=d_embed,
                                       max_episodes=limit, seed=seed)
    for r in rows:
        print(f"[zt_stats_decodability] {r['stat']:8s} zt_linear_R2={r['zt_linear_R2']:+.4f} "
              f"zt_mlp_R2={r['zt_mlp_R2']:+.4f} trivial_R2={r['trivial_upper_bound_R2']:+.4f} "
              f"shuffle_R2={r['shuffle_R2']:+.4f}  n_test={r['n_test']}  "
              f"var={r['target_var']:.3g}  range=[{r['target_min']:g},{r['target_max']:g}]")

    csv_path = out / "e2_zt_stats_decodability.csv"
    fieldnames = ["stat", "zt_linear_R2", "zt_mlp_R2", "trivial_upper_bound_R2", "shuffle_R2",
                  "n_test", "target_var", "target_min", "target_max"]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[zt_stats_decodability] wrote {csv_path}")

    pdf_path = _plot(rows, out)
    print(f"[zt_stats_decodability] wrote {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on validation episodes (smoke slice). Default: full split.")
    parser.add_argument("--seed", type=int, default=0, help="Split / init seed.")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: <eval.out_dir>/diagnosis).")
    args = parser.parse_args()
    run(args.config, args.limit, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
