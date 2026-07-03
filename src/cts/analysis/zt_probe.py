"""z_t tree-stat decodability probe: can the encoder ROOT EMBEDDING recover topology?

The deployed MCHalt controller reads a PURE ``z_t`` head (``controller_inputs =
['z_t', 'T_t']``); the tree-stats baseline (:class:`cts.models.readout.StatsReadout`)
instead reads the hand-crafted ``[height, width, n_nodes]``. If ``z_t`` cannot recover
those three stats, the pure-``z_t`` head is structurally missing information the stats
baseline keys off. The encoder was trained on child-WDL, NOT topology, so decodability
is a real, open question — this probe measures it directly.

For each target in ``{height, width, n_nodes}`` we fit two decoders ``z_t -> target``
on a held-out episode split of the VALIDATION set and report test R^2:

    1. Linear : standardize ``z_t`` (fit on probe-train), Ridge, held-out R^2.
    2. MLP    : small 2x64 ReLU torch net, early-stopped on an inner val slice.
    3. Shuffle: row-permuted ``z_t`` Ridge control -> the R^2 floor (~0).

Alignment guard (CRITICAL). The materialized cache stores ``[z_t (d_embed), N_t, T_t]``
rows in dataset (episode) order; the TRAIN cache is materialized with ``shuffle=True`` so
its row order need NOT match the dataset, but VALIDATION uses ``shuffle=False`` so it does.
We therefore probe on VALIDATION only and SELF-CHECK the pairing row-for-row: the packed
episodes' per-snapshot ``n_nodes`` (== ``tree_sizes``) MUST equal ``N_t = feat[:, d_embed]``
from the cache. If they mismatch the pairing is wrong and we abort rather than emit garbage.

The ``[height, width, n_nodes]`` stats come from the SAME code path the eval harness uses
(:func:`analysis.evaluate._load_split_episodes`, which derives per-step height/width from
the trajectory ``depth`` array + ``step_node_cutoffs`` and reads ``n_nodes`` from the
packed shards), so the probe targets are computed identically to the StatsReadout inputs.

    python -m cts.analysis.zt_probe --config config_minply15_maxply75.yaml
    python -m cts.analysis.zt_probe --config config_minply15_maxply75.yaml --limit 200
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analysis.utils.helpers import load_config_section
from analysis.evaluate import _load_split_episodes
from analysis.mchalt_scorer import _load_materialized_cache_unchecked


# target name -> per-episode key emitted by _load_split_episodes
_TARGET_EPISODE_KEY = {"height": "heights", "width": "widths", "n_nodes": "tree_sizes"}
_TARGETS = ("height", "width", "n_nodes")


def _load_cache_features(cache, total_steps: int) -> torch.Tensor:
    """Concatenate the FIRST ``total_steps`` cache rows in shard order.

    Mirrors :func:`cts.train.pg_controller_train._episode_feature_chunks`: the cache
    snapshot order matches the validation dataset's episode order, so the first
    ``total_steps`` rows are exactly the snapshots of the first episodes we loaded.
    """
    parts: list[torch.Tensor] = []
    produced = 0
    for shard_path in cache.shard_paths:
        if produced >= total_steps:
            break
        feats = torch.load(shard_path, weights_only=False)["features"]
        take = min(feats.shape[0], total_steps - produced)
        parts.append(feats[:take])
        produced += take
    allf = torch.cat(parts, dim=0)
    if allf.shape[0] != total_steps:
        raise ValueError(
            f"cache yielded {allf.shape[0]} snapshots but {total_steps} were requested "
            f"from the packed episodes — cache and manifest are out of sync."
        )
    return allf


def _snapshot_arrays(episodes: list[dict]) -> dict[str, np.ndarray]:
    """Flatten per-episode per-step lists into per-snapshot vectors, in episode order."""
    out: dict[str, np.ndarray] = {}
    for target in _TARGETS:
        key = _TARGET_EPISODE_KEY[target]
        out[target] = np.concatenate([np.asarray(ep[key], dtype=np.float64) for ep in episodes])
    return out


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination on the raw target scale (1 - SS_res / SS_tot)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def _linear_r2(z_train, y_train, z_test, y_test, alpha: float = 1.0) -> float:
    """Standardize z on probe-train, Ridge z->target, held-out R^2 on probe-test."""
    scaler = StandardScaler().fit(z_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(z_train), y_train)
    return _r2(y_test, model.predict(scaler.transform(z_test)))


class _MLP(nn.Module):
    """Small 2x64 ReLU regressor (matches the StatsReadout / GnnMetaController head width)."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _mlp_r2(z_train, y_train, z_test, y_test, *, seed: int,
            epochs: int = 400, patience: int = 25, val_frac: float = 0.15) -> float:
    """Torch MLP z->target, early-stopped on an inner val slice; held-out R^2 on probe-test.

    Features are standardized on the inner-train split; the target is standardized too so
    the MSE objective is well-scaled, then predictions are mapped back to the raw scale for
    a comparable R^2.
    """
    rng = np.random.default_rng(seed)
    n = z_train.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    scaler = StandardScaler().fit(z_train[tr_idx])
    zt = torch.tensor(scaler.transform(z_train[tr_idx]), dtype=torch.float32)
    zv = torch.tensor(scaler.transform(z_train[val_idx]), dtype=torch.float32)
    ze = torch.tensor(scaler.transform(z_test), dtype=torch.float32)

    y_mean = float(y_train[tr_idx].mean())
    y_std = float(y_train[tr_idx].std()) or 1.0
    yt = torch.tensor((y_train[tr_idx] - y_mean) / y_std, dtype=torch.float32)
    yv = torch.tensor((y_train[val_idx] - y_mean) / y_std, dtype=torch.float32)

    torch.manual_seed(seed)
    model = _MLP(z_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(zt), yt).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(zv), yv))
        if v < best_val - 1e-5:
            best_val = v
            best_state = {k: val.clone() for k, val in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(ze).numpy()
    return _r2(y_test, pred_std * y_std + y_mean)


def _episode_split_mask(episode_step_counts: list[int], total_steps: int, *,
                        seed: int, train_frac: float = 0.7) -> np.ndarray:
    """Per-snapshot boolean train mask from a 70/30 split BY EPISODE (no step leakage)."""
    n_ep = len(episode_step_counts)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ep)
    n_train_ep = int(round(train_frac * n_ep))
    train_eps = set(perm[:n_train_ep].tolist())
    is_train = np.zeros(total_steps, dtype=bool)
    off = 0
    for i, num_steps in enumerate(episode_step_counts):
        if i in train_eps:
            is_train[off:off + num_steps] = True
        off += num_steps
    return is_train


def run(config_path: str, limit: int | None, seed: int) -> None:
    eval_cfg = load_config_section("eval", config_path)
    train_cfg = load_config_section("train", config_path)
    packed_root = Path(eval_cfg["packed_root"])
    val_cache_path = Path(eval_cfg["materialized_validation_cache"])
    out_dir = Path(eval_cfg["out_dir"])
    d_embed = int(train_cfg["d_embed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[zt_probe] packed_root={packed_root}")
    print(f"[zt_probe] validation_cache={val_cache_path}")
    print(f"[zt_probe] d_embed={d_embed} out_dir={out_dir} limit={limit}")

    # 1. Packed VALIDATION episodes (same stats-source code path as the eval harness) ----
    episodes = _load_split_episodes(packed_root, "validation", max_episodes=limit)
    step_counts = [len(ep["tree_sizes"]) for ep in episodes]
    total_steps = int(sum(step_counts))
    print(f"[zt_probe] loaded {len(episodes)} validation episodes, {total_steps} snapshots")

    # 2. Materialized cache rows for exactly those snapshots (dataset order) --------------
    cache = _load_materialized_cache_unchecked(val_cache_path)
    feats = _load_cache_features(cache, total_steps)
    if feats.shape[1] < d_embed + 1:
        raise ValueError(f"cache feature width {feats.shape[1]} < d_embed+1 ({d_embed + 1})")
    z = feats[:, :d_embed].to(torch.float64).numpy()
    n_t_cache = feats[:, d_embed].to(torch.float64).numpy()  # N_t == n_nodes column

    targets = _snapshot_arrays(episodes)

    # 3. ALIGNMENT SELF-CHECK: packed n_nodes == cache N_t, row-for-row -------------------
    n_nodes_packed = targets["n_nodes"]
    max_abs_diff = float(np.max(np.abs(n_nodes_packed - n_t_cache))) if total_steps else 0.0
    aligned = np.allclose(n_nodes_packed, n_t_cache, atol=1e-4)
    n_mismatch = int(np.sum(~np.isclose(n_nodes_packed, n_t_cache, atol=1e-4)))
    print(f"[zt_probe] ALIGNMENT self-check: n_nodes(packed)==N_t(cache)  "
          f"aligned={aligned}  max_abs_diff={max_abs_diff:g}  mismatches={n_mismatch}/{total_steps}")
    if not aligned:
        raise SystemExit(
            "[zt_probe] ABORT: packed n_nodes != cache N_t row-for-row "
            f"(max_abs_diff={max_abs_diff:g}, {n_mismatch}/{total_steps} mismatched). "
            "The cache/dataset pairing is wrong — do NOT trust the probe outputs."
        )

    # 4. Held-out episode split ----------------------------------------------------------
    is_train = _episode_split_mask(step_counts, total_steps, seed=seed)
    is_test = ~is_train
    z_train, z_test = z[is_train], z[is_test]
    n_test = int(is_test.sum())
    print(f"[zt_probe] probe-train snapshots={int(is_train.sum())}  probe-test snapshots={n_test}")

    # Row-shuffled z control (break the z<->target correspondence) -> R^2 floor.
    shuffle_perm = np.random.default_rng(seed + 1).permutation(total_steps)
    z_shuf = z[shuffle_perm]
    z_shuf_train, z_shuf_test = z_shuf[is_train], z_shuf[is_test]

    rows: list[dict] = []
    for target in _TARGETS:
        y = targets[target]
        y_train, y_test = y[is_train], y[is_test]
        linear_r2 = _linear_r2(z_train, y_train, z_test, y_test)
        mlp_r2 = _mlp_r2(z_train, y_train, z_test, y_test, seed=seed)
        shuffle_r2 = _linear_r2(z_shuf_train, y_train, z_shuf_test, y_test)
        row = {
            "stat": target,
            "linear_R2": round(linear_r2, 6),
            "mlp_R2": round(mlp_r2, 6),
            "shuffle_R2": round(shuffle_r2, 6),
            "n_test": n_test,
            "target_var": round(float(y_test.var()), 6),
            "target_min": float(y_test.min()),
            "target_max": float(y_test.max()),
        }
        rows.append(row)
        print(f"[zt_probe] {target:8s} linear_R2={linear_r2:+.4f}  mlp_R2={mlp_r2:+.4f}  "
              f"shuffle_R2={shuffle_r2:+.4f}  var={y_test.var():.3g}  "
              f"range=[{y_test.min():g},{y_test.max():g}]")

    # 5. Outputs -------------------------------------------------------------------------
    csv_path = out_dir / "zt_decodability.csv"
    fieldnames = ["stat", "linear_R2", "mlp_R2", "shuffle_R2", "n_test",
                  "target_var", "target_min", "target_max"]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[zt_probe] wrote {csv_path}")

    _plot(rows, out_dir / "zt_decodability.png", limit=limit)
    print(f"[zt_probe] wrote {out_dir / 'zt_decodability.png'}")


def _plot(rows: list[dict], png_path: Path, *, limit: int | None) -> None:
    """Grouped bar chart: per stat, {linear, mlp, shuffle} held-out R^2."""
    stats = [r["stat"] for r in rows]
    series = {
        "linear (Ridge)": [r["linear_R2"] for r in rows],
        "MLP (2x64)": [r["mlp_R2"] for r in rows],
        "shuffle (floor)": [r["shuffle_R2"] for r in rows],
    }
    x = np.arange(len(stats))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylabel("held-out $R^2$")
    suffix = "" if limit is None else f"  (limit={limit})"
    ax.set_title(f"Decodability of tree stats from encoder root embedding $z_t${suffix}")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on validation episodes (smoke slice). Default: full split.")
    parser.add_argument("--seed", type=int, default=0, help="Split / init seed.")
    args = parser.parse_args()
    run(args.config, args.limit, args.seed)


if __name__ == "__main__":
    main()
