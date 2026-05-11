from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_episode_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_relative_q_trajectories(records: list[dict], out_path: Path) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        oracle_vals = record["oracle_move_q_trajectory"]
        meta_vals = record["metacontroller_move_q_trajectory"]
        oracle_stop = int(record["oracle_stop_step"])
        for step_index, (oracle_q, meta_q) in enumerate(zip(oracle_vals, meta_vals)):
            grouped[step_index - oracle_stop].append(
                {
                    "oracle_q": oracle_q,
                    "meta_q": meta_q,
                    "oracle_final_value": float(record["oracle_move_final_value"]),
                    "meta_final_value": float(record["metacontroller_move_final_value"]),
                }
            )

    xs = sorted(grouped)
    xs_arr = np.array(xs, dtype=np.int32)
    oracle_means = []
    oracle_p10 = []
    oracle_p90 = []
    meta_means = []
    meta_p10 = []
    meta_p90 = []
    oracle_final_means = []
    meta_final_means = []
    oracle_availability = []
    meta_availability = []

    for x in xs:
        step_rows = grouped[x]
        oracle_values = np.array([float(row["oracle_q"]) for row in step_rows if row["oracle_q"] is not None], dtype=np.float32)
        meta_values = np.array([float(row["meta_q"]) for row in step_rows if row["meta_q"] is not None], dtype=np.float32)
        oracle_means.append(float(oracle_values.mean()) if oracle_values.size else np.nan)
        oracle_p10.append(float(np.percentile(oracle_values, 10)) if oracle_values.size else np.nan)
        oracle_p90.append(float(np.percentile(oracle_values, 90)) if oracle_values.size else np.nan)
        meta_means.append(float(meta_values.mean()) if meta_values.size else np.nan)
        meta_p10.append(float(np.percentile(meta_values, 10)) if meta_values.size else np.nan)
        meta_p90.append(float(np.percentile(meta_values, 90)) if meta_values.size else np.nan)
        oracle_final_means.append(float(np.mean([row["oracle_final_value"] for row in step_rows])))
        meta_final_means.append(float(np.mean([row["meta_final_value"] for row in step_rows])))
        oracle_availability.append(float(np.mean([float(row["oracle_q"] is not None) for row in step_rows])))
        meta_availability.append(float(np.mean([float(row["meta_q"] is not None) for row in step_rows])))

    fig, (ax_q, ax_avail) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    ax_q.fill_between(xs_arr, oracle_p10, oracle_p90, alpha=0.2, color="tab:blue")
    ax_q.plot(xs_arr, oracle_means, linewidth=2, color="tab:blue", label="Oracle move Q")
    ax_q.fill_between(xs_arr, meta_p10, meta_p90, alpha=0.2, color="tab:orange")
    ax_q.plot(xs_arr, meta_means, linewidth=2, color="tab:orange", label="Metacontroller move Q")
    ax_q.plot(xs_arr, oracle_final_means, linewidth=1.5, linestyle="--", color="tab:blue", label="Oracle move final value")
    ax_q.plot(xs_arr, meta_final_means, linewidth=1.5, linestyle="--", color="tab:orange", label="Metacontroller move final value")
    ax_q.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_q.set_ylabel("Snapshot root Q-value")
    ax_q.set_title("Oversearch Episodes: Oracle vs Metacontroller Choice Q-Trajectories")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best")

    ax_avail.plot(xs_arr, oracle_availability, linewidth=2, color="tab:blue", label="Oracle move available")
    ax_avail.plot(xs_arr, meta_availability, linewidth=2, color="tab:orange", label="Metacontroller move available")
    ax_avail.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_avail.set_xlabel("Step relative to oracle stop")
    ax_avail.set_ylabel("Availability fraction")
    ax_avail.set_ylim(-0.02, 1.02)
    ax_avail.grid(True, alpha=0.3)
    ax_avail.legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "relative_steps": xs,
        "oracle_move_q_mean": oracle_means,
        "metacontroller_move_q_mean": meta_means,
        "oracle_move_final_value_mean": oracle_final_means,
        "metacontroller_move_final_value_mean": meta_final_means,
        "oracle_move_available_fraction": oracle_availability,
        "metacontroller_move_available_fraction": meta_availability,
    }


def plot_relative_q_gap(records: list[dict], out_path: Path) -> dict:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        oracle_vals = record["oracle_move_q_trajectory"]
        meta_vals = record["metacontroller_move_q_trajectory"]
        oracle_stop = int(record["oracle_stop_step"])
        for step_index, (oracle_q, meta_q) in enumerate(zip(oracle_vals, meta_vals)):
            if oracle_q is None or meta_q is None:
                continue
            grouped[step_index - oracle_stop].append(float(meta_q) - float(oracle_q))

    xs = sorted(grouped)
    xs_arr = np.array(xs, dtype=np.int32)
    means = []
    medians = []
    p10s = []
    p90s = []
    counts = []
    for x in xs:
        values = np.array(grouped[x], dtype=np.float32)
        means.append(float(values.mean()))
        medians.append(float(np.median(values)))
        p10s.append(float(np.percentile(values, 10)))
        p90s.append(float(np.percentile(values, 90)))
        counts.append(int(values.size))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.fill_between(xs_arr, p10s, p90s, alpha=0.2, color="tab:purple", label="10-90 pct")
    ax.plot(xs_arr, medians, linewidth=2, color="tab:purple", label="Median Q gap")
    ax.plot(xs_arr, means, linewidth=1.5, linestyle="--", color="tab:red", label="Mean Q gap")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Step relative to oracle stop")
    ax.set_ylabel("Q(meta choice) - Q(oracle choice)")
    ax.set_title("Oversearch Episodes: Relative Root-Q Preference Gap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "relative_steps": xs,
        "mean_q_gap": means,
        "median_q_gap": medians,
        "p10_q_gap": p10s,
        "p90_q_gap": p90s,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot oversearch preference-evolution outputs from reconstructed JSONL.")
    parser.add_argument("--records-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records = load_episode_records(Path(args.records_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "relative_q_plot": plot_relative_q_trajectories(records, output_dir / "oversearch_choice_q_trajectories.png"),
        "relative_q_gap_plot": plot_relative_q_gap(records, output_dir / "oversearch_choice_q_gap.png"),
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
