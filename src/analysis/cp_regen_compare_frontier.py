"""Side-by-side comparison of the ``cp_regen`` variant's real eval frontier against the
WDL-baseline eval frontier already used throughout plan.md (S1: SingleHalt*=0.1204,
Stats=0.1138; Z: e2e z_t; etc.) -- plan.md Agent 3's required "proof of success" plot.

Reads BOTH runs' ``frontier_data.json`` (written by ``analysis.evaluate``'s
``_render_frontier``/``plot_regret_effort_frontier`` at eval time -- ``lo``/``hi`` on
each point are already bootstrap CIs, computed by evaluate.py's own ``_mean_ci``, so
this script does not recompute anything, it only re-plots the already-bootstrapped
numbers side by side) and draws one grouped-bar panel per controller: two bars
(WDL-baseline vs. cp_regen), error bars = the CI, colored by CONTROLLER identity
(matching every other frontier plot in this investigation, `_C` in evaluate.py) with
corpus distinguished by a hatch texture (solid = WDL-baseline, hatched = cp_regen) --
not a second hue, per this repo's color-follows-the-entity convention.

Usage:
    python -m analysis.cp_regen_compare_frontier \
        --baseline outputs/figures/minply15_maxply75/normative/frontier_data.json \
        --cp-regen outputs/figures/minply15_maxply75/variants/cp_regen/normative/frontier_data.json \
        --out-dir outputs/figures/minply15_maxply75/variants/cp_regen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.utils.plots import save_pdf_png

# Matches evaluate.py's `_C` exactly, so controller colors are identical across every
# frontier figure in this investigation (baseline, e2e-z3, and this comparison).
_C = {"stats": "#12A19A", "zt": "#3F4DA0", "always": "#C0392B", "never": "#8B97A3",
      "singlehalt": "#F2A9A6"}
_LABEL_TO_KEY = {
    "SingleHalt* (fixed stop)": "singlehalt",
    "Stats-Controller": "stats",
    "$z_t$-Controller": "zt",
    "Always Stop (k=0)": "always",
    "Always Continue (k=max)": "never",
}
# Fixed display order (excludes the two degenerate anchors by default -- see --include-anchors).
_ORDER = ["SingleHalt* (fixed stop)", "Stats-Controller", "$z_t$-Controller"]


def _load_points(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    return {p["label"]: p for p in data["all_points"]}


def render_comparison(baseline_points: dict[str, dict], cp_regen_points: dict[str, dict],
                       out_dir: str | Path, *, include_anchors: bool = False) -> dict:
    """Grouped bar chart: regret (y, with CI) per controller, WDL-baseline vs cp_regen.
    Returns the plotted numbers as a JSON-serializable dict (for the report / regression)."""
    labels = list(_ORDER) + (["Always Stop (k=0)", "Always Continue (k=max)"] if include_anchors else [])
    labels = [l for l in labels if l in baseline_points and l in cp_regen_points]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 2.0, 4.6))

    rows = []
    for i, label in enumerate(labels):
        key = _LABEL_TO_KEY[label]
        color = _C[key]
        b, c = baseline_points[label], cp_regen_points[label]
        ax.bar(x[i] - width / 2, b["y"], width, color=color, edgecolor="white", linewidth=0.6,
               label="WDL-baseline" if i == 0 else None)
        ax.errorbar(x[i] - width / 2, b["y"], yerr=[[b["y"] - b["lo"]], [b["hi"] - b["y"]]],
                    fmt="none", ecolor="#2A2A2A", elinewidth=1.3, capsize=3.5, zorder=6)
        ax.bar(x[i] + width / 2, c["y"], width, color=color, edgecolor="white", linewidth=0.6,
               hatch="///", alpha=0.85, label="cp_regen (tanh300, pruned)" if i == 0 else None)
        ax.errorbar(x[i] + width / 2, c["y"], yerr=[[c["y"] - c["lo"]], [c["hi"] - c["y"]]],
                    fmt="none", ecolor="#2A2A2A", elinewidth=1.3, capsize=3.5, zorder=6)
        rows.append({"label": label, "baseline_regret": b["y"], "baseline_lo": b["lo"], "baseline_hi": b["hi"],
                     "cp_regen_regret": c["y"], "cp_regen_lo": c["lo"], "cp_regen_hi": c["hi"]})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Regret")
    ax.set_title("cp_regen (real regeneration under tanh(cp_order/300) + pruning) vs. WDL-baseline")
    ax.grid(axis="y", color="#E3E7EB", lw=1)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    fig.tight_layout()
    save_pdf_png(fig, str(out_dir), "cp_regen_frontier_comparison", dpi=200)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cp_regen_frontier_comparison_data.json").write_text(json.dumps(rows, indent=2))
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cp-regen", required=True)
    parser.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/variants/cp_regen")
    parser.add_argument("--include-anchors", action="store_true",
                        help="also plot the AlwaysStop/AlwaysContinue degenerate anchors")
    args = parser.parse_args()

    baseline_points = _load_points(args.baseline)
    cp_regen_points = _load_points(args.cp_regen)
    result = render_comparison(baseline_points, cp_regen_points, args.out_dir,
                               include_anchors=args.include_anchors)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
