"""Cross-Elo "ladder" plot: the 4 halt policies vs Stockfish strength rung.

Reads one per-rung results JSON per Elo rung (each emitted by
``evaluate.py --results-json``) and draws, for the four headline halt
policies (Always / Never / Fraction / MCHalt), how their **Regret** and
**P(stop==OSS)** move as the opponent Stockfish strength climbs the ladder
(elo1800 -> 2000 -> 2200). The StatsReadout (tree-stats) tier is an optional
5th line, included only if every rung scored it.

The per-rung JSON schema (written by ``evaluate._maybe_write_results``):

    {
      "always":   {"regret": <float>, "stop_acc": <float>},
      "never":    {"regret": <float>, "stop_acc": <float>},
      "fraction": {"regret": <float>, "stop_acc": <float>},
      "stats":    {"regret": <float>, "stop_acc": <float>},
      "mchalt":   {"regret": <float|null>, "stop_acc": <float|null>}
    }

``mchalt`` carries ``null`` if that rung's controller was not yet scored
(pending); such a rung is dropped from the MCHalt line rather than plotted as 0.

    PYTHONPATH=src python -m analysis.ladder_plot \
        --rung 1800 elo1800_results.json \
        --rung 2000 elo2000_results.json \
        --rung 2200 elo2200_results.json \
        --out-dir figures/normative
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Stable model key -> (display label, line color). Order is the legend order.
_MODELS: list[tuple[str, str, str]] = [
    ("always", "Always Stop", "#969696"),
    ("never", "Never Stop", "#525252"),
    ("fraction", "Fraction", "#6baed6"),
    ("stats", "Readout(stats)", "#fd8d3c"),
    ("mchalt", "MCHalt (GNN)", "#08519c"),
]


def _load_rungs(rung_args: list[tuple[int, str]]) -> tuple[list[int], dict[str, dict[str, list]]]:
    """Read each rung's results JSON; return sorted Elos + per-model {elo->metrics}.

    Returns ``(elos, table)`` where ``table[model_key][metric]`` is a list of
    ``(elo, value)`` for every rung that has a non-null value for that model.
    """
    parsed = sorted(((int(elo), json.loads(Path(path).read_text())) for elo, path in rung_args),
                    key=lambda item: item[0])
    elos = [elo for elo, _ in parsed]
    table: dict[str, dict[str, list]] = {
        key: {"regret": [], "stop_acc": []} for key, _, _ in _MODELS
    }
    for elo, payload in parsed:
        for key, _, _ in _MODELS:
            entry = payload.get(key)
            if entry is None:
                continue
            for metric in ("regret", "stop_acc"):
                value = entry.get(metric)
                if value is not None:
                    table[key][metric].append((elo, float(value)))
    return elos, table


def _plot_metric(ax, table: dict[str, dict[str, list]], metric: str, ylabel: str) -> None:
    for key, label, color in _MODELS:
        points = table[key][metric]
        if not points:
            continue
        xs = [elo for elo, _ in points]
        ys = [value for _, value in points]
        ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=1.6, markersize=5)
    ax.set_xlabel("Stockfish Elo rung")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.8)


def plot_ladder(rung_args: list[tuple[int, str]], out_dir: Path, *, title: str | None = None) -> Path:
    """Draw the two-panel ladder figure (regret + P(stop==OSS) vs Elo) and save it."""
    elos, table = _load_rungs(rung_args)
    fig, (ax_regret, ax_stop) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _plot_metric(ax_regret, table, "regret", "mean regret (lower = better)")
    _plot_metric(ax_stop, table, "stop_acc", "P(stop == OSS)")
    for ax in (ax_regret, ax_stop):
        ax.set_xticks(elos)
    ax_regret.set_title("Regret across the SF ladder")
    ax_stop.set_title("Stop-accuracy across the SF ladder")
    ax_stop.legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ladder_by_model.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def _parse_rung(values: list[str]) -> tuple[int, str]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("--rung takes exactly: ELO PATH")
    return int(values[0]), values[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", nargs=2, action="append", metavar=("ELO", "JSON"), required=True,
                    help="an Elo rung and its results JSON; repeatable (e.g. --rung 1800 a.json)")
    ap.add_argument("--out-dir", default="outputs/figures/normative")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    rung_args = [(int(elo), path) for elo, path in args.rung]
    plot_ladder(rung_args, Path(args.out_dir), title=args.title)


if __name__ == "__main__":
    main()
