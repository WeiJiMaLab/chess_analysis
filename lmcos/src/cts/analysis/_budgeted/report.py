"""Human-readable Markdown report stitched from the in-memory summary dict.

Renders ``report.md`` with run summary, best epochs, best baseline,
clean oracle decision factors, error-episode oracle boundary, regret
decomposition, and same-tree multi-budget consistency sections.
Each section degrades gracefully when the corresponding analysis was
skipped in core-report mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    """Render the human-readable ``report.md`` from the in-memory summary dict.

    The report is a hand-built Markdown stitch-up that lifts the most useful
    headline numbers out of ``summary`` (regret/return/expansions, best
    epochs, best baseline, clean-oracle factors, regret decomposition,
    multi-budget consistency). Sections degrade gracefully when the
    corresponding analysis was skipped in core-report mode.
    """
    # Per-section setup: pull out the specific summary slices we need, handling
    # the absent-key case for each (some sections only run in --report-mode full).
    large_plus_six = summary["regret_decomposition"]["by_bucket"].get("large", {}).get("6")
    large_plus_six_line = (
        f"- Large-bucket `+6` episodes: regret={large_plus_six['mean_regret']:.4f}, "
        f"halt={large_plus_six['mean_halt_reward_term']:.4f}, "
        f"maint={large_plus_six['mean_maintenance_term']:.4f}, "
        f"time={large_plus_six['mean_time_term']:.4f}"
        if large_plus_six is not None
        else "- Large-bucket `+6` episodes: unavailable"
    )
    future_worse = summary.get("future_worse_move_switch")
    if future_worse is not None:
        future_worse_counts = future_worse["overall_relation_counts"]
        future_worse_line = (
            f"- `future_already_worse` split: same_move={future_worse_counts.get('same_move', 0)}, "
            f"move_switch={future_worse_counts.get('move_switch', 0)}"
        )
    else:
        future_worse_line = "- `future_already_worse` split: skipped (source trees unavailable locally)"
    clean_oracle = summary.get("oracle_stop_clean_factors")
    clean_large = summary.get("oracle_stop_clean_driver_by_bucket", {}).get("large", {})
    oversearch_oracle_stop = summary.get("error_episode_oracle_stop_factors", {}).get("large", {})
    churn_summary = summary.get("root_action_churn")
    if churn_summary is not None:
        churn_line = (
            f"- Root-churn motifs: A={churn_summary['top_level_counts'].get('A', 0)}, "
            f"X*A={churn_summary['top_level_counts'].get('X*A', 0)}, "
            f"X*AB*A={churn_summary['top_level_counts'].get('X*AB*A', 0)}"
        )
        lucky_line = (
            f"- Lucky early halts: {churn_summary['lucky_early_halt_episodes']} "
            f"({churn_summary['lucky_early_halt_fraction']:.3f} of budgeted episodes)"
        )
    else:
        churn_line = "- Root-churn motifs: skipped (source trees unavailable locally)"
        lucky_line = "- Lucky early halts: skipped (source trees unavailable locally)"
    if clean_oracle is not None:
        clean_oracle_lines = [
            "## Clean Oracle Decision Factors",
            "- At each step, oracle advantage is decomposed as:",
            "  `target_advantage = cost_free_future_reward_gain - downstream_future_cost - current_step_cost`",
            "  where `cost_free_future_reward_gain` is the best later halt-reward improvement ignoring all future costs,",
            "  `downstream_future_cost` is the cumulative cost from the next step until that best future stop,",
            "  and `current_step_cost` is the immediate cost of taking one more step now.",
            f"- Oracle-stop residual check: {clean_oracle['max_abs_target_residual']:.6g}",
            f"- Oracle-stop overall means: reward_gain={clean_oracle['overall']['mean_cost_free_future_reward_gain']:.4f}, "
            f"downstream_cost={clean_oracle['overall']['mean_downstream_future_cost']:.4f}, "
            f"current_cost={clean_oracle['overall']['mean_current_step_cost']:.4f}, "
            f"target_adv={clean_oracle['overall']['mean_target_advantage']:.4f}",
            f"- Large-bucket clean driver mix: no_gain={clean_large.get('no_cost_free_gain', 0.0):.3f}, "
            f"downstream_cost={clean_large.get('downstream_cost_dominated', 0.0):.3f}, "
            f"current_step_cost={clean_large.get('current_step_cost_dominated', 0.0):.3f}",
            future_worse_line,
            churn_line,
            lucky_line,
            "",
        ]
    else:
        clean_oracle_lines = [
            "## Clean Oracle Decision Factors",
            "- Skipped in core report mode.",
            future_worse_line,
            churn_line,
            lucky_line,
            "",
        ]

    if oversearch_oracle_stop:
        error_oracle_lines = [
            "## Error-Episode Oracle Boundary",
            "- For each erroneous episode, the report now records the oracle-stop decomposition and the model's predicted advantage at that same step.",
            f"- Large-bucket oversearch at oracle stop: reward_gain={oversearch_oracle_stop.get('mean_cost_free_future_reward_gain', 0.0):.4f}, "
            f"downstream_cost={oversearch_oracle_stop.get('mean_downstream_future_cost', 0.0):.4f}, "
            f"maint={oversearch_oracle_stop.get('mean_current_step_maintenance_cost', 0.0):.4f}, "
            f"time={oversearch_oracle_stop.get('mean_current_step_time_cost', 0.0):.4f}, "
            f"pred_adv={oversearch_oracle_stop.get('mean_predicted_advantage_at_oracle_stop', 0.0):.4f}",
            "",
        ]
    else:
        error_oracle_lines = [
            "## Error-Episode Oracle Boundary",
            "- Skipped in core report mode.",
            "",
        ]

    # Final stitched-together Markdown body.
    lines = [
        "# Budgeted Controller Analysis",
        "",
        "## Run Summary",
        f"- Episodes: {summary['episodes']}",
        f"- States: {summary['states']}",
        f"- Final greedy regret: {summary['final_greedy']['average_regret']:.4f}",
        f"- Final greedy return: {summary['final_greedy']['average_return']:.4f}",
        f"- Final greedy expansions: {summary['final_greedy']['average_expansions']:.3f}",
        "",
        "## Best Epochs",
        f"- Best validation MSE epoch: {summary['best_epochs']['best_validation_mse_epoch']}",
        f"- Best greedy return epoch: {summary['best_epochs']['best_greedy_return_epoch']}",
        f"- Best greedy regret epoch: {summary['best_epochs']['best_greedy_regret_epoch']}",
        "",
        "## Best Baseline",
        f"- Best baseline by regret: `{summary['baselines']['best_by_regret']['name']}`",
        f"- Baseline regret: {summary['baselines']['best_by_regret']['average_regret']:.4f}",
        f"- Baseline return: {summary['baselines']['best_by_regret']['average_return']:.4f}",
        "",
        *clean_oracle_lines,
        *error_oracle_lines,
        "## Regret Decomposition",
        "- Regret is decomposed as:",
        "  `oracle_value - predicted_value = (halt_reward@oracle - halt_reward@predicted) + (predicted maintenance paid - oracle maintenance paid) + (predicted time cost paid - oracle time cost paid)`",
        f"- Max absolute decomposition residual: {summary['regret_decomposition']['max_abs_residual']:.6g}",
        large_plus_six_line,
        "",
        "## Same-Tree Different-Budget Consistency",
        f"- Source paths with multiple budgets: {summary['consistency']['num_multibudget_source_paths']}",
        f"- Oracle monotone fraction: {summary['consistency']['oracle_stop_monotone_fraction']:.3f}",
        f"- Predicted monotone fraction: {summary['consistency']['predicted_stop_monotone_fraction']:.3f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
