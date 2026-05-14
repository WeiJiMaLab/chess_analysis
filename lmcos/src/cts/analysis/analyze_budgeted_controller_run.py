"""End-to-end diagnostics driver for a budgeted-controller training run.

Wires up the themed sub-modules under ``cts.analysis._budgeted/``: training
curves, regret breakdowns, calibration, oracle-stop drivers, root-action
churn, future-worse move switching, halt-reward trajectories, baselines,
and the Markdown report. Heavy / expensive analyses are gated behind
``--report-mode full`` so a quick "core" pass can run without the source
trees being locally available.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class AnalyzeBudgetedControllerRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostics_path: str
    log_path: str
    output_dir: str
    report_mode: str = "full"
    path_rewrite: list[str] = []
    max_state_rows: int = 500000
    state_sample_seed: int = 0


from cts.analysis._budgeted._shared import (
    _episode_error_type,
    _source_paths_available,
    _write_jsonl,
    _write_summary,
    build_state_rows,
    load_diagnostics,
    torch,
)
from cts.analysis._budgeted.baselines import _baseline_sweep
from cts.analysis._budgeted.calibration import (
    _calibration_by_margin,
    _plot_accuracy_by_oracle_stop_bin,
    _plot_false_action_rates_by_time,
    _plot_predicted_vs_target,
    _plot_sign_accuracy_by_time,
    _plot_target_distribution_by_time,
    _same_tree_budget_consistency,
)
from cts.analysis._budgeted.future_worse import (
    _best_move_sequences,
    _future_worse_move_switch_summary,
    _plot_future_worse_move_switch,
)
from cts.analysis._budgeted.halt_trajectories import (
    _plot_halt_reward_delta_trajectory_distribution,
    _plot_halt_reward_trajectory_distribution,
)
from cts.analysis._budgeted.oracle_drivers import (
    _error_episode_decomposition_records,
    _oracle_step_factor_rows,
    _oracle_stop_clean_factor_rows,
    _oracle_stop_clean_factor_summary,
    _oracle_stop_factor_summary,
    _oversearch_tail_step_records,
    _plot_error_episode_oracle_stop_factors,
    _plot_false_continue_by_clean_oracle_driver,
    _plot_false_continue_by_oracle_driver,
    _plot_oracle_stop_clean_driver_by_bucket,
    _plot_oracle_stop_clean_factor_magnitudes,
    _plot_oracle_stop_driver_by_bucket,
    _plot_oracle_stop_factor_magnitudes,
    _plot_oversearch_regret_components_by_bucket,
    _plot_oversearch_tail_components_by_extra_step,
    _summarize_error_episode_decomposition,
    _summarize_oversearch_tail,
)
from cts.analysis._budgeted.regret import (
    _oracle_stop_distribution,
    _plot_episode_regret_by_bucket,
    _plot_partial_dependence_heatmaps,
    _plot_regret_by_oracle_stop,
    _plot_regret_by_stop_step_delta,
    _plot_regret_by_tree_size,
    _plot_regret_decomposition_by_delta,
    _plot_stop_error_by_bucket,
    _regret_decomposition_summary,
    _stop_step_delta_summary,
)
from cts.analysis._budgeted.report import _write_markdown_report
from cts.analysis._budgeted.root_churn import (
    _move_trace_cache,
    _plot_root_action_churn_lengths,
    _plot_root_action_churn_top_level,
    _root_action_churn_records,
    _summarize_root_action_churn,
)
from cts.analysis._budgeted.training_curves import (
    _best_epoch_summary,
    _plot_greedy_metrics,
    _plot_loss_curves,
)
from cts.analysis._common import (
    TeacherSearchConfig,
    _analysis_quality_config,
    _apply_path_rewrites,
    _parse_path_rewrites,
    parse_log,
)
from cts.data.preprocess_mc.oracle import budgeted_oracle_config_from_metadata


def main(config: AnalyzeBudgetedControllerRunConfig) -> None:
    """CLI entry point: parse args, load inputs, run every plot/summary, write outputs.

    The "core" vs "full" report-mode switch gates the expensive analyses
    (oracle-stop factor sweeps, error-episode decomposition, halt-reward
    trajectory plots, root-action churn). Most of the core figures are
    cheap enough to always compute.
    """
    # ---- Load + remap inputs ---------------------------------------------
    diagnostics_path = Path(config.diagnostics_path)
    log_path = Path(config.log_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, train_rows, validation_rows, greedy_rows = parse_log(log_path)
    diagnostics = load_diagnostics(diagnostics_path)
    path_rewrites = _parse_path_rewrites(config.path_rewrite)
    if path_rewrites:
        # Apply --path-rewrite up front so every downstream analysis sees
        # the same (possibly local) source paths.
        diagnostics = [
            {
                **episode,
                "source_path": _apply_path_rewrites(str(episode["source_path"]), path_rewrites),
            }
            for episode in diagnostics
        ]
    state_rows, total_state_rows = build_state_rows(
        diagnostics,
        max_rows=config.max_state_rows,
        seed=config.state_sample_seed,
    )
    oracle_config = budgeted_oracle_config_from_metadata(metadata)
    if oracle_config is None:
        raise ValueError("Could not recover budgeted oracle metadata from the .out file.")
    include_extended = config.report_mode == "full"

    # ---- Core plots (always produced) ------------------------------------
    _plot_loss_curves(train_rows, validation_rows, output_dir / "advantage_loss.png")
    _plot_greedy_metrics(greedy_rows, output_dir / "greedy_metrics.png")
    regret_by_bucket = _plot_episode_regret_by_bucket(diagnostics, output_dir / "regret_by_budget_bucket.png")
    stop_by_bucket = _plot_stop_error_by_bucket(diagnostics, output_dir / "stop_error_by_budget_bucket.png")
    regret_by_oracle_stop = _plot_regret_by_oracle_stop(diagnostics, output_dir / "regret_by_oracle_stop.png")
    accuracy_by_oracle_stop_bin = _plot_accuracy_by_oracle_stop_bin(
        diagnostics,
        oracle_config,
        output_dir / "accuracy_by_oracle_stop_bin.png",
    )
    stop_step_delta = _stop_step_delta_summary(diagnostics)
    _plot_regret_by_stop_step_delta(diagnostics, output_dir / "regret_by_stop_step_delta.png")
    regret_decomposition = _regret_decomposition_summary(diagnostics, oracle_config)
    _plot_regret_decomposition_by_delta(diagnostics, oracle_config, output_dir / "regret_decomposition_by_stop_step_delta.png")

    # ---- Extended analyses (full mode only) ------------------------------
    # Pre-declare optional summary slots so the assembly at the bottom can
    # unconditionally reference them (None values JSON-serialize fine).
    oracle_stop_factors: dict[str, Any] | None = None
    oracle_stop_driver_by_bucket: dict[str, Any] = {}
    oracle_stop_factor_magnitudes: dict[str, Any] = {}
    false_continue_by_oracle_driver: dict[str, Any] = {}
    oracle_stop_clean_factors: dict[str, Any] | None = None
    oracle_stop_clean_driver_by_bucket: dict[str, Any] = {}
    oracle_stop_clean_factor_magnitudes: dict[str, Any] = {}
    false_continue_by_clean_oracle_driver: dict[str, Any] = {}
    oversearch_regret_components: dict[str, Any] | None = None
    future_worse_move_switch: dict[str, Any] | None = None
    error_episode_summary: dict[str, Any] | None = None
    oversearch_tail_summary: dict[str, Any] | None = None
    error_episode_oracle_stop_factors: dict[str, Any] = {}
    oversearch_tail_plot_summary: dict[str, Any] | None = None
    halt_reward_trajectories: dict[str, Any] | None = None
    halt_reward_delta_trajectories: dict[str, Any] | None = None
    root_action_churn: dict[str, Any] | None = None
    if include_extended:
        # Build the per-step factor rows once (expensive) and reuse them
        # across multiple summary/plot functions.
        oracle_step_rows = _oracle_step_factor_rows(diagnostics, oracle_config)
        oracle_step_rows_clean = _oracle_stop_clean_factor_rows(diagnostics, oracle_config)
        oracle_stop_factors = _oracle_stop_factor_summary(oracle_step_rows)
        oracle_stop_driver_by_bucket = _plot_oracle_stop_driver_by_bucket(
            oracle_step_rows,
            output_dir / "oracle_stop_driver_by_bucket.png",
        )
        oracle_stop_factor_magnitudes = _plot_oracle_stop_factor_magnitudes(
            oracle_step_rows,
            output_dir / "oracle_stop_factor_magnitudes.png",
        )
        false_continue_by_oracle_driver = _plot_false_continue_by_oracle_driver(
            oracle_step_rows,
            output_dir / "false_continue_by_oracle_driver.png",
        )
        oracle_stop_clean_factors = _oracle_stop_clean_factor_summary(oracle_step_rows_clean)
        oracle_stop_clean_driver_by_bucket = _plot_oracle_stop_clean_driver_by_bucket(
            oracle_step_rows_clean,
            output_dir / "oracle_stop_clean_driver_by_bucket.png",
        )
        oracle_stop_clean_factor_magnitudes = _plot_oracle_stop_clean_factor_magnitudes(
            oracle_step_rows_clean,
            output_dir / "oracle_stop_clean_factor_magnitudes.png",
        )
        false_continue_by_clean_oracle_driver = _plot_false_continue_by_clean_oracle_driver(
            oracle_step_rows_clean,
            output_dir / "false_continue_by_clean_oracle_driver.png",
        )
        oversearch_regret_components = _plot_oversearch_regret_components_by_bucket(
            diagnostics,
            oracle_config,
            output_dir / "oversearch_regret_components_by_bucket.png",
        )
        # Move-switch analysis requires re-running the teacher search on the
        # source trees, so skip when those files aren't locally available.
        if _source_paths_available(diagnostics) and TeacherSearchConfig is not None and torch is not None:
            quality_config = _analysis_quality_config(metadata)
            best_move_cache = _best_move_sequences(diagnostics, quality_config)
            future_worse_move_switch = _future_worse_move_switch_summary(
                diagnostics,
                oracle_step_rows,
                best_move_cache,
                oracle_config,
            )
            _plot_future_worse_move_switch(
                future_worse_move_switch,
                output_dir / "future_worse_move_switch.png",
            )
        error_episode_records = _error_episode_decomposition_records(diagnostics, oracle_config)
        oversearch_tail_records = _oversearch_tail_step_records(diagnostics, oracle_config)
        _write_jsonl(output_dir / "error_episode_decomposition.jsonl", error_episode_records)
        _write_jsonl(output_dir / "oversearch_tail_decomposition.jsonl", oversearch_tail_records)
        error_episode_summary = _summarize_error_episode_decomposition(error_episode_records)
        oversearch_tail_summary = _summarize_oversearch_tail(oversearch_tail_records)
        error_episode_oracle_stop_factors = _plot_error_episode_oracle_stop_factors(
            error_episode_records,
            output_dir / "error_episode_oracle_stop_factors.png",
        )
        oversearch_tail_plot_summary = _plot_oversearch_tail_components_by_extra_step(
            oversearch_tail_records,
            output_dir / "oversearch_tail_components_by_extra_step.png",
        )

    # ---- State-level plots (always produced) -----------------------------
    sign_by_time = _plot_sign_accuracy_by_time(state_rows, output_dir / "sign_accuracy_by_time_budget.png")
    target_by_time = _plot_target_distribution_by_time(state_rows, output_dir / "target_advantage_by_time_budget.png")
    pred_vs_target = _plot_predicted_vs_target(state_rows, output_dir / "predicted_vs_target_advantage.png")
    false_action_by_time = _plot_false_action_rates_by_time(state_rows, output_dir / "false_action_rates_by_time_budget.png")
    regret_by_tree_size = _plot_regret_by_tree_size(diagnostics, output_dir / "regret_by_initial_tree_size.png")
    partial_dependence = _plot_partial_dependence_heatmaps(state_rows, output_dir / "partial_dependence_heatmaps.png")
    if include_extended:
        halt_reward_trajectories = _plot_halt_reward_trajectory_distribution(
            diagnostics,
            output_dir / "halt_reward_trajectory_distribution.png",
        )
        halt_reward_delta_trajectories = _plot_halt_reward_delta_trajectory_distribution(
            diagnostics,
            output_dir / "halt_reward_delta_trajectory_distribution.png",
        )
    consistency = _same_tree_budget_consistency(diagnostics)
    oracle_stop_dist = _oracle_stop_distribution(diagnostics)
    calibration = _calibration_by_margin(state_rows, output_dir / "calibration_by_margin.png")
    best_epochs = _best_epoch_summary(validation_rows, greedy_rows)
    baselines = _baseline_sweep(diagnostics, oracle_config)
    # Root-action churn also needs the source trees; gate it the same way.
    if include_extended and _source_paths_available(diagnostics) and TeacherSearchConfig is not None and torch is not None:
        quality_config = _analysis_quality_config(metadata)
        move_trace_cache = _move_trace_cache(diagnostics, quality_config)
        root_action_churn_source_records, root_action_churn_episode_records = _root_action_churn_records(
            diagnostics,
            move_trace_cache,
        )
        _write_jsonl(output_dir / "root_action_churn_sources.jsonl", root_action_churn_source_records)
        _write_jsonl(output_dir / "root_action_churn_episodes.jsonl", root_action_churn_episode_records)
        root_action_churn = _summarize_root_action_churn(
            root_action_churn_source_records,
            root_action_churn_episode_records,
        )
        root_action_churn["top_level_plot"] = _plot_root_action_churn_top_level(
            root_action_churn_source_records,
            output_dir / "root_action_churn_top_level.png",
        )
        root_action_churn["length_plot"] = _plot_root_action_churn_lengths(
            root_action_churn_source_records,
            output_dir / "root_action_churn_lengths.png",
        )

    # ---- Summary assembly + write-out ------------------------------------
    episode_error_counts = Counter(_episode_error_type(episode) for episode in diagnostics)
    episode_error_regret = {
        error_type: statistics.mean(float(episode["regret"]) for episode in diagnostics if _episode_error_type(episode) == error_type)
        for error_type in sorted(episode_error_counts)
    }
    bucket_counts = Counter(episode["budget_bucket_name"] for episode in diagnostics)

    summary = {
        "episodes": len(diagnostics),
        "states": total_state_rows,
        "sampled_states": len(state_rows),
        "report_mode": config.report_mode,
        "bucket_counts": dict(bucket_counts),
        "final_greedy": greedy_rows[-1] if greedy_rows else None,
        "best_epochs": best_epochs,
        "oracle_metadata": metadata,
        "oracle_stop_distribution": oracle_stop_dist,
        "regret_by_budget_bucket": regret_by_bucket,
        "stop_error_by_budget_bucket": stop_by_bucket,
        "regret_by_oracle_stop_step": regret_by_oracle_stop,
        "accuracy_by_oracle_stop_bin": accuracy_by_oracle_stop_bin,
        "regret_by_stop_step_delta": stop_step_delta,
        "regret_decomposition": regret_decomposition,
        "oracle_stop_factors": oracle_stop_factors,
        "oracle_stop_driver_by_bucket": oracle_stop_driver_by_bucket,
        "oracle_stop_factor_magnitudes": oracle_stop_factor_magnitudes,
        "false_continue_by_oracle_driver": false_continue_by_oracle_driver,
        "oracle_stop_clean_factors": oracle_stop_clean_factors,
        "oracle_stop_clean_driver_by_bucket": oracle_stop_clean_driver_by_bucket,
        "oracle_stop_clean_factor_magnitudes": oracle_stop_clean_factor_magnitudes,
        "false_continue_by_clean_oracle_driver": false_continue_by_clean_oracle_driver,
        "oversearch_regret_components": oversearch_regret_components,
        "future_worse_move_switch": future_worse_move_switch,
        "error_episode_decomposition_summary": error_episode_summary,
        "oversearch_tail_decomposition_summary": oversearch_tail_summary,
        "error_episode_oracle_stop_factors": error_episode_oracle_stop_factors,
        "oversearch_tail_plot_summary": oversearch_tail_plot_summary,
        "sign_accuracy_by_time_budget": sign_by_time,
        "target_advantage_by_time_budget": target_by_time,
        "predicted_vs_target": pred_vs_target,
        "false_action_rates_by_time_budget": false_action_by_time,
        "regret_by_initial_tree_size": regret_by_tree_size,
        "partial_dependence": partial_dependence,
        "halt_reward_trajectory_distribution": halt_reward_trajectories,
        "halt_reward_delta_trajectory_distribution": halt_reward_delta_trajectories,
        "consistency": consistency,
        "root_action_churn": root_action_churn,
        "calibration_by_margin": calibration,
        "episode_error_counts": dict(episode_error_counts),
        "episode_error_regret": episode_error_regret,
        "baselines": baselines,
    }
    _write_summary(output_dir / "summary.json", summary)
    _write_markdown_report(output_dir / "report.md", summary)
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(AnalyzeBudgetedControllerRunConfig, main)
