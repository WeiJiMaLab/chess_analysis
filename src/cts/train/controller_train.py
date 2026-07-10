"""Train the budget-aware compute-advantage controller via fitted-Q regression.

This script is the controller-training stage of the CTS pipeline. It loads
packed episode shards (sequences of partial-tree snapshots with target
advantages) and, when the encoder is frozen, materializes the encoder once
into a feature cache keyed by ``[z_root, N_t, T_t]``. It then trains a small
MLP advantage head on those features with MSE + auxiliary sign BCE loss,
optionally reweighted by oracle stop-step frequency or a hand-tuned
non-trivial boost. The output is a controller checkpoint plus a JSONL of
per-episode greedy-policy diagnostics.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.utils.data import DataLoader, Dataset

from cts.core.schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts.core.tensorizer import TreeBatch
from cts.data.preprocess_mc.oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
    budgeted_oracle_metadata,
    compute_budgeted_oracle,
    predicted_stop_from_advantages,
    return_for_stop_step,
)
from cts.models.gnn import TreeEncoderOutput, TreeEncoder
from cts.models.mc import (
    CONTROLLER_INPUT_NAMES,
    MetaController,
    validate_controller_inputs,
)
from cts.train.gnn_pretrain import load_encoder_checkpoint


class ControllerTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packed_train_data: str
    packed_validation_data: str
    encoder_checkpoint: str
    output_checkpoint: Optional[str] = None
    # plan.md Agent 2 (our_trees_continued) -- warm-start the FULL model (encoder + head)
    # from a previously-trained checkpoint's ``model_state_dict``, instead of the usual
    # fresh-head-on-top-of-``encoder_checkpoint`` init. Loaded AFTER ``encoder_checkpoint``
    # in ``_build_model_and_optimizer`` (so it takes precedence), which means
    # ``encoder_checkpoint`` can safely stay pointed at the ORIGINAL base encoder --
    # ``resume_checkpoint``'s full state dict overwrites those weights anyway. Neither
    # ``pg_controller_train.py`` nor ``e2e_controller_train.py`` otherwise supports
    # continuing a run; this is the single shared hook both use.
    resume_checkpoint: Optional[str] = None
    # Dump a checkpoint EVERY epoch (not just on val-regret improvement), named
    # ``<output_checkpoint stem>_epoch{N:03d}<suffix>`` -- lets a continued run be
    # paired-significance-evaluated at several points along its curve, not just the
    # final/best epoch. Mirrors ``e2e_controller_train.py``'s existing per-epoch save
    # (that script always saves every epoch); ``pg_controller_train.py`` previously only
    # saved on improvement, so this flag opt-in extends it without changing default behavior.
    save_every_epoch: bool = False
    materialized_train_cache: Optional[str] = None
    materialized_validation_cache: Optional[str] = None
    materialized_cache_shard_snapshots: int = 250000
    device: str = "cpu"
    seed: int = 0
    k: int = 2
    node_embed_hidden: int = 128
    d_embed: int = 128
    d_message: int = 128
    n_heads: int = 4
    d_att: int = 32
    hidden_dim: int = 256
    hidden_layers: int = 3
    unfreeze_encoder: bool = False
    batch_size: int = 1024
    episode_batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    min_lr: float = 0.0
    weight_decay: float = 0.0
    sign_loss_weight: float = 1.0
    # PROTOTYPE — asymmetric over-search penalty on the sign-BCE. That loss treats the
    # predicted advantage as a logit for "continue"; pos_weight < 1 down-weights the
    # "continue" class, so FALSE-CONTINUE (predicting continue when the oracle says STOP)
    # is penalised more than false-stop. This pulls the advantage zero-crossing earlier,
    # directly countering the controller's chronic over-search. 1.0 = symmetric/original.
    # Checkpoint selection stays on validation regret, not this loss (the §10b rule).
    sign_pos_weight: float = 1.0
    nontrivial_loss_weight: float = 1.0
    # --- policy-gradient (exact expected-return) trainer: cts.train.pg_controller_train ---
    # Directly optimizes E[regret] under the stochastic stop policy (continue prob = sigmoid(A_t)),
    # with the stop step marginalized in closed form over the full trace (no REINFORCE sampling).
    pg_episode_batch: int = 1024        # episodes per PG gradient step
    pg_max_episodes: Optional[int] = None  # cap train/val episodes (smoke / quick runs)
    # Stop-policy temperature for the PG trainer: continue prob = sigmoid(A_t / tau).
    # tau > 1 keeps the soft policy AWAY from the 0/1 saturation boundary (where the
    # sigmoid slope p(1-p) -> 0 and the gradient dies), so it explores to the right
    # stop step before sharpening. Annealed linearly stop_temperature -> _final over
    # the epochs. tau does NOT affect the deployed hard greedy rule (sign(A_t)), so
    # checkpoint selection on hard val regret stays comparable. 1.0/1.0 = original.
    stop_temperature: float = 1.0
    stop_temperature_final: float = 1.0
    inverse_freq_weights: bool = False
    separate_sign_head: bool = False
    max_grad_norm: float = 1.0
    num_workers: int = 0
    log_interval: int = 25
    validation_interval: int = 1
    validation_step_interval: Optional[int] = None  # validate every N train batches; disables epoch validation_interval when set
    greedy_eval_step_interval: Optional[int] = None  # greedy eval every N train batches; defaults to validation_step_interval when unset
    greedy_eval_interval: int = 1
    output_diagnostics: Optional[str] = None
    metrics_path: Optional[str] = None
    metrics_run_name: Optional[str] = None  # plot title / comparison legend; default: output_checkpoint stem
    metrics_plot_path: Optional[str] = None  # default: <run>.png beside metrics YAML in the same stage dir
    plot_refresh_step_interval: Optional[int] = None  # default: validation_step_interval
    metrics_log_interval: int = 10
    train_batches: Optional[int] = None  # gradient steps per epoch; unset = full pass over train data
    max_validation_batches: Optional[int] = None  # rarely used; smoke leaves unset (full val cache)
    max_greedy_eval_episodes: Optional[int] = None  # rarely used; smoke leaves unset (full val episodes)
    maintenance_scale: float = 0.0
    maintenance_ref_nodes: float = 30.0
    maintenance_exponent: float = 1.1
    time_lambda: float = 18.537
    time_p: float = 2.8
    time_tau: float = 2.5
    time_delta: int = 1
    timeout_value: float = -1.0
    time_mode: str = "power_law"
    samples_per_bucket: int = 2
    fixed_budget: Optional[int] = None  # if set, every episode gets this one starting budget
                                         # (must match the packing-time value — see _oracle_config)
    scramble_min_time: int = 1
    scramble_max_time: int = 3
    medium_small_min_time: int = 4
    medium_small_max_time: int = 10
    medium_large_min_time: int = 11
    medium_large_max_time: int = 25
    large_min_time: int = 26
    large_max_time: int = 60
    very_large_min_time: int = 61
    very_large_max_time: int = 120
    # Which canonical inputs the advantage head consumes. Order in the list is
    # not significant; the model concatenates the selected features in
    # canonical ``(z_t, N_t, T_t)`` order regardless. Default keeps the
    # pre-refactor behavior (all three inputs).
    controller_inputs: List[str] = Field(default_factory=lambda: list(CONTROLLER_INPUT_NAMES))

    @field_validator("controller_inputs")
    @classmethod
    def _validate_controller_inputs(cls, value: List[str]) -> List[str]:
        return list(validate_controller_inputs(value))


@dataclass(frozen=True)
class ControllerBatch:
    """One collated minibatch flowing through the live encoder training path.

    Holds the packed ``TreeBatch`` for the encoder plus all per-snapshot
    scalars and per-episode metadata the loss/eval code needs downstream.
    """

    tree_batch: TreeBatch  # packed (multi-tree) batch consumed by TreeEncoder
    target_advantages: torch.Tensor  # [num_snapshots] regression targets (one per controller step)
    tree_sizes: torch.Tensor  # [num_snapshots] N_t scalar fed into the advantage head
    time_budgets: torch.Tensor  # [num_snapshots] T_t scalar fed into the advantage head
    paths: List[str]  # episode keys (one per episode in this batch)
    source_paths: List[str]  # underlying trajectory source paths (one per episode)
    path_lengths: List[int]  # number of snapshots per episode; used to repeat per-episode scalars to per-snapshot
    halt_rewards: List[List[float]]  # per-episode halt reward sequence
    oracle_stop_steps: List[int]  # per-episode oracle-optimal stop step (eval / loss weighting)
    oracle_values: List[float]  # per-episode oracle-optimal return (eval baseline for regret)
    starting_budgets: List[int]  # per-episode initial time budget T_0
    budget_bucket_names: List[str]  # per-episode bucket label (scramble / medium-small / ...)


@dataclass(frozen=True)
class EpisodeMetadata:
    """Per-episode metadata extracted from packed shards without rebuilding trees.

    Used by the greedy policy evaluator, which only needs scalars per step
    and never reconstructs the encoder input — the predicted advantages
    come from the materialized cache.
    """

    path: str  # episode key
    source_path: str  # trajectory source path
    halt_rewards: List[float]  # per-step halt reward
    tree_sizes: List[int]  # per-step N_t
    time_budgets: List[int]  # per-step T_t
    target_advantages: List[float]  # per-step regression target
    oracle_stop_step: int  # oracle-optimal stop step for the episode
    oracle_value: float  # oracle-optimal return for the episode
    starting_budget: int  # initial T_0 budget
    budget_bucket_name: str  # bucket label for grouping in diagnostics
    num_steps: int  # episode length in controller steps


@dataclass(frozen=True)
class MaterializedCache:
    """Sharded on-disk cache of encoder outputs (``[z_root, N_t, T_t]``).

    Computed once per (manifest, encoder checkpoint) pair so that frozen-
    encoder training can skip the encoder forward pass entirely on every
    epoch. Each shard stores its own contiguous block of snapshots.
    """

    shard_paths: List[str]  # absolute paths to per-shard .pt files
    shard_sizes: List[int]  # number of snapshots in each shard (parallel to shard_paths)
    examples: int  # total snapshots across all shards


def _load_materialized_cache_unchecked(cache_index_path) -> "MaterializedCache":
    """Load a materialized ``z_t`` cache index (shard list) WITHOUT the encoder-identity check that
    :func:`_load_materialized_cache` enforces.

    Used when scoring an already-trained checkpoint over a cache built for the SAME encoder (identity
    holds by construction — same encoder produced both), so only the shard list is needed. Imported by
    :mod:`cts.train.pg_controller_train`, :mod:`cts.analysis.zt_probe` and :mod:`analysis.evaluate`.
    """
    payload = torch.load(cache_index_path, weights_only=False)
    if payload.get("format") != "cts_materialized_advantage_cache_v2":
        raise ValueError(f"Unexpected materialized cache format: {cache_index_path}")
    shards = payload.get("shards", [])
    return MaterializedCache(shard_paths=[str(e["path"]) for e in shards],
                             shard_sizes=[int(e["examples"]) for e in shards],
                             examples=int(payload.get("examples", sum(int(e["examples"]) for e in shards))))


# Oracle stop-step bucket boundaries used by --inverse-freq-weights. Bins are
# right-open intervals against torch.bucketize: 0, 1, 2, 3, [4,8), [8,16), 16+.
# These match the distribution buckets the dataset typically exhibits.
ORACLE_STOP_BIN_BOUNDARIES = torch.tensor([1, 2, 3, 4, 8, 16])
ORACLE_STOP_NUM_BINS = len(ORACLE_STOP_BIN_BOUNDARIES) + 1  # 7


def _compute_inverse_freq_bin_weights(
    cache: MaterializedCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-bin inverse frequency weights from cached oracle_stop_steps.

    Bins: 0, 1, 2, 3, 4-7, 8-15, 16+
    Returns (bin_weights, bin_boundaries) where bin_weights[i] is the loss weight
    for bin i, normalized so the expected weight per snapshot is 1.
    """
    # Histogram oracle stop steps across all cache shards. Loading each shard
    # eagerly here is fine — this runs once at startup, not per step.
    bin_counts = torch.zeros(ORACLE_STOP_NUM_BINS, dtype=torch.long)
    for shard_path in cache.shard_paths:
        payload = torch.load(shard_path, weights_only=False)
        oracle_steps = payload.get("oracle_stop_steps")
        if oracle_steps is None:
            raise ValueError(
                "Materialized cache missing oracle_stop_steps. "
                "Delete the cache and re-materialize to use --inverse-freq-weights."
            )
        bins = torch.bucketize(oracle_steps, ORACLE_STOP_BIN_BOUNDARIES)
        bin_counts += torch.bincount(bins, minlength=ORACLE_STOP_NUM_BINS)
    # Normalize so the expected weight per snapshot is 1 (i.e. weights sum to
    # ORACLE_STOP_NUM_BINS in inverse proportion to count). clamp_min(1) keeps
    # empty bins from blowing up the divide.
    total = bin_counts.sum().float()
    bin_counts_safe = bin_counts.float().clamp(min=1)
    bin_weights = total / (ORACLE_STOP_NUM_BINS * bin_counts_safe)
    bin_names = ["0", "1", "2", "3", "4-7", "8-15", "16+"]
    for name, count, weight in zip(bin_names, bin_counts.tolist(), bin_weights.tolist()):
        print(f"  bin={name:>4s}  count={count:>8d}  weight={weight:.3f}", flush=True)
    return bin_weights, ORACLE_STOP_BIN_BOUNDARIES


@dataclass(frozen=True)
class AdvantageMetrics:
    """Aggregated regression metrics for one train/eval epoch."""

    total_loss: float  # advantage_mse + sign_loss_weight * sign_bce, averaged over snapshots
    advantage_mse: float  # MSE between predicted and target advantages
    sign_bce: float  # binary cross-entropy on sign(target_advantage)
    mean_abs_advantage_error: float  # MAE between predicted and target advantages
    sign_accuracy: float  # fraction of snapshots where predicted sign matches target sign
    examples: int  # number of snapshots aggregated


@dataclass(frozen=True)
class GreedyPolicyMetrics:
    """Aggregated rollout metrics from the greedy stop-when-advantage<=0 policy."""

    exact_stop_step_accuracy: float  # fraction of episodes where predicted stop step == oracle stop step
    first_action_accuracy: float  # fraction where (predicted_stop == 0) matches (oracle_stop == 0)
    average_return: float  # mean greedy-policy return across episodes
    average_oracle_value: float  # mean oracle-optimal return across episodes
    average_regret: float  # mean (oracle_value - greedy_return); the primary checkpoint-selection criterion
    average_expansions: float  # mean predicted stop step (number of expansions taken)
    evaluated_episodes: int  # number of episodes evaluated


def _advantage_metrics_dict(metrics: AdvantageMetrics) -> Dict[str, float | int]:
    return {
        "total_loss": metrics.total_loss,
        "advantage_mse": metrics.advantage_mse,
        "sign_bce": metrics.sign_bce,
        "mean_abs_advantage_error": metrics.mean_abs_advantage_error,
        "sign_accuracy": metrics.sign_accuracy,
        "snapshots": metrics.examples,
    }


def _greedy_metrics_dict(metrics: GreedyPolicyMetrics) -> Dict[str, float | int]:
    return {
        "exact_stop_step_accuracy": metrics.exact_stop_step_accuracy,
        "first_action_accuracy": metrics.first_action_accuracy,
        "average_return": metrics.average_return,
        "average_oracle_value": metrics.average_oracle_value,
        "average_regret": metrics.average_regret,
        "average_expansions": metrics.average_expansions,
        "evaluated_episodes": metrics.evaluated_episodes,
    }


def _load_metrics_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"batches": []}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file {path} must contain a YAML mapping at the top level.")
    data.setdefault("batches", [])
    return data


def _save_metrics_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _plot_title_from_run_start(run_start: dict[str, Any] | None) -> str | None:
    if not run_start:
        return None
    if run_start.get("run_name"):
        return str(run_start["run_name"])
    output_checkpoint = run_start.get("output_checkpoint")
    if output_checkpoint:
        return Path(str(output_checkpoint)).stem
    return None


def _style_plot_axes(ax: Any, *, legend: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    if legend:
        ax.legend(frameon=False)


_PLOT_LINE_STYLE = {"linewidth": 1.5, "markersize": 3}

_COMPARISON_MODEL_COLORS = (
    "#2171b5",
    "#525252",
    "#6baed6",
    "#969696",
    "#08519c",
    "#bdbdbd",
)


def _metric_series(
    batch_rows: list[dict[str, Any]],
    section: str,
    field: str,
) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in batch_rows:
        payload = row.get(section)
        if payload is None:
            continue
        xs.append(int(row["n_batches"]))
        ys.append(float(payload[field]))
    return xs, ys


def _default_metrics_plot_path(metrics_path: Path) -> Path:
    """Same stage dir: ``<run>.yaml`` → ``<run>.png``."""
    metrics_path = Path(metrics_path)
    if metrics_path.suffix in {".yaml", ".yml"}:
        return metrics_path.with_suffix(".png")
    return metrics_path.with_name("training_curves.png")


@dataclass
class ControllerTrainMetricsLogger:
    """Single sink for controller training metrics: YAML log + curve plot refresh."""

    metrics_path: Path
    plot_path: Path
    metrics_log_interval: int = 10
    plot_refresh_step_interval: int | None = None
    epochs: int = 1
    title: str | None = None
    run_name: str | None = None
    n_batches: int = 0

    def _append_batch_record(self, epoch: int, record: Dict[str, Any]) -> None:
        payload: Dict[str, Any] = {
            "n_batches": self.n_batches,
            "epoch": epoch,
            "epochs": self.epochs,
        }
        payload.update(record)
        data = _load_metrics_file(self.metrics_path)
        data["batches"].append(payload)
        _save_metrics_file(self.metrics_path, data)

    def write_run_start(
        self,
        config: ControllerTrainConfig,
        oracle_config: BudgetedOracleConfig,
        *,
        train_batches_per_epoch: int,
        full_train_batches_per_epoch: int | None,
    ) -> None:
        _save_metrics_file(
            self.metrics_path,
            {
                "run_start": {
                    "run_name": self.run_name,
                    "packed_train_data": config.packed_train_data,
                    "packed_validation_data": config.packed_validation_data,
                    "encoder_checkpoint": config.encoder_checkpoint,
                    "output_checkpoint": config.output_checkpoint,
                    "controller_inputs": list(config.controller_inputs),
                    "device": config.device,
                    "epochs": config.epochs,
                    "train_batches_per_epoch": train_batches_per_epoch,
                    "full_train_batches_per_epoch": full_train_batches_per_epoch,
                    "train_batches": config.train_batches,
                    "validation_step_interval": config.validation_step_interval,
                    "greedy_eval_step_interval": config.greedy_eval_step_interval,
                    "max_validation_batches": config.max_validation_batches,
                    "max_greedy_eval_episodes": config.max_greedy_eval_episodes,
                    "metrics_log_interval": config.metrics_log_interval,
                    **budgeted_oracle_metadata(oracle_config),
                },
                "batches": [],
            },
        )

    def after_train_batch(
        self,
        epoch: int,
        *,
        total_loss_sum: float,
        total_advantage_mse: float,
        total_sign_bce: float,
        total_mean_abs_advantage_error: float,
        total_correct: int,
        total_examples: int,
    ) -> None:
        self.n_batches += 1
        interval = self.metrics_log_interval
        if interval <= 0 or self.n_batches % interval != 0:
            return
        examples = max(total_examples, 1)
        self._append_batch_record(
            epoch,
            {
                "train": _advantage_metrics_dict(
                    AdvantageMetrics(
                        total_loss=total_loss_sum / examples,
                        advantage_mse=total_advantage_mse / examples,
                        sign_bce=total_sign_bce / examples,
                        mean_abs_advantage_error=total_mean_abs_advantage_error / examples,
                        sign_accuracy=total_correct / examples,
                        examples=total_examples,
                    )
                ),
            },
        )

    def after_epoch_eval(
        self,
        epoch: int,
        validation_metrics: AdvantageMetrics | None,
        greedy_metrics: GreedyPolicyMetrics | None,
    ) -> None:
        if validation_metrics is None and greedy_metrics is None:
            return
        record: Dict[str, Any] = {}
        if validation_metrics is not None:
            record["validation"] = _advantage_metrics_dict(validation_metrics)
        if greedy_metrics is not None:
            record["greedy"] = _greedy_metrics_dict(greedy_metrics)
        self._append_batch_record(epoch, record)
        self.maybe_refresh_plot()

    def finish(self) -> None:
        self.refresh_plot(force=True)

    @staticmethod
    def load_metrics(metrics_path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Parse metrics YAML into ``(run_start, batch_rows)``."""
        if not metrics_path.exists():
            return None, []
        data = _load_metrics_file(metrics_path)
        run_start = data.get("run_start")
        batch_rows = list(data.get("batches") or [])
        batch_rows.sort(key=lambda row: int(row["n_batches"]))
        return run_start, batch_rows

    @classmethod
    def from_config(cls, config: ControllerTrainConfig) -> ControllerTrainMetricsLogger | None:
        if not config.metrics_path:
            return None
        metrics_path = Path(config.metrics_path)
        plot_path = Path(config.metrics_plot_path or _default_metrics_plot_path(metrics_path))
        plot_interval = config.plot_refresh_step_interval
        if plot_interval is None:
            plot_interval = config.validation_step_interval
        run_name = (
            config.metrics_run_name
            or (Path(config.output_checkpoint).stem if config.output_checkpoint else metrics_path.stem)
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            metrics_path=metrics_path,
            plot_path=plot_path,
            metrics_log_interval=config.metrics_log_interval,
            plot_refresh_step_interval=plot_interval,
            epochs=config.epochs,
            run_name=run_name,
        )

    @classmethod
    def from_metrics_path(
        cls,
        metrics_path: Path | str,
        *,
        output_path: Path | str | None = None,
        title: str | None = None,
    ) -> ControllerTrainMetricsLogger:
        metrics_path = Path(metrics_path)
        plot_path = Path(output_path) if output_path is not None else _default_metrics_plot_path(metrics_path)
        return cls(metrics_path=metrics_path, plot_path=plot_path, title=title)

    def refresh_plot(self, *, force: bool = False) -> bool:
        """Rewrite ``plot_path`` from the current ``metrics_path`` contents."""
        run_start, batch_rows = self.load_metrics(self.metrics_path)
        if not batch_rows:
            if force:
                raise ValueError(f"No batch records found in {self.metrics_path}")
            return False

        import matplotlib.pyplot as plt

        train_x = [int(row["n_batches"]) for row in batch_rows if row.get("train") is not None]
        train_loss = [float(row["train"]["total_loss"]) for row in batch_rows if row.get("train") is not None]
        val_x = [int(row["n_batches"]) for row in batch_rows if row.get("validation") is not None]
        validation_loss = [
            float(row["validation"]["total_loss"]) for row in batch_rows if row.get("validation") is not None
        ]
        regret_x = [int(row["n_batches"]) for row in batch_rows if row.get("greedy") is not None]
        greedy_regret = [
            float(row["greedy"]["average_regret"]) for row in batch_rows if row.get("greedy") is not None
        ]

        plot_title = self.title or _plot_title_from_run_start(run_start) or self.run_name or "Controller training"
        line_style = _PLOT_LINE_STYLE

        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax_loss, ax_regret) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

        if train_x:
            ax_loss.plot(
                train_x,
                train_loss,
                marker="o",
                color="#9ecae1",
                label="Train loss",
                **line_style,
            )
        if val_x:
            ax_loss.plot(
                val_x,
                validation_loss,
                marker="o",
                color="#2171b5",
                label="Val loss",
                **line_style,
            )
        ax_loss.set_ylabel("Total loss")
        ax_loss.set_title(plot_title)
        _style_plot_axes(ax_loss, legend=bool(train_x or val_x))

        if regret_x:
            ax_regret.plot(
                regret_x,
                greedy_regret,
                marker="o",
                color="black",
                **line_style,
            )
            ax_regret.set_ylabel("Average regret")
        else:
            ax_regret.text(
                0.5,
                0.5,
                "No greedy eval records",
                ha="center",
                va="center",
                transform=ax_regret.transAxes,
            )
        ax_regret.set_xlabel("Training batches (n_batches)")
        _style_plot_axes(ax_regret, legend=False)

        fig.savefig(self.plot_path, dpi=180)
        plt.close(fig)
        print(f"[controller_train_metrics_logger] wrote {self.plot_path}", flush=True)
        return True

    def maybe_refresh_plot(self) -> bool:
        interval = self.plot_refresh_step_interval
        if interval is None or interval <= 0 or self.n_batches % interval != 0:
            return False
        if not self.metrics_path.exists():
            return False
        return self.refresh_plot()

    @staticmethod
    def plot_comparison(
        metrics_paths: Sequence[Path | str],
        output_path: Path | str,
        *,
        title: str | None = None,
    ) -> None:
        """Overlay val loss (top) and average regret (bottom) across runs."""
        import matplotlib.pyplot as plt

        labeled_runs: list[tuple[str, list[dict[str, Any]]]] = []
        for metrics_path in metrics_paths:
            path = Path(metrics_path)
            run_start, batch_rows = ControllerTrainMetricsLogger.load_metrics(path)
            if not batch_rows:
                raise ValueError(f"No batch records found in {path}")
            label = _plot_title_from_run_start(run_start) or path.parent.name
            labeled_runs.append((label, batch_rows))

        fig, (ax_val, ax_regret) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
        has_val = False
        has_regret = False
        for index, (label, batch_rows) in enumerate(labeled_runs):
            color = _COMPARISON_MODEL_COLORS[index % len(_COMPARISON_MODEL_COLORS)]
            val_x, val_y = _metric_series(batch_rows, "validation", "total_loss")
            regret_x, regret_y = _metric_series(batch_rows, "greedy", "average_regret")
            if val_x:
                has_val = True
                ax_val.plot(val_x, val_y, marker="o", color=color, label=label, **_PLOT_LINE_STYLE)
            if regret_x:
                has_regret = True
                ax_regret.plot(regret_x, regret_y, marker="o", color=color, label=label, **_PLOT_LINE_STYLE)

        ax_val.set_ylabel("Val loss")
        ax_val.set_title(title or "Controller comparison")
        if has_val:
            _style_plot_axes(ax_val, legend=True)
        else:
            ax_val.text(
                0.5,
                0.5,
                "No validation records",
                ha="center",
                va="center",
                transform=ax_val.transAxes,
            )
            _style_plot_axes(ax_val, legend=False)

        if has_regret:
            ax_regret.set_ylabel("Average regret")
            _style_plot_axes(ax_regret, legend=True)
        else:
            ax_regret.text(
                0.5,
                0.5,
                "No greedy eval records",
                ha="center",
                va="center",
                transform=ax_regret.transAxes,
            )
            _style_plot_axes(ax_regret, legend=False)
        ax_regret.set_xlabel("Training batches (n_batches)")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(f"[controller_train_metrics_logger] wrote {output}", flush=True)


@dataclass(frozen=True)
class ControllerEpisode:
    """One episode unpacked from a v4 shard, with per-step encoder inputs.

    Each ``step_*`` list has one entry per controller step and contains the
    tensors needed to reconstruct that step's partial tree (the prefix of
    the trajectory's nodes/edges visible up to and including that step).
    """

    path: str  # episode key
    source_path: str  # underlying trajectory source path
    step_node_features: List[torch.Tensor]  # per-step [N_t, F] node features (prefix-truncated)
    step_parent_index: List[torch.Tensor]  # per-step [N_t] parent indices for the encoder
    step_edge_parent: List[torch.Tensor]  # per-step [E_t] edge parent ids (already filtered to active expansions)
    step_edge_child: List[torch.Tensor]  # per-step [E_t] edge child ids
    step_edge_slot: List[torch.Tensor]  # per-step [E_t] child slot indices
    step_depth: List[torch.Tensor]  # per-step [N_t] node depths
    halt_rewards: List[float]  # per-step halt reward
    target_advantages: torch.Tensor  # [num_steps] regression target
    tree_sizes: torch.Tensor  # [num_steps] N_t
    time_budgets: torch.Tensor  # [num_steps] T_t = starting_budget - step
    oracle_stop_step: int  # oracle-optimal stop step for the episode
    oracle_value: float  # oracle-optimal return for the episode
    starting_budget: int  # T_0
    budget_bucket_name: str  # bucket label


class ControllerEpisodeDataset(Dataset):
    """Index into the v4 packed-episode manifest, lazily loading shards.

    The manifest tells us how many episodes live in each shard; we keep a
    cumulative-size index and rebuild the per-step encoder inputs from the
    shard's compressed (trajectory + ptr) layout on demand. The most-recent
    shard is cached in memory so consecutive ``__getitem__`` calls within a
    shard don't pay the torch.load cost.
    """

    def __init__(self, manifest_path: str) -> None:
        """Parse the manifest and build the cumulative-size index.

        Args:
            manifest_path: path to a ``cts_budgeted_controller_episode_manifest_v4`` JSON file.
        """
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("format") != "cts_budgeted_controller_episode_manifest_v4":
            raise ValueError(f"Unexpected manifest format: {manifest_path}")

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed shard entries in manifest: {manifest_path}")

        # cumulative_sizes[i] is the total number of episodes in shards 0..i,
        # which lets bisect map a global episode index to (shard, offset) in O(log n).
        self.shard_paths: List[str] = []
        self.shard_num_episodes: List[int] = []
        self.cumulative_sizes: List[int] = []
        total = 0
        for entry in entries:
            num_episodes = int(entry["num_episodes"])
            if num_episodes <= 0:
                continue
            total += num_episodes
            self.shard_paths.append(entry["path"])
            self.shard_num_episodes.append(num_episodes)
            self.cumulative_sizes.append(total)
        if not self.shard_paths:
            raise ValueError(f"All shards empty in manifest: {manifest_path}")

        # Single-shard memo so we don't re-load on every consecutive __getitem__.
        self._loaded_shard_index: Optional[int] = None
        self._loaded_payload: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    # Shard formats this dataset can read: the original one-shot-frozen-value
    # format, and packhistory_trees' (Task 1's) format, which adds the sparse
    # per-edge update log (see history.md "Stage: packhistory_trees"). Both are
    # accepted so this dataset keeps working against pre-Task-1 packed data too.
    _SUPPORTED_SHARD_FORMATS = (
        "cts_budgeted_controller_episode_shard_v4",
        "cts_packhistory_trees_shard_v1",
    )

    def _load_shard(self, shard_index: int) -> Dict[str, Any]:
        """Return the shard payload, loading from disk only on a cache miss."""
        if self._loaded_shard_index != shard_index:
            payload = torch.load(self.shard_paths[shard_index], weights_only=False)
            if payload.get("format") not in self._SUPPORTED_SHARD_FORMATS:
                raise ValueError(f"Unexpected shard format: {self.shard_paths[shard_index]}")
            self._loaded_shard_index = shard_index
            self._loaded_payload = payload
        assert self._loaded_payload is not None
        return self._loaded_payload

    def __getitem__(self, index: int) -> ControllerEpisode:
        """Materialize per-step encoder inputs for one episode.

        The v4 shard stores each *trajectory* once and lets multiple episodes
        share it (one trajectory, many starting budgets). We slice into the
        trajectory's nodes/edges/child-pointers and rebuild a per-step prefix
        view (truncated by ``step_node_cutoffs`` + active expansion list) for
        every step in the episode.
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        # Locate the shard and the episode's offset within it.
        shard_index = bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        episode_offset = index - shard_start
        payload = self._load_shard(shard_index)

        # --- Episode-level slices into the shard's flat arrays ---
        episode_step_ptr = payload["episode_step_ptr"]
        episode_step_begin = int(episode_step_ptr[episode_offset].item())
        episode_step_end = int(episode_step_ptr[episode_offset + 1].item())
        num_steps = episode_step_end - episode_step_begin

        episode_trajectory_index = payload["episode_trajectory_index"]
        trajectory_index = int(episode_trajectory_index[episode_offset].item())

        # --- Trajectory-level slices (shared by all episodes from this trajectory) ---
        trajectory_node_ptr = payload["trajectory_node_ptr"]
        node_begin = int(trajectory_node_ptr[trajectory_index].item())
        node_end = int(trajectory_node_ptr[trajectory_index + 1].item())
        full_node_features = payload["node_features"][node_begin:node_end]
        full_parent_index = payload["parent_index"][node_begin:node_end]
        full_depth = payload["depth"][node_begin:node_end]

        # --- history.md "Stage: packhistory_MCmaterialize" (Task 5) ---
        # ``full_node_features`` above is the pre-existing one-shot, frozen
        # tensorization (each node's own raw leaf eval, baked once from the
        # complete tree — see history.md "What's wrong with the current code").
        # If this shard was produced by the new packhistory_trees stage (Task 1),
        # it also carries a sparse per-edge update log recording each node's real
        # backed-up value/WDL as backprop revised it during search (schema fixed
        # in history.md "Stage: packhistory_trees" > "Update log schema":
        # step_index/node_id/visit_count/q_value/wdl[3], sorted by
        # (node_id, step_index), node ids LOCAL to this trajectory — same
        # convention as ``parent_index``/``edge_child`` above).
        #
        # Field names below (``update_log_step_index``, ``update_log_node_id``,
        # ``update_log_q_value``, ``update_log_wdl``, ``trajectory_update_log_ptr``)
        # were this task's own guess, made before Task 1 landed, mirroring the
        # existing ``trajectory_edge_ptr`` / ``trajectory_child_ptr_ptr``
        # pointer-array convention used elsewhere in this same method.
        # Reconciled 2026-07-10 against Task 1's actual, landed implementation
        # (preprocess_mc/pack.py's ``_serialize_shard_payload``): every one of
        # these field names matches exactly — no renaming needed. (Task 1 also
        # ships ``update_log_visit_count``, ``node_update_ptr``, and
        # ``trajectory_node_update_ptr_ptr`` for CSR point-lookups, which this
        # method doesn't need — its access pattern is a single forward sweep
        # over steps, not random-access per-node queries — see
        # ``packhistory_GNNpretrain``/``pack_history.py`` for where those are
        # actually used.) The step-index convention was NOT a correct guess —
        # see the fixed off-by-one in the per-step loop below and the Task 5
        # Progress Log entry in history.md for the reconciliation. Falls back
        # to the old frozen-slice behavior below when these fields are absent
        # (older shard format, or Task 1 not yet run),
        # so this dataset keeps working against today's packed data too.
        update_log_step_index_all = payload.get("update_log_step_index")
        has_update_log = update_log_step_index_all is not None
        if has_update_log:
            trajectory_update_log_ptr = payload["trajectory_update_log_ptr"]
            log_begin = int(trajectory_update_log_ptr[trajectory_index].item())
            log_end = int(trajectory_update_log_ptr[trajectory_index + 1].item())
            log_step_index = update_log_step_index_all[log_begin:log_end]
            log_node_id = payload["update_log_node_id"][log_begin:log_end]
            log_q_value = payload["update_log_q_value"][log_begin:log_end]
            log_wdl = payload["update_log_wdl"][log_begin:log_end]
            # The packed log is sorted by (node_id, step_index) — the layout
            # that gives O(log M) per-node lookup, per history.md — not by
            # step_index alone. Re-sort by step_index (stable, so entries that
            # share a step keep their node_id-ascending relative order) so the
            # per-step loop below can apply updates with one linear sweep
            # instead of re-scanning per node at every step.
            step_order = torch.argsort(log_step_index, stable=True)
            log_step_index = log_step_index[step_order].tolist()
            log_node_id = log_node_id[step_order].tolist()
            log_q_value = log_q_value[step_order]
            log_wdl = log_wdl[step_order]
            num_log_entries = len(log_step_index)
            log_cursor = 0
            # Running per-node state, forward-filled step by step. Starts from
            # the same baseline the old code used unconditionally (each node's
            # pre-backprop static value) and is overwritten in place as log
            # entries are applied — this IS the fix: node ROWS now carry
            # step-accurate values, not just a longer/shorter prefix of one
            # static tensor. Column layout is TREE_ENCODER_FEATURE_NAMES =
            # ("value", "wdl_win", "wdl_draw", "wdl_loss", "wdl_var")
            # (cts/core/schema.py); wdl_var (column 4) has no corresponding
            # field in the update-log schema and is left at its baseline value
            # throughout — variance-over-visits isn't tracked by replay.
            running_node_features = full_node_features.clone()

        trajectory_edge_ptr = payload["trajectory_edge_ptr"]
        edge_begin = int(trajectory_edge_ptr[trajectory_index].item())
        edge_end = int(trajectory_edge_ptr[trajectory_index + 1].item())
        full_edge_child = payload["edge_child"][edge_begin:edge_end]
        full_edge_slot = payload["edge_slot"][edge_begin:edge_end]

        # child_ptr is per-parent CSR-style; one pointer per node in the
        # trajectory's full tree, telling us where that parent's child edges
        # begin/end in full_edge_child.
        trajectory_child_ptr_ptr = payload["trajectory_child_ptr_ptr"]
        child_ptr_begin = int(trajectory_child_ptr_ptr[trajectory_index].item())
        child_ptr_end = int(trajectory_child_ptr_ptr[trajectory_index + 1].item())
        full_child_ptr = payload["child_ptr"][child_ptr_begin:child_ptr_end]

        # Expansion sequence: which parent nodes got expanded in which order
        # during the original tree-search run. Used to gate edges per step.
        trajectory_expansion_parent_ptr = payload["trajectory_expansion_parent_ptr"]
        expansion_parent_begin = int(trajectory_expansion_parent_ptr[trajectory_index].item())
        expansion_parent_end = int(trajectory_expansion_parent_ptr[trajectory_index + 1].item())
        expansion_parent_ids = payload["expansion_parent_ids"][expansion_parent_begin:expansion_parent_end]

        trajectory_step_ptr = payload["trajectory_step_ptr"]
        trajectory_step_begin = int(trajectory_step_ptr[trajectory_index].item())
        trajectory_step_end = int(trajectory_step_ptr[trajectory_index + 1].item())
        step_node_cutoffs = payload["step_node_cutoffs"][trajectory_step_begin:trajectory_step_end]
        full_halt_rewards = payload["trajectory_halt_rewards"][trajectory_step_begin:trajectory_step_end]
        first_decision_expansion_count = int(payload["first_decision_expansion_counts"][trajectory_index].item())

        # --- Rebuild per-step encoder inputs ---
        # For each step we want the prefix of the tree that existed at that
        # decision point: first `node_cutoff` nodes, and only the edges whose
        # parent was among the first `expansion_count` expanded nodes.
        step_nf = []
        step_pi = []
        step_ep = []
        step_ec = []
        step_es = []
        step_d = []

        for local_step in range(num_steps):
            node_cutoff = int(step_node_cutoffs[local_step].item())
            expansion_count = first_decision_expansion_count + local_step

            if has_update_log:
                # Advance the log cursor to absorb every update whose
                # step_index has already happened as of this step (cumulative
                # values, per schema — a later entry for the same node simply
                # overwrites, no accumulation needed here).
                #
                # Reconciled against Task 1's actual implementation
                # (preprocess_mc/pack.py, ``_replay_backprop_history`` /
                # ``_build_compact_trajectory``): ``step_index`` is the
                # 0-indexed position of an expansion event within
                # ``expansion_parent_ids`` (tagged BEFORE increment — a log row
                # with ``step_index == k`` reflects the state immediately after
                # the ``(k+1)``-th real expansion). ``step_node_cutoffs[local_step]``
                # (this method's structural cutoff) is built the same way: it
                # reflects the tree after processing
                # ``expansion_parent_ids[0 : first_decision_expansion_count + local_step]``,
                # i.e. after the expansion event at 0-indexed position
                # ``expansion_count - 1`` (== ``first_decision_expansion_count
                # + local_step - 1``) was just processed. So the update-log
                # query that matches this step's structural cutoff must use
                # ``step_index <= expansion_count - 1``, not ``<= expansion_count``
                # — this is the exact same before/after-increment off-by-one
                # Task 4 found and fixed in ``pack_history.py:_build_tree_n``
                # (see history.md, Task 4 Progress Log, "Off-by-one fixed").
                # Using ``<= expansion_count`` here would leak one extra
                # expansion's worth of backprop into this step's values —
                # confirmed concretely against real data (see Task 5 Progress
                # Log entry for 2026-07-10, reconciliation pass).
                while log_cursor < num_log_entries and log_step_index[log_cursor] <= expansion_count - 1:
                    node_id = log_node_id[log_cursor]
                    if node_id < running_node_features.shape[0]:
                        running_node_features[node_id, 0] = log_q_value[log_cursor]
                        running_node_features[node_id, 1:4] = log_wdl[log_cursor]
                    log_cursor += 1
                # Clone: running_node_features is mutated in place on later
                # iterations, so a bare slice (a view) would retroactively
                # change earlier steps' already-appended tensors.
                step_nf.append(running_node_features[:node_cutoff].clone())
            else:
                # Fallback for shards without an update log (pre-Task-1 packed
                # data): reproduces today's frozen-slice behavior exactly. This
                # is the documented bug (history.md "What's wrong with the
                # current code"), kept only so this dataset still runs against
                # packed data that predates packhistory_trees landing.
                step_nf.append(full_node_features[:node_cutoff])
            step_pi.append(full_parent_index[:node_cutoff])
            step_d.append(full_depth[:node_cutoff])

            # Collect edges grouped by parent in sorted-parent order. Sorting
            # makes the resulting edge_parent/edge_child arrays deterministic
            # and matches what TreeBatch downstream expects.
            active_parents = sorted(int(parent_id) for parent_id in expansion_parent_ids[:expansion_count].tolist())
            edge_parent_parts = []
            edge_child_parts = []
            edge_slot_parts = []
            for parent_id in active_parents:
                local_edge_start = int(full_child_ptr[parent_id].item())
                local_edge_end = int(full_child_ptr[parent_id + 1].item())
                if local_edge_end <= local_edge_start:
                    continue
                count = local_edge_end - local_edge_start
                edge_parent_parts.append(torch.full((count,), parent_id, dtype=torch.long))
                edge_child_parts.append(full_edge_child[local_edge_start:local_edge_end])
                edge_slot_parts.append(full_edge_slot[local_edge_start:local_edge_end])
            if edge_parent_parts:
                step_ep.append(torch.cat(edge_parent_parts, dim=0))
                step_ec.append(torch.cat(edge_child_parts, dim=0))
                step_es.append(torch.cat(edge_slot_parts, dim=0))
            else:
                # Root-only step: empty edge tensors with the right dtype.
                step_ep.append(torch.empty(0, dtype=torch.long))
                step_ec.append(torch.empty(0, dtype=torch.long))
                step_es.append(torch.empty(0, dtype=torch.long))

        # Time budget decays by 1 per step from the starting budget.
        starting_budget = int(payload["starting_budgets"][episode_offset].item())
        time_budgets = torch.arange(
            starting_budget,
            starting_budget - num_steps,
            -1,
            dtype=torch.long,
        )
        halt_rewards = full_halt_rewards[:num_steps].tolist()
        tree_sizes = step_node_cutoffs[:num_steps].to(dtype=torch.long)

        return ControllerEpisode(
            path=payload["episode_keys"][episode_offset],
            source_path=payload["trajectory_source_paths"][trajectory_index],
            step_node_features=step_nf,
            step_parent_index=step_pi,
            step_edge_parent=step_ep,
            step_edge_child=step_ec,
            step_edge_slot=step_es,
            step_depth=step_d,
            halt_rewards=halt_rewards,
            target_advantages=payload["target_advantages"][episode_step_begin:episode_step_end],
            tree_sizes=tree_sizes,
            time_budgets=time_budgets,
            oracle_stop_step=int(payload["oracle_stop_steps"][episode_offset].item()),
            oracle_value=float(payload["oracle_values"][episode_offset].item()),
            starting_budget=starting_budget,
            budget_bucket_name=payload["budget_bucket_names"][episode_offset],
        )


def _extract_all_episode_metadata(
    dataset: ControllerEpisodeDataset,
    max_episodes: int | None = None,
) -> List[EpisodeMetadata]:
    """Extract per-episode metadata from packed shards without tree reconstruction."""
    metadata: List[EpisodeMetadata] = []
    # Walk shards directly rather than via __getitem__ — we don't need the
    # per-step encoder tensors, just the scalar metadata, which is much cheaper.
    for shard_index in range(len(dataset.shard_paths)):
        payload = torch.load(dataset.shard_paths[shard_index], weights_only=False)
        episode_step_ptr = payload["episode_step_ptr"]
        episode_trajectory_index = payload["episode_trajectory_index"]
        trajectory_step_ptr = payload["trajectory_step_ptr"]
        step_node_cutoffs = payload["step_node_cutoffs"]
        trajectory_halt_rewards = payload["trajectory_halt_rewards"]
        target_advantages = payload["target_advantages"]
        oracle_stop_steps = payload["oracle_stop_steps"]
        oracle_values = payload["oracle_values"]
        starting_budgets = payload["starting_budgets"]
        budget_bucket_names = payload["budget_bucket_names"]
        episode_keys = payload["episode_keys"]
        trajectory_source_paths = payload["trajectory_source_paths"]

        num_episodes = dataset.shard_num_episodes[shard_index]
        for i in range(num_episodes):
            ep_step_begin = int(episode_step_ptr[i].item())
            ep_step_end = int(episode_step_ptr[i + 1].item())
            num_steps = ep_step_end - ep_step_begin

            trajectory_index = int(episode_trajectory_index[i].item())
            traj_step_begin = int(trajectory_step_ptr[trajectory_index].item())

            starting_budget = int(starting_budgets[i].item())
            metadata.append(EpisodeMetadata(
                path=episode_keys[i],
                source_path=trajectory_source_paths[trajectory_index],
                halt_rewards=trajectory_halt_rewards[traj_step_begin:traj_step_begin + num_steps].tolist(),
                tree_sizes=step_node_cutoffs[traj_step_begin:traj_step_begin + num_steps].tolist(),
                time_budgets=list(range(starting_budget, starting_budget - num_steps, -1)),
                target_advantages=target_advantages[ep_step_begin:ep_step_end].tolist(),
                oracle_stop_step=int(oracle_stop_steps[i].item()),
                oracle_value=float(oracle_values[i].item()),
                starting_budget=starting_budget,
                budget_bucket_name=budget_bucket_names[i],
                num_steps=num_steps,
            ))
            if max_episodes is not None and len(metadata) >= max_episodes:
                return metadata
    return metadata




def collate_controller_episodes(episodes: Sequence[ControllerEpisode]) -> ControllerBatch | None:
    """Collate a list of ``ControllerEpisode`` into a single ``ControllerBatch``.

    Stacks every episode's per-step partial trees into one packed ``TreeBatch``
    (offsetting node/edge ids so trees stay disjoint) and concatenates the
    per-snapshot scalars / per-episode metadata in matching order. Returns
    ``None`` if the input is empty.
    """
    if not episodes:
        return None

    all_node_features = []
    all_parent_index = []
    all_edge_parent = []
    all_edge_child = []
    all_edge_slot = []
    all_depth = []
    all_tree_index = []
    root_index = []

    target_advantages_list = []
    tree_sizes_list = []
    time_budgets_list = []
    paths = []
    source_paths = []
    path_lengths = []
    halt_rewards = []
    oracle_stop_steps = []
    oracle_values = []
    starting_budgets = []
    budget_bucket_names = []

    # node_offset is the running base index for the next tree's nodes.
    # tree_idx counts trees-in-the-packed-batch (== number of snapshots
    # across all episodes), used to fill tree_index and root_index.
    node_offset = 0
    tree_idx = 0

    for episode in episodes:
        for step_nf, step_pi, step_ep, step_ec, step_es, step_d in zip(
            episode.step_node_features,
            episode.step_parent_index,
            episode.step_edge_parent,
            episode.step_edge_child,
            episode.step_edge_slot,
            episode.step_depth,
        ):
            num_nodes = step_nf.shape[0]
            all_node_features.append(step_nf)
            all_depth.append(step_d)
            all_tree_index.append(torch.full((num_nodes,), tree_idx, dtype=torch.long))
            # Root is always the first node in each step's prefix tree;
            # node_offset is exactly that root's id after packing.
            root_index.append(node_offset)

            # Offset parent_index for all non-root entries; -1 sentinels
            # for root nodes stay -1 because the mask excludes them.
            pi = step_pi.clone()
            has_parent = pi >= 0
            pi[has_parent] += node_offset
            all_parent_index.append(pi)

            if step_ep.numel() > 0:
                all_edge_parent.append(step_ep + node_offset)
                all_edge_child.append(step_ec + node_offset)
                all_edge_slot.append(step_es)

            node_offset += num_nodes
            tree_idx += 1

        target_advantages_list.append(episode.target_advantages)
        tree_sizes_list.append(episode.tree_sizes)
        time_budgets_list.append(episode.time_budgets)
        paths.append(episode.path)
        source_paths.append(episode.source_path)
        path_lengths.append(len(episode.step_node_features))
        halt_rewards.append(episode.halt_rewards)
        oracle_stop_steps.append(episode.oracle_stop_step)
        oracle_values.append(episode.oracle_value)
        starting_budgets.append(episode.starting_budget)
        budget_bucket_names.append(episode.budget_bucket_name)

    node_features = torch.cat(all_node_features, dim=0)
    parent_index = torch.cat(all_parent_index, dim=0)
    depth = torch.cat(all_depth, dim=0)
    tree_index = torch.cat(all_tree_index, dim=0)
    root_index_tensor = torch.tensor(root_index, dtype=torch.long)

    # Build CSR child_ptr from the concatenated edge_parent list. If the
    # batch is entirely edgeless (e.g. all root-only snapshots), fall
    # back to empty tensors with the right dtype so the encoder's
    # short-circuit path triggers cleanly.
    if all_edge_parent:
        edge_parent = torch.cat(all_edge_parent, dim=0)
        edge_child = torch.cat(all_edge_child, dim=0)
        edge_slot = torch.cat(all_edge_slot, dim=0)
        children_index = edge_child
        child_counts = torch.bincount(edge_parent, minlength=node_features.shape[0])
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long)
        child_ptr[1:] = torch.cumsum(child_counts, dim=0)
    else:
        edge_parent = torch.empty(0, dtype=torch.long)
        edge_child = torch.empty(0, dtype=torch.long)
        edge_slot = torch.empty(0, dtype=torch.long)
        children_index = torch.empty(0, dtype=torch.long)
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long)

    tree_batch = TreeBatch(
        node_features=node_features,
        tree_index=tree_index,
        parent_index=parent_index,
        root_index=root_index_tensor,
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_slot=edge_slot,
        child_ptr=child_ptr,
        children_index=children_index,
        depth=depth,
        feature_names=("packed",),
        batch_size=tree_idx,
        num_nodes=int(node_features.shape[0]),
        num_edges=int(edge_parent.shape[0]),
    )

    return ControllerBatch(
        tree_batch=tree_batch,
        target_advantages=torch.cat(target_advantages_list, dim=0),
        tree_sizes=torch.cat(tree_sizes_list, dim=0),
        time_budgets=torch.cat(time_budgets_list, dim=0),
        paths=paths,
        source_paths=source_paths,
        path_lengths=path_lengths,
        halt_rewards=halt_rewards,
        oracle_stop_steps=oracle_stop_steps,
        oracle_values=oracle_values,
        starting_budgets=starting_budgets,
        budget_bucket_names=budget_bucket_names,
    )


def _feature_schema() -> NodeFeatureSchema:
    """Return the canonical encoder feature schema; thin wrapper for readability."""
    return tree_encoder_feature_schema()


def _oracle_config(config: ControllerTrainConfig) -> BudgetedOracleConfig:
    """Build a ``BudgetedOracleConfig`` from the parsed CLI arguments.

    ``fixed_budget`` collapses the usual 5-bucket stratification to a single bucket — must
    stay byte-identical to the packing-time choice (`cts.data.preprocess_mc.pack._oracle_config`)
    or `_validate_packed_manifest_oracle` below will reject the packed manifest.
    """
    if config.fixed_budget is not None:
        return BudgetedOracleConfig(
            maintenance_scale=config.maintenance_scale,
            maintenance_ref_nodes=config.maintenance_ref_nodes,
            maintenance_exponent=config.maintenance_exponent,
            time_lambda=config.time_lambda,
            time_p=config.time_p,
            time_tau=config.time_tau,
            time_delta=config.time_delta,
            timeout_value=config.timeout_value,
            time_mode=config.time_mode,
            budget_buckets=(BudgetBucket("fixed", config.fixed_budget, config.fixed_budget),),
            samples_per_bucket=1,
            seed=config.seed,
        )
    return BudgetedOracleConfig(
        maintenance_scale=config.maintenance_scale,
        maintenance_ref_nodes=config.maintenance_ref_nodes,
        maintenance_exponent=config.maintenance_exponent,
        time_lambda=config.time_lambda,
        time_p=config.time_p,
        time_tau=config.time_tau,
        time_delta=config.time_delta,
        timeout_value=config.timeout_value,
        time_mode=config.time_mode,
        budget_buckets=(
            BudgetBucket("scramble", config.scramble_min_time, config.scramble_max_time),
            BudgetBucket("medium-small", config.medium_small_min_time, config.medium_small_max_time),
            BudgetBucket("medium-large", config.medium_large_min_time, config.medium_large_max_time),
            BudgetBucket("large", config.large_min_time, config.large_max_time),
            BudgetBucket("very-large", config.very_large_min_time, config.very_large_max_time),
        ),
        samples_per_bucket=config.samples_per_bucket,
        seed=config.seed,
    )


def _build_packed_loader(
    manifest_path: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    """Wrap a ``ControllerEpisodeDataset`` in a DataLoader with the project collator."""
    dataset = ControllerEpisodeDataset(manifest_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_controller_episodes,
        generator=generator if shuffle else None,
    )


def _default_materialized_cache_path(manifest_path: str, encoder_checkpoint: str, split_name: str) -> str:
    """Compose a cache filename keyed by manifest stem + encoder stem + split.

    Two different encoders or two different manifests get different cache
    paths, so cache reuse is automatically scoped to compatible runs.
    """
    manifest = Path(manifest_path)
    encoder_stem = Path(encoder_checkpoint).stem
    manifest_stem = manifest.stem
    return str(manifest.with_name(f"{manifest_stem}.materialized_{split_name}_{encoder_stem}.pt"))


def _materialized_cache_shard_dir(path: str) -> Path:
    """Return the sidecar directory holding shard_*.pt files for a cache index."""
    return Path(f"{path}.d")


def _save_materialized_cache_shards(
    path: str,
    model: MetaController,
    loader: DataLoader,
    *,
    manifest_path: str,
    encoder_checkpoint: str,
    log_interval: int,
    split_name: str,
    max_snapshots_per_shard: int,
) -> MaterializedCache:
    """Run the encoder over the full loader once and stream features to sharded files.

    The encoder is the by-far most expensive part of each training step when
    it's frozen. Materializing its outputs once and reading back from disk
    on every epoch is a large speedup. Shards keep per-file size bounded so
    each individual torch.load stays fast and we don't OOM on huge splits.
    """
    output_path = Path(path)
    shard_dir = _materialized_cache_shard_dir(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Purge any stale shards from a previous run; the index file is rewritten
    # below so leftovers would be referenced inconsistently.
    for stale in shard_dir.glob("shard_*.pt"):
        stale.unlink()
    if output_path.exists():
        output_path.unlink()

    model.eval()
    shard_paths: List[str] = []
    shard_sizes: List[int] = []
    shard_features: List[torch.Tensor] = []
    shard_targets: List[torch.Tensor] = []
    shard_oracle_stop_steps: List[torch.Tensor] = []
    shard_snapshots = 0
    total_snapshots = 0
    total_episodes = 0
    shard_index = 0
    started = time.time()

    def flush_shard() -> None:
        """Write any buffered tensors to a new shard file and reset the buffers."""
        nonlocal shard_features, shard_targets, shard_oracle_stop_steps, shard_snapshots, shard_index
        if not shard_features:
            return
        features = torch.cat(shard_features, dim=0)
        targets = torch.cat(shard_targets, dim=0)
        oracle_steps = torch.cat(shard_oracle_stop_steps, dim=0)
        shard_path = shard_dir / f"shard_{shard_index:05d}.pt"
        torch.save(
            {
                "format": "cts_materialized_advantage_cache_shard_v2",
                "features": features,
                "target_advantages": targets,
                "oracle_stop_steps": oracle_steps,
            },
            shard_path,
        )
        shard_paths.append(str(shard_path))
        shard_sizes.append(int(features.shape[0]))
        shard_features = []
        shard_targets = []
        shard_oracle_stop_steps = []
        shard_snapshots = 0
        shard_index += 1

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if batch is None:
                continue
            # Encode and immediately detach to CPU; we don't need gradients
            # and we don't want GPU memory to accumulate across batches.
            features = model.encode_with_state_features(batch.tree_batch, batch.tree_sizes, batch.time_budgets).detach().cpu()
            targets = batch.target_advantages.detach().cpu()
            oracle_steps = torch.tensor(batch.oracle_stop_steps, dtype=torch.int32)
            path_lengths = torch.tensor(batch.path_lengths, dtype=torch.int32)
            # oracle_stop_steps is per-episode but the cache is per-snapshot,
            # so repeat each value path_length times to align with features.
            oracle_steps_per_snapshot = oracle_steps.repeat_interleave(path_lengths)
            shard_features.append(features)
            shard_targets.append(targets)
            shard_oracle_stop_steps.append(oracle_steps_per_snapshot)
            batch_snapshots = int(features.shape[0])
            shard_snapshots += batch_snapshots
            total_snapshots += batch_snapshots
            total_episodes += len(batch.paths)
            if shard_snapshots >= max_snapshots_per_shard:
                flush_shard()
            if log_interval > 0 and (batch_index % log_interval == 0 or batch_index == len(loader)):
                elapsed = time.time() - started
                print(
                    f"materialize_{split_name}_encoder_batch={batch_index}/{len(loader)} "
                    f"episodes={total_episodes} snapshots={total_snapshots} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    flush_shard()
    if total_snapshots == 0:
        raise ValueError(f"No usable {split_name} episodes were materialized.")
    # Index file referencing every shard, with the manifest/encoder pair the
    # cache was built against so _load_materialized_cache can validate reuse.
    payload = {
        "format": "cts_materialized_advantage_cache_v2",
        "metadata": {
            "manifest_path": str(manifest_path),
            "encoder_checkpoint": str(encoder_checkpoint),
        },
        "shards": [
            {"path": shard_path, "examples": examples}
            for shard_path, examples in zip(shard_paths, shard_sizes)
        ],
        "examples": total_snapshots,
    }
    torch.save(payload, output_path)
    print(f"[compute_advantage] saved_materialized_cache={output_path}", flush=True)
    return MaterializedCache(shard_paths=shard_paths, shard_sizes=shard_sizes, examples=total_snapshots)


def _load_materialized_cache(
    path: str,
    *,
    manifest_path: str,
    encoder_checkpoint: str,
) -> MaterializedCache:
    """Load a previously-saved cache, asserting its manifest/encoder match the caller's."""
    payload = torch.load(path, weights_only=False)
    if payload.get("format") != "cts_materialized_advantage_cache_v2":
        raise ValueError("Unexpected materialized cache format.")
    metadata = payload.get("metadata", {})
    # Refuse to reuse a cache that was built against a different manifest or
    # encoder — the stored features would be silently wrong for the current run.
    if metadata.get("manifest_path") != str(manifest_path):
        raise ValueError(f"Materialized cache manifest mismatch: {path}")
    if metadata.get("encoder_checkpoint") != str(encoder_checkpoint):
        raise ValueError(f"Materialized cache encoder mismatch: {path}")
    shard_paths = [str(entry["path"]) for entry in payload.get("shards", [])]
    shard_sizes = [int(entry["examples"]) for entry in payload.get("shards", [])]
    return MaterializedCache(
        shard_paths=shard_paths,
        shard_sizes=shard_sizes,
        examples=int(payload.get("examples", sum(shard_sizes))),
    )


def _advantage_loss_components(
    predicted_advantages: torch.Tensor,
    target_advantages: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (MSE, MAE, sign_accuracy), optionally with per-element weights.

    Weights are applied to MSE and MAE (weighted mean) but never to
    sign_accuracy, which is reported as the raw unweighted rate so it
    stays comparable across weighted/unweighted runs.
    """
    if weights is not None:
        per_element_mse = (predicted_advantages - target_advantages) ** 2
        advantage_mse = (per_element_mse * weights).sum() / weights.sum()
        per_element_abs = torch.abs(predicted_advantages - target_advantages)
        mean_abs_advantage_error = (per_element_abs * weights).sum() / weights.sum()
    else:
        advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
        mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))
    sign_accuracy = ((predicted_advantages > 0) == (target_advantages > 0)).float().mean()
    return advantage_mse, mean_abs_advantage_error, sign_accuracy


# Prototype over-search knob; set once from ControllerTrainConfig.sign_pos_weight at the
# top of main(). <1 penalises false-continue (over-search) in the sign-BCE below.
_SIGN_POS_WEIGHT: float = 1.0


def _sign_auxiliary_loss(
    predicted_advantages: torch.Tensor,
    target_advantages: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE-with-logits between the predicted scalar and ``sign(target) > 0``.

    The auxiliary loss the regression head uses to anchor its sign decision
    even when small-magnitude regression errors would otherwise leave it
    ambiguous. ``_SIGN_POS_WEIGHT`` < 1 makes it asymmetric, penalising
    false-continue (over-search) more than false-stop.
    """
    sign_targets = (target_advantages > 0).to(dtype=predicted_advantages.dtype)
    pos_weight = (
        torch.as_tensor(_SIGN_POS_WEIGHT, dtype=predicted_advantages.dtype, device=predicted_advantages.device)
        if _SIGN_POS_WEIGHT != 1.0 else None
    )
    if weights is not None:
        per_element = F.binary_cross_entropy_with_logits(
            predicted_advantages, sign_targets, pos_weight=pos_weight, reduction="none")
        return (per_element * weights).sum() / weights.sum()
    return F.binary_cross_entropy_with_logits(predicted_advantages, sign_targets, pos_weight=pos_weight)


def _train_epoch(
    model: MetaController,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    sign_loss_weight: float,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
    max_batches: int | None = None,
    logger: ControllerTrainMetricsLogger | None = None,
    after_step: Callable[[int, int], None] | None = None,
) -> AdvantageMetrics:
    """One training epoch over the *live* encoder path (no materialized cache).

    Used when ``--unfreeze-encoder`` is set, so the encoder must be re-run on
    every batch. Slower than the cached path; preserved for the rare fine-
    tuning runs that need encoder gradients.
    """
    model.train()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    started = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        targets = batch.target_advantages.to(device, non_blocking=True)
        predicted, sign_logits = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
        advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, targets)
        sign_loss = _sign_auxiliary_loss(sign_logits, targets)
        total_loss = advantage_mse + sign_loss_weight * sign_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Snapshot-weighted running sums so the epoch-level averages stay
        # correct even when the last batch is short.
        examples = int(targets.shape[0])
        total_loss_sum += float(total_loss.item()) * examples
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_sign_bce += float(sign_loss.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += int(((sign_logits.detach() > 0) == (targets > 0)).sum().item())

        if log_interval > 0 and batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"snapshots={total_examples} "
                f"total_loss={total_loss_sum / max(total_examples, 1):.3f} "
                f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                f"sign_bce={total_sign_bce / max(total_examples, 1):.3f} "
                f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )
        if logger is not None:
            logger.after_train_batch(
                epoch,
                total_loss_sum=total_loss_sum,
                total_advantage_mse=total_advantage_mse,
                total_sign_bce=total_sign_bce,
                total_mean_abs_advantage_error=total_mean_abs_advantage_error,
                total_correct=total_correct,
                total_examples=total_examples,
            )
            if after_step is not None:
                after_step(epoch, logger.n_batches)
        if max_batches is not None and batch_index >= max_batches:
            break

    if total_examples == 0:
        raise ValueError("Training loader produced no valid controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_advantage_predictions(
    model: MetaController,
    loader: DataLoader,
    *,
    device: torch.device,
    sign_loss_weight: float,
    max_batches: int | None = None,
) -> AdvantageMetrics:
    """Live-encoder validation pass; mirror of ``_train_epoch`` without the backward."""
    model.eval()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    batch_index = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            batch_index += 1
            targets = batch.target_advantages.to(device, non_blocking=True)
            predicted, sign_logits = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, targets)
            sign_loss = _sign_auxiliary_loss(sign_logits, targets)
            total_loss = advantage_mse + sign_loss_weight * sign_loss
            examples = int(targets.shape[0])
            total_loss_sum += float(total_loss.item()) * examples
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_sign_bce += float(sign_loss.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((sign_logits > 0) == (targets > 0)).sum().item())
            if max_batches is not None and batch_index >= max_batches:
                break
    if total_examples == 0:
        raise ValueError("Evaluation loader produced no valid controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )




def _train_materialized_cache_epoch(
    model: MetaController,
    cache: MaterializedCache,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
    nontrivial_loss_weight: float,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
    seed: int,
    bin_weights: torch.Tensor | None = None,
    bin_boundaries: torch.Tensor | None = None,
    max_batches: int | None = None,
    logger: ControllerTrainMetricsLogger | None = None,
    after_step: Callable[[int, int], None] | None = None,
) -> AdvantageMetrics:
    """One training epoch over the materialized cache (frozen-encoder path).

    Shuffles shards per epoch and shuffles snapshots within each shard so
    every gradient step sees a fresh mix. The advantage head is trained on
    cached ``[z_root, N_t, T_t]`` features without ever running the encoder.

    Args:
        bin_weights: optional per-bin loss weights (from ``_compute_inverse_freq_bin_weights``).
            When supplied, takes precedence over ``nontrivial_loss_weight``.
        bin_boundaries: bucket boundaries paired with ``bin_weights``.
        nontrivial_loss_weight: applied to snapshots with ``oracle_stop_step > 1``
            when bin weighting is disabled and the value differs from 1.0.
    """
    model.train()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    started = time.time()
    batch_counter = 0
    # bin weighting fully replaces nontrivial weighting when both are set;
    # nontrivial weighting is a no-op when the user-supplied weight is 1.0.
    use_bin_weighting = bin_weights is not None
    use_nontrivial_weighting = not use_bin_weighting and nontrivial_loss_weight != 1.0

    # Reshuffle shard visitation order per epoch so we don't always start
    # gradient descent from the same shard's distribution.
    shard_order = list(range(len(cache.shard_paths)))
    random.Random(seed + epoch).shuffle(shard_order)
    stop_epoch = False
    for shard_index in shard_order:
        if stop_epoch:
            break
        payload = torch.load(cache.shard_paths[shard_index], weights_only=False)
        features = payload["features"]
        target_advantages = payload["target_advantages"]
        shard_oracle = payload.get("oracle_stop_steps")
        if (use_bin_weighting or use_nontrivial_weighting) and shard_oracle is None:
            raise ValueError(
                "Materialized cache missing oracle_stop_steps. "
                "Delete the cache and re-materialize to use loss weighting."
            )
        # Within-shard shuffle: new random order each epoch.
        order = torch.randperm(features.shape[0])
        for start in range(0, int(features.shape[0]), batch_size):
            batch_counter += 1
            batch_index = order[start : start + batch_size]
            batch_features = features[batch_index].to(device, non_blocking=True)
            batch_targets = target_advantages[batch_index].to(device, non_blocking=True)
            if use_bin_weighting:
                batch_bins = torch.bucketize(shard_oracle[batch_index], bin_boundaries)
                weights = bin_weights[batch_bins].to(device, non_blocking=True)
            elif use_nontrivial_weighting:
                # Boost any snapshot whose oracle action is non-trivial
                # (stops after at least one expansion). Singleton episodes
                # keep weight 1.
                batch_oracle = shard_oracle[batch_index]
                weights = torch.where(batch_oracle > 1, nontrivial_loss_weight, 1.0).to(device, non_blocking=True)
            else:
                weights = None
            predicted, sign_logits = model.predict_from_features(batch_features)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, batch_targets, weights)
            sign_loss = _sign_auxiliary_loss(sign_logits, batch_targets, weights)
            total_loss = advantage_mse + sign_loss_weight * sign_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            examples = int(batch_targets.shape[0])
            total_loss_sum += float(total_loss.item()) * examples
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_sign_bce += float(sign_loss.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((sign_logits.detach() > 0) == (batch_targets > 0)).sum().item())

            if log_interval > 0 and batch_counter % log_interval == 0:
                elapsed = time.time() - started
                print(
                    f"epoch={epoch} batch={batch_counter} "
                    f"snapshots={total_examples} "
                    f"total_loss={total_loss_sum / max(total_examples, 1):.3f} "
                    f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                    f"sign_bce={total_sign_bce / max(total_examples, 1):.3f} "
                    f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                    f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            if logger is not None:
                logger.after_train_batch(
                    epoch,
                    total_loss_sum=total_loss_sum,
                    total_advantage_mse=total_advantage_mse,
                    total_sign_bce=total_sign_bce,
                    total_mean_abs_advantage_error=total_mean_abs_advantage_error,
                    total_correct=total_correct,
                    total_examples=total_examples,
                )
                if after_step is not None:
                    after_step(epoch, logger.n_batches)
            if max_batches is not None and batch_counter >= max_batches:
                stop_epoch = True
                break

    if total_examples == 0:
        raise ValueError("Materialized cache training produced no controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_materialized_cache_predictions(
    model: MetaController,
    cache: MaterializedCache,
    *,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
    max_batches: int | None = None,
) -> AdvantageMetrics:
    """Cached-feature validation: deterministic order, no weighting, no shuffle."""
    model.eval()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    batch_counter = 0
    with torch.inference_mode():
        for shard_path in cache.shard_paths:
            payload = torch.load(shard_path, weights_only=False)
            features = payload["features"]
            target_advantages = payload["target_advantages"]
            for start in range(0, int(features.shape[0]), batch_size):
                batch_counter += 1
                batch_features = features[start : start + batch_size].to(device, non_blocking=True)
                batch_targets = target_advantages[start : start + batch_size].to(device, non_blocking=True)
                predicted, sign_logits = model.predict_from_features(batch_features)
                advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, batch_targets)
                sign_loss = _sign_auxiliary_loss(sign_logits, batch_targets)
                total_loss = advantage_mse + sign_loss_weight * sign_loss
                examples = int(batch_targets.shape[0])
                total_loss_sum += float(total_loss.item()) * examples
                total_advantage_mse += float(advantage_mse.item()) * examples
                total_sign_bce += float(sign_loss.item()) * examples
                total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
                total_examples += examples
                total_correct += int(((sign_logits > 0) == (batch_targets > 0)).sum().item())
                if max_batches is not None and batch_counter >= max_batches:
                    break
            if max_batches is not None and batch_counter >= max_batches:
                break
    if total_examples == 0:
        raise ValueError("Materialized cache evaluation produced no controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )




def _collect_episode_metadata_and_step_count(
    dataset: ControllerEpisodeDataset,
    max_episodes: int | None = None,
) -> Tuple[List[EpisodeMetadata], int]:
    """Pull episode scalars out of packed shards and sum total controller steps."""
    episode_metadata = _extract_all_episode_metadata(dataset, max_episodes=max_episodes)
    if max_episodes is None:
        assert len(episode_metadata) == len(dataset), (
            f"Metadata extraction returned {len(episode_metadata)} episodes but dataset has {len(dataset)}"
        )
    total_steps = sum(m.num_steps for m in episode_metadata)
    return episode_metadata, total_steps


def _predict_advantages_for_cache(
    model: MetaController,
    cache: MaterializedCache,
    predict_batch_size: int,
    max_snapshots: int | None = None,
) -> torch.Tensor:
    """Run a batched forward pass over cached features and return one flat advantage tensor.

    The cache's snapshot order matches the dataset's episode order, so the
    returned tensor can be re-split per episode by ``num_steps`` downstream.
    """
    model.eval()
    device = next(model.parameters()).device
    all_predicted: List[torch.Tensor] = []
    produced = 0
    with torch.inference_mode():
        for shard_path in cache.shard_paths:
            if max_snapshots is not None and produced >= max_snapshots:
                break
            payload = torch.load(shard_path, weights_only=False)
            features = payload["features"]
            for start in range(0, features.shape[0], predict_batch_size):
                if max_snapshots is not None and produced >= max_snapshots:
                    break
                end = start + predict_batch_size
                if max_snapshots is not None:
                    end = min(end, produced + (max_snapshots - produced))
                batch_features = features[start:end].to(device, non_blocking=True)
                if batch_features.shape[0] == 0:
                    continue
                predicted, _ = model.predict_from_features(batch_features)
                all_predicted.append(predicted.detach().cpu())
                produced += int(batch_features.shape[0])
    return torch.cat(all_predicted, dim=0)


def _aggregate_greedy_rollout_metrics(
    episode_metadata: List[EpisodeMetadata],
    all_advantages: torch.Tensor,
    oracle_config: BudgetedOracleConfig,
    *,
    log_interval: int,
    started: float,
    diagnostics_out: List[Dict[str, Any]] | None,
) -> GreedyPolicyMetrics:
    """Walk per-episode slices, apply the stop-when-advantage<=0 rule, sum metrics, optionally write diagnostics.

    Walks episodes in order, takes a contiguous slice of predicted advantages,
    and stops at the first non-positive advantage (or falls back to the last
    step if every advantage stays positive).
    """
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    offset = 0

    for index, meta in enumerate(episode_metadata):
        episode_advantages = all_advantages[offset:offset + meta.num_steps].tolist()
        offset += meta.num_steps

        predicted_stop = predicted_stop_from_advantages(episode_advantages)

        predicted_return = return_for_stop_step(
            meta.halt_rewards,
            meta.tree_sizes,
            meta.time_budgets,
            predicted_stop,
            oracle_config,
        )
        # Recompute oracle_value under the current oracle_config so that cost-sweep
        # runs (different lambda) report regret consistently, even when the packed
        # shard oracle_value was computed under a different cost config.
        live_oracle_value = compute_budgeted_oracle(
            meta.halt_rewards, meta.tree_sizes, meta.starting_budget, oracle_config
        ).oracle_value
        regret = live_oracle_value - predicted_return
        exact += int(predicted_stop == meta.oracle_stop_step)
        first_action += int((predicted_stop == 0) == (meta.oracle_stop_step == 0))
        total_return += predicted_return
        total_oracle_value += live_oracle_value
        total_expansions += predicted_stop

        if diagnostics_out is not None:
            diagnostics_out.append(
                {
                    "path": meta.path,
                    "source_path": meta.source_path,
                    "episode_length": meta.num_steps,
                    "starting_budget": meta.starting_budget,
                    "budget_bucket_name": meta.budget_bucket_name,
                    "tree_sizes": meta.tree_sizes,
                    "time_budgets": meta.time_budgets,
                    "halt_rewards": meta.halt_rewards,
                    "oracle_stop_step": meta.oracle_stop_step,
                    "oracle_value": live_oracle_value,
                    "predicted_stop_step": predicted_stop,
                    "predicted_value": predicted_return,
                    "regret": regret,
                    "predicted_advantages": episode_advantages,
                    "target_advantages": meta.target_advantages,
                }
            )

        if log_interval > 0 and ((index + 1) % log_interval == 0 or index + 1 == len(episode_metadata)):
            elapsed = time.time() - started
            print(
                f"greedy_eval_progress={index + 1}/{len(episode_metadata)} "
                f"exact_stop_step_accuracy={exact / max(index + 1, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    assert offset == all_advantages.shape[0]
    evaluated = len(episode_metadata)
    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no episodes.")
    return GreedyPolicyMetrics(
        exact_stop_step_accuracy=exact / evaluated,
        first_action_accuracy=first_action / evaluated,
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle_value / evaluated,
        average_regret=(total_oracle_value - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
    )


def evaluate_batched_greedy_policy(
    model: MetaController,
    dataset: ControllerEpisodeDataset,
    cache: MaterializedCache,
    oracle_config: BudgetedOracleConfig,
    *,
    log_interval: int,
    diagnostics_out: List[Dict[str, Any]] | None = None,
    predict_batch_size: int = 65536,
    max_eval_episodes: int | None = None,
) -> GreedyPolicyMetrics:
    """Greedy eval using materialized cache features in batched forward passes."""
    started = time.time()
    episode_metadata, total_steps = _collect_episode_metadata_and_step_count(
        dataset,
        max_episodes=max_eval_episodes,
    )
    all_advantages = _predict_advantages_for_cache(
        model,
        cache,
        predict_batch_size,
        max_snapshots=total_steps,
    )
    assert all_advantages.shape[0] == total_steps, (
        f"Cache has {all_advantages.shape[0]} steps but truncated dataset has {total_steps}"
    )
    return _aggregate_greedy_rollout_metrics(
        episode_metadata,
        all_advantages,
        oracle_config,
        log_interval=log_interval,
        started=started,
        diagnostics_out=diagnostics_out,
    )


def _write_diagnostics(diagnostics: List[Dict[str, Any]], output_path: str) -> None:
    """Write the per-episode diagnostics list to disk as JSONL + log summary stats."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in diagnostics:
            handle.write(json.dumps(record) + "\n")
    regrets = [record["regret"] for record in diagnostics]
    print(f"[diagnostics] wrote {len(diagnostics)} episodes to {path}", flush=True)
    print(
        f"[diagnostics] regret_mean={statistics.mean(regrets):.4f} "
        f"regret_std={statistics.pstdev(regrets):.4f} "
        f"regret_max={max(regrets):.4f}",
        flush=True,
    )


def _save_checkpoint(path: str, model: nn.Module, metadata: dict) -> None:
    """Write ``{model_state_dict, metadata}`` to ``path``, creating parent dirs as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, output_path)


def _validate_packed_manifest_oracle(manifest_path: str, oracle_config: BudgetedOracleConfig) -> None:
    """Fail fast if the manifest was packed with different oracle params than requested.

    Prevents the silent corner case where a user changes the oracle CLI
    flags but reuses an old packed manifest — the targets in the manifest
    are baked at packing time, so a mismatch makes regression meaningless.
    """
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_config = budgeted_oracle_config_from_metadata(manifest)
    if manifest_config is None:
        raise ValueError(f"Packed manifest is missing budgeted oracle metadata: {manifest_path}")

    float_fields = (
        "maintenance_scale",
        "maintenance_ref_nodes",
        "maintenance_exponent",
        "time_lambda",
        "time_p",
        "time_tau",
        "timeout_value",
    )
    for field in float_fields:
        if abs(float(getattr(manifest_config, field)) - float(getattr(oracle_config, field))) > 1e-12:
            raise ValueError(f"Packed manifest {field} does not match requested training config.")
    int_fields = ("time_delta", "samples_per_bucket")
    for field in int_fields:
        if int(getattr(manifest_config, field)) != int(getattr(oracle_config, field)):
            raise ValueError(f"Packed manifest {field} does not match requested training config.")
    if manifest_config.time_mode != oracle_config.time_mode:
        raise ValueError(
            f"Packed manifest time_mode={manifest_config.time_mode!r} does not match "
            f"requested training config time_mode={oracle_config.time_mode!r}."
        )
    manifest_buckets = [(bucket.name, bucket.min_time, bucket.max_time) for bucket in manifest_config.budget_buckets]
    requested_buckets = [(bucket.name, bucket.min_time, bucket.max_time) for bucket in oracle_config.budget_buckets]
    if manifest_buckets != requested_buckets:
        raise ValueError("Packed manifest budget buckets do not match requested training config.")


def _seed_and_resolve_paths(
    config: ControllerTrainConfig,
    *,
    require_packed_oracle_match: bool = True,
) -> Tuple[torch.device, str, str, BudgetedOracleConfig, NodeFeatureSchema]:
    """Seed RNGs, resolve cache paths, validate the packed manifests against the requested oracle.

    ``require_packed_oracle_match`` gates ``_validate_packed_manifest_oracle`` (default True,
    matching every caller before 2026-07-09). It exists only for callers whose training loss
    never reads the manifest's baked cost-dependent fields (``target_advantages``,
    ``oracle_values``) and instead recomputes the equivalent quantities fresh from raw
    ``halt_rewards``/``tree_sizes``/``time_budgets`` under the CURRENT ``oracle_config`` --
    for those callers a packed-vs-requested cost-param mismatch is harmless (the manifest's
    baked oracle metadata is simply never consulted), so gating on it would only block runs
    that reuse an existing manifest under new cost params for no correctness reason.
    ``controller_train.py``'s own MSE trainer (regresses directly against baked
    ``target_advantages``) and ``pg_controller_train.py`` (its ``_episode_returns`` still
    trusts the baked ``em.oracle_value``, see 2026-07-09 e2e-probe notebook entry) both still
    depend on the manifest matching, so they must keep the default True.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    train_cache_path = config.materialized_train_cache or _default_materialized_cache_path(
        config.packed_train_data,
        config.encoder_checkpoint,
        "train",
    )
    validation_cache_path = config.materialized_validation_cache or _default_materialized_cache_path(
        config.packed_validation_data,
        config.encoder_checkpoint,
        "validation",
    )
    oracle_config = _oracle_config(config)
    if require_packed_oracle_match:
        _validate_packed_manifest_oracle(config.packed_train_data, oracle_config)
        _validate_packed_manifest_oracle(config.packed_validation_data, oracle_config)
    schema = _feature_schema()
    return device, train_cache_path, validation_cache_path, oracle_config, schema


def _build_model_and_optimizer(
    config: ControllerTrainConfig,
    schema: NodeFeatureSchema,
) -> Tuple[MetaController, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Construct the MetaController, load encoder weights, freeze if requested, and build optimizer + scheduler."""
    print("[compute_advantage] stage=build_model", flush=True)
    model = MetaController(
        k=config.k,
        node_feat=len(schema.feature_names),
        device=config.device,
        node_embed_hidden=config.node_embed_hidden,
        d_embed=config.d_embed,
        d_message=config.d_message,
        n_heads=config.n_heads,
        d_att=config.d_att,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        separate_sign_head=config.separate_sign_head,
        controller_inputs=config.controller_inputs,
    )
    load_encoder_checkpoint(config.encoder_checkpoint, model.encoder)
    if config.resume_checkpoint:
        # Warm-start the WHOLE model (encoder + head) from a prior run's checkpoint --
        # overwrites the encoder weights ``load_encoder_checkpoint`` just loaded above (by
        # design: resume_checkpoint is authoritative when set). Checkpoint format is the
        # shared ``{"model_state_dict": ..., "metadata": {...}}`` both pg_controller_train.py
        # and e2e_controller_train.py already save (see each module's save call), so this one
        # loader works for resuming either lineage, frozen or unfrozen.
        payload = torch.load(config.resume_checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        print(f"[compute_advantage] resumed full model state (encoder+head) from "
              f"{config.resume_checkpoint}", flush=True)
    if not config.unfreeze_encoder:
        model.freeze_encoder()

    # Optimizer only sees parameters whose requires_grad is True, so frozen
    # encoder weights are excluded automatically.
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.min_lr > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr,
        )
    return model, optimizer, scheduler


def _build_train_and_validation_loaders(
    config: ControllerTrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build packed-episode DataLoaders for the train and validation splits."""
    print(
        f"[compute_advantage] packed_train={config.packed_train_data} packed_validation={config.packed_validation_data}",
        flush=True,
    )
    train_loader_raw = _build_packed_loader(
        config.packed_train_data,
        batch_size=config.episode_batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader_raw = _build_packed_loader(
        config.packed_validation_data,
        batch_size=config.episode_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    return train_loader_raw, validation_loader_raw


def _materialize_or_load_caches(
    config: ControllerTrainConfig,
    model: MetaController,
    train_loader_raw: DataLoader,
    validation_loader_raw: DataLoader,
    train_cache_path: str,
    validation_cache_path: str,
) -> Tuple[MaterializedCache | None, MaterializedCache | None]:
    """Return (train_cache, validation_cache) when the encoder is frozen, else (None, None).

    When the encoder is unfrozen we must re-run it each batch, so we skip
    caching entirely and use the live-encoder train path.
    """
    if config.unfreeze_encoder:
        return None, None

    if Path(train_cache_path).exists():
        print(f"[compute_advantage] stage=load_train_cache path={train_cache_path}", flush=True)
        train_cache = _load_materialized_cache(
            train_cache_path,
            manifest_path=config.packed_train_data,
            encoder_checkpoint=config.encoder_checkpoint,
        )
    else:
        print("[compute_advantage] stage=materialize_train_encoder", flush=True)
        train_cache = _save_materialized_cache_shards(
            train_cache_path,
            model,
            train_loader_raw,
            manifest_path=config.packed_train_data,
            encoder_checkpoint=config.encoder_checkpoint,
            log_interval=config.log_interval,
            split_name="train",
            max_snapshots_per_shard=config.materialized_cache_shard_snapshots,
        )
    if Path(validation_cache_path).exists():
        print(f"[compute_advantage] stage=load_validation_cache path={validation_cache_path}", flush=True)
        validation_cache = _load_materialized_cache(
            validation_cache_path,
            manifest_path=config.packed_validation_data,
            encoder_checkpoint=config.encoder_checkpoint,
        )
    else:
        print("[compute_advantage] stage=materialize_validation_encoder", flush=True)
        validation_cache = _save_materialized_cache_shards(
            validation_cache_path,
            model,
            validation_loader_raw,
            manifest_path=config.packed_validation_data,
            encoder_checkpoint=config.encoder_checkpoint,
            log_interval=config.log_interval,
            split_name="validation",
            max_snapshots_per_shard=config.materialized_cache_shard_snapshots,
        )
    return train_cache, validation_cache


def _effective_greedy_eval_step_interval(config: ControllerTrainConfig) -> int | None:
    """Return the step interval for greedy eval, or None if disabled."""
    if config.greedy_eval_step_interval is not None:
        return config.greedy_eval_step_interval if config.greedy_eval_step_interval > 0 else None
    if config.validation_step_interval is not None and config.validation_step_interval > 0:
        return config.validation_step_interval
    return None


@dataclass
class _StepEvalState:
    best_greedy_regret: float = float("inf")
    best_metadata: dict | None = None


def _run_one_epoch(
    config: ControllerTrainConfig,
    epoch: int,
    model: MetaController,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_cache: MaterializedCache | None,
    train_loader: DataLoader | None,
    bin_weights: torch.Tensor | None,
    bin_boundaries: torch.Tensor | None,
    logger: ControllerTrainMetricsLogger | None = None,
    validation_cache: MaterializedCache | None = None,
    validation_loader: DataLoader | None = None,
    validation_dataset: ControllerEpisodeDataset | None = None,
    oracle_config: BudgetedOracleConfig | None = None,
    step_eval_state: _StepEvalState | None = None,
) -> AdvantageMetrics:
    """Drive one training epoch, dispatching to the cached or live-encoder path.

    Cached path is used whenever the encoder is frozen (the common case);
    live path runs the encoder on every batch.
    """
    after_step: Callable[[int, int], None] | None = None
    step_interval = config.validation_step_interval
    greedy_step_interval = _effective_greedy_eval_step_interval(config)
    if (
        logger is not None
        and step_eval_state is not None
        and validation_dataset is not None
        and oracle_config is not None
        and (
            (step_interval is not None and step_interval > 0)
            or greedy_step_interval is not None
        )
    ):
        def after_step(epoch: int, n_batches: int) -> None:
            _maybe_eval_at_training_step(
                config,
                epoch,
                n_batches,
                model,
                device,
                validation_cache,
                validation_loader,
                validation_dataset,
                oracle_config,
                logger,
                step_eval_state,
            )

    if train_cache is not None:
        train_metrics = _train_materialized_cache_epoch(
            model,
            train_cache,
            optimizer,
            device=device,
            batch_size=config.batch_size,
            sign_loss_weight=config.sign_loss_weight,
            nontrivial_loss_weight=config.nontrivial_loss_weight,
            max_grad_norm=config.max_grad_norm,
            epoch=epoch,
            log_interval=config.log_interval,
            seed=config.seed,
            bin_weights=bin_weights,
            bin_boundaries=bin_boundaries,
            max_batches=config.train_batches,
            logger=logger,
            after_step=after_step,
        )
    else:
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            sign_loss_weight=config.sign_loss_weight,
            max_grad_norm=config.max_grad_norm,
            epoch=epoch,
            log_interval=config.log_interval,
            max_batches=config.train_batches,
            logger=logger,
            after_step=after_step,
        )
    print(
        f"epoch={epoch}/{config.epochs} "
        f"train_total_loss={train_metrics.total_loss:.3f} "
        f"train_advantage_mse={train_metrics.advantage_mse:.3f} "
        f"train_sign_bce={train_metrics.sign_bce:.3f} "
        f"train_mean_abs_advantage_error={train_metrics.mean_abs_advantage_error:.3f} "
        f"train_sign_accuracy={train_metrics.sign_accuracy:.3f} "
        f"train_snapshots={train_metrics.examples}",
        flush=True,
    )
    return train_metrics


def _train_batches_per_epoch(
    config: ControllerTrainConfig,
    train_cache: MaterializedCache | None,
    train_loader: DataLoader | None,
) -> tuple[int, int | None]:
    """Return ``(batches run per epoch, full uncapped batches or None)``."""
    if train_cache is not None:
        full_batches = (train_cache.examples + config.batch_size - 1) // config.batch_size
    elif train_loader is not None:
        full_batches = len(train_loader)
    else:
        return 0, None
    run_batches = full_batches
    if config.train_batches is not None:
        run_batches = min(full_batches, config.train_batches)
    return run_batches, full_batches


def _run_validation(
    config: ControllerTrainConfig,
    epoch: int,
    model: MetaController,
    device: torch.device,
    validation_cache: MaterializedCache | None,
    validation_loader: DataLoader | None,
) -> AdvantageMetrics:
    if validation_cache is not None:
        validation_metrics = evaluate_materialized_cache_predictions(
            model,
            validation_cache,
            device=device,
            batch_size=config.batch_size,
            sign_loss_weight=config.sign_loss_weight,
            max_batches=config.max_validation_batches,
        )
    else:
        validation_metrics = evaluate_advantage_predictions(
            model,
            validation_loader,
            device=device,
            sign_loss_weight=config.sign_loss_weight,
            max_batches=config.max_validation_batches,
        )
    print(
        f"validation_epoch={epoch}/{config.epochs} "
        f"validation_total_loss={validation_metrics.total_loss:.3f} "
        f"validation_advantage_mse={validation_metrics.advantage_mse:.3f} "
        f"validation_sign_bce={validation_metrics.sign_bce:.3f} "
        f"validation_mean_abs_advantage_error={validation_metrics.mean_abs_advantage_error:.3f} "
        f"validation_sign_accuracy={validation_metrics.sign_accuracy:.3f} "
        f"validation_snapshots={validation_metrics.examples}",
        flush=True,
    )
    return validation_metrics


def _maybe_run_validation(
    config: ControllerTrainConfig,
    epoch: int,
    model: MetaController,
    device: torch.device,
    validation_cache: MaterializedCache | None,
    validation_loader: DataLoader | None,
) -> AdvantageMetrics | None:
    """Run validation if this epoch hits the validation interval. Reported but not used for checkpoint selection."""
    if config.validation_step_interval is not None and config.validation_step_interval > 0:
        return None
    if config.validation_interval <= 0:
        return None
    if not (epoch % config.validation_interval == 0 or epoch == config.epochs):
        return None
    return _run_validation(
        config,
        epoch,
        model,
        device,
        validation_cache,
        validation_loader,
    )


def _maybe_eval_at_training_step(
    config: ControllerTrainConfig,
    epoch: int,
    n_batches: int,
    model: MetaController,
    device: torch.device,
    validation_cache: MaterializedCache | None,
    validation_loader: DataLoader | None,
    validation_dataset: ControllerEpisodeDataset,
    oracle_config: BudgetedOracleConfig,
    logger: ControllerTrainMetricsLogger | None,
    step_eval_state: _StepEvalState,
) -> None:
    if n_batches <= 0:
        return
    validation_interval = config.validation_step_interval
    greedy_interval = _effective_greedy_eval_step_interval(config)
    run_validation = validation_interval is not None and validation_interval > 0 and n_batches % validation_interval == 0
    run_greedy = greedy_interval is not None and n_batches % greedy_interval == 0
    if not run_validation and not run_greedy:
        return

    validation_metrics: AdvantageMetrics | None = None
    greedy_metrics: GreedyPolicyMetrics | None = None
    if run_validation:
        validation_metrics = _run_validation(
            config,
            epoch,
            model,
            device,
            validation_cache,
            validation_loader,
        )
    if run_greedy:
        step_eval_state.best_greedy_regret, step_eval_state.best_metadata, greedy_metrics = _run_greedy_eval(
            config,
            epoch,
            model,
            validation_dataset,
            validation_cache,
            oracle_config,
            save_best=True,
            best_greedy_regret=step_eval_state.best_greedy_regret,
            best_metadata=step_eval_state.best_metadata,
        )
    if logger is not None:
        logger.after_epoch_eval(epoch, validation_metrics, greedy_metrics)


def _run_greedy_eval(
    config: ControllerTrainConfig,
    epoch: int,
    model: MetaController,
    validation_dataset: ControllerEpisodeDataset,
    validation_cache: MaterializedCache | None,
    oracle_config: BudgetedOracleConfig,
    *,
    save_best: bool,
    best_greedy_regret: float,
    best_metadata: dict | None,
) -> Tuple[float, dict | None, GreedyPolicyMetrics]:
    if validation_cache is None:
        raise RuntimeError(
            "Greedy evaluation requires a materialized validation cache. "
            "Either drop --unfreeze-encoder or set --greedy-eval-interval 0."
        )
    diagnostics: List[Dict[str, Any]] | None = (
        [] if (save_best and epoch == config.epochs and config.output_diagnostics) else None
    )
    greedy_metrics = evaluate_batched_greedy_policy(
        model,
        validation_dataset,
        validation_cache,
        oracle_config,
        log_interval=config.log_interval,
        diagnostics_out=diagnostics,
        max_eval_episodes=config.max_greedy_eval_episodes,
    )
    print(
        f"greedy_epoch={epoch}/{config.epochs} "
        f"exact_stop_step_accuracy={greedy_metrics.exact_stop_step_accuracy:.3f} "
        f"first_action_accuracy={greedy_metrics.first_action_accuracy:.3f} "
        f"average_return={greedy_metrics.average_return:.3f} "
        f"average_oracle_value={greedy_metrics.average_oracle_value:.3f} "
        f"average_regret={greedy_metrics.average_regret:.3f} "
        f"average_expansions={greedy_metrics.average_expansions:.3f} "
        f"evaluated_episodes={greedy_metrics.evaluated_episodes}",
        flush=True,
    )
    if save_best and greedy_metrics.average_regret < best_greedy_regret:
        best_greedy_regret = greedy_metrics.average_regret
        best_metadata = {
            "stage": "compute_advantage_controller",
            "epoch": epoch,
            "average_regret": greedy_metrics.average_regret,
            "average_return": greedy_metrics.average_return,
            "exact_stop_step_accuracy": greedy_metrics.exact_stop_step_accuracy,
            "encoder_checkpoint": config.encoder_checkpoint,
            "unfreeze_encoder": config.unfreeze_encoder,
            "sign_loss_weight": config.sign_loss_weight,
            "nontrivial_loss_weight": config.nontrivial_loss_weight,
            "inverse_freq_weights": config.inverse_freq_weights,
            "separate_sign_head": config.separate_sign_head,
            "controller_inputs": list(config.controller_inputs),
            **budgeted_oracle_metadata(oracle_config),
        }
        if config.output_checkpoint:
            print(f"[compute_advantage] stage=save_best regret={best_greedy_regret:.6f} path={config.output_checkpoint}", flush=True)
            _save_checkpoint(config.output_checkpoint, model, best_metadata)
    if diagnostics is not None:
        _write_diagnostics(diagnostics, config.output_diagnostics)
    return best_greedy_regret, best_metadata, greedy_metrics


def _maybe_run_greedy_eval_and_save_best(
    config: ControllerTrainConfig,
    epoch: int,
    model: MetaController,
    validation_dataset: ControllerEpisodeDataset,
    validation_cache: MaterializedCache | None,
    oracle_config: BudgetedOracleConfig,
    best_greedy_regret: float,
    best_metadata: dict | None,
) -> Tuple[float, dict | None, GreedyPolicyMetrics | None]:
    if config.greedy_eval_interval <= 0:
        return best_greedy_regret, best_metadata, None
    if _effective_greedy_eval_step_interval(config) is not None:
        return best_greedy_regret, best_metadata, None
    if not (epoch % config.greedy_eval_interval == 0 or epoch == config.epochs):
        return best_greedy_regret, best_metadata, None
    best_greedy_regret, best_metadata, greedy_metrics = _run_greedy_eval(
        config,
        epoch,
        model,
        validation_dataset,
        validation_cache,
        oracle_config,
        save_best=True,
        best_greedy_regret=best_greedy_regret,
        best_metadata=best_metadata,
    )
    return best_greedy_regret, best_metadata, greedy_metrics


def _run_initial_eval(
    config: ControllerTrainConfig,
    model: MetaController,
    device: torch.device,
    validation_cache: MaterializedCache | None,
    validation_loader: DataLoader | None,
    validation_dataset: ControllerEpisodeDataset,
    oracle_config: BudgetedOracleConfig,
    logger: ControllerTrainMetricsLogger | None,
) -> None:
    """Validation + greedy baseline at ``n_batches=0`` before any training steps."""
    print("[compute_advantage] stage=initial_eval", flush=True)
    validation_metrics: AdvantageMetrics | None = None
    greedy_metrics: GreedyPolicyMetrics | None = None
    if config.validation_interval > 0 or (
        config.validation_step_interval is not None and config.validation_step_interval > 0
    ):
        validation_metrics = _run_validation(
            config,
            0,
            model,
            device,
            validation_cache,
            validation_loader,
        )
    if config.greedy_eval_interval > 0 or _effective_greedy_eval_step_interval(config) is not None:
        _, _, greedy_metrics = _run_greedy_eval(
            config,
            0,
            model,
            validation_dataset,
            validation_cache,
            oracle_config,
            save_best=False,
            best_greedy_regret=float("inf"),
            best_metadata=None,
        )
    if logger is not None:
        logger.after_epoch_eval(0, validation_metrics, greedy_metrics)
        logger.maybe_refresh_plot()


def _run_training_loop(
    config: ControllerTrainConfig,
    model: MetaController,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    train_cache: MaterializedCache | None,
    validation_cache: MaterializedCache | None,
    train_loader: DataLoader | None,
    validation_loader: DataLoader | None,
    validation_dataset: ControllerEpisodeDataset,
    oracle_config: BudgetedOracleConfig,
    bin_weights: torch.Tensor | None,
    bin_boundaries: torch.Tensor | None,
) -> dict | None:
    """Iterate epochs: train, validate, run greedy eval, track best regret. Returns the best-epoch metadata dict (or None if greedy eval never ran/improved)."""
    # We track best-regret-so-far and only save when greedy regret improves.
    # The greedy eval is the primary checkpoint-selection criterion, not
    # validation MSE.
    best_greedy_regret = float("inf")
    best_metadata: dict | None = None
    step_eval_state = _StepEvalState()
    logger = ControllerTrainMetricsLogger.from_config(config)
    train_batches, full_train_batches = _train_batches_per_epoch(config, train_cache, train_loader)
    if config.train_batches is not None and full_train_batches is not None:
        print(
            f"[compute_advantage] train_batches={train_batches} "
            f"(capped from full_epoch={full_train_batches})",
            flush=True,
        )
    else:
        print(f"[compute_advantage] train_batches_per_epoch={train_batches}", flush=True)
    if logger is not None:
        logger.write_run_start(
            config,
            oracle_config,
            train_batches_per_epoch=train_batches,
            full_train_batches_per_epoch=full_train_batches,
        )
    _run_initial_eval(
        config,
        model,
        device,
        validation_cache,
        validation_loader,
        validation_dataset,
        oracle_config,
        logger,
    )
    print("[compute_advantage] stage=train_start", flush=True)

    for epoch in range(1, config.epochs + 1):
        _run_one_epoch(
            config,
            epoch,
            model,
            optimizer,
            device,
            train_cache,
            train_loader,
            bin_weights,
            bin_boundaries,
            logger=logger,
            validation_cache=validation_cache,
            validation_loader=validation_loader,
            validation_dataset=validation_dataset,
            oracle_config=oracle_config,
            step_eval_state=step_eval_state,
        )
        if scheduler is not None:
            scheduler.step()
        best_greedy_regret = step_eval_state.best_greedy_regret
        best_metadata = step_eval_state.best_metadata
        validation_metrics = _maybe_run_validation(
            config,
            epoch,
            model,
            device,
            validation_cache,
            validation_loader,
        )
        best_greedy_regret, best_metadata, greedy_metrics = _maybe_run_greedy_eval_and_save_best(
            config,
            epoch,
            model,
            validation_dataset,
            validation_cache,
            oracle_config,
            best_greedy_regret,
            best_metadata,
        )
        if logger is not None:
            logger.after_epoch_eval(epoch, validation_metrics, greedy_metrics)
    if logger is not None:
        logger.finish()
    return best_metadata


def _save_fallback_checkpoint_if_needed(
    config: ControllerTrainConfig,
    model: MetaController,
    oracle_config: BudgetedOracleConfig,
    best_metadata: dict | None,
) -> None:
    """If greedy eval never produced a checkpoint, save the final-epoch weights as a fallback so the run still produces an output."""
    if config.output_checkpoint and best_metadata is None:
        _save_checkpoint(
            config.output_checkpoint,
            model,
            {
                "stage": "compute_advantage_controller",
                "epoch": config.epochs,
                "encoder_checkpoint": config.encoder_checkpoint,
                "unfreeze_encoder": config.unfreeze_encoder,
                "controller_inputs": list(config.controller_inputs),
                **budgeted_oracle_metadata(oracle_config),
            },
        )
    print("[compute_advantage] stage=done", flush=True)


def main(config: ControllerTrainConfig) -> None:
    """End-to-end driver: validate args, materialize cache (if frozen), train, eval, save."""
    global _SIGN_POS_WEIGHT
    _SIGN_POS_WEIGHT = config.sign_pos_weight
    if _SIGN_POS_WEIGHT != 1.0:
        print(f"[compute_advantage] sign_pos_weight={_SIGN_POS_WEIGHT} "
              f"(<1 penalises over-search in the sign-BCE)", flush=True)
    device, train_cache_path, validation_cache_path, oracle_config, schema = _seed_and_resolve_paths(config)
    model, optimizer, scheduler = _build_model_and_optimizer(config, schema)
    train_loader_raw, validation_loader_raw = _build_train_and_validation_loaders(config)
    train_cache, validation_cache = _materialize_or_load_caches(
        config,
        model,
        train_loader_raw,
        validation_loader_raw,
        train_cache_path,
        validation_cache_path,
    )
    # Live-encoder path keeps the raw loaders as the training source; cached
    # path uses the materialized features and the raw loaders are unused.
    train_loader = train_loader_raw if train_cache is None else None
    validation_loader = validation_loader_raw if validation_cache is None else None

    validation_dataset = ControllerEpisodeDataset(config.packed_validation_data)
    print(json.dumps(budgeted_oracle_metadata(oracle_config), sort_keys=True), flush=True)

    bin_weights: torch.Tensor | None = None
    bin_boundaries: torch.Tensor | None = None
    if config.inverse_freq_weights and train_cache is not None:
        print("[compute_advantage] stage=compute_inverse_freq_weights", flush=True)
        bin_weights, bin_boundaries = _compute_inverse_freq_bin_weights(train_cache)

    best_metadata = _run_training_loop(
        config,
        model,
        optimizer,
        scheduler,
        device,
        train_cache,
        validation_cache,
        train_loader,
        validation_loader,
        validation_dataset,
        oracle_config,
        bin_weights,
        bin_boundaries,
    )
    _save_fallback_checkpoint_if_needed(config, model, oracle_config, best_metadata)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot-metrics":
        import argparse

        sys.argv.pop(1)
        plot_parser = argparse.ArgumentParser(description="Plot controller metrics from metrics YAML")
        plot_parser.add_argument("metrics_path", help="Path to metrics.yaml from controller_train")
        plot_parser.add_argument(
            "-o",
            "--output",
            help="Output PNG path (default: <run>.png beside metrics YAML)",
        )
        plot_parser.add_argument("--title", help="Optional plot title")
        plot_args = plot_parser.parse_args()
        logger = ControllerTrainMetricsLogger.from_metrics_path(
            plot_args.metrics_path,
            output_path=plot_args.output,
            title=plot_args.title,
        )
        logger.refresh_plot(force=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "plot-metrics-compare":
        import argparse

        sys.argv.pop(1)
        compare_parser = argparse.ArgumentParser(
            description="Overlay controller val loss and regret curves across runs",
        )
        compare_parser.add_argument(
            "metrics_paths",
            nargs="+",
            help="Paths to metrics.yaml files from controller_train",
        )
        compare_parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Output PNG path for the comparison plot",
        )
        compare_parser.add_argument("--title", help="Optional plot title")
        compare_args = compare_parser.parse_args()
        ControllerTrainMetricsLogger.plot_comparison(
            compare_args.metrics_paths,
            compare_args.output,
            title=compare_args.title,
        )
    else:
        from cts._config import run_with_config_cli

        run_with_config_cli(ControllerTrainConfig, main)
