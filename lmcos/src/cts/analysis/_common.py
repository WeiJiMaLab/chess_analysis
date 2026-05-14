"""Shared helpers used by multiple analysis scripts in ``cts.analysis``.

Consolidates regex parsers for training-log lines emitted by
``cts.train.controller_train``, plus a handful of utility helpers
(path rewriting, regret decomposition, oracle-stop binning, root-move
motif classification) that several analysis drivers had duplicated
verbatim. Importing the canonical versions from one place keeps the
regex schema in lockstep with the trainer's log writer and avoids the
sort of silent drift that produced an earlier divergence between
``analyze_oversearch_preference_evolution`` and the actual log format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    maintenance_cost,
    time_cost,
)

# TeacherSearchConfig lives behind a torch-dependent import chain; some
# analysis contexts intentionally run without torch installed (log-only
# re-summaries on a workstation). Mirror the try/except sentinel pattern
# the original scripts used so the rest of this module still imports.
try:
    from cts.data.preprocess_gnn.teacher_targets import TeacherSearchConfig
except ModuleNotFoundError:
    TeacherSearchConfig = None  # type: ignore[assignment]


# Per-epoch training-loss line emitted by ``cts.train.controller_train``:
#   ``epoch=N/N train_total_loss=X train_advantage_mse=X train_sign_bce=X
#    train_mean_abs_advantage_error=X train_sign_accuracy=X train_snapshots=N``
# The two ``train_total_loss`` / ``train_sign_bce`` subgroups are optional
# so older log formats (pre-total_loss/sign_bce) still parse with NaN for
# the missing fields. Canonical version: chosen from
# ``analyze_budgeted_controller_run`` because it matches the current
# trainer emitter exactly and keeps backward compatibility via the
# optional groups (the version in ``analyze_oversearch_preference_evolution``
# had drifted and omitted those new fields entirely).
TRAIN_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"(?:(?:train_total_loss=(?P<train_total_loss>-?\d+(?:\.\d+)?) )?)"
    r"train_advantage_mse=(?P<train_advantage_mse>-?\d+(?:\.\d+)?) "
    r"(?:(?:train_sign_bce=(?P<train_sign_bce>-?\d+(?:\.\d+)?) )?)"
    r"train_mean_abs_advantage_error=(?P<train_mean_abs_advantage_error>-?\d+(?:\.\d+)?) "
    r"train_sign_accuracy=(?P<train_sign_accuracy>-?\d+(?:\.\d+)?) "
    r"train_snapshots=(?P<train_snapshots>\d+)$"
)

# Per-epoch validation-loss line; mirrors TRAIN_RE with ``validation_``
# prefixes. Same canonical-version rationale as TRAIN_RE.
VALIDATION_RE = re.compile(
    r"^validation_epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"(?:(?:validation_total_loss=(?P<validation_total_loss>-?\d+(?:\.\d+)?) )?)"
    r"validation_advantage_mse=(?P<validation_advantage_mse>-?\d+(?:\.\d+)?) "
    r"(?:(?:validation_sign_bce=(?P<validation_sign_bce>-?\d+(?:\.\d+)?) )?)"
    r"validation_mean_abs_advantage_error=(?P<validation_mean_abs_advantage_error>-?\d+(?:\.\d+)?) "
    r"validation_sign_accuracy=(?P<validation_sign_accuracy>-?\d+(?:\.\d+)?) "
    r"validation_snapshots=(?P<validation_snapshots>\d+)$"
)

# Per-epoch greedy-policy eval line. Canonical version: matches the current
# trainer emitter, which does NOT print ``skipped_episodes``. The version
# in ``analyze_compute_advantage_training_log`` required a trailing
# ``skipped_episodes`` group and would silently fail to match real logs.
GREEDY_RE = re.compile(
    r"^greedy_epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"exact_stop_step_accuracy=(?P<exact_stop_step_accuracy>-?\d+(?:\.\d+)?) "
    r"first_action_accuracy=(?P<first_action_accuracy>-?\d+(?:\.\d+)?) "
    r"average_return=(?P<average_return>-?\d+(?:\.\d+)?) "
    r"average_oracle_value=(?P<average_oracle_value>-?\d+(?:\.\d+)?) "
    r"average_regret=(?P<average_regret>-?\d+(?:\.\d+)?) "
    r"average_expansions=(?P<average_expansions>-?\d+(?:\.\d+)?) "
    r"evaluated_episodes=(?P<evaluated_episodes>\d+)"
)


def _parse_number(value: str) -> int | float:
    """Parse a metric value as int unless it looks float-shaped."""
    return float(value) if "." in value or "e" in value.lower() else int(value)


def _parse_match_row(match: re.Match[str]) -> dict[str, Any]:
    """Convert a regex match into a dict; absent optional groups become NaN.

    Canonical version: chosen from ``analyze_budgeted_controller_run`` because
    it decides int-vs-float from the matched string syntax, which generalises
    to any new metric field without an explicit whitelist update.
    """
    row: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        row[key] = float("nan") if value is None else _parse_number(value)
    return row


def parse_log(log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a training ``.out`` log into (metadata, train, validation, greedy) rows.

    Canonical version: chosen from ``analyze_budgeted_controller_run`` because
    it recognises the JSON oracle-metadata blob emitted by the trainer and
    absorbs bare ``key=value`` lines into metadata — the
    ``analyze_oversearch_preference_evolution`` copy did the same things but
    inlined the row-dict construction instead of going through
    ``_parse_match_row``.
    """
    metadata: dict[str, Any] = {}
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    greedy_rows: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # JSON-shaped lines are the oracle metadata blob; absorb them into metadata.
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("oracle_type") == "budgeted_controller_v1":
                metadata.update(payload)
                continue

        match = TRAIN_RE.match(line)
        if match is not None:
            train_rows.append(_parse_match_row(match))
            continue
        match = VALIDATION_RE.match(line)
        if match is not None:
            validation_rows.append(_parse_match_row(match))
            continue
        match = GREEDY_RE.match(line)
        if match is not None:
            greedy_rows.append({key: _parse_number(value) for key, value in match.groupdict().items()})
            continue
        # Bare ``key=value`` lines (no spaces) end up in metadata too.
        if "=" in line and " " not in line:
            key, value = line.split("=", 1)
            if key and value:
                try:
                    metadata[key] = _parse_number(value)
                except ValueError:
                    metadata[key] = value
    return metadata, train_rows, validation_rows, greedy_rows


def _parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    """Parse ``OLD=NEW`` CLI strings into substitution pairs.

    Canonical version: both copies (budgeted, oversearch) were byte-identical.
    """
    rewrites: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected path rewrite OLD=NEW, got {value!r}.")
        old_prefix, new_prefix = value.split("=", 1)
        if not old_prefix:
            raise ValueError(f"Rewrite prefix cannot be empty: {value!r}.")
        rewrites.append((old_prefix, new_prefix))
    return rewrites


def _apply_path_rewrites(source_path: str, rewrites: list[tuple[str, str]]) -> str:
    """Apply prefix substitutions to remap cluster paths to local mounts.

    Canonical version: both copies (budgeted, oversearch) were byte-identical.
    """
    rewritten = source_path
    for old_prefix, new_prefix in rewrites:
        if rewritten.startswith(old_prefix):
            rewritten = new_prefix + rewritten[len(old_prefix):]
    return rewritten


def _analysis_quality_config(metadata: dict[str, Any]) -> "TeacherSearchConfig":
    """Reconstruct the teacher-search config used to build trimmed episodes.

    Canonical version: chosen from ``analyze_budgeted_controller_run`` because
    it explicitly guards against ``TeacherSearchConfig`` being None when torch
    isn't installed; the other two copies would have raised an opaque
    ``TypeError`` on construction in that environment.
    """
    if TeacherSearchConfig is None:
        raise RuntimeError("TeacherSearchConfig is unavailable; install the CTS environment to reconstruct moves.")
    return TeacherSearchConfig(
        max_depth=int(metadata.get("max_depth", 10)),
        search_budget=int(metadata.get("search_budget", 64)),
        c_puct=float(metadata.get("c_puct", 1.0)),
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def _episode_regret_decomposition(
    episode: dict[str, Any],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, float | int | str]:
    """Break episode regret into halt-reward, maintenance, and time components.

    Returns a dict with ``delta`` and ``budget_bucket_name`` for context plus
    the three regret terms (``halt_reward``, ``maintenance``, ``time``), a
    decomposed-regret sum, and a ``residual`` for any unattributed delta.
    """
    oracle_stop = int(episode["oracle_stop_step"])
    predicted_stop = int(episode["predicted_stop_step"])
    halt_rewards = [float(value) for value in episode["halt_rewards"]]
    tree_sizes = [int(value) for value in episode["tree_sizes"]]
    time_budgets = [int(value) for value in episode["time_budgets"]]

    oracle_maintenance = sum(maintenance_cost(tree_sizes[idx], oracle_config) for idx in range(oracle_stop))
    predicted_maintenance = sum(maintenance_cost(tree_sizes[idx], oracle_config) for idx in range(predicted_stop))
    oracle_time = sum(time_cost(time_budgets[idx], oracle_config) for idx in range(oracle_stop))
    predicted_time = sum(time_cost(time_budgets[idx], oracle_config) for idx in range(predicted_stop))

    halt_reward_term = halt_rewards[oracle_stop] - halt_rewards[predicted_stop]
    maintenance_term = predicted_maintenance - oracle_maintenance
    time_term = predicted_time - oracle_time
    total = halt_reward_term + maintenance_term + time_term

    return {
        "delta": predicted_stop - oracle_stop,
        "budget_bucket_name": str(episode["budget_bucket_name"]),
        "halt_reward_term": float(halt_reward_term),
        "maintenance_term": float(maintenance_term),
        "time_term": float(time_term),
        "decomposed_regret": float(total),
        "saved_regret": float(episode["regret"]),
        "residual": float(total - float(episode["regret"])),
    }


def _oracle_stop_bin_label(stop_step: int) -> str:
    """Coarse bin for the oracle stop step so plots aren't dominated by sparse tails.

    Canonical version: both copies (budgeted, compare_controller_diagnostics)
    were byte-identical.
    """
    if stop_step == 0:
        return "0"
    if stop_step == 1:
        return "1"
    if stop_step == 2:
        return "2"
    if stop_step == 3:
        return "3"
    if 4 <= stop_step <= 7:
        return "4-7"
    if 8 <= stop_step <= 15:
        return "8-15"
    return "16+"


def _compressed_move_sequence(best_moves: list[str]) -> list[str]:
    """Collapse consecutive repeats so ``[a,a,b,b,a]`` becomes ``[a,b,a]``.

    Canonical version: both copies (budgeted, source_root_churn) computed the
    same compressed sequence; we use the shorter docstring form.
    """
    compressed: list[str] = []
    for move in best_moves:
        if not compressed or compressed[-1] != move:
            compressed.append(move)
    return compressed


def _first_appearance_step(best_moves: list[str], target_move: str) -> int:
    """Return the first index at which ``target_move`` appears in ``best_moves``.

    Canonical version: both copies were byte-identical.
    """
    return next(index for index, move in enumerate(best_moves) if move == target_move)


def _stabilization_step(best_moves: list[str], target_move: str) -> int:
    """Return the first index after which every entry equals ``target_move``.

    Canonical version: both copies were byte-identical.
    Raises ``ValueError`` if no such index exists.
    """
    for index in range(len(best_moves)):
        if all(move == target_move for move in best_moves[index:]):
            return index
    raise ValueError("Target move never stabilizes.")


def _canonicalize_final_anchored_motif(compressed_moves: list[str]) -> str:
    """Relabel a compressed sequence with the final move as A, others as B, C, ...

    Canonical version: both copies were byte-identical.
    """
    final_move = compressed_moves[-1]
    mapping: dict[str, str] = {final_move: "A"}
    next_label = ord("B")
    labels: list[str] = []
    for move in compressed_moves:
        if move not in mapping:
            mapping[move] = chr(next_label)
            next_label += 1
        labels.append(mapping[move])
    return "".join(labels)
