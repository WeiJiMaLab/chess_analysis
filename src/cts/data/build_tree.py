"""Single CLI entry point for the supervised branch of the pipeline.

Hosts two subcommands: ``generate-dataset``, which iterates over root FENs
and runs PUCT-bounded expansion through ``Lc0DirectEvalProvider`` to write
one JSON example per root into a shard directory; and
``pretrain-child-wdl-encoder``, which consumes those examples to train the
tree encoder against the per-child WDL pretraining objective. Dataset
generation is the slow, engine-bound first stage; it supports ``--resume``
and ``--start-index``/``--end-index`` for shard parallelism, plus a periodic
``--cache-clear-interval`` so the provider caches don't blow memory.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator

from cts.core.providers import (
    Lc0DirectEvalProvider,
    StockfishDirectEvalProvider,
    UciEngineConfig,
    UciEngineProcess,
)
from cts.core.schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeExample
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    TeacherSearchConfig,
    build_pretrain_example,
    load_pretrain_example_dataset,
    save_pretrain_example_to_directory,
)
from cts.models.gnn import ChildWdlModel
from cts.train.gnn_pretrain import (
    ChildWdlPretrainConfig,
    ChildWdlPretrainer,
)


# Stockfish 14 exposes UCI_Elo with ``min 1350``; setting it lower segfaults
# the process. The strength-ladder bottom rung must respect this floor.
STOCKFISH_MIN_ELO = 1350


def _default_lc0_engine_path() -> str:
    """Default lc0 binary location on macOS Homebrew installs."""
    return "/opt/homebrew/bin/lc0"


def _default_lc0_weights_path() -> str | None:
    """Look for a bundled weights file alongside this module; return None if absent."""
    candidate = os.path.join(os.path.dirname(__file__), "weights", "t1-256x10-distilled-swa-2432500.pb.gz")
    return candidate if os.path.exists(candidate) else None


class BuildTreeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["generate-dataset", "pretrain-child-wdl-encoder"]

    # generate-dataset
    # Which engine drives expansion. "lc0" runs the two-engine prior+value
    # provider; "stockfish" runs the single-process StockfishDirectEvalProvider
    # (uniform priors, WDL from Stockfish's internal eval->WDL model). Both
    # emit the same per-edge child-WDL targets the encoder pretrains on.
    provider: Literal["lc0", "stockfish"] = "lc0"
    # Stockfish-only knobs (ignored when provider == "lc0").
    sf_elo: Optional[int] = None  # UCI_Elo strength limit; None => full strength
    sf_search_limit_nodes: int = 100  # nodes per per-position Stockfish eval
    sf_search_limit_depth: Optional[int] = None  # optional fixed depth per eval
    engine_path: str = _default_lc0_engine_path()
    weights_path: Optional[str] = _default_lc0_weights_path()
    backend: Optional[str] = None
    prior_backend: Optional[str] = None
    value_backend: Optional[str] = None
    fens: Optional[str] = None
    output_dir: Optional[str] = None
    multipv: int = 8
    max_depth: int = 4
    search_budget: int = 64
    c_puct: float = 1.0
    # Generation-time leaf selection rule. "puct" (only option): AlphaZero
    # PUCT. A "befs" best-first-minimax alternative existed but was removed
    # 2026-07-08: its selection rule tunnel-visioned (confirmed, unfixed bug)
    # and it was already unreachable from the live pipeline -- see
    # TeacherSearchConfig.selection in teacher_targets.py and labnotebook.md.
    selection: Literal["puct"] = "puct"
    # Per-node scalar feature used as the search value (selection priority +
    # backup targets). "value" = win-loss in [-1,1] from WDL; "cp_order" =
    # Stockfish-native centipawns with mate scores folded into a ±20000 band
    # (see cts.core.providers.parsers.score_order_features); "tanh_cp_value" =
    # tanh(cp_order / tanh_cp_temperature), a desaturated alternative to
    # "value" computed at generation time (see cp_regen.md, Agent 3) --
    # requires tanh_cp_temperature to be set (stockfish provider only).
    value_feature: str = "value"
    # Temperature for the "tanh_cp_value" value_feature (ignored otherwise).
    # None (default) leaves every node's features exactly as before -- a
    # zero-cost no-op for every existing config.
    tanh_cp_temperature: Optional[float] = None
    min_nodes: int = 16
    max_nodes: int = 128
    prune_epsilon: Optional[float] = None  # value-prune knob (depth>=1); None = no prune. See R-PRUNING.
    prune_mode: Optional[str] = None       # None/"relative" | "absolute" (win-prob floor) | "rank" (top-k)
    seed: int = 0
    start_index: int = 0
    end_index: Optional[int] = None
    cache_clear_interval: int = 128
    log_interval: int = 25
    resume: bool = False

    # pretrain-child-wdl-encoder
    train_dir: Optional[str] = None
    validation_dir: Optional[str] = None
    output_checkpoint: Optional[str] = None
    device: str = "cpu"
    k: int = 2
    node_embed_hidden: int = 128
    d_embed: int = 128
    d_message: int = 128
    n_heads: int = 4
    d_att: int = 32
    decoder_hidden: int = 128
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    disable_persistent_workers: bool = False
    loss_type: Literal["huber", "mse"] = "huber"
    huber_delta: float = 1.0
    # When set, the pretrainer's validation pass also accumulates per-edge KL
    # into a (child_subtree_size, parent_depth) grid, and appends one JSONL
    # row per epoch to ``log_bucketed_kl_path`` containing the per-bin mean
    # KL + edge counts + marginals. Same bucketing as ``cts.analysis.audit_encoder_kl``
    # so the time series is directly comparable to the post-hoc audit.
    log_bucketed_kl_path: Optional[str] = None
    bucket_max_depth_bin: int = 12
    bucket_subtree_size_log_max: int = 10
    # When True, encoder pretraining weights each edge's cross-entropy by
    # its child's subtree size — see ``ChildWdlPretrainConfig`` for the
    # full rationale.
    loss_weight_by_subtree_size: bool = False
    # Per-epoch encoder checkpointing so an interrupted/incomplete run still yields
    # downstream-loadable weights. ``checkpoint_dir`` defaults to ``output_checkpoint``'s
    # directory; a tagged ``encoder_epoch{NNN}.pt`` (every ``checkpoint_every_epochs``) plus a
    # rolling ``encoder_latest.pt`` are written each epoch with the current encoder weights.
    checkpoint_dir: Optional[str] = None
    checkpoint_every_epochs: int = 1


def _feature_schema() -> NodeFeatureSchema:
    """Return the canonical encoder feature schema."""
    return tree_encoder_feature_schema()


def _load_fens(path: str) -> List[str]:
    """Read newline-separated FENs from ``path``, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _build_child_wdl_model(config: BuildTreeConfig, schema: NodeFeatureSchema) -> ChildWdlModel:
    """Instantiate the child-WDL pretraining model wired to the CLI hyperparameters."""
    return ChildWdlModel(
        k=config.k,
        node_feat=len(schema.feature_names),
        device=config.device,
        node_embed_hidden=config.node_embed_hidden,
        d_embed=config.d_embed,
        d_message=config.d_message,
        n_heads=config.n_heads,
        d_att=config.d_att,
        decoder_hidden=config.decoder_hidden,
    )


def _build_quality_config(
    config: BuildTreeConfig,
    *,
    target_normalization_version: str,
    search_config_id: str,
) -> TeacherSearchConfig:
    """Bundle the PUCT/depth/budget knobs into a ``TeacherSearchConfig``.

    Args:
        config: parsed config; reads ``max_depth``, ``search_budget``, ``c_puct``.
        target_normalization_version: version tag stamped onto the produced
            examples so downstream consumers can detect format drift.
        search_config_id: short label identifying this search config in
            saved examples (e.g. ``"supervised_branch_v1"``). A non-default
            selection rule would be appended so saved trees self-describe how
            they were built; dead in practice now that ``selection`` only
            accepts ``"puct"`` (the alternative, ``"befs"``, was removed
            2026-07-08), kept as a harmless no-op rather than ripped out.
    """
    if config.selection != "puct":
        search_config_id = f"{search_config_id}_{config.selection}"
    return TeacherSearchConfig(
        max_depth=config.max_depth,
        search_budget=config.search_budget,
        c_puct=config.c_puct,
        prior_feature="prior",
        value_feature=config.value_feature,
        target_normalization_version=target_normalization_version,
        search_config_id=search_config_id,
        prune_epsilon=config.prune_epsilon,
        prune_mode=config.prune_mode,
        selection=config.selection,
    )


def _selected_root_records(fens: List[str], start_index: int, end_index: int | None) -> List[tuple[int, str]]:
    """Slice the FEN list to the caller's shard range and pair each with its global index.

    The global index is preserved so output filenames and ``root_position_id``
    fields stay stable regardless of which shard processed them — critical for
    shard parallelism with ``--start-index``/``--end-index``.

    Args:
        fens: full list of root FENs loaded from disk.
        start_index: first index to include (inclusive); must be non-negative.
        end_index: one-past-the-last index; ``None`` means "to the end".
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    resolved_end = len(fens) if end_index is None else min(end_index, len(fens))
    if resolved_end < start_index:
        raise ValueError("end_index must be >= start_index.")
    return list(enumerate(fens[start_index:resolved_end], start=start_index))


def _output_example_path(output_dir: str, index: int) -> str:
    """Canonical filename for the example produced from root ``index``.

    Zero-padded prefix gives lexicographic ordering matching numeric order,
    which makes shard merging and ``--resume`` skip checks trivial.
    """
    return os.path.join(output_dir, f"{index:06d}_root_{index}.pt")


def _generate_and_save_examples(
    root_records: List[tuple[int, str]],
    provider,
    search_config: TeacherSearchConfig,
    node_budget_distribution: NodeBudgetDistribution,
    rng: random.Random,
    config: BuildTreeConfig,
) -> List[str]:
    """Run expansion + save for each root, handling resume/cache-clear/logging.

    For each (index, fen): if ``--resume`` and the output already exists,
    skip. Otherwise build a pretrain example via PUCT-bounded expansion and
    write it to disk. Periodically clear the provider's caches to keep
    memory bounded over long shards, and emit a progress line every
    ``--log-interval`` roots.

    Args:
        root_records: (global_index, fen) pairs to process; the index is
            embedded in the output filename and ``root_position_id``.
        provider: tree expansion provider (typically ``Lc0DirectEvalProvider``).
        search_config: PUCT/depth/budget settings forwarded to ``build_pretrain_example``.
        node_budget_distribution: sampler for per-root max-node budgets.
        rng: shared RNG for budget sampling; ``--seed`` controls reproducibility.
        config: parsed config; reads ``resume``, ``output_dir``,
            ``cache_clear_interval``, ``log_interval``.
    """
    saved_paths: List[str] = []
    total = len(root_records)
    started = time.time()

    for completed, (index, fen) in enumerate(root_records, start=1):
        # Resume path: a previously completed shard run will have left the
        # file on disk; skip without re-running the expensive expansion.
        if config.resume:
            existing_path = _output_example_path(config.output_dir, index)
            if os.path.exists(existing_path):
                saved_paths.append(existing_path)
                if completed % config.log_interval == 0 or completed == total:
                    elapsed = time.time() - started
                    print(
                        f"progress={completed}/{total} skipped_existing={len(saved_paths)} "
                        f"elapsed_s={elapsed:.1f}"
                    )
                continue

        example = build_pretrain_example(
            fen,
            provider,
            search_config,
            node_budget_distribution=node_budget_distribution,
            rng=rng,
            root_position_id=f"root_{index}",
            # Emit per-edge child-WDL targets — the supervision ysagiv's encoder
            # pretrain objective consumes. v5 made this opt-in and the worker
            # never opted in, so our trees had no edge targets (all-NaN on load),
            # diverging from the ysagiv reference.
            include_edge_wdl_targets=True,
        )
        path = save_pretrain_example_to_directory(config.output_dir, example, index)
        saved_paths.append(path)

        # Periodic cache flush: the provider memoizes per-position engine
        # results, which is great for hit rate but grows unbounded over a
        # long shard. Flushing every N roots caps RSS.
        if config.cache_clear_interval > 0 and completed % config.cache_clear_interval == 0:
            provider.clear_caches()

        if completed % config.log_interval == 0 or completed == total:
            elapsed = time.time() - started
            print(
                f"progress={completed}/{total} saved={len(saved_paths)} "
                f"elapsed_s={elapsed:.1f} roots_per_s={completed / max(elapsed, 1e-6):.2f}"
            )

    provider.clear_caches()
    return saved_paths


def _require_engine_ready_root_fens(root_records: List[tuple[int, str]]) -> None:
    """Reject malformed root FENs *before* any reach the engine.

    Stockfish 14 segfaults on a ``position fen <board-only>`` line when the
    FEN is missing its side-to-move/castling/en-passant fields (python-chess
    silently defaults them, so the bug only surfaces at the engine). A bare
    board placement is therefore ambiguous *and* engine-fatal: we fail loudly,
    naming the offending root, rather than crashing mid-search or — worse —
    silently corrupting the dataset with a guessed side-to-move.
    """
    bad: List[tuple[int, str]] = []
    for index, fen in root_records:
        # A well-formed FEN carries at least placement + side-to-move +
        # castling + en-passant (4 fields). The half/full-move counters are
        # optional for Stockfish, but the first four are load-bearing.
        if len(fen.split()) < 4:
            bad.append((index, fen))
    if bad:
        preview = ", ".join(f"index={i} fen={f!r}" for i, f in bad[:5])
        raise ValueError(
            f"{len(bad)} root FEN(s) are not well-formed (need >=4 space-separated "
            f"fields: placement, side-to-move, castling, en-passant). Stockfish "
            f"segfaults on board-only FENs. Fix the FEN source; do not feed these "
            f"to the engine. First offenders: {preview}"
        )


def generate_dataset_stockfish_command(config: BuildTreeConfig) -> None:
    """``generate-dataset`` with ``provider: stockfish``: one Stockfish process, expand each root, save.

    Unlike the lc0 path (two engines for priors + value), Stockfish uses a
    single process via ``StockfishDirectEvalProvider``: priors are uniform and
    WDL comes from Stockfish's internal eval->WDL model. ``sf_elo`` limits
    playing strength (the strength-ladder rungs); ``sf_search_limit_nodes``
    sets the per-position node budget. Edge child-WDL targets are emitted
    exactly as in the lc0 path.
    """
    search_config = _build_quality_config(
        config,
        target_normalization_version="v1",
        search_config_id="supervised_branch_sf_v1",
    )
    # Stockfish 14's UCI_Elo floor is 1350; anything lower segfaults the
    # process on the first analyse. Guard explicitly rather than letting the
    # crash surface as a cryptic EOF deep in the search loop.
    if config.sf_elo is not None and config.sf_elo < STOCKFISH_MIN_ELO:
        raise ValueError(
            f"sf_elo={config.sf_elo} is below Stockfish's UCI_Elo floor "
            f"({STOCKFISH_MIN_ELO}); the engine segfaults below it. "
            f"Use sf_elo>={STOCKFISH_MIN_ELO} or full strength (omit sf_elo)."
        )
    if config.value_feature == "tanh_cp_value" and config.tanh_cp_temperature is None:
        raise ValueError(
            "value_feature='tanh_cp_value' requires tanh_cp_temperature to be set "
            "(e.g. 300.0) -- it selects the desaturated tanh(cp_order/T) feature, "
            "which the provider only computes when a temperature is configured."
        )

    node_budget_distribution = NodeBudgetDistribution(config.min_nodes, config.max_nodes)
    rng = random.Random(config.seed)
    fens = _load_fens(config.fens)
    root_records = _selected_root_records(fens, config.start_index, config.end_index)
    if not root_records:
        raise ValueError("No root positions selected for dataset generation.")
    # Validate FEN well-formedness up front so a malformed root can never reach
    # (and segfault) the engine; this also stops silent dataset corruption.
    _require_engine_ready_root_fens(root_records)

    engine_config = UciEngineConfig(
        engine_path=config.engine_path,
        engine_kind="stockfish",
        movetime_ms=0,
        multipv=1,
        depth=config.sf_search_limit_depth,
        nodes=config.sf_search_limit_nodes,
    )
    with UciEngineProcess(engine_config) as engine:
        provider = StockfishDirectEvalProvider(
            engine,
            search_limit_nodes=config.sf_search_limit_nodes,
            search_limit_depth=config.sf_search_limit_depth,
            elo=config.sf_elo,
            metadata={"value_source": "stockfish_wdl", "prior_source": "uniform"},
            tanh_cp_temperature=config.tanh_cp_temperature,
        )
        saved_paths = _generate_and_save_examples(
            root_records,
            provider,
            search_config,
            node_budget_distribution,
            rng,
            config,
        )

    print(
        f"saved_examples={len(saved_paths)} output_dir={config.output_dir} "
        f"provider=stockfish sf_elo={config.sf_elo} "
        f"start_index={config.start_index} end_index={config.end_index if config.end_index is not None else len(fens)}"
    )


def generate_dataset_command(config: BuildTreeConfig) -> None:
    """Entry point for ``generate-dataset``: spin up engines, expand each root, save.

    Constructs two lc0 ``UciEngineProcess`` instances — one in ``classic``
    mode for priors via ``verbose-move-stats``, one in ``valuehead`` mode for
    WDL value targets — wraps them in an ``Lc0DirectEvalProvider``, and
    drives ``_generate_and_save_examples`` over the selected shard range.
    The two-engine split is deliberate: priors need multipv > 1 with
    classic search, while the valuehead needs ``UCI_ShowWDL`` and a
    different engine mode.
    """
    if config.provider == "stockfish":
        generate_dataset_stockfish_command(config)
        return
    search_config = _build_quality_config(
        config,
        target_normalization_version="v1",
        search_config_id="supervised_branch_v1",
    )
    node_budget_distribution = NodeBudgetDistribution(config.min_nodes, config.max_nodes)
    rng = random.Random(config.seed)
    fens = _load_fens(config.fens)
    root_records = _selected_root_records(fens, config.start_index, config.end_index)
    if not root_records:
        raise ValueError("No root positions selected for dataset generation.")

    # Per-engine backend overrides: ``--backend`` sets both; ``--prior-backend``
    # and ``--value-backend`` override individually. Useful when prior and
    # value engines should run on different devices (e.g. CPU prior + CUDA value).
    prior_uci_options: dict[str, str] = {}
    value_uci_options: dict[str, str] = {}
    if config.backend:
        prior_uci_options["Backend"] = config.backend
        value_uci_options["Backend"] = config.backend
    if config.prior_backend:
        prior_uci_options["Backend"] = config.prior_backend
    if config.value_backend:
        value_uci_options["Backend"] = config.value_backend
    value_uci_options.setdefault("UCI_ShowWDL", "true")

    # Prior engine: classic search with multipv to harvest sibling priors
    # from ``verbose-move-stats``. ``nodes=1`` keeps the search cheap; we
    # only need the prior distribution, not a deep evaluation.
    prior_config = UciEngineConfig(
        engine_path=config.engine_path,
        engine_kind="lc0",
        engine_mode="classic",
        movetime_ms=0,
        multipv=config.multipv,
        depth=None,
        nodes=1,
        weights_path=config.weights_path,
        uci_options=prior_uci_options,
        set_multipv=False,
        enable_verbose_move_stats=True,
    )
    # Value engine: valuehead mode emits the raw WDL distribution from the
    # network without running search. multipv=1; we only need the position's
    # own value, not move-conditioned values.
    value_config = UciEngineConfig(
        engine_path=config.engine_path,
        engine_kind="lc0",
        engine_mode="valuehead",
        movetime_ms=0,
        multipv=1,
        depth=None,
        nodes=1,
        weights_path=config.weights_path,
        uci_options=value_uci_options,
        set_multipv=False,
        enable_verbose_move_stats=False,
    )
    with UciEngineProcess(prior_config) as prior_engine, UciEngineProcess(value_config) as value_engine:
        provider = Lc0DirectEvalProvider(
            prior_engine,
            value_engine,
            metadata={"engine_kind": "lc0", "value_source": "valuehead", "prior_source": "classic_nodes_1"},
        )
        saved_paths = _generate_and_save_examples(
            root_records,
            provider,
            search_config,
            node_budget_distribution,
            rng,
            config,
        )

    print(
        f"saved_examples={len(saved_paths)} output_dir={config.output_dir} "
        f"start_index={config.start_index} end_index={config.end_index if config.end_index is not None else len(fens)}"
    )


def _save_encoder_training_curves(history: list[dict], output_checkpoint: str) -> None:
    """Write the per-epoch train/val loss trajectory (CSV) and a curve figure next to the
    checkpoint — ``loss_gap`` (total_loss minus target_entropy; "the real KL we're driving toward
    zero", per ``ChildWdlMetrics``) is the primary metric plotted, since ``total_loss`` alone isn't
    comparable across batches with different target-entropy floors. Same CSV+PNG convention as
    ``cts.train.pg_controller_train._save_training_curves`` (the readout-fit curves), kept as a
    separate, simpler implementation here since ``history`` entries are ``ChildWdlMetrics``
    dataclasses (train/validation pair), not plain dicts."""
    if not history:
        return
    import csv
    from pathlib import Path
    base = Path(output_checkpoint).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "epoch": i + 1,
            "train_total_loss": h["train"].total_loss,
            "train_target_entropy": h["train"].target_entropy,
            "train_loss_gap": h["train"].loss_gap,
            "train_num_supervised_edges": h["train"].num_supervised_edges,
            "val_total_loss": h["validation"].total_loss,
            "val_target_entropy": h["validation"].target_entropy,
            "val_loss_gap": h["validation"].loss_gap,
            "val_num_supervised_edges": h["validation"].num_supervised_edges,
        }
        for i, h in enumerate(history)
    ]
    csv_path = Path(f"{base}_training_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[child-wdl-pretrain] training curve -> {csv_path}", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(ep, [r["train_loss_gap"] for r in rows], "-o", label="train loss_gap (KL)")
        ax.plot(ep, [r["val_loss_gap"] for r in rows], "-s", label="val loss_gap (KL)")
        ax.plot(ep, [r["train_total_loss"] for r in rows], ":", color="tab:blue", alpha=0.5,
               label="train total_loss")
        ax.plot(ep, [r["val_total_loss"] for r in rows], ":", color="tab:orange", alpha=0.5,
               label="val total_loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(loc="upper right")
        ax.set_title("Child-WDL encoder pretrain — loss over epochs")
        fig.tight_layout()
        fig.savefig(f"{base}_training_curve.png", dpi=150)
        plt.close(fig)
        print(f"[child-wdl-pretrain] training curve figure -> {base}_training_curve.png", flush=True)
    except Exception as e:  # plotting is best-effort; the CSV is the source of truth
        print(f"[child-wdl-pretrain] curve plot skipped: {e}", flush=True)


def pretrain_child_wdl_encoder_command(config: BuildTreeConfig) -> None:
    """Entry point for ``pretrain-child-wdl-encoder``: load examples, fit, save.

    Loads the generated train/validation example directories, constructs the
    encoder + child-WDL decoder, and runs the supervised pretraining loop.
    Supports auto-resume via a sibling ``*_resume.pt`` file written by the
    trainer at each epoch boundary. On completion, saves the best encoder
    and decoder checkpoints separately so downstream stages can load just
    the piece they need.
    """
    print(f"[child-wdl-pretrain] stage=load_train_dataset path={config.train_dir}", flush=True)
    train_examples = load_pretrain_example_dataset(config.train_dir)
    print(f"[child-wdl-pretrain] stage=load_validation_dataset path={config.validation_dir}", flush=True)
    validation_examples = load_pretrain_example_dataset(config.validation_dir)
    print(
        f"[child-wdl-pretrain] stage=datasets_ready train_examples={len(train_examples)} "
        f"validation_examples={len(validation_examples)}",
        flush=True,
    )

    print("[child-wdl-pretrain] stage=build_schema_and_tensorizer", flush=True)
    first_example = train_examples[0]
    if isinstance(first_example, TensorizedTreeExample):
        schema = NodeFeatureSchema.from_ordered_features(first_example.feature_names)
    else:
        schema = tree_encoder_feature_schema()
    print("[child-wdl-pretrain] stage=build_model", flush=True)
    model = _build_child_wdl_model(config, schema)
    print("[child-wdl-pretrain] stage=build_trainer", flush=True)
    trainer = ChildWdlPretrainer(
        model=model,
        schema=schema,
        device=config.device,
        train_examples=train_examples,
        validation_examples=validation_examples,
        config=ChildWdlPretrainConfig(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=not config.disable_persistent_workers,
            log_bucketed_kl=bool(config.log_bucketed_kl_path),
            bucket_max_depth_bin=config.bucket_max_depth_bin,
            bucket_subtree_size_log_max=config.bucket_subtree_size_log_max,
            loss_weight_by_subtree_size=config.loss_weight_by_subtree_size,
        ),
    )
    # Resume convention: the trainer writes ``<output>_resume.pt`` after
    # every completed epoch. If present, pick up at the next epoch rather
    # than restarting from scratch.
    resume_path = config.output_checkpoint.replace(".pt", "_resume.pt")
    start_epoch = 1
    if os.path.isfile(resume_path):
        print(f"[child-wdl-pretrain] stage=resume path={resume_path}", flush=True)
        start_epoch = trainer.load_training_state(resume_path) + 1
        print(f"[child-wdl-pretrain] resuming_from_epoch={start_epoch}", flush=True)

    print("[child-wdl-pretrain] stage=train_start", flush=True)

    # Per-epoch encoder checkpoints (current weights) so an interrupted run is still usable.
    ckpt_dir = config.checkpoint_dir or os.path.dirname(config.output_checkpoint) or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    # Per-phase last-logged-batch counter so the callback can throttle on a
    # batch-count interval rather than firing every step. Mutating dict
    # (not int) so the closures below can update it in place.
    last_logged_batch = {"train": 0, "validation": 0}

    def _log_batch_progress(
        epoch_index,
        phase,
        batch_index,
        total_batches,
        seen_examples,
        seen_edges,
        total_loss,
        target_entropy,
        loss_gap,
    ):
        """Per-batch progress line; emits at first/last batch or every ``--log-interval`` batches."""
        interval = config.log_interval
        should_log = (
            batch_index == 1
            or batch_index == total_batches
            or batch_index - last_logged_batch[phase] >= interval
        )
        if not should_log:
            return
        last_logged_batch[phase] = batch_index
        print(
            f"[child-wdl-pretrain] epoch={epoch_index}/{config.epochs} phase={phase} "
            f"batch={batch_index}/{total_batches} seen_examples={seen_examples} "
            f"seen_edges={seen_edges} total_loss={total_loss:.6f} "
            f"target_entropy={target_entropy:.6f} loss_gap={loss_gap:.6f}",
            flush=True,
        )

    def _log_epoch(epoch_index, train_metrics, validation_metrics):
        """End-of-epoch summary line with train + validation metrics side by side."""
        print(
            f"epoch={epoch_index}/{config.epochs} "
            f"train_total_loss={train_metrics.total_loss:.6f} "
            f"train_target_entropy={train_metrics.target_entropy:.6f} "
            f"train_loss_gap={train_metrics.loss_gap:.6f} "
            f"train_supervised_edges={train_metrics.num_supervised_edges} "
            f"val_total_loss={validation_metrics.total_loss:.6f} "
            f"val_target_entropy={validation_metrics.target_entropy:.6f} "
            f"val_loss_gap={validation_metrics.loss_gap:.6f} "
            f"val_supervised_edges={validation_metrics.num_supervised_edges}",
            flush=True,
        )
        # Save the CURRENT encoder this epoch (best is only restored at the very end of fit),
        # so a partial run still yields downstream-loadable weights.
        if config.checkpoint_every_epochs > 0 and epoch_index % config.checkpoint_every_epochs == 0:
            meta = {
                "stage": "supervised_pretrain_epoch",
                "epoch": int(epoch_index),
                "val_loss_gap": float(validation_metrics.loss_gap),
                "pretrain_objective": "search_consolidated_edge_wdl_v1",
            }
            trainer.save_best_encoder(os.path.join(ckpt_dir, f"encoder_epoch{epoch_index:03d}.pt"), metadata=meta)
            trainer.save_best_encoder(os.path.join(ckpt_dir, "encoder_latest.pt"), metadata=meta)
            print(f"[child-wdl-pretrain] stage=epoch_checkpoint epoch={epoch_index} dir={ckpt_dir}", flush=True)

    # Optional JSONL writer for the per-validation-epoch bucketed-KL time series.
    bucketed_kl_callback = None
    if config.log_bucketed_kl_path:
        os.makedirs(os.path.dirname(config.log_bucketed_kl_path) or ".", exist_ok=True)
        if start_epoch == 1 and os.path.isfile(config.log_bucketed_kl_path):
            # Fresh run: truncate any stale JSONL from a previous run targeting the same path.
            os.remove(config.log_bucketed_kl_path)

        def _write_bucketed_row(epoch_index: int, summary: dict) -> None:
            row = {"epoch": int(epoch_index), **summary}
            with open(config.log_bucketed_kl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

        bucketed_kl_callback = _write_bucketed_row

    history = trainer.fit(
        progress_callback=_log_epoch,
        batch_progress_callback=_log_batch_progress,
        start_epoch=start_epoch,
        resume_path=resume_path,
        bucketed_kl_callback=bucketed_kl_callback,
    )
    # Best encoder + best decoder go to separate files: downstream training
    # stages load the encoder alone, while analysis tools may need the
    # decoder to reconstruct WDL predictions.
    print(f"[child-wdl-pretrain] stage=save_checkpoint path={config.output_checkpoint}", flush=True)
    trainer.save_best_encoder(
        config.output_checkpoint,
        metadata={
            "stage": "supervised_pretrain",
            "pretrain_objective": "search_consolidated_edge_wdl_v1",
        },
    )
    decoder_path = config.output_checkpoint.replace(".pt", "_decoder.pt")
    trainer.save_best_decoder(
        decoder_path,
        metadata={
            "stage": "supervised_pretrain",
            "encoder_checkpoint": config.output_checkpoint,
        },
    )
    print(f"[child-wdl-pretrain] stage=save_decoder path={decoder_path}", flush=True)
    _save_encoder_training_curves(history, config.output_checkpoint)
    final_validation = history[-1]["validation"]
    print(
        f"validation_total_loss={final_validation.total_loss:.6f} "
        f"validation_target_entropy={final_validation.target_entropy:.6f} "
        f"validation_loss_gap={final_validation.loss_gap:.6f} "
        f"validation_supervised_edges={final_validation.num_supervised_edges} "
        f"checkpoint={config.output_checkpoint}",
        flush=True,
    )


def main(config: BuildTreeConfig) -> None:
    """Dispatch to the selected subcommand based on ``config.command``."""
    if config.command == "generate-dataset":
        generate_dataset_command(config)
    elif config.command == "pretrain-child-wdl-encoder":
        pretrain_child_wdl_encoder_command(config)


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(BuildTreeConfig, main)
