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
from cts.core.schema import nodetargets_target_feature_names
from cts.models.gnn import ChildWdlModel, NodeTargetsModel
from cts.train.gnn_pretrain import (
    ChildWdlPretrainConfig,
    ChildWdlPretrainer,
    NodePretrainConfig,
    NodePretrainer,
)


def _default_lc0_engine_path() -> str:
    """Default lc0 binary location on macOS Homebrew installs."""
    return "/opt/homebrew/bin/lc0"


def _default_lc0_weights_path() -> str | None:
    """Look for a bundled weights file alongside this module; return None if absent."""
    candidate = os.path.join(os.path.dirname(__file__), "weights", "t1-256x10-distilled-swa-2432500.pb.gz")
    return candidate if os.path.exists(candidate) else None


class BuildTreeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["generate-dataset", "pretrain-child-wdl-encoder", "pretrain-nodetargets-encoder"]

    # generate-dataset
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
    min_nodes: int = 16
    max_nodes: int = 128
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
    #: Phased node targets Huber curriculum: weights aligned with :func:`nodetargets_target_feature_names`.
    nodetargets_target_weights: Optional[Tuple[float, ...]] = None
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

    @field_validator("nodetargets_target_weights", mode="before")
    @classmethod
    def _coerce_nodetargets_target_weights(cls, value: object) -> Optional[Tuple[float, ...]]:
        if value is None:
            return None
        seq = tuple(float(x) for x in value)  # type: ignore[arg-type]
        return seq
def _feature_schema() -> NodeFeatureSchema:
    """Return the canonical encoder feature schema."""
    return tree_encoder_feature_schema()


def _load_fens(path: str) -> List[str]:
    """Read newline-separated FENs from ``path``, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _build_nodetargets_model(config: BuildTreeConfig, schema: NodeFeatureSchema) -> NodeTargetsModel:
    """Instantiate the node-targets-supervised pretraining model (5-wide encoder input)."""
    return NodeTargetsModel(
        k=config.k,
        node_feat=len(schema.feature_names),
        device=config.device,
        node_embed_hidden=config.node_embed_hidden,
        d_embed=config.d_embed,
        d_message=config.d_message,
        n_heads=config.n_heads,
        d_att=config.d_att,
        decoder_hidden=config.decoder_hidden,
        num_node_targets=len(nodetargets_target_feature_names()),
    )


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
            saved examples (e.g. ``"supervised_branch_v1"``).
    """
    return TeacherSearchConfig(
        max_depth=config.max_depth,
        search_budget=config.search_budget,
        c_puct=config.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version=target_normalization_version,
        search_config_id=search_config_id,
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
            # diverging from the ysagiv reference. Additive: node-topology targets
            # (value_gap / policy_drift) are still produced.
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
    final_validation = history[-1]["validation"]
    print(
        f"validation_total_loss={final_validation.total_loss:.6f} "
        f"validation_target_entropy={final_validation.target_entropy:.6f} "
        f"validation_loss_gap={final_validation.loss_gap:.6f} "
        f"validation_supervised_edges={final_validation.num_supervised_edges} "
        f"checkpoint={config.output_checkpoint}",
        flush=True,
    )


def pretrain_nodetargets_encoder_command(config: BuildTreeConfig) -> None:
    """Entry point for ``pretrain-nodetargets-encoder``: fit encoder + ``NodeTargetsHead`` on packed node targets."""
    print(f"[nodetargets-pretrain] stage=load_train_dataset path={config.train_dir}", flush=True)
    train_examples = load_pretrain_example_dataset(config.train_dir)
    print(f"[nodetargets-pretrain] stage=load_validation_dataset path={config.validation_dir}", flush=True)
    validation_examples = load_pretrain_example_dataset(config.validation_dir)
    print(
        f"[nodetargets-pretrain] stage=datasets_ready train_examples={len(train_examples)} "
        f"validation_examples={len(validation_examples)}",
        flush=True,
    )

    schema = tree_encoder_feature_schema(node_targets=False)
    model = _build_nodetargets_model(config, schema)
    trainer = NodePretrainer(
        model=model,
        device=config.device,
        train_examples=train_examples,
        validation_examples=validation_examples,
        config=NodePretrainConfig(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            nodetargets_target_weights=config.nodetargets_target_weights,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=not config.disable_persistent_workers,
        ),
    )

    resume_path = config.output_checkpoint.replace(".pt", "_resume.pt")
    start_epoch = 1
    if os.path.isfile(resume_path):
        print(f"[nodetargets-pretrain] stage=resume path={resume_path}", flush=True)
        start_epoch = trainer.load_training_state(resume_path) + 1

    last_logged_batch = {"train": 0, "validation": 0}

    def _log_batch_progress(epoch_index, phase, batch_index, total_batches, seen_examples, seen_nodes, total_loss):
        interval = config.log_interval
        if not (
            batch_index == 1
            or batch_index == total_batches
            or batch_index - last_logged_batch[phase] >= interval
        ):
            return
        last_logged_batch[phase] = batch_index
        print(
            f"[nodetargets-pretrain] epoch={epoch_index}/{config.epochs} phase={phase} "
            f"batch={batch_index}/{total_batches} seen_examples={seen_examples} "
            f"seen_nodes={seen_nodes} total_loss={total_loss:.6f}",
            flush=True,
        )

    def _log_epoch(epoch_index, train_metrics, validation_metrics):
        print(
            f"epoch={epoch_index}/{config.epochs} "
            f"train_total_loss={train_metrics.total_loss:.6f} "
            f"train_supervised_nodes={train_metrics.num_supervised_nodes} "
            f"val_total_loss={validation_metrics.total_loss:.6f} "
            f"val_supervised_nodes={validation_metrics.num_supervised_nodes}",
            flush=True,
        )

    history = trainer.fit(
        progress_callback=_log_epoch,
        batch_progress_callback=_log_batch_progress,
        start_epoch=start_epoch,
        resume_path=resume_path,
    )

    trainer.save_best_encoder(
        config.output_checkpoint,
        metadata={"stage": "supervised_pretrain"},
    )
    head_path = config.output_checkpoint.replace(".pt", "_nodetargets_head.pt")
    trainer.save_best_node_targets_head(
        head_path,
        metadata={"encoder_checkpoint": config.output_checkpoint},
    )
    final_validation = history[-1]["validation"]
    print(
        f"validation_total_loss={final_validation.total_loss:.6f} "
        f"validation_supervised_nodes={final_validation.num_supervised_nodes} "
        f"checkpoint={config.output_checkpoint} nodetargets_head={head_path}",
        flush=True,
    )


def main(config: BuildTreeConfig) -> None:
    """Dispatch to the selected subcommand based on ``config.command``."""
    if config.command == "generate-dataset":
        generate_dataset_command(config)
    elif config.command == "pretrain-child-wdl-encoder":
        pretrain_child_wdl_encoder_command(config)
    elif config.command == "pretrain-nodetargets-encoder":
        pretrain_nodetargets_encoder_command(config)


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(BuildTreeConfig, main)
