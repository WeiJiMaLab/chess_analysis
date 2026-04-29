from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional

from GNN import ChildWdlModel, NodeValueModel, PolicyValueTreeSearchModel
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from supervised_branch import (
    ChildWdlPretrainConfig,
    ChildWdlPretrainer,
    FrozenEncoderControllerTrainer,
    GeneratedTreeHaltEnv,
    NodeBudgetDistribution,
    PPOConfig,
    ReinforceConfig,
    ReinforceControllerTrainer,
    SnapshotEpisodeCache,
    load_raw_pretrain_example_paths,
    load_pretrain_example_dataset,
    save_policy_value_checkpoint,
    SupervisedPretrainConfig,
    SupervisedPretrainer,
    TeacherSearchConfig,
    ToyHaltEnv,
    build_pretrain_example,
    evaluate_controller,
    save_pretrain_example_to_directory,
)
from tensorizer import TreeTensorizer
from tree import ExpansionChild, SearchTree
from uci_provider import (
    Lc0DirectEvalProvider,
    StockfishAnalysisParser,
    UciEngineConfig,
    UciEngineProcess,
    UciTreeExpansionProvider,
)


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


def _default_lc0_engine_path() -> str:
    return "/opt/homebrew/bin/lc0"


def _default_lc0_weights_path() -> str | None:
    candidate = os.path.join(os.path.dirname(__file__), "weights", "t1-256x10-distilled-swa-2432500.pb.gz")
    return candidate if os.path.exists(candidate) else None


def _load_fens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _build_tensorizer(args: argparse.Namespace, schema: NodeFeatureSchema) -> TreeTensorizer:
    return TreeTensorizer(schema, device=args.device)


def _build_node_value_model(args: argparse.Namespace, schema: NodeFeatureSchema) -> NodeValueModel:
    return NodeValueModel(
        k=args.k,
        node_feat=len(schema.feature_names),
        device=args.device,
        node_embed_hidden=args.node_embed_hidden,
        d_embed=args.d_embed,
        d_message=args.d_message,
        n_heads=args.n_heads,
        d_att=args.d_att,
        value_hidden=args.value_hidden,
    )


def _build_child_wdl_model(args: argparse.Namespace, schema: NodeFeatureSchema) -> ChildWdlModel:
    return ChildWdlModel(
        k=args.k,
        node_feat=len(schema.feature_names),
        device=args.device,
        node_embed_hidden=args.node_embed_hidden,
        d_embed=args.d_embed,
        d_message=args.d_message,
        n_heads=args.n_heads,
        d_att=args.d_att,
        decoder_hidden=args.decoder_hidden,
    )


def _build_policy_value_model(args: argparse.Namespace, schema: NodeFeatureSchema) -> PolicyValueTreeSearchModel:
    return PolicyValueTreeSearchModel(
        k=args.k,
        node_feat=len(schema.feature_names),
        device=args.device,
        node_embed_hidden=args.node_embed_hidden,
        d_embed=args.d_embed,
        d_message=args.d_message,
        n_heads=args.n_heads,
        d_att=args.d_att,
        controller_hidden=args.controller_hidden,
        value_hidden=args.value_hidden,
    )


def _build_quality_config(
    args: argparse.Namespace,
    *,
    target_normalization_version: str,
    search_config_id: str,
) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version=target_normalization_version,
        search_config_id=search_config_id,
    )


def _load_generated_data_paths(args: argparse.Namespace, stage: str) -> tuple[List[str], List[str]]:
    print(f"[{stage}] stage=load_train_paths path={args.train_data}", flush=True)
    train_paths = load_raw_pretrain_example_paths(args.train_data)
    print(f"[{stage}] stage=load_validation_paths path={args.validation_data}", flush=True)
    validation_paths = load_raw_pretrain_example_paths(args.validation_data)
    print(
        f"[{stage}] stage=paths_ready train_examples={len(train_paths)} validation_examples={len(validation_paths)}",
        flush=True,
    )
    return train_paths, validation_paths


def _build_generated_env(
    example_paths: List[str],
    quality_config: TeacherSearchConfig,
    args: argparse.Namespace,
    *,
    seed: int,
    shuffle: bool,
    max_cache_size: Optional[int] = None,
    episode_cache: Optional[SnapshotEpisodeCache] = None,
) -> GeneratedTreeHaltEnv:
    resolved_max_cache_size = args.max_cache_size if max_cache_size is None else max_cache_size
    return GeneratedTreeHaltEnv(
        example_paths=example_paths,
        quality_config=quality_config,
        continue_cost=args.continue_cost,
        seed=seed,
        shuffle=shuffle,
        max_cache_size=resolved_max_cache_size,
        episode_cache=episode_cache,
    )


def _build_generated_train_envs(
    example_paths: List[str],
    quality_config: TeacherSearchConfig,
    args: argparse.Namespace,
) -> List[GeneratedTreeHaltEnv]:
    shared_cache = SnapshotEpisodeCache(args.max_cache_size)
    return [
        _build_generated_env(
            example_paths,
            quality_config,
            args,
            seed=args.seed + env_index,
            shuffle=True,
            episode_cache=shared_cache,
        )
        for env_index in range(args.num_envs)
    ]


def _make_generated_eval_env_factory(
    example_paths: List[str],
    quality_config: TeacherSearchConfig,
    args: argparse.Namespace,
):
    shared_cache = SnapshotEpisodeCache(min(args.max_cache_size, 8))

    def _make_eval_env() -> GeneratedTreeHaltEnv:
        return _build_generated_env(
            example_paths,
            quality_config,
            args,
            seed=args.seed,
            shuffle=False,
            max_cache_size=min(args.max_cache_size, 8),
            episode_cache=shared_cache,
        )

    return _make_eval_env


def _add_shared_model_args(parser: argparse.ArgumentParser, *, include_controller_hidden: bool) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    if include_controller_hidden:
        parser.add_argument("--controller-hidden", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=128)


def _add_generated_quality_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--continue-cost", type=float, default=0.05)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--max-cache-size", type=int, default=32)


def _selected_root_records(fens: List[str], start_index: int, end_index: int | None) -> List[tuple[int, str]]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    resolved_end = len(fens) if end_index is None else min(end_index, len(fens))
    if resolved_end < start_index:
        raise ValueError("end_index must be >= start_index.")
    return list(enumerate(fens[start_index:resolved_end], start=start_index))


def _output_example_path(output_dir: str, index: int) -> str:
    return os.path.join(output_dir, f"{index:06d}_root_{index}.pt")


def _generate_and_save_examples(
    root_records: List[tuple[int, str]],
    provider,
    config: TeacherSearchConfig,
    node_budget_distribution: NodeBudgetDistribution,
    rng: random.Random,
    args: argparse.Namespace,
) -> List[str]:
    saved_paths: List[str] = []
    total = len(root_records)
    started = time.time()

    for completed, (index, fen) in enumerate(root_records, start=1):
        if args.resume:
            existing_path = _output_example_path(args.output_dir, index)
            if os.path.exists(existing_path):
                saved_paths.append(existing_path)
                if completed % args.log_interval == 0 or completed == total:
                    elapsed = time.time() - started
                    print(
                        f"progress={completed}/{total} skipped_existing={len(saved_paths)} "
                        f"elapsed_s={elapsed:.1f}"
                    )
                continue

        example = build_pretrain_example(
            fen,
            provider,
            config,
            node_budget_distribution=node_budget_distribution,
            rng=rng,
            root_position_id=f"root_{index}",
        )
        path = save_pretrain_example_to_directory(args.output_dir, example, index)
        saved_paths.append(path)

        if args.cache_clear_interval > 0 and completed % args.cache_clear_interval == 0:
            provider.clear_caches()

        if completed % args.log_interval == 0 or completed == total:
            elapsed = time.time() - started
            print(
                f"progress={completed}/{total} saved={len(saved_paths)} "
                f"elapsed_s={elapsed:.1f} roots_per_s={completed / max(elapsed, 1e-6):.2f}"
            )

    provider.clear_caches()
    return saved_paths


def _make_toy_env():
    def make_snapshot(best_value: float) -> SearchTree:
        tree = SearchTree()
        root_id = tree.create_root("toy-root", {"value": 0.0, "prior": 1.0})
        tree.add_children(
            root_id,
            [
                ExpansionChild("best", f"best-{best_value}", {"value": best_value, "prior": 0.7}),
                ExpansionChild("other", f"other-{best_value}", {"value": 0.1, "prior": 0.3}),
            ],
        )
        return tree

    return ToyHaltEnv(
        [make_snapshot(0.2), make_snapshot(0.5), make_snapshot(0.9)],
        continue_cost=0.05,
    )


def generate_dataset_command(args: argparse.Namespace) -> None:
    config = _build_quality_config(
        args,
        target_normalization_version="v1",
        search_config_id="supervised_branch_v1",
    )
    node_budget_distribution = NodeBudgetDistribution(args.min_nodes, args.max_nodes)
    rng = random.Random(args.seed)
    fens = _load_fens(args.fens)
    root_records = _selected_root_records(fens, args.start_index, args.end_index)
    if not root_records:
        raise ValueError("No root positions selected for dataset generation.")

    uci_options = {}
    if args.engine_kind == "lc0" and args.backend:
        uci_options["Backend"] = args.backend

    engine_config = UciEngineConfig(
        engine_path=args.engine_path,
        engine_kind=args.engine_kind,
        movetime_ms=args.movetime_ms,
        multipv=args.multipv,
        depth=args.depth,
        nodes=args.nodes,
        weights_path=args.weights_path,
        uci_options=uci_options,
    )
    if args.engine_kind == "lc0":
        prior_uci_options = dict(uci_options)
        value_uci_options = dict(uci_options)
        if args.prior_backend:
            prior_uci_options["Backend"] = args.prior_backend
        if args.value_backend:
            value_uci_options["Backend"] = args.value_backend
        value_uci_options.setdefault("UCI_ShowWDL", "true")
        prior_config = UciEngineConfig(
            engine_path=args.engine_path,
            engine_kind="lc0",
            engine_mode="classic",
            movetime_ms=0,
            multipv=args.multipv,
            depth=None,
            nodes=1,
            weights_path=args.weights_path,
            uci_options=prior_uci_options,
            set_multipv=False,
            enable_verbose_move_stats=True,
        )
        value_config = UciEngineConfig(
            engine_path=args.engine_path,
            engine_kind="lc0",
            engine_mode="valuehead",
            movetime_ms=0,
            multipv=1,
            depth=None,
            nodes=1,
            weights_path=args.weights_path,
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
                config,
                node_budget_distribution,
                rng,
                args,
            )
    else:
        with UciEngineProcess(engine_config) as engine_process:
            provider = UciTreeExpansionProvider(
                engine_process,
                StockfishAnalysisParser(),
                metadata={"engine_kind": engine_process.config.engine_kind},
            )
            saved_paths = _generate_and_save_examples(
                root_records,
                provider,
                config,
                node_budget_distribution,
                rng,
                args,
            )

    print(
        f"saved_examples={len(saved_paths)} output_dir={args.output_dir} "
        f"start_index={args.start_index} end_index={args.end_index if args.end_index is not None else len(fens)}"
    )


def pretrain_encoder_command(args: argparse.Namespace) -> None:
    print(f"[pretrain] stage=load_train_dataset path={args.train_dir}", flush=True)
    train_examples = load_pretrain_example_dataset(args.train_dir)
    print(f"[pretrain] stage=load_validation_dataset path={args.validation_dir}", flush=True)
    validation_examples = load_pretrain_example_dataset(args.validation_dir)
    print(
        f"[pretrain] stage=datasets_ready train_examples={len(train_examples)} "
        f"validation_examples={len(validation_examples)}",
        flush=True,
    )

    print("[pretrain] stage=build_schema_and_tensorizer", flush=True)
    schema = _feature_schema()
    tensorizer = _build_tensorizer(args, schema)
    print("[pretrain] stage=build_model", flush=True)
    model = _build_node_value_model(args, schema)
    print("[pretrain] stage=build_trainer", flush=True)
    trainer = SupervisedPretrainer(
        model=model,
        tensorizer=tensorizer,
        train_examples=train_examples,
        validation_examples=validation_examples,
        config=SupervisedPretrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            root_loss_weight=args.root_loss_weight,
            epochs=args.epochs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=not args.disable_persistent_workers,
        ),
    )
    print("[pretrain] stage=train_start", flush=True)

    last_logged_batch = {"train": 0, "validation": 0}

    def _log_batch_progress(phase, batch_index, total_batches, seen_examples, total_loss, node_mse, root_mse):
        interval = args.log_interval
        should_log = (
            batch_index == 1
            or batch_index == total_batches
            or batch_index - last_logged_batch[phase] >= interval
        )
        if not should_log:
            return
        last_logged_batch[phase] = batch_index
        print(
            f"[pretrain] phase={phase} batch={batch_index}/{total_batches} "
            f"seen_examples={seen_examples} total_loss={total_loss:.6f} "
            f"node_mse={node_mse:.6f} root_mse={root_mse:.6f}",
            flush=True,
        )

    def _log_epoch(epoch_index, train_metrics, validation_metrics):
        print(
            f"epoch={epoch_index}/{args.epochs} "
            f"train_total_loss={train_metrics.total_loss:.6f} "
            f"train_node_mse={train_metrics.node_mse:.6f} "
            f"train_root_mse={train_metrics.root_mse:.6f} "
            f"val_total_loss={validation_metrics.total_loss:.6f} "
            f"val_node_mse={validation_metrics.node_mse:.6f} "
            f"val_root_mse={validation_metrics.root_mse:.6f}",
            flush=True,
        )

    history = trainer.fit(
        progress_callback=_log_epoch,
        batch_progress_callback=_log_batch_progress,
    )
    print(f"[pretrain] stage=save_checkpoint path={args.output_checkpoint}", flush=True)
    trainer.save_best_encoder(args.output_checkpoint, metadata={"stage": "supervised_pretrain"})
    final_validation = history[-1]["validation"]
    print(
        f"validation_total_loss={final_validation.total_loss:.6f} "
        f"validation_node_mse={final_validation.node_mse:.6f} "
        f"validation_root_mse={final_validation.root_mse:.6f} "
        f"checkpoint={args.output_checkpoint}",
        flush=True,
    )
    print("[pretrain] stage=done", flush=True)


def pretrain_child_wdl_encoder_command(args: argparse.Namespace) -> None:
    print(f"[child-wdl-pretrain] stage=load_train_dataset path={args.train_dir}", flush=True)
    train_examples = load_pretrain_example_dataset(args.train_dir)
    print(f"[child-wdl-pretrain] stage=load_validation_dataset path={args.validation_dir}", flush=True)
    validation_examples = load_pretrain_example_dataset(args.validation_dir)
    print(
        f"[child-wdl-pretrain] stage=datasets_ready train_examples={len(train_examples)} "
        f"validation_examples={len(validation_examples)}",
        flush=True,
    )

    print("[child-wdl-pretrain] stage=build_schema_and_tensorizer", flush=True)
    schema = _feature_schema()
    tensorizer = _build_tensorizer(args, schema)
    print("[child-wdl-pretrain] stage=build_model", flush=True)
    model = _build_child_wdl_model(args, schema)
    print("[child-wdl-pretrain] stage=build_trainer", flush=True)
    trainer = ChildWdlPretrainer(
        model=model,
        tensorizer=tensorizer,
        train_examples=train_examples,
        validation_examples=validation_examples,
        config=ChildWdlPretrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=not args.disable_persistent_workers,
        ),
    )
    resume_path = args.output_checkpoint.replace(".pt", "_resume.pt")
    start_epoch = 1
    if os.path.isfile(resume_path):
        print(f"[child-wdl-pretrain] stage=resume path={resume_path}", flush=True)
        start_epoch = trainer.load_training_state(resume_path) + 1
        print(f"[child-wdl-pretrain] resuming_from_epoch={start_epoch}", flush=True)

    print("[child-wdl-pretrain] stage=train_start", flush=True)

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
        interval = args.log_interval
        should_log = (
            batch_index == 1
            or batch_index == total_batches
            or batch_index - last_logged_batch[phase] >= interval
        )
        if not should_log:
            return
        last_logged_batch[phase] = batch_index
        print(
            f"[child-wdl-pretrain] epoch={epoch_index}/{args.epochs} phase={phase} "
            f"batch={batch_index}/{total_batches} seen_examples={seen_examples} "
            f"seen_edges={seen_edges} total_loss={total_loss:.6f} "
            f"target_entropy={target_entropy:.6f} loss_gap={loss_gap:.6f}",
            flush=True,
        )

    def _log_epoch(epoch_index, train_metrics, validation_metrics):
        print(
            f"epoch={epoch_index}/{args.epochs} "
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

    history = trainer.fit(
        progress_callback=_log_epoch,
        batch_progress_callback=_log_batch_progress,
        start_epoch=start_epoch,
        resume_path=resume_path,
    )
    print(f"[child-wdl-pretrain] stage=save_checkpoint path={args.output_checkpoint}", flush=True)
    trainer.save_best_encoder(
        args.output_checkpoint,
        metadata={
            "stage": "supervised_pretrain",
            "pretrain_objective": "search_consolidated_edge_wdl_v1",
        },
    )
    decoder_path = args.output_checkpoint.replace(".pt", "_decoder.pt")
    trainer.save_best_decoder(
        decoder_path,
        metadata={
            "stage": "supervised_pretrain",
            "encoder_checkpoint": args.output_checkpoint,
        },
    )
    print(f"[child-wdl-pretrain] stage=save_decoder path={decoder_path}", flush=True)
    final_validation = history[-1]["validation"]
    print(
        f"validation_total_loss={final_validation.total_loss:.6f} "
        f"validation_target_entropy={final_validation.target_entropy:.6f} "
        f"validation_loss_gap={final_validation.loss_gap:.6f} "
        f"validation_supervised_edges={final_validation.num_supervised_edges} "
        f"checkpoint={args.output_checkpoint}",
        flush=True,
    )
    print("[child-wdl-pretrain] stage=done", flush=True)


def frozen_controller_toy_command(args: argparse.Namespace) -> None:
    schema = _feature_schema()
    tensorizer = _build_tensorizer(args, schema)
    model = _build_policy_value_model(args, schema)
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=[_make_toy_env() for _ in range(args.num_envs)],
        config=PPOConfig(
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        ),
        encoder_checkpoint_path=args.encoder_checkpoint,
    )
    trainer.train(args.num_updates)
    metrics = evaluate_controller(model, tensorizer, _make_toy_env, num_episodes=args.eval_episodes)
    print(
        f"average_return={metrics.average_return:.6f} "
        f"average_expansions={metrics.average_expansions:.6f} "
        f"average_terminal_quality={metrics.average_terminal_quality:.6f}"
    )


def frozen_controller_generated_command(args: argparse.Namespace) -> None:
    train_paths, validation_paths = _load_generated_data_paths(args, "controller")

    schema = _feature_schema()
    tensorizer = _build_tensorizer(args, schema)
    quality_config = _build_quality_config(
        args,
        target_normalization_version="controller_v1",
        search_config_id="controller_quality",
    )

    print("[controller] stage=build_model", flush=True)
    model = _build_policy_value_model(args, schema)

    print("[controller] stage=build_envs", flush=True)
    envs = _build_generated_train_envs(train_paths, quality_config, args)

    print("[controller] stage=build_trainer", flush=True)
    trainer = FrozenEncoderControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=envs,
        config=PPOConfig(
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
        ),
        encoder_checkpoint_path=args.encoder_checkpoint,
        freeze_encoder=not args.unfreeze_encoder,
    )

    print("[controller] stage=train_start", flush=True)
    metrics = []

    _make_eval_env = _make_generated_eval_env_factory(validation_paths, quality_config, args)

    for update_index in range(1, args.num_updates + 1):
        update_metrics = trainer.train_update()
        metrics.append(update_metrics)
        if update_index % args.log_interval == 0 or update_index == args.num_updates:
            print(
                f"update={update_index}/{args.num_updates} "
                f"policy_loss={update_metrics.policy_loss:.6f} "
                f"value_loss={update_metrics.value_loss:.6f} "
                f"entropy={update_metrics.entropy:.6f} "
                f"approx_kl={update_metrics.approx_kl:.6f} "
                f"clipfrac={update_metrics.clipfrac:.6f} "
                f"explained_variance={update_metrics.explained_variance:.6f} "
                f"mean_episode_return={update_metrics.mean_episode_return:.6f} "
                f"mean_episode_length={update_metrics.mean_episode_length:.3f} "
                f"halt_rate={update_metrics.halt_rate:.3f} "
                f"episodes={update_metrics.num_episodes}",
                flush=True,
            )

        if args.validation_interval > 0 and (
            update_index % args.validation_interval == 0 or update_index == args.num_updates
        ):
            validation_metrics = evaluate_controller(
                model,
                tensorizer,
                _make_eval_env,
                num_episodes=args.eval_episodes,
            )
            print(
                f"validation_update={update_index}/{args.num_updates} "
                f"average_return={validation_metrics.average_return:.6f} "
                f"average_expansions={validation_metrics.average_expansions:.6f} "
                f"average_terminal_quality={validation_metrics.average_terminal_quality:.6f}",
                flush=True,
            )

    print("[controller] stage=evaluate", flush=True)

    eval_metrics = evaluate_controller(
        model,
        tensorizer,
        _make_eval_env,
        num_episodes=args.eval_episodes,
    )
    print(
        f"average_return={eval_metrics.average_return:.6f} "
        f"average_expansions={eval_metrics.average_expansions:.6f} "
        f"average_terminal_quality={eval_metrics.average_terminal_quality:.6f}",
        flush=True,
    )

    print(f"[controller] stage=save_checkpoint path={args.output_checkpoint}", flush=True)
    save_policy_value_checkpoint(
        args.output_checkpoint,
        model,
        metadata={
            "stage": "frozen_controller_rl",
            "encoder_checkpoint": args.encoder_checkpoint,
            "num_updates": args.num_updates,
            "eval_average_return": eval_metrics.average_return,
            "eval_average_expansions": eval_metrics.average_expansions,
            "eval_average_terminal_quality": eval_metrics.average_terminal_quality,
        },
    )
    print("[controller] stage=done", flush=True)


def reinforce_generated_command(args: argparse.Namespace) -> None:
    train_paths, validation_paths = _load_generated_data_paths(args, "reinforce")

    schema = _feature_schema()
    tensorizer = _build_tensorizer(args, schema)
    quality_config = _build_quality_config(
        args,
        target_normalization_version="controller_v1",
        search_config_id="controller_quality",
    )

    print("[reinforce] stage=build_model", flush=True)
    model = _build_policy_value_model(args, schema)

    print("[reinforce] stage=build_envs", flush=True)
    envs = _build_generated_train_envs(train_paths, quality_config, args)

    print("[reinforce] stage=build_trainer", flush=True)
    trainer = ReinforceControllerTrainer(
        model=model,
        tensorizer=tensorizer,
        envs=envs,
        config=ReinforceConfig(
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            batch_episodes=args.batch_episodes,
            entropy_coef=args.entropy_coef,
            use_return_normalization=not args.disable_return_normalization,
        ),
        encoder_checkpoint_path=args.encoder_checkpoint,
        freeze_encoder=not args.unfreeze_encoder,
    )

    print("[reinforce] stage=train_start", flush=True)

    _make_eval_env = _make_generated_eval_env_factory(validation_paths, quality_config, args)

    for update_index in range(1, args.num_updates + 1):
        update_metrics = trainer.train_update()
        if update_index % args.log_interval == 0 or update_index == args.num_updates:
            print(
                f"update={update_index}/{args.num_updates} "
                f"policy_loss={update_metrics.policy_loss:.6f} "
                f"entropy={update_metrics.entropy:.6f} "
                f"mean_episode_return={update_metrics.mean_episode_return:.6f} "
                f"mean_episode_length={update_metrics.mean_episode_length:.3f} "
                f"halt_rate={update_metrics.halt_rate:.3f} "
                f"episodes={update_metrics.num_episodes}",
                flush=True,
            )

        if args.validation_interval > 0 and (
            update_index % args.validation_interval == 0 or update_index == args.num_updates
        ):
            validation_metrics = evaluate_controller(
                model,
                tensorizer,
                _make_eval_env,
                num_episodes=args.eval_episodes,
            )
            print(
                f"validation_update={update_index}/{args.num_updates} "
                f"average_return={validation_metrics.average_return:.6f} "
                f"average_expansions={validation_metrics.average_expansions:.6f} "
                f"average_terminal_quality={validation_metrics.average_terminal_quality:.6f}",
                flush=True,
            )

    print("[reinforce] stage=evaluate", flush=True)
    eval_metrics = evaluate_controller(
        model,
        tensorizer,
        _make_eval_env,
        num_episodes=args.eval_episodes,
    )
    print(
        f"average_return={eval_metrics.average_return:.6f} "
        f"average_expansions={eval_metrics.average_expansions:.6f} "
        f"average_terminal_quality={eval_metrics.average_terminal_quality:.6f}",
        flush=True,
    )

    print(f"[reinforce] stage=save_checkpoint path={args.output_checkpoint}", flush=True)
    save_policy_value_checkpoint(
        args.output_checkpoint,
        model,
        metadata={
            "stage": "reinforce_controller_rl",
            "encoder_checkpoint": args.encoder_checkpoint,
            "num_updates": args.num_updates,
            "eval_average_return": eval_metrics.average_return,
            "eval_average_expansions": eval_metrics.average_expansions,
            "eval_average_terminal_quality": eval_metrics.average_terminal_quality,
        },
    )
    print("[reinforce] stage=done", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised branch utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-dataset")
    generate.add_argument("--engine-path", default=_default_lc0_engine_path())
    generate.add_argument("--engine-kind", choices=["lc0", "stockfish"], default="lc0")
    generate.add_argument("--weights-path", default=_default_lc0_weights_path())
    generate.add_argument("--backend")
    generate.add_argument("--prior-backend")
    generate.add_argument("--value-backend")
    generate.add_argument("--fens", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--movetime-ms", type=int, default=200)
    generate.add_argument("--depth", type=int)
    generate.add_argument("--nodes", type=int)
    generate.add_argument("--multipv", type=int, default=8)
    generate.add_argument("--max-depth", type=int, default=4)
    generate.add_argument("--search-budget", type=int, default=64)
    generate.add_argument("--c-puct", type=float, default=1.0)
    generate.add_argument("--min-nodes", type=int, default=16)
    generate.add_argument("--max-nodes", type=int, default=128)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--start-index", type=int, default=0)
    generate.add_argument("--end-index", type=int)
    generate.add_argument("--cache-clear-interval", type=int, default=128)
    generate.add_argument("--log-interval", type=int, default=25)
    generate.add_argument("--resume", action="store_true")
    generate.set_defaults(func=generate_dataset_command)

    pretrain = subparsers.add_parser("pretrain-encoder")
    pretrain.add_argument("--train-dir", required=True)
    pretrain.add_argument("--validation-dir", required=True)
    pretrain.add_argument("--output-checkpoint", required=True)
    _add_shared_model_args(pretrain, include_controller_hidden=False)
    pretrain.add_argument("--batch-size", type=int, default=128)
    pretrain.add_argument("--learning-rate", type=float, default=1e-3)
    pretrain.add_argument("--weight-decay", type=float, default=0.0)
    pretrain.add_argument("--root-loss-weight", type=float, default=1.0)
    pretrain.add_argument("--epochs", type=int, default=10)
    pretrain.add_argument("--log-interval", type=int, default=25)
    pretrain.add_argument("--num-workers", type=int, default=4)
    pretrain.add_argument("--prefetch-factor", type=int, default=2)
    pretrain.add_argument("--pin-memory", action="store_true")
    pretrain.add_argument("--disable-persistent-workers", action="store_true")
    pretrain.set_defaults(func=pretrain_encoder_command)

    pretrain_child_wdl = subparsers.add_parser("pretrain-child-wdl-encoder")
    pretrain_child_wdl.add_argument("--train-dir", required=True)
    pretrain_child_wdl.add_argument("--validation-dir", required=True)
    pretrain_child_wdl.add_argument("--output-checkpoint", required=True)
    _add_shared_model_args(pretrain_child_wdl, include_controller_hidden=False)
    pretrain_child_wdl.add_argument("--decoder-hidden", type=int, default=128)
    pretrain_child_wdl.add_argument("--batch-size", type=int, default=128)
    pretrain_child_wdl.add_argument("--learning-rate", type=float, default=1e-3)
    pretrain_child_wdl.add_argument("--weight-decay", type=float, default=0.0)
    pretrain_child_wdl.add_argument("--epochs", type=int, default=10)
    pretrain_child_wdl.add_argument("--log-interval", type=int, default=25)
    pretrain_child_wdl.add_argument("--num-workers", type=int, default=0)
    pretrain_child_wdl.add_argument("--prefetch-factor", type=int, default=2)
    pretrain_child_wdl.add_argument("--pin-memory", action="store_true")
    pretrain_child_wdl.add_argument("--disable-persistent-workers", action="store_true")
    pretrain_child_wdl.set_defaults(func=pretrain_child_wdl_encoder_command)

    frozen = subparsers.add_parser("frozen-rl-toy")
    frozen.add_argument("--encoder-checkpoint", required=True)
    _add_shared_model_args(frozen, include_controller_hidden=True)
    frozen.add_argument("--num-envs", type=int, default=2)
    frozen.add_argument("--rollout-steps", type=int, default=32)
    frozen.add_argument("--learning-rate", type=float, default=1e-3)
    frozen.add_argument("--ppo-epochs", type=int, default=4)
    frozen.add_argument("--minibatch-size", type=int, default=8)
    frozen.add_argument("--num-updates", type=int, default=5)
    frozen.add_argument("--eval-episodes", type=int, default=8)
    frozen.set_defaults(func=frozen_controller_toy_command)

    controller = subparsers.add_parser("frozen-rl-generated")
    controller.add_argument("--train-data", required=True)
    controller.add_argument("--validation-data", required=True)
    controller.add_argument("--encoder-checkpoint", required=True)
    controller.add_argument("--output-checkpoint", required=True)
    controller.add_argument("--seed", type=int, default=0)
    _add_shared_model_args(controller, include_controller_hidden=True)
    controller.add_argument("--num-envs", type=int, default=4)
    _add_generated_quality_args(controller)
    controller.add_argument("--rollout-steps", type=int, default=64)
    controller.add_argument("--learning-rate", type=float, default=1e-3)
    controller.add_argument("--gamma", type=float, default=0.99)
    controller.add_argument("--gae-lambda", type=float, default=0.95)
    controller.add_argument("--clip-epsilon", type=float, default=0.2)
    controller.add_argument("--entropy-coef", type=float, default=0.01)
    controller.add_argument("--value-loss-coef", type=float, default=0.5)
    controller.add_argument("--ppo-epochs", type=int, default=4)
    controller.add_argument("--minibatch-size", type=int, default=32)
    controller.add_argument("--max-grad-norm", type=float, default=1.0)
    controller.add_argument("--num-updates", type=int, default=200)
    controller.add_argument("--eval-episodes", type=int, default=32)
    controller.add_argument("--log-interval", type=int, default=10)
    controller.add_argument("--validation-interval", type=int, default=0)
    controller.add_argument("--unfreeze-encoder", action="store_true")
    controller.set_defaults(func=frozen_controller_generated_command)

    reinforce = subparsers.add_parser("reinforce-generated")
    reinforce.add_argument("--train-data", required=True)
    reinforce.add_argument("--validation-data", required=True)
    reinforce.add_argument("--encoder-checkpoint", required=True)
    reinforce.add_argument("--output-checkpoint", required=True)
    reinforce.add_argument("--seed", type=int, default=0)
    _add_shared_model_args(reinforce, include_controller_hidden=True)
    reinforce.add_argument("--num-envs", type=int, default=4)
    _add_generated_quality_args(reinforce)
    reinforce.add_argument("--learning-rate", type=float, default=1e-3)
    reinforce.add_argument("--entropy-coef", type=float, default=0.0)
    reinforce.add_argument("--max-grad-norm", type=float, default=1.0)
    reinforce.add_argument("--batch-episodes", type=int, default=128)
    reinforce.add_argument("--num-updates", type=int, default=200)
    reinforce.add_argument("--eval-episodes", type=int, default=32)
    reinforce.add_argument("--log-interval", type=int, default=10)
    reinforce.add_argument("--validation-interval", type=int, default=0)
    reinforce.add_argument("--disable-return-normalization", action="store_true")
    reinforce.add_argument("--unfreeze-encoder", action="store_true")
    reinforce.set_defaults(func=reinforce_generated_command)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
