from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

from GNN import ChildWdlModel
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts_pretrain import (
    ChildWdlPretrainConfig,
    ChildWdlPretrainer,
    NodeBudgetDistribution,
    TeacherSearchConfig,
    build_pretrain_example,
    load_pretrain_example_dataset,
    save_pretrain_example_to_directory,
)
from tensorizer import TreeTensorizer
from uci_provider import (
    Lc0DirectEvalProvider,
    UciEngineConfig,
    UciEngineProcess,
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


def _add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)


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

    prior_uci_options: dict[str, str] = {}
    value_uci_options: dict[str, str] = {}
    if args.backend:
        prior_uci_options["Backend"] = args.backend
        value_uci_options["Backend"] = args.backend
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

    print(
        f"saved_examples={len(saved_paths)} output_dir={args.output_dir} "
        f"start_index={args.start_index} end_index={args.end_index if args.end_index is not None else len(fens)}"
    )


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
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised branch utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-dataset")
    generate.add_argument("--engine-path", default=_default_lc0_engine_path())
    generate.add_argument("--weights-path", default=_default_lc0_weights_path())
    generate.add_argument("--backend")
    generate.add_argument("--prior-backend")
    generate.add_argument("--value-backend")
    generate.add_argument("--fens", required=True)
    generate.add_argument("--output-dir", required=True)
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

    pretrain_child_wdl = subparsers.add_parser("pretrain-child-wdl-encoder")
    pretrain_child_wdl.add_argument("--train-dir", required=True)
    pretrain_child_wdl.add_argument("--validation-dir", required=True)
    pretrain_child_wdl.add_argument("--output-checkpoint", required=True)
    _add_shared_model_args(pretrain_child_wdl)
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

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
