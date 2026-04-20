from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from cts_pretrain import PretrainExample, RawPretrainExampleRecord, _normalize_wdl_target
from cts_uci_common import append_move_to_position_spec
from tree import SearchNode, SearchTree


def _gather_examples(source_root: Path) -> list[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    examples = sorted(source_root.rglob("*.pt"))
    if not examples:
        raise ValueError(f"No .pt examples found under {source_root}")
    return examples


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _legacy_tree_setstate(self: SearchTree, state) -> None:
    if state.get("__format__") != "search_tree_v2":
        self.__dict__.update(state)
        return

    root_id = state["root_id"]
    parent_ids = list(state["parent_ids"])
    incoming_moves = list(state["incoming_moves"])
    root_position_spec = state.get("root_position_spec")
    fens = state.get("fens")
    depths = list(state["depths"])
    is_terminal = list(state["is_terminal"])
    is_expanded = list(state["is_expanded"])
    scalar_feature_names = state.get("scalar_feature_names")
    scalar_feature_values = state.get("scalar_feature_values")
    node_scalar_features = state.get("node_scalar_features")
    sparse_metadata_items = state.get("sparse_metadata", [])

    metadata_by_node = {int(node_id): dict(metadata) for node_id, metadata in sparse_metadata_items}
    self.root_id = int(root_id) if root_id is not None else None
    self._nodes = []
    self._children = {}

    if fens is not None:
        resolved_fens = [str(fen) for fen in fens]
    else:
        if root_position_spec is None:
            raise ValueError("Legacy serialized tree must include root_position_spec when fens are omitted.")
        resolved_fens = []
        for node_id, (parent_id, move) in enumerate(zip(parent_ids, incoming_moves)):
            if parent_id is None:
                resolved_fens.append(str(root_position_spec))
                continue
            if parent_id >= node_id:
                raise ValueError("Legacy serialized parent_ids must be topologically ordered.")
            if move is None:
                raise ValueError("Legacy serialized non-root node is missing an incoming move.")
            resolved_fens.append(append_move_to_position_spec(resolved_fens[parent_id], str(move)))

    if scalar_feature_names is not None:
        scalar_feature_names = [str(name) for name in scalar_feature_names]
        resolved_scalar_features = [
            {name: float(value) for name, value in zip(scalar_feature_names, row)}
            for row in scalar_feature_values
        ]
    else:
        resolved_scalar_features = [
            {str(name): float(value) for name, value in dict(features).items()}
            for features in node_scalar_features
        ]

    for node_id, (parent_id, move, fen, depth, terminal, expanded, scalar_features) in enumerate(
        zip(
            parent_ids,
            incoming_moves,
            resolved_fens,
            depths,
            is_terminal,
            is_expanded,
            resolved_scalar_features,
        )
    ):
        self._nodes.append(
            SearchNode(
                node_id=node_id,
                parent_id=int(parent_id) if parent_id is not None else None,
                incoming_move_uci=str(move) if move is not None else None,
                fen=str(fen),
                depth=int(depth),
                is_terminal=bool(terminal),
                is_expanded=bool(expanded),
                scalar_features=scalar_features,
                metadata=dict(metadata_by_node.get(node_id, {})),
            )
        )
        self._children[node_id] = []

    for node in self._nodes:
        if node.parent_id is not None:
            self._children[node.parent_id].append(node.node_id)


def _legacy_pretrain_example_setstate(self: PretrainExample, state) -> None:
    if state.get("__format__") != "pretrain_example_v2":
        self.__dict__.update(state)
        if "edge_wdl_targets" not in self.__dict__:
            self.edge_wdl_targets = {}
        if "oracle_trace_expansion_counts" not in self.__dict__:
            self.oracle_trace_expansion_counts = []
        if "oracle_root_moves" not in self.__dict__:
            self.oracle_root_moves = []
        if "oracle_root_q_trace" not in self.__dict__:
            self.oracle_root_q_trace = []
        if "oracle_best_move_trace" not in self.__dict__:
            self.oracle_best_move_trace = []
        if "oracle_final_root_q_values" not in self.__dict__:
            self.oracle_final_root_q_values = {}
        self.__post_init__()
        return

    edge_parent_ids = list(state.get("edge_target_parent_ids", []))
    edge_child_ids = list(state.get("edge_target_child_ids", []))
    edge_target_wdls = list(state.get("edge_target_wdls", []))
    if not (len(edge_parent_ids) == len(edge_child_ids) == len(edge_target_wdls)):
        raise ValueError("Legacy stored edge target arrays must have the same length.")

    oracle_root_moves = list(state.get("oracle_root_moves", []))
    oracle_final_root_q_values_aligned = state.get("oracle_final_root_q_values_aligned")
    oracle_final_root_q_values_sparse = state.get("oracle_final_root_q_values_sparse")
    if oracle_final_root_q_values_aligned is not None:
        oracle_final_root_q_values = {
            str(move): float(value)
            for move, value in zip(oracle_root_moves, oracle_final_root_q_values_aligned)
        }
    else:
        oracle_final_root_q_values = dict(oracle_final_root_q_values_sparse or {})

    self.__dict__.update(
        {
            "tree": state["tree"],
            "node_target_values": list(state["node_target_values"]),
            "edge_wdl_targets": {
                (int(parent_id), int(child_id)): _normalize_wdl_target(target)
                for parent_id, child_id, target in zip(edge_parent_ids, edge_child_ids, edge_target_wdls)
            },
            "metadata": dict(state.get("metadata", {})),
            "oracle_trace_expansion_counts": list(state.get("oracle_trace_expansion_counts", [])),
            "oracle_root_moves": oracle_root_moves,
            "oracle_root_q_trace": [list(row) for row in state.get("oracle_root_q_trace", [])],
            "oracle_best_move_trace": list(state.get("oracle_best_move_trace", [])),
            "oracle_final_root_q_values": oracle_final_root_q_values,
        }
    )
    self.__post_init__()


def _install_legacy_unpicklers() -> None:
    SearchTree.__setstate__ = _legacy_tree_setstate  # type: ignore[method-assign]
    PretrainExample.__setstate__ = _legacy_pretrain_example_setstate  # type: ignore[method-assign]


def _load_record_from_source(source: Path) -> RawPretrainExampleRecord:
    _install_legacy_unpicklers()
    payload = torch.load(source, weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == "cts_raw_pretrain_example_v1":
        return RawPretrainExampleRecord.from_payload(payload)
    if isinstance(payload, PretrainExample):
        return RawPretrainExampleRecord.from_example(payload)
    raise ValueError(f"Unsupported raw example payload at {source}: {type(payload).__name__}")


def _rewrite_file(source: Path, destination: Path) -> tuple[int, int]:
    record = _load_record_from_source(source)
    before = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    record.save(destination)
    after = destination.stat().st_size
    return before, after


def _rewrite_in_place(source: Path) -> tuple[int, int]:
    temp_path = source.with_suffix(source.suffix + ".compact_tmp")
    if temp_path.exists():
        temp_path.unlink()
    before, after = _rewrite_file(source, temp_path)
    os.replace(temp_path, source)
    return before, after


def _rewrite_to_output(source: Path, source_root: Path, output_root: Path) -> tuple[int, int, Path]:
    destination = output_root / source.relative_to(source_root)
    before, after = _rewrite_file(source, destination)
    return before, after, destination


def _rewrite_task(task: tuple[str, str, str | None]) -> tuple[int, int]:
    mode, source_str, extra_str = task
    source = Path(source_str)
    if mode == "in_place":
        return _rewrite_in_place(source)
    if mode == "copy":
        if extra_str is None:
            raise ValueError("copy mode requires source_root and output_root context.")
        source_root_str, output_root_str = extra_str.split("::", 1)
        before, after, _ = _rewrite_to_output(source, Path(source_root_str), Path(output_root_str))
        return before, after
    raise ValueError(f"Unknown rewrite mode: {mode}")


def _iter_selected(paths: Iterable[Path], limit: int | None) -> Iterable[Path]:
    if limit is None:
        yield from paths
        return
    for index, path in enumerate(paths):
        if index >= limit:
            break
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite raw pretrain examples using the compact serializer.")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory containing raw pretrain .pt examples.",
    )
    parser.add_argument(
        "--output-root",
        help="Write rewritten examples to a parallel directory tree under this root.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite examples atomically in place.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only rewrite the first N examples.")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes to use. 0 or 1 = serial.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="When using --output-root, skip files that already exist at the destination.",
    )
    args = parser.parse_args()

    if args.in_place == bool(args.output_root):
        raise ValueError("Specify exactly one of --in-place or --output-root.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when provided.")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")

    source_root = Path(args.source_root)
    output_root = Path(args.output_root) if args.output_root else None
    examples = _gather_examples(source_root)

    total_before = 0
    total_after = 0
    rewritten = 0
    skipped = 0

    selected_examples = list(_iter_selected(examples, args.limit))
    pending_tasks: list[tuple[str, str, str | None]] = []
    for source in selected_examples:
        if output_root is not None:
            destination = output_root / source.relative_to(source_root)
            if args.skip_existing and destination.exists():
                skipped += 1
                continue
            pending_tasks.append(("copy", str(source), f"{source_root}::{output_root}"))
        else:
            pending_tasks.append(("in_place", str(source), None))

    def _log_progress() -> None:
        saved = total_before - total_after
        print(
            f"rewritten={rewritten} skipped={skipped} "
            f"before={_human_bytes(total_before)} after={_human_bytes(total_after)} "
            f"saved={_human_bytes(saved)}"
        )

    if args.num_workers <= 1:
        for task in pending_tasks:
            before, after = _rewrite_task(task)
            rewritten += 1
            total_before += before
            total_after += after
            if rewritten % args.log_interval == 0:
                _log_progress()
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(_rewrite_task, task) for task in pending_tasks]
                for future in as_completed(futures):
                    before, after = future.result()
                    rewritten += 1
                    total_before += before
                    total_after += after
                    if rewritten % args.log_interval == 0:
                        _log_progress()
        except (PermissionError, OSError):
            for task in pending_tasks:
                before, after = _rewrite_task(task)
                rewritten += 1
                total_before += before
                total_after += after
                if rewritten % args.log_interval == 0:
                    _log_progress()

    if rewritten % args.log_interval != 0 and rewritten > 0:
        _log_progress()

    saved = total_before - total_after
    print(f"source_root={source_root}")
    if output_root is not None:
        print(f"output_root={output_root}")
    print(f"mode={'in_place' if args.in_place else 'copy'}")
    print(f"rewritten={rewritten}")
    print(f"skipped={skipped}")
    print(f"before_bytes={total_before}")
    print(f"after_bytes={total_after}")
    print(f"saved_bytes={saved}")
    if total_before > 0:
        print(f"saved_fraction={saved / total_before:.6f}")


if __name__ == "__main__":
    main()
