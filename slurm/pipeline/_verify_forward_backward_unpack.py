"""Standalone (non-pytest) GPU verification: real Lc0 tree generation vs. the
retroactive backward-unpack replay. Needs a GPU at useful scale and takes tens
of minutes, so it runs via slurm/pipeline/verify_forward_backward_unpack.slurm
instead of routine pytest. Checks, per FEN: (1) two independent generations of
the same FEN+seed are identical (determinism), (2) the backward-unpack replay
matches the live oracle root-Q trace at every step. Exit code is 0 iff every
FEN passes both checks.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cts.data.preprocess_gnn.teacher_targets import (
    TeacherSearchConfig,
    NodeBudgetDistribution,
    RawPretrainExampleRecord,
    build_pretrain_example,
    load_raw_pretrain_record,
    save_pretrain_example_to_directory,
)
from cts.data.preprocess_mc.pack import _ordered_expansion_parent_ids, _replay_backprop_history

_TOLERANCE = 1e-3
_SEARCH_BUDGET = 96
_DEFAULT_HUMAN_TREES_DIR = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"


def _pick_real_fens(n: int, human_trees_dir: str, seed: int = 0) -> List[Tuple[str, str]]:
    """Pull N real root FENs from real human_trees records, deterministically sampled."""
    files = sorted(Path(human_trees_dir).glob("*_root_*.pt"))
    if not files:
        raise FileNotFoundError(f"no *_root_*.pt files under {human_trees_dir}")
    rng = random.Random(seed)
    chosen = rng.sample(files, min(n, len(files)))
    out = []
    for f in chosen:
        record = load_raw_pretrain_record(str(f))
        if record.root_position_spec:
            out.append((f.stem, record.root_position_spec))
        if len(out) >= n:
            break
    return out


def _make_lc0_provider(lc0_binary: str, lc0_weights: str, backend: str):
    from cts.core.providers import Lc0DirectEvalProvider, UciEngineConfig, UciEngineProcess

    prior_config = UciEngineConfig(
        engine_path=lc0_binary, engine_kind="lc0", engine_mode="classic",
        movetime_ms=0, multipv=8, depth=None, nodes=1,
        weights_path=lc0_weights, uci_options={"Backend": backend},
        set_multipv=False, enable_verbose_move_stats=True,
    )
    value_config = UciEngineConfig(
        engine_path=lc0_binary, engine_kind="lc0", engine_mode="valuehead",
        movetime_ms=0, multipv=1, depth=None, nodes=1,
        weights_path=lc0_weights, uci_options={"Backend": backend, "UCI_ShowWDL": "true"},
        set_multipv=False, enable_verbose_move_stats=False,
    )
    prior_engine = UciEngineProcess(prior_config)
    value_engine = UciEngineProcess(value_config)
    prior_engine.start()
    value_engine.start()
    provider = Lc0DirectEvalProvider(prior_engine, value_engine, metadata={"engine_kind": "lc0"})
    return provider, (prior_engine, value_engine)


def _generate_from_scratch(fen: str, budget: int, seed: int, tag: str, lc0_binary: str, lc0_weights: str, backend: str):
    provider, procs = _make_lc0_provider(lc0_binary, lc0_weights, backend)
    try:
        search_config = TeacherSearchConfig(
            max_depth=40, search_budget=budget, c_puct=1.0,
            prior_feature="prior", value_feature="value",
            target_normalization_version="v1",
            search_config_id=f"verify_from_scratch_{tag}", selection="puct",
        )
        rng = random.Random(seed)
        example = build_pretrain_example(
            fen, provider, search_config,
            node_budget_distribution=NodeBudgetDistribution(budget, budget),
            rng=rng, root_position_id=f"root_{tag}", include_edge_wdl_targets=True,
        )
    finally:
        for proc in procs:
            proc.close()
    return example


def _replay_vs_live_trace(record: RawPretrainExampleRecord) -> Dict[str, Any]:
    expansion_parent_ids = _ordered_expansion_parent_ids(record)
    replay = _replay_backprop_history(record, expansion_parent_ids, value_feature="value")

    parent_index = record.parent_index.tolist()
    root_children = sorted(i for i, p in enumerate(parent_index) if p == 0)
    node_update_ptr = replay["node_update_ptr"]
    steps = replay["update_log_step_index"]
    qvals = replay["update_log_q_value"]

    def value_as_of_step(node_id: int, step_t: int) -> Optional[float]:
        start, end = int(node_update_ptr[node_id]), int(node_update_ptr[node_id + 1])
        best = None
        for s, q in zip(steps[start:end].tolist(), qvals[start:end].tolist()):
            if s <= step_t:
                best = float(q)
            else:
                break
        return best

    num_steps = int(record.oracle_root_q_trace.shape[0])
    n_compared = 0
    max_abs_err = 0.0
    worst = None
    mismatches = []
    for node_id in root_children:
        move = record.incoming_moves[node_id]
        move_idx = record.oracle_root_moves.index(move)
        for row in range(num_steps):
            replayed = value_as_of_step(node_id, row)
            if replayed is None:
                continue
            oracle = float(record.oracle_root_q_trace[row, move_idx].item())
            err = abs(replayed - oracle)
            n_compared += 1
            if err > max_abs_err:
                max_abs_err = err
                worst = (node_id, move, row, replayed, oracle)
            if err >= _TOLERANCE:
                mismatches.append((node_id, move, row, replayed, oracle))
    return {"n_root_children": len(root_children), "num_steps": num_steps, "n_compared": n_compared,
            "max_abs_err": max_abs_err, "worst": worst, "mismatches": mismatches}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-fens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lc0-binary", default="/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0")
    ap.add_argument("--lc0-weights", default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz")
    ap.add_argument("--backend", default="cuda", help="lc0 Backend UCI option (cuda, cuda-fp16, eigen, ...)")
    ap.add_argument("--human-trees-dir", default=_DEFAULT_HUMAN_TREES_DIR)
    ap.add_argument("--out-dir", default=None, help="dir to save generated trees to (temp dir if omitted)")
    args = ap.parse_args()

    if not os.path.isfile(args.lc0_binary) or not os.access(args.lc0_binary, os.X_OK):
        print(f"FATAL: lc0 binary not usable at {args.lc0_binary}", flush=True)
        return 2
    if not os.path.isfile(args.lc0_weights):
        print(f"FATAL: lc0 weights not found at {args.lc0_weights}", flush=True)
        return 2

    import tempfile
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="verify_fwd_bwd_")
    os.makedirs(out_dir, exist_ok=True)

    fens = _pick_real_fens(args.num_fens, args.human_trees_dir, seed=args.seed)
    print(f"picked {len(fens)} real FENs (seed={args.seed}) from {args.human_trees_dir}", flush=True)
    print(f"backend={args.backend} search_budget={_SEARCH_BUDGET} out_dir={out_dir}", flush=True)

    all_ok = True
    for i, (tag, fen) in enumerate(fens):
        t0 = time.time()
        print(f"[{i+1}/{len(fens)}] {tag}: {fen}", flush=True)

        # --- determinism: two independent generations, same FEN+seed ---
        example_a = _generate_from_scratch(fen, _SEARCH_BUDGET, seed=0, tag=f"{tag}_detA",
                                            lc0_binary=args.lc0_binary, lc0_weights=args.lc0_weights, backend=args.backend)
        example_b = _generate_from_scratch(fen, _SEARCH_BUDGET, seed=0, tag=f"{tag}_detB",
                                            lc0_binary=args.lc0_binary, lc0_weights=args.lc0_weights, backend=args.backend)
        path_a = save_pretrain_example_to_directory(out_dir, example_a, i * 2)
        path_b = save_pretrain_example_to_directory(out_dir, example_b, i * 2 + 1)
        record_a = load_raw_pretrain_record(path_a)
        record_b = load_raw_pretrain_record(path_b)

        det_ok = (record_a.parent_index.tolist() == record_b.parent_index.tolist()
                  and record_a.oracle_root_q_trace.tolist() == record_b.oracle_root_q_trace.tolist())
        if not det_ok:
            print(f"  DETERMINISM FAIL: {tag} -- two independent generations diverged", flush=True)
            all_ok = False
        else:
            print(f"  determinism OK ({len(record_a.parent_index)} nodes)", flush=True)

        # --- replay vs. the live trace generation just recorded ---
        if len(example_a.oracle_root_q_trace) != _SEARCH_BUDGET:
            print(f"  GENERATION SHORT: {tag} -- {len(example_a.oracle_root_q_trace)}/{_SEARCH_BUDGET} rows", flush=True)
            all_ok = False
            continue

        result = _replay_vs_live_trace(record_a)
        elapsed = time.time() - t0
        status = "OK" if (result["n_compared"] > 0 and not result["mismatches"]) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  replay {status}: n_compared={result['n_compared']} max_abs_err={result['max_abs_err']:.2e} "
              f"mismatches={len(result['mismatches'])} worst={result['worst']} elapsed={elapsed:.1f}s", flush=True)

    print(f"\n{'ALL PASSED' if all_ok else 'FAILURES FOUND'} -- {len(fens)} FENs checked", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
