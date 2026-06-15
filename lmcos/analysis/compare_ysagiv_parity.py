"""Parity test: does OUR tree generator reproduce a ysagiv reference tree from the same root FEN?

Throwaway comparison harness. For each ysagiv reference tree (v2 dict on disk):
  1. read root_position_spec (the exact FEN it was built from),
  2. regenerate our tree from that FEN with the matching regime
     (max_depth=10, search_budget=96, c_puct=1.0, multipv=8, both engines Backend=cuda),
  3. serialize our PretrainExample to the same tensor layout via RawPretrainExampleRecord,
  4. compare the COMPARABLE fields, flagging v2->v5 schema-only diffs separately.

Run from the lmcos dir with cts importable:
    cd /home/hl4291/chess_analysis/lmcos && python analysis/compare_ysagiv_parity.py
"""
from __future__ import annotations

import os
import random
import sys

import torch

from cts.core.providers import (
    Lc0DirectEvalProvider,
    UciEngineConfig,
    UciEngineProcess,
)
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    RawPretrainExampleRecord,
    TeacherSearchConfig,
    build_pretrain_example,
)

ENGINE = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"

REF_PATHS = [
    "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_oracle96_trace_filtered/shard_00000/000000_root_0.pt",
    "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_oracle96_trace_filtered/shard_00007/014112_root_14112.pt",
    "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_oracle96_trace_filtered/shard_00015/030240_root_30240.pt",
]


def _make_provider():
    prior_config = UciEngineConfig(
        engine_path=ENGINE, engine_kind="lc0", engine_mode="classic",
        movetime_ms=0, multipv=8, depth=None, nodes=1,
        weights_path=WEIGHTS, uci_options={"Backend": "cuda"},
        set_multipv=False, enable_verbose_move_stats=True,
    )
    value_config = UciEngineConfig(
        engine_path=ENGINE, engine_kind="lc0", engine_mode="valuehead",
        movetime_ms=0, multipv=1, depth=None, nodes=1,
        weights_path=WEIGHTS, uci_options={"Backend": "cuda", "UCI_ShowWDL": "true"},
        set_multipv=False, enable_verbose_move_stats=False,
    )
    return prior_config, value_config


def _t(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _maxdiff(a, b):
    a, b = _t(a).float(), _t(b).float()
    if a.shape != b.shape:
        return f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.numel() == 0:
        return 0.0
    m = ~(torch.isnan(a) | torch.isnan(b))
    if m.sum() == 0:
        return "all-nan-on-one-or-both"
    return float((a[m] - b[m]).abs().max())


def compare(ref_path, prior_engine, value_engine):
    ref = torch.load(ref_path, weights_only=False)
    fen = ref["root_position_spec"]
    print(f"\n{'='*100}\nREF: {ref_path}\nFEN: {fen}\nformat: {ref['format']}")

    provider = Lc0DirectEvalProvider(
        prior_engine, value_engine,
        metadata={"engine_kind": "lc0", "value_source": "valuehead", "prior_source": "classic_nodes_1"},
    )
    cfg = TeacherSearchConfig(
        max_depth=10, search_budget=96, c_puct=1.0,
        prior_feature="prior", value_feature="value",
        target_normalization_version="v1", search_config_id="cmp",
    )
    ex = build_pretrain_example(
        fen, provider, cfg,
        node_budget_distribution=NodeBudgetDistribution(96, 96),
        rng=random.Random(0), root_position_id="cmp",
    )
    rec = RawPretrainExampleRecord.from_example(ex)
    provider.clear_caches()

    results = {}

    # --- structure / topology ---
    results["N_nodes"] = (len(ref["parent_index"]), int(rec.parent_index.shape[0]))
    results["num_edges"] = (len(ref["children_index"]), int(rec.children_index.shape[0]))
    results["is_expanded_sum"] = (int(_t(ref["is_expanded"]).sum()), int(rec.is_expanded.sum()))
    results["is_terminal_sum"] = (int(_t(ref["is_terminal"]).sum()), int(rec.is_terminal.sum()))

    same_N = len(ref["parent_index"]) == int(rec.parent_index.shape[0])
    if same_N:
        results["parent_index_match"] = bool(torch.equal(_t(ref["parent_index"]).int(), rec.parent_index.int()))
        results["child_ptr_match"] = bool(torch.equal(_t(ref["child_ptr"]).int(), rec.child_ptr.int()))
        results["children_index_match"] = bool(torch.equal(_t(ref["children_index"]).int(), rec.children_index.int()))
        results["depth_match"] = bool(torch.equal(_t(ref["depth"]).int(), rec.depth.int()))
        results["incoming_moves_match"] = list(ref["incoming_moves"]) == list(rec.incoming_moves)

        # --- node_features: align by feature name ---
        ref_names = list(ref["feature_names"])
        our_names = list(rec.feature_names)
        results["feature_names"] = (ref_names, our_names)
        for fname in ref_names:
            if fname in our_names:
                ri, oi = ref_names.index(fname), our_names.index(fname)
                results[f"feat[{fname}]_maxdiff"] = _maxdiff(
                    _t(ref["node_features"])[:, ri], rec.node_features[:, oi]
                )
        results["node_targets_maxdiff"] = _maxdiff(ref["node_targets"], rec.node_targets)

    # --- oracle trace ---
    results["oracle_root_moves_match"] = list(ref["oracle_root_moves"]) == list(rec.oracle_root_moves)
    results["oracle_root_moves_set_match"] = set(ref["oracle_root_moves"]) == set(rec.oracle_root_moves)
    results["oracle_n_moves"] = (len(ref["oracle_root_moves"]), len(rec.oracle_root_moves))
    results["oracle_expansion_counts_match"] = bool(
        torch.equal(_t(ref["oracle_trace_expansion_counts"]).int(), rec.oracle_trace_expansion_counts.int())
    )
    if list(ref["oracle_root_moves"]) == list(rec.oracle_root_moves):
        results["oracle_root_q_trace_maxdiff"] = _maxdiff(ref["oracle_root_q_trace"], rec.oracle_root_q_trace)
        results["oracle_final_root_q_maxdiff"] = _maxdiff(ref["oracle_final_root_q_values"], rec.oracle_final_root_q_values)
        # best-move-index sequence
        rb = [int(x) for x in _t(ref["oracle_best_move_index"]).tolist()]
        ob = [int(x) for x in rec.oracle_best_move_index.tolist()]
        results["oracle_best_move_index_match"] = rb == ob
    # best move by NAME (robust to move-ordering differences)
    rb_idx = [int(x) for x in _t(ref["oracle_best_move_index"]).tolist()]
    ob_idx = [int(x) for x in rec.oracle_best_move_index.tolist()]
    rb_names = [ref["oracle_root_moves"][i] for i in rb_idx]
    ob_names = [rec.oracle_root_moves[i] for i in ob_idx]
    results["oracle_best_move_NAME_match"] = rb_names == ob_names
    results["oracle_best_move_NAME_first/last"] = ((rb_names[0], ob_names[0]), (rb_names[-1], ob_names[-1]))

    # --- edge_wdl_targets (schema drift: v2 populated, v5 default NaN) ---
    ref_ew = _t(ref["edge_wdl_targets"]).float()
    our_ew = rec.edge_wdl_targets.float()
    results["edge_wdl_ref_all_finite"] = bool(torch.isfinite(ref_ew).all()) if ref_ew.numel() else None
    results["edge_wdl_our_all_nan"] = bool(torch.isnan(our_ew).all()) if our_ew.numel() else None
    if same_N and ref_ew.shape == our_ew.shape and not bool(torch.isnan(our_ew).all()):
        results["edge_wdl_maxdiff"] = _maxdiff(ref_ew, our_ew)

    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    prior_config, value_config = _make_provider()
    with UciEngineProcess(prior_config) as prior_engine, UciEngineProcess(value_config) as value_engine:
        for p in REF_PATHS:
            if not os.path.exists(p):
                print(f"MISSING {p}", file=sys.stderr)
                continue
            compare(p, prior_engine, value_engine)


if __name__ == "__main__":
    main()
