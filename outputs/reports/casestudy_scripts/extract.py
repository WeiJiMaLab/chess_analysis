"""One-off (not part of the pipeline): find episodes where the Action-Gap and Meta-Controller
(z_t) stop controllers diverge significantly, on the validated xaba100k_minply15_maxply75_history
regime (linear time_lambda=0.005 -- see history.md L124-127), and dump enough raw-tree data for a
handful of top candidates to render board/tree case-study figures offline (no cluster resources
needed for that second step). Also re-fits the zt/ag readout heads standalone (same seed, verified
bit-identical to the fit inside _fit_stop_controllers) to expose each controller's raw per-step
PREDICTED advantage -- the actual signal each stop decision is made from (stop at the first step
with advantage <= 0, see cts.models.readout.stop_step_from_advantages) -- not just the resulting
stop step, so the case-study plots can show when each controller's own belief crosses zero.

Run via slurm/pipeline (see submit script) -- reuses analysis.evaluate's private fit/load helpers
verbatim so the numbers here are exactly what regret-scatter/frontier compute, not a re-derivation.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from analysis.evaluate import (
    _load_assessment_data, _return_curves, _fit_stop_controllers, _regret_at, _oracle_config,
    _steps_zt_tensor, _steps_ag_tensor,
)
from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from cts.models.readout import build_advantage_head, stop_step_from_advantages
from cts.train.pg_controller_train import fit_readout_pg

PACKED_ROOT = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba100k_minply15_maxply75/pack_history")
CACHE_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba100k_minply15_maxply75/materialize_history/validation_cache.pt"
D_EMBED = 32
SEED = 0
MAX_EPISODES = 15000
TIME_MODE = "linear"
TIME_LAMBDA = 0.005  # validated regime, history.md L124-127
OUT_PATH = Path("/home/hl4291/chess_analysis/outputs/reports/case_study_candidates.json")
N_CANDIDATES = 40  # top-N by |regret gap| (tie-break: smaller explored tree), saved for manual selection


def _raw_row(first_decision_expansion_count: int, packed_step: int) -> int:
    return first_decision_expansion_count - 1 + packed_step


def _train_readout_with_head(fit_feats, fit_curves, *, in_dim, epochs, lr, seed):
    """Verbatim copy of ``analysis.evaluate._train_readout``'s training procedure, except it returns
    the trained head + normalization stats instead of only greedy eval stop steps -- so the SAME
    fitted controller can be replayed on arbitrary episodes afterward to get raw per-step predicted
    advantages, not just the final stop-step decision. Every argument matches what
    ``_fit_stop_controllers`` passes for the "zt"/"ag" controllers exactly, so with the same `seed`
    this reproduces bit-identical weights (`torch.manual_seed(seed)` is called fresh at the same two
    points -- once here, once inside `fit_readout_pg` -- regardless of what was fit before it)."""
    full = torch.cat(fit_feats, 0)
    mean, std = full.mean(0), full.std(0).clamp_min(1e-6)
    torch.manual_seed(seed)
    head = build_advantage_head(in_dim, 64, 2)
    fit_readout_pg(head, [(f - mean) / std for f in fit_feats], fit_curves, epochs=epochs, lr=lr, seed=seed)
    head.eval()
    return head, mean, std


def _advantage_trace(head, mean, std, feat: torch.Tensor) -> list[float]:
    with torch.no_grad():
        return head((feat - mean) / std).reshape(-1).tolist()


def main() -> None:
    config = replace(_oracle_config(PACKED_ROOT), time_mode=TIME_MODE, time_lambda=TIME_LAMBDA,
                     maintenance_scale=0.0, maintenance_exponent=1.0)
    print(f"[1/4] loading assessment data (packed_root={PACKED_ROOT})...", flush=True)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(
        PACKED_ROOT, CACHE_PATH, D_EMBED, MAX_EPISODES, SEED, train_frac=0.7, load_action_gaps=True)
    print(f"  {len(episodes)} episodes ({len(fit_idx)} fit / {len(ev_idx)} eval)", flush=True)

    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    print("[2/4] fitting stop controllers (singlehalt / stats / zt / ag)...", flush=True)
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, D_EMBED, SEED)
    print(f"  k_singlehalt={ctrl['k_singlehalt']}", flush=True)

    print("[2b/4] re-fitting zt/ag heads standalone (same seed) to expose raw per-step advantages...",
          flush=True)
    fit_feats_zt = [_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in fit_idx]
    fit_feats_ag = [_steps_ag_tensor(episodes[i]) for i in fit_idx]
    head_zt, mean_zt, std_zt = _train_readout_with_head(
        fit_feats_zt, fit_curves, in_dim=D_EMBED + 1, epochs=200, lr=1e-3, seed=SEED)
    head_ag, mean_ag, std_ag = _train_readout_with_head(
        fit_feats_ag, fit_curves, in_dim=2, epochs=200, lr=1e-3, seed=SEED)
    # Sanity check: this standalone re-fit must reproduce _fit_stop_controllers' own ctrl["zt"]/
    # ctrl["ag"] stop steps EXACTLY (same seed, same data -> bit-identical weights) -- if it doesn't,
    # the advantage traces below are not actually what determined those stop decisions and must not
    # be trusted or reported.
    mismatches = 0
    for j, i in enumerate(ev_idx):
        adv_zt = _advantage_trace(head_zt, mean_zt, std_zt, _steps_zt_tensor(episodes[i], z_by_ep[i]))
        adv_ag = _advantage_trace(head_ag, mean_ag, std_ag, _steps_ag_tensor(episodes[i]))
        if stop_step_from_advantages(torch.tensor(adv_zt)) != int(ctrl["zt"][j]):
            mismatches += 1
        if stop_step_from_advantages(torch.tensor(adv_ag)) != int(ctrl["ag"][j]):
            mismatches += 1
    print(f"  reproducibility check: {mismatches}/{2 * len(ev_idx)} stop-step mismatches "
          f"(must be 0 to trust the advantage traces below)", flush=True)
    assert mismatches == 0, "standalone re-fit did not reproduce _fit_stop_controllers -- aborting"

    print("[3/4] ranking eval episodes by |Action-Gap regret - Meta-Controller regret|...", flush=True)
    rows = []
    for j, i in enumerate(ev_idx):
        ep = episodes[i]
        num_steps = len(ep["halt_rewards"])
        stop_zt, stop_ag = int(ctrl["zt"][j]), int(ctrl["ag"][j])
        regret_zt = float(_regret_at([curves[i]], [stop_zt])[0])
        regret_ag = float(_regret_at([curves[i]], [stop_ag])[0])
        rows.append(dict(
            eval_idx=j, episode_idx=int(i), trajectory_key=ep["trajectory_key"], num_steps=num_steps,
            stop_zt=stop_zt, stop_ag=stop_ag, regret_zt=regret_zt, regret_ag=regret_ag,
            delta_ag_minus_zt=regret_ag - regret_zt, oracle_stop_step=ep["oracle_stop_step"],
            tree_sizes=ep["tree_sizes"], return_curve=curves[i].tolist(),
            halt_rewards=ep["halt_rewards"],
            advantage_zt=_advantage_trace(head_zt, mean_zt, std_zt, _steps_zt_tensor(ep, z_by_ep[i])),
            advantage_ag=_advantage_trace(head_ag, mean_ag, std_ag, _steps_ag_tensor(ep)),
        ))
    # num_steps is constant (96, the fixed search budget every episode shares -- confirmed by a
    # dry run, so it carries no filtering signal); "short/simple" instead means a small explored
    # tree (tree_sizes[-1], the final packed node count). Rank candidates by |regret gap| first
    # (the actual "diverge significantly" criterion), break ties toward smaller trees.
    rows.sort(key=lambda r: (-abs(r["delta_ag_minus_zt"]), r["tree_sizes"][-1]))
    top = rows[:N_CANDIDATES]
    print(f"  top {len(top)} candidates by |Δregret|: sizes {[r['tree_sizes'][-1] for r in top]}", flush=True)

    print(f"[4/4] pulling raw tree data for the top {len(top)} candidates...", flush=True)
    shard_cache: dict[str, dict] = {}
    out_candidates = []
    for r in top:
        shard_name, traj_s = r["trajectory_key"].split("#")
        traj = int(traj_s)
        if shard_name not in shard_cache:
            shard_cache[shard_name] = torch.load(PACKED_ROOT / "validation" / shard_name, weights_only=False)
        p = shard_cache[shard_name]
        source_path = p["trajectory_source_paths"][traj]
        fdec = int(p["first_decision_expansion_counts"][traj].item())
        rec = RawPretrainExampleRecord.load(source_path)

        num_steps = r["num_steps"]
        raw_rows = [_raw_row(fdec, s) for s in range(num_steps)]
        max_raw_row = rec.oracle_root_q_trace.shape[0] - 1
        raw_rows = [min(rr, max_raw_row) for rr in raw_rows]
        q_trace = rec.oracle_root_q_trace[raw_rows].tolist()  # [num_steps, n_root_moves]
        best_idx = rec.oracle_best_move_index[raw_rows].tolist()

        out_candidates.append(dict(
            **r,
            source_path=source_path,
            root_fen=rec.root_position_spec,
            oracle_root_moves=rec.oracle_root_moves,
            oracle_final_root_q_values=rec.oracle_final_root_q_values.tolist(),
            q_trace=q_trace,
            best_move_index_per_step=best_idx,
            first_decision_expansion_count=fdec,
            num_raw_tree_nodes=int(rec.node_features.shape[0]),
        ))
        print(f"  {r['trajectory_key']}: num_steps={num_steps} stop_zt={r['stop_zt']} "
              f"stop_ag={r['stop_ag']} delta={r['delta_ag_minus_zt']:+.4f} "
              f"root_fen={rec.root_position_spec}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(dict(config=dict(time_mode=TIME_MODE, time_lambda=TIME_LAMBDA, seed=SEED,
                                   packed_root=str(PACKED_ROOT), cache_path=CACHE_PATH),
                       candidates=out_candidates), f, indent=1)
    print(f"Saved {len(out_candidates)} candidates -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
