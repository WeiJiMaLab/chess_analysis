#!/usr/bin/env python
"""minimal_mc.py — a MINIMAL, transparent meta-controller pipeline (Phase 1+2).

One readable script, driven by ONE yaml (minimal_mc.yaml), runs end-to-end in
<=30 min on CPU and is fully self-documenting. It replaces the sprawling 7-stage
cluster pipeline whose results we no longer trust (see mc_pipeline.md /
mc_minimal_plan.md).

What it does, in order:
  1. Load a fixed-seed RANDOM subset of the CURRENT filtered tree set
     (clean_trees.txt basenames -> human_trees/*.pt).
  2. Build budgeted-oracle episodes per tree, REUSING the existing oracle so the
     metric is byte-identical (cts.data.preprocess_mc.{pack,oracle}). The cost
     model is the canonical budgeted_controller_v1 (mc_pipeline.md §6).
  3. ROOT FEATURES: a TRIVIAL hand-crafted per-snapshot vector (a few
     interpretable scalars) — NOT the GNN. The GNN is a clean later swap-in
     (see `snapshot_root_features` / the `features.root` config seam).
  4. ONE eval function for ALL models: regret = oracle_value -
     return_for_stop_step(stop_step); plus P(stop==OSS) and avg expansions, on
     ONE fixed held-out val split.
  5. Fit ALL models on REGRET DIRECTLY, on the SAME train split, judged by the
     SAME eval fn on the SAME val split:
        - Always-Stop, Never-Stop                       (fixed)
        - Fraction-of-Budget f*                         (1 scalar, train regret)
        - budget-only MC head (b)                       (nested FLOOR)
        - full MC head over (root_feats, budget) (a)    (must beat the floor)
  6. Emit ONE comparison table + the headline number
        value of root state = regret(budget-only) - regret(full)
     and a figure. FAIRNESS / nestedness is checked and flagged LOUDLY if broken.

Every model is reduced to the SAME shape: a per-episode STOP RULE that returns a
stop step. Regret is then computed by the SAME eval fn for everyone — fairness is
structural, not hoped-for.

ENV:
  source /home/hl4291/venv/bin/activate
  export PYTHONPATH=/home/hl4291/chess_analysis/lmcos/src
  python /home/hl4291/chess_analysis/minimal_mc.py --config /home/hl4291/chess_analysis/minimal_mc.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml

# --- REUSE the existing oracle + episode builder (identical metric) ----------
from cts.data.preprocess_mc.oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    predicted_stop_from_advantages,
    return_for_stop_step,
)
from cts.data.preprocess_mc.pack import (
    build_compact_trajectory_from_payload,
    budgeted_oracle_from_trajectory,
)
from cts.data.preprocess_gnn.teacher_targets import load_raw_pretrain_record


# =============================================================================
# Logging: everything printed also goes to the run log, so the run is
# reproducible from script+yaml+log alone.
# =============================================================================
class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._fh = open(log_path, "w")
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._fh.write(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def log(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# Config
# =============================================================================
def build_oracle_config(oc: dict[str, Any]) -> BudgetedOracleConfig:
    """Construct the canonical BudgetedOracleConfig from the yaml (no silent edits)."""
    buckets = tuple(
        BudgetBucket(name=b["name"], min_time=int(b["min_time"]), max_time=int(b["max_time"]))
        for b in oc["budget_buckets"]
    )
    return BudgetedOracleConfig(
        maintenance_scale=float(oc["maintenance_scale"]),
        maintenance_ref_nodes=float(oc["maintenance_ref_nodes"]),
        maintenance_exponent=float(oc["maintenance_exponent"]),
        time_lambda=float(oc["time_lambda"]),
        time_p=float(oc["time_p"]),
        time_tau=float(oc["time_tau"]),
        time_delta=int(oc["time_delta"]),
        timeout_value=float(oc["timeout_value"]),
        budget_buckets=buckets,
        samples_per_bucket=int(oc["samples_per_bucket"]),
        seed=int(oc["seed"]),
    )


# =============================================================================
# Trivial hand-crafted root features (the GNN swap-in seam lives here)
# =============================================================================
def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy (nats) of a non-negative weight vector, normalized to a distribution."""
    p = np.clip(probs.astype(np.float64), 0.0, None)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def snapshot_root_features(
    record: Any,
    trajectory: dict[str, Any],
    step: int,
    starting_budget: int,
    feature_names: Sequence[str],
) -> list[float]:
    """A few INTERPRETABLE scalars describing the root state at episode `step`.

    This is the trivial stand-in for a GNN root embedding `z_t`. To swap in the
    GNN later, replace this function's body with an encoder forward pass over the
    step-`step` tree snapshot; the rest of the pipeline is untouched (clean seam).

    Available per-step signal (all from the raw record / trajectory, no search):
      - root_value       : root Q of the current best move at this step (= halt reward)
      - policy_entropy   : entropy of the root move Q-distribution at this step
      - q_margin         : top1 - top2 of the root move Q-values (decisiveness)
      - bmi_stable       : 1.0 if argmax root move == final argmax, else 0.0
      - log_tree_size    : log(1 + current tree size)
      - frac_budget_used : step / starting_budget (a budget-derived but root-agnostic scalar)
    """
    halt_rewards = np.asarray(trajectory["halt_rewards"], dtype=np.float64)
    tree_sizes = np.asarray(trajectory["tree_sizes"], dtype=np.int64)
    q_trace = np.asarray(record.oracle_root_q_trace, dtype=np.float64)  # (96, n_moves)
    bmi = np.asarray(record.oracle_best_move_index, dtype=np.int64)      # (96,)
    final_bmi = int(bmi[-1])

    # The trajectory trims pre-root steps; map episode step -> raw trace row.
    # first_decision_expansion_count is (root_rank + 1); raw row = root_rank + step.
    raw_row = int(trajectory["first_decision_expansion_count"]) - 1 + step
    raw_row = max(0, min(raw_row, q_trace.shape[0] - 1))

    row = q_trace[raw_row]
    # Shift Q-values to non-negative weights for an entropy/softmax read.
    w = row - row.min()
    sorted_q = np.sort(row)[::-1]
    q_margin = float(sorted_q[0] - sorted_q[1]) if row.shape[0] >= 2 else 0.0

    feats = {
        "root_value": float(halt_rewards[step]),
        "policy_entropy": _entropy(w),
        "q_margin": q_margin,
        "bmi_stable": 1.0 if int(bmi[raw_row]) == final_bmi else 0.0,
        "log_tree_size": float(math.log1p(int(tree_sizes[step]))),
        "frac_budget_used": float(step) / float(max(1, starting_budget)),
    }
    return [feats[name] for name in feature_names]


# =============================================================================
# Episode construction (REUSES the existing oracle -> identical metric)
# =============================================================================
@dataclass
class Episode:
    """One (tree, starting_budget) episode: the oracle ground truth + per-step features."""
    halt_rewards: list[float]      # per-step halt reward (scaled, == oracle input)
    tree_sizes: list[int]          # per-step node count
    time_budgets: list[int]        # per-step remaining budget
    oracle_value: float            # V*(0): the achievable return (oracle)
    oracle_stop_step: int          # OSS from step 0
    target_advantages: list[float] # post-DP continue-halt advantage (regression target)
    starting_budget: int
    root_features: np.ndarray      # (num_steps, n_root_feats), trivial hand-crafted
    budget_feature: np.ndarray     # (num_steps,), remaining budget per step (the only feat the floor sees)


def build_episodes_for_tree(
    pt_path: str,
    oracle_config: BudgetedOracleConfig,
    reward_scale: float,
    feature_names: Sequence[str],
) -> list[Episode]:
    """Build all budget-bucket episodes for one tree, exactly as the pack would.

    Mirrors pack._build_packed_tree_result: same trajectory, same per-tree
    deterministic budget draws, same oracle DP. We intentionally do NOT apply the
    min_halt_reward_range / min_decision_margin pack filters here — the eval fn is
    computed over whatever episodes we build, and ALL models see the same set, so
    fairness is preserved. (Keeping all episodes also keeps the subset honest.)
    """
    from cts.data.preprocess_mc.oracle import deterministic_starting_budgets

    record = load_raw_pretrain_record(pt_path)
    if record is None:
        return []
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    trajectory = build_compact_trajectory_from_payload(payload, source_path=pt_path)
    if trajectory is None:
        return []

    # Reward-scale the halt rewards, matching pack (_build_packed_tree_result).
    raw_halt = np.asarray(trajectory["halt_rewards"], dtype=np.float64)
    trajectory = dict(trajectory)
    trajectory["halt_rewards"] = (reward_scale * raw_halt).astype(np.float32)

    episodes: list[Episode] = []
    for sampled in deterministic_starting_budgets(pt_path, oracle_config):
        budget = sampled.starting_budget
        policy = budgeted_oracle_from_trajectory(trajectory, budget, oracle_config)
        num_steps = len(policy.halt_rewards)
        if num_steps == 0:
            continue
        root_feats = np.asarray(
            [
                snapshot_root_features(record, trajectory, step, budget, feature_names)
                for step in range(num_steps)
            ],
            dtype=np.float64,
        )
        episodes.append(
            Episode(
                halt_rewards=list(policy.halt_rewards),
                tree_sizes=list(policy.tree_sizes),
                time_budgets=list(policy.time_budgets),
                oracle_value=float(policy.oracle_value),
                oracle_stop_step=int(policy.optimal_stop_step),
                target_advantages=list(policy.target_advantages),
                starting_budget=budget,
                root_features=root_feats,
                budget_feature=np.asarray(policy.time_budgets, dtype=np.float64),
            )
        )
    return episodes


# =============================================================================
# THE ONE eval function (used by EVERY model)
# =============================================================================
def evaluate(
    episodes: list[Episode],
    oracle_config: BudgetedOracleConfig,
    stop_rule: Callable[[Episode], int],
) -> dict[str, float]:
    """Regret + P(stop==OSS) + avg expansions for a per-episode stop rule.

    Identical semantics to alt_models_eval / controller greedy-eval:
      regret_i  = oracle_value_i - return_for_stop_step(stop_i)
      stop_i    clamped to [0, num_steps-1]
      P(stop==OSS) = mean[ stop_i == oracle_stop_step_i ]
      avg_expansions = mean[ stop_i ]   (a stop at step k followed k expansions)
    """
    regrets, exact, expansions = [], 0, []
    for ep in episodes:
        last = len(ep.halt_rewards) - 1
        stop = max(0, min(stop_rule(ep), last))
        ret = return_for_stop_step(ep.halt_rewards, ep.tree_sizes, ep.time_budgets, stop, oracle_config)
        regrets.append(ep.oracle_value - ret)
        exact += int(stop == ep.oracle_stop_step)
        expansions.append(stop)
    n = max(1, len(episodes))
    return {
        "average_regret": float(np.mean(regrets)),
        "p_stop_eq_oss": exact / n,
        "average_expansions": float(np.mean(expansions)),
        "n_episodes": len(episodes),
    }


# =============================================================================
# Models — every model is a STOP RULE. All fit on TRAIN regret.
# =============================================================================
def always_stop(_: Episode) -> int:
    return 0


def never_stop(ep: Episode) -> int:
    return len(ep.halt_rewards) - 1


def fraction_rule(f: float) -> Callable[[Episode], int]:
    """Fraction-of-Budget: STOP at round(f * B). 1 scalar."""
    def rule(ep: Episode) -> int:
        return int(round(f * ep.starting_budget))
    return rule


def fit_fraction(train: list[Episode], cfg: BudgetedOracleConfig) -> tuple[float, float]:
    """Grid-search f in (0,1) to MINIMIZE MEAN TRAIN REGRET (piecewise-constant in f)."""
    best_f, best_r = 0.0, math.inf
    for k in range(1, 100):
        f = round(0.01 * k, 2)
        r = evaluate(train, cfg, fraction_rule(f))["average_regret"]
        if r < best_r:
            best_f, best_r = f, r
    return best_f, best_r


def budget_threshold_rule(theta: float) -> Callable[[Episode], int]:
    """The 1-parameter BUDGET-ONLY head: advantage(step) = theta - step/B; STOP when adv<=0.

    Equivalently: STOP at the first step whose fraction-of-budget-used
    `step / starting_budget` reaches `theta`. The decision depends ONLY on the
    budget state (the step index and the known budget B) — never on the root
    features. This is the nested FLOOR:
      - theta=0  -> Always-Stop
      - theta>=1 -> Never-Stop
      - general  -> fraction-of-budget with f=theta
    so its train-regret-fitted value can only be <= Fraction-f*'s (it sweeps the
    SAME family on a FINER grid). The full head, which also sees `frac_budget`,
    must in turn beat this.
    """
    # Use the SAME stop-step formula as fraction_rule (round(theta*B)) so the
    # budget-only family STRICTLY contains fraction-of-budget; only the grid is
    # finer. (Any other rounding would make them merely overlapping, not nested,
    # and could let budget-only lose to fraction on TRAIN — a spurious bug flag.)
    def rule(ep: Episode) -> int:
        return int(round(theta * ep.starting_budget))
    return rule


def fit_budget_only(train: list[Episode], cfg: BudgetedOracleConfig, n_points: int) -> tuple[float, float]:
    """Fit the 1-parameter budget-only head by DIRECTLY minimizing train regret.

    Sweep theta in [0, 1] (fraction-of-budget threshold) on a grid that is a
    SUPERSET of the Fraction-f* grid (n_points >= 101 covers all k/100). Since the
    rule is identical to fraction_rule(theta), this is a true nested floor:
    its train regret can never exceed Fraction-f*'s.
    """
    # Superset of the fraction grid (k/100, k=0..100) UNION a finer linspace, so
    # budget-only can always match-or-beat Fraction-f* on TRAIN by construction.
    fraction_grid = [round(0.01 * k, 2) for k in range(0, 101)]
    grid = sorted(set(fraction_grid) | set(np.linspace(0.0, 1.0, n_points).tolist()))
    best_t, best_r = 0.0, math.inf
    for t in grid:
        r = evaluate(train, cfg, budget_threshold_rule(float(t)))["average_regret"]
        if r < best_r:
            best_t, best_r = float(t), r
    return best_t, best_r


# --- the full MC head: tiny MLP advantage(root_feats, budget) -> STOP at adv<=thr
class AdvantageMLP(torch.nn.Module):
    """A few-unit MLP that predicts the oracle's post-DP advantage per step.

    Inputs = [z-scored root features, z-scored remaining budget]. Because budget
    is one of the inputs, this head CAN represent the budget-only rule — so by
    nestedness its train-regret-selected stop policy must do at least as well.
    """
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _episode_feature_matrix(ep: Episode, col_idx: Sequence[int] | None) -> np.ndarray:
    """Per-step feature matrix for one episode: [root_features | budget], optionally
    restricted to the column indices in `col_idx` (None = all columns)."""
    full = np.column_stack([ep.root_features, ep.budget_feature])
    return full if col_idx is None else full[:, list(col_idx)]


def _stack_steps(episodes: list[Episode], col_idx: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten selected (root_features, budget) step rows and their advantage targets."""
    X = np.concatenate([_episode_feature_matrix(ep, col_idx) for ep in episodes], axis=0)
    y = np.concatenate([np.asarray(ep.target_advantages, dtype=np.float64) for ep in episodes])
    return X, y


def fit_surrogate_head(
    train: list[Episode],
    val: list[Episode],
    head_cfg: dict[str, Any],
    seed: int,
    col_idx: Sequence[int] | None,
    sign_loss_weight: float,
) -> tuple[Callable[[Episode], int], dict[str, Any]]:
    """Fit an advantage MLP on the CLUSTER's surrogate recipe, then use its FIXED rule.

    Faithfully mirrors controller_train.py (mc_pipeline.md §8):
      loss = advantage_MSE + sign_loss_weight * sign_BCE
        advantage_MSE = MSE(pred_adv, target_advantage)            (post-DP target)
        sign_BCE      = BCE( sigmoid(pred_adv), 1{target_adv > 0} ) (decision-direction)
      decision rule = predicted_stop_from_advantages  (STOP at first pred_adv <= 0)

    Crucially, and unlike the regret-direct heads: the stop threshold is the FIXED
    untuned adv<=0 rule (NOT selected on regret), and there is NO budget-only
    floor fallback. This is the exact configuration we suspect is the root cause
    of the cluster controller losing to its own nested baselines, reproduced in
    miniature. `col_idx=None` -> all features (full head); a single-column index
    -> budget-only head.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr = _stack_steps(train, col_idx)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0

    Xn = (Xtr - mu) / sd
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    sign_t = (yt > 0).to(torch.float32)  # 1 = continue is better (target_adv > 0)

    model = AdvantageMLP(in_dim=Xn.shape[1], hidden=int(head_cfg["hidden_units"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(head_cfg["lr"]),
                           weight_decay=float(head_cfg["weight_decay"]))
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()  # logits = predicted advantage (sign decision)
    model.train()
    for _ in range(int(head_cfg["epochs"])):
        opt.zero_grad()
        pred = model(Xt)
        loss = mse(pred, yt) + float(sign_loss_weight) * bce(pred, sign_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_all = model(Xt)
        final_mse = float(mse(pred_all, yt).item())
        final_bce = float(bce(pred_all, sign_t).item())
        sign_acc = float(((pred_all > 0).to(torch.float32) == sign_t).to(torch.float32).mean().item())

    def predict_adv(ep: Episode) -> np.ndarray:
        x = _episode_feature_matrix(ep, col_idx)
        xn = (x - mu) / sd
        with torch.no_grad():
            return model(torch.tensor(xn, dtype=torch.float32)).numpy()

    adv_cache = {id(ep): predict_adv(ep) for ep in (train + val)}

    def rule(ep: Episode) -> int:
        # FIXED adv<=0 rule via the SAME helper the cluster eval uses.
        return predicted_stop_from_advantages(adv_cache[id(ep)])

    prov = {
        "objective": "MSE + %.2f*signBCE (fixed adv<=0; no threshold tuning; no floor)" % sign_loss_weight,
        "advantage_mse_train": final_mse,
        "sign_bce_train": final_bce,
        "sign_accuracy_train": sign_acc,
        "n_input_features": int(Xn.shape[1]),
    }
    return rule, prov


def fit_full_head(
    train: list[Episode],
    val: list[Episode],
    cfg: BudgetedOracleConfig,
    head_cfg: dict[str, Any],
    seed: int,
    floor_rule: Callable[[Episode], int] | None = None,
    floor_train_regret: float = math.inf,
) -> tuple[Callable[[Episode], int], dict[str, Any]]:
    """Fit the advantage MLP, then SELECT the STOP threshold ON TRAIN REGRET.

    Selection == the judged metric (train regret), so the comparison is fair.

    Nestedness, made structural: the budget-only rule (`floor_rule`) is literally
    inside the full head's hypothesis class (the head may ignore the root features
    and key only off `frac_budget_used`, which is among its inputs). So we let the
    full head FALL BACK to the budget-only rule when the MLP's best train regret
    fails to beat it. This guarantees regret(full) <= regret(budget-only) by
    construction, on the SAME train-regret selection criterion — exactly the
    nestedness the plan requires. Any residual loss to fraction-f* would then be a
    real budget-only-vs-fraction bug, surfaced by the floor itself.

    Returns the stop rule (closure over the trained net + chosen threshold) and a
    provenance dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr = _stack_steps(train)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0  # guard against constant columns

    Xn = (Xtr - mu) / sd
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)

    model = AdvantageMLP(in_dim=Xn.shape[1], hidden=int(head_cfg["hidden_units"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(head_cfg["lr"]),
                           weight_decay=float(head_cfg["weight_decay"]))
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(int(head_cfg["epochs"])):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    final_mse = float(loss.item())

    def predict_adv(ep: Episode) -> np.ndarray:
        x = np.column_stack([ep.root_features, ep.budget_feature])
        xn = (x - mu) / sd
        with torch.no_grad():
            return model(torch.tensor(xn, dtype=torch.float32)).numpy()

    # cache predictions so threshold sweep is cheap
    adv_train = {id(ep): predict_adv(ep) for ep in train}
    adv_val = {id(ep): predict_adv(ep) for ep in val}

    def stop_rule_for(thr: float, adv_cache: dict[int, np.ndarray]) -> Callable[[Episode], int]:
        def rule(ep: Episode) -> int:
            adv = adv_cache[id(ep)]
            for step, a in enumerate(adv):
                if a <= thr:
                    return step
            return len(ep.halt_rewards) - 1
        return rule

    all_adv = np.concatenate(list(adv_train.values()))
    lo, hi = float(all_adv.min()), float(all_adv.max())
    grid = np.linspace(lo, hi, int(head_cfg["threshold_grid_points"]))
    best_thr, best_train_r = grid[0], math.inf
    for thr in grid:
        r = evaluate(train, cfg, stop_rule_for(float(thr), adv_train))["average_regret"]
        if r < best_train_r:
            best_thr, best_train_r = float(thr), r

    # Return a rule that works on BOTH train and val episodes (lookup by id).
    merged = {**adv_train, **adv_val}

    def mlp_rule(ep: Episode) -> int:
        adv = merged[id(ep)]
        for step, a in enumerate(adv):
            if a <= best_thr:
                return step
        return len(ep.halt_rewards) - 1

    # Structural nestedness: pick whichever of {MLP, budget-only floor} has lower
    # TRAIN regret (the floor is inside the full head's class). Same criterion.
    used_floor = False
    if floor_rule is not None and floor_train_regret < best_train_r - 1e-12:
        final_rule = floor_rule
        used_floor = True
    else:
        final_rule = mlp_rule

    prov = {
        "advantage_mse_train": final_mse,
        "selected_threshold": best_thr,
        "mlp_train_regret": best_train_r,
        "floor_train_regret": (None if floor_train_regret == math.inf else floor_train_regret),
        "fell_back_to_budget_only_floor": used_floor,
        "train_regret_at_selection": min(best_train_r, floor_train_regret),
        "n_step_rows_train": int(Xn.shape[0]),
        "feature_means": mu.tolist(),
        "feature_stds": sd.tolist(),
    }
    return final_rule, prov


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="/home/hl4291/chess_analysis/minimal_mc.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run = cfg["run"]
    out_dir = Path(run["out_dir"])
    tee = _Tee(out_dir / run["log_file"])
    sys.stdout = tee  # everything below is logged
    try:
        _run(cfg)
    finally:
        sys.stdout = tee._stdout
        tee.close()


def _cache_key(cfg: dict[str, Any], oracle_config: BudgetedOracleConfig, feature_names: Sequence[str]) -> str:
    """Hash everything that determines the built episodes (so a change auto-invalidates)."""
    import hashlib
    from cts.data.preprocess_mc.oracle import budgeted_oracle_metadata
    payload = {
        "seed": int(cfg["run"]["seed"]),
        "subset_size": int(cfg["data"]["subset_size"]),
        "val_fraction": float(cfg["data"]["val_fraction"]),
        "reward_scale": float(cfg["oracle"]["reward_scale"]),
        "feature_names": list(feature_names),
        "oracle": budgeted_oracle_metadata(oracle_config),
        "clean_trees_txt": cfg["data"]["clean_trees_txt"],
        "human_trees_dir": cfg["data"]["human_trees_dir"],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_or_load_episodes(
    cfg: dict[str, Any], oracle_config: BudgetedOracleConfig, reward_scale: float,
    feature_names: Sequence[str], seed: int,
) -> tuple[list[Episode], list[Episode], int]:
    """Return (train_eps, val_eps, n_trees_used), using a disk cache keyed on the
    subset/seed/oracle/feature config. Skips the ~20-min build on a cache hit."""
    data = cfg["data"]
    use_cache = bool(data.get("use_cache", True))
    cache_dir = Path(data.get("cache_dir", ""))
    key = _cache_key(cfg, oracle_config, feature_names)
    cache_path = cache_dir / f"episodes_{key}.pt" if str(cache_dir) else None

    if use_cache and cache_path is not None and cache_path.exists():
        log(f"\n--- EPISODE CACHE HIT: {cache_path} (key={key}) — skipping build ---")
        blob = torch.load(cache_path, weights_only=False)
        return blob["train_eps"], blob["val_eps"], int(blob["n_trees_used"])

    basenames = Path(data["clean_trees_txt"]).read_text().split()
    log(f"\n--- DATA ---\n  clean_trees.txt: {data['clean_trees_txt']}  ({len(basenames):,} basenames)")
    rng = random.Random(seed)
    subset = rng.sample(basenames, min(int(data["subset_size"]), len(basenames)))
    subset.sort()             # deterministic order independent of sample internals
    rng.shuffle(subset)       # then a seeded shuffle for the split
    human_dir = Path(data["human_trees_dir"])
    log(f"  subset_size={len(subset)} (seed={seed}) from {human_dir}")

    log("\n--- BUILDING EPISODES (reusing pack/oracle; identical to the cluster pack) ---")
    n_val_trees = int(round(len(subset) * float(data["val_fraction"])))
    val_trees = set(subset[:n_val_trees])
    train_eps: list[Episode] = []
    val_eps: list[Episode] = []
    n_trees_used = 0
    for i, base in enumerate(subset):
        pt = str(human_dir / base)
        if not Path(pt).exists():
            continue
        eps = build_episodes_for_tree(pt, oracle_config, reward_scale, feature_names)
        if not eps:
            continue
        n_trees_used += 1
        (val_eps if base in val_trees else train_eps).extend(eps)
        if (i + 1) % 500 == 0:
            log(f"  ...{i + 1}/{len(subset)} trees  (train_eps={len(train_eps):,} val_eps={len(val_eps):,})")
    log(f"  trees used={n_trees_used}  train_eps={len(train_eps):,}  val_eps={len(val_eps):,}")
    log(f"  split: {1 - float(data['val_fraction']):.0%} train / {float(data['val_fraction']):.0%} val "
        f"by TREE (no episode leaks across the split)")

    if use_cache and cache_path is not None and train_eps and val_eps:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"train_eps": train_eps, "val_eps": val_eps, "n_trees_used": n_trees_used}, cache_path)
        log(f"  cached episodes -> {cache_path} (key={key})")
    return train_eps, val_eps, n_trees_used


def _run(cfg: dict[str, Any]) -> None:
    run = cfg["run"]
    seed = int(run["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(run["out_dir"])
    oracle_config = build_oracle_config(cfg["oracle"])
    reward_scale = float(cfg["oracle"]["reward_scale"])
    feature_names = list(cfg["features"]["root"])

    log("=" * 78)
    log("minimal_mc.py — minimal transparent meta-controller (Phase 1+2)")
    log("=" * 78)
    log("\n--- CONFIG (verbatim; the run is reproducible from script + this) ---")
    log(yaml.safe_dump(cfg, sort_keys=False).rstrip())
    log("\n--- ORACLE (REUSED budgeted_controller_v1; metric identical) ---")
    log(f"  time_lambda={oracle_config.time_lambda} time_p={oracle_config.time_p} "
        f"time_tau={oracle_config.time_tau} time_delta={oracle_config.time_delta}")
    log(f"  maintenance_scale={oracle_config.maintenance_scale} reward_scale={reward_scale} "
        f"timeout_value={oracle_config.timeout_value} samples_per_bucket={oracle_config.samples_per_bucket}")
    log(f"  buckets={[ (b.name,b.min_time,b.max_time) for b in oracle_config.budget_buckets ]}")

    # --- 1+2. fixed-seed subset -> budgeted-oracle episodes (REUSED oracle) ----
    train_eps, val_eps, n_trees_used = build_or_load_episodes(
        cfg, oracle_config, reward_scale, feature_names, seed
    )
    if not train_eps or not val_eps:
        raise RuntimeError("empty train or val set — check data paths")

    # frac_budget_used column index (the ONLY input the budget-only surrogate sees).
    # Column layout in the stacked feature matrix = [root_features..., budget].
    frac_col = feature_names.index("frac_budget_used")

    # --- 3-5. fit models; judge on VAL with ONE eval fn -----------------------
    log("\n--- FITTING ---")
    results: dict[str, dict[str, Any]] = {}

    # Baselines (no fit / fixed).
    results["Always-Stop"] = evaluate(val_eps, oracle_config, always_stop)
    results["Never-Stop"] = evaluate(val_eps, oracle_config, never_stop)

    # Fraction-of-Budget f* (1 scalar, fit on TRAIN regret).
    best_f, ftrain = fit_fraction(train_eps, oracle_config)
    log(f"  Fraction-of-Budget f* = {best_f:.2f}  (train regret {ftrain:.4f})")
    results[f"Fraction f*={best_f:.2f}"] = evaluate(val_eps, oracle_config, fraction_rule(best_f))

    # === COLUMN 1: REGRET-DIRECT (fit/select on TRAIN regret) ================
    log("\n  [REGRET-DIRECT column] — fit/select on TRAIN regret")
    best_t, btrain = fit_budget_only(train_eps, oracle_config, int(cfg["budget_only"]["threshold_grid_points"]))
    log(f"    budget-only(regret): theta={best_t:.3f}  (train regret {btrain:.4f})  [nested FLOOR]")
    results["budget-only (regret)"] = evaluate(val_eps, oracle_config, budget_threshold_rule(best_t))

    full_rule, full_prov = fit_full_head(
        train_eps, val_eps, oracle_config, cfg["head"], seed,
        floor_rule=budget_threshold_rule(best_t), floor_train_regret=btrain,
    )
    log(f"    full(regret): adv-MSE(train)={full_prov['advantage_mse_train']:.5f}  "
        f"thr={full_prov['selected_threshold']:.4f}  "
        f"selected train regret={full_prov['train_regret_at_selection']:.4f}"
        + ("  [fell back to budget-only floor]" if full_prov["fell_back_to_budget_only_floor"] else ""))
    results["full (regret)"] = evaluate(val_eps, oracle_config, full_rule)

    # === COLUMN 2: MSE+BCE SURROGATE (fixed adv<=0; no tuning; no floor) =====
    # Faithfully replicates controller_train: regress target_advantage with an
    # MSE + 0.1*sign-BCE loss, then read STOP off the FIXED adv<=0 rule.
    sign_w = float(cfg["surrogate"]["sign_loss_weight"])
    log(f"\n  [MSE+BCE column] — loss = MSE + {sign_w}*signBCE; FIXED adv<=0 stop (no tuning, no floor)")

    bo_rule, bo_prov = fit_surrogate_head(
        train_eps, val_eps, cfg["head"], seed, col_idx=[frac_col], sign_loss_weight=sign_w
    )
    log(f"    budget-only(MSE+BCE): MSE={bo_prov['advantage_mse_train']:.5f} "
        f"BCE={bo_prov['sign_bce_train']:.4f} sign_acc={bo_prov['sign_accuracy_train']:.3f} "
        f"(input: frac_budget_used only)")
    results["budget-only (MSE+BCE)"] = evaluate(val_eps, oracle_config, bo_rule)

    full_s_rule, full_s_prov = fit_surrogate_head(
        train_eps, val_eps, cfg["head"], seed, col_idx=None, sign_loss_weight=sign_w
    )
    log(f"    full(MSE+BCE): MSE={full_s_prov['advantage_mse_train']:.5f} "
        f"BCE={full_s_prov['sign_bce_train']:.4f} sign_acc={full_s_prov['sign_accuracy_train']:.3f} "
        f"(inputs: all {full_s_prov['n_input_features']} feats)")
    results["full (MSE+BCE)"] = evaluate(val_eps, oracle_config, full_s_rule)
    log(f"    root features = {feature_names} (+ remaining_budget); hidden={cfg['head']['hidden_units']}")

    # --- 6. tables + interpretations + fairness check -------------------------
    frac_name = f"Fraction f*={best_f:.2f}"

    def row(name: str) -> str:
        m = results[name]
        return (f"{name:<24s} {m['average_regret']:>10.4f} {m['p_stop_eq_oss']:>14.3f} "
                f"{m['average_expansions']:>15.2f}")

    log("\n" + "=" * 78)
    log("BASELINES (VAL split — the SAME held-out + SAME eval fn for every model)")
    log("=" * 78)
    log(f"{'model':<24s} {'regret':>10s} {'P(stop==OSS)':>14s} {'avg_expansions':>15s}")
    log("-" * 78)
    for name in ["Always-Stop", "Never-Stop", frac_name]:
        log(row(name))

    log("\n" + "=" * 78)
    log("THE 2x2: rows {budget-only, full} x cols {regret-direct, MSE+BCE}  (VAL regret)")
    log("=" * 78)
    log(f"{'head':<24s} {'regret':>10s} {'P(stop==OSS)':>14s} {'avg_expansions':>15s}")
    log("-" * 78)
    cell_names = {
        ("budget-only", "regret"): "budget-only (regret)",
        ("budget-only", "MSE+BCE"): "budget-only (MSE+BCE)",
        ("full", "regret"): "full (regret)",
        ("full", "MSE+BCE"): "full (MSE+BCE)",
    }
    for r_head in ["budget-only", "full"]:
        for c_obj in ["regret", "MSE+BCE"]:
            log(row(cell_names[(r_head, c_obj)]))

    # compact 2x2 regret-only grid
    g = {k: results[v]["average_regret"] for k, v in cell_names.items()}
    log("\n  regret grid:")
    log(f"    {'':<14s}{'regret-direct':>16s}{'MSE+BCE':>14s}")
    log(f"    {'budget-only':<14s}{g[('budget-only','regret')]:>16.4f}{g[('budget-only','MSE+BCE')]:>14.4f}")
    log(f"    {'full':<14s}{g[('full','regret')]:>16.4f}{g[('full','MSE+BCE')]:>14.4f}")

    # --- the two labeled reads ------------------------------------------------
    r_frac = results[frac_name]["average_regret"]
    bo_reg = g[("budget-only", "regret")]
    bo_sur = g[("budget-only", "MSE+BCE")]
    full_reg = g[("full", "regret")]
    full_sur = g[("full", "MSE+BCE")]

    log("\n" + "=" * 78)
    log("READ 1 — SAME MODEL, TWO OBJECTIVES: the surrogate's cost on the 1-PARAM head")
    log("=" * 78)
    log(f"  budget-only(MSE+BCE) regret = {bo_sur:.4f}   vs   budget-only(regret) = {bo_reg:.4f}"
        f"   (Fraction f* = {r_frac:.4f})")
    surrogate_cost = bo_sur - bo_reg
    log(f"  surrogate cost on the 1-parameter model = {surrogate_cost:+.4f}")
    if bo_sur > r_frac + 1e-3:
        log("  => MSE+BCE + fixed adv<=0 DEGRADES even the 1-PARAMETER budget-only head below f*.")
        log("     Capacity / representation / root-state are RULED OUT: the failure is the OBJECTIVE.")
    elif surrogate_cost > 1e-3:
        log("  => the surrogate costs regret even on the 1-param head, but it still beats f*.")
    else:
        log("  => the surrogate does NOT degrade the 1-param head here (objective not implicated at this scale).")

    log("\n" + "=" * 78)
    log("READ 2 — SAME OBJECTIVE, TWO MODELS: value of the root state (full vs budget-only)")
    log("=" * 78)
    log(f"  regret-direct column: budget-only={bo_reg:.4f}  full={full_reg:.4f}  "
        f"=> root-state value = {bo_reg - full_reg:+.4f}")
    log(f"  MSE+BCE     column:  budget-only={bo_sur:.4f}  full={full_sur:.4f}  "
        f"=> root-state value = {bo_sur - full_sur:+.4f}")
    headline = bo_reg - full_reg  # the canonical headline = regret-direct column
    log(f"  HEADLINE (regret-direct): value of root state = {headline:+.4f}")

    # nestedness check (regret-direct column only — the structurally fair one).
    # The MSE+BCE column is NOT regret-selected, so it carries no nestedness
    # guarantee by design (that is the whole point of READ 1).
    r_full = full_reg
    r_budget = bo_reg
    # nestedness check. TWO regimes:
    #  (i)  TRAIN nestedness MUST hold by construction (each class strictly
    #       contains the simpler one and all are fit on TRAIN regret). A TRAIN
    #       violation is a real BUG and is flagged loudly.
    #  (ii) VAL nestedness usually holds too, but CAN break by generalization
    #       (a finer-grid fit overfitting train). A VAL-only break is reported as
    #       a generalization gap, not a bug.
    full_train = float(full_prov["train_regret_at_selection"])
    log("\n--- FAIRNESS / NESTEDNESS CHECK ---")
    log("  all three fit on the SAME TRAIN regret; full ⊇ budget-only ⊇ fraction.")
    log("  TRAIN regret  (must be non-increasing: fraction >= budget-only >= full):")
    log(f"    fraction={ftrain:.4f}  >=  budget-only={btrain:.4f}  >=  full={full_train:.4f}")
    log("  VAL regret    (the judged metric; nestedness expected, not guaranteed):")
    log(f"    fraction={r_frac:.4f}      budget-only={r_budget:.4f}      full={r_full:.4f}")
    tol = 1e-9
    train_ok = (btrain <= ftrain + tol) and (full_train <= btrain + tol)
    if not train_ok:
        log("  !!!! TRAIN NESTEDNESS VIOLATED — THERE IS A BUG (flagging loudly) !!!!")
        if btrain > ftrain + tol:
            log(f"  !!!! budget-only train ({btrain:.4f}) > fraction train ({ftrain:.4f}); "
                "budget-only sweeps a FINER superset grid and can never lose on TRAIN.")
        if full_train > btrain + tol:
            log(f"  !!!! full train ({full_train:.4f}) > budget-only train ({btrain:.4f}); "
                "the full head CONTAINS the budget-only rule (floor fallback) and can never lose on TRAIN.")
    else:
        log("  OK (TRAIN): nestedness holds exactly — the comparison is structurally fair.")
    val_tol = 1e-3
    ok_floor = r_full <= r_budget + val_tol
    ok_frac = r_budget <= r_frac + val_tol
    if ok_floor and ok_frac:
        log("  OK (VAL): nestedness also holds on the held-out split.")
    elif not val_tol:  # never; placeholder for readability
        pass
    else:
        log("  NOTE (VAL): val-side nestedness gap — generalization, not a bug (TRAIN is exact):")
        if not ok_floor:
            log(f"    full ({r_full:.4f}) > budget-only ({r_budget:.4f}) on VAL by {r_full - r_budget:+.4f}.")
        if not ok_frac:
            log(f"    budget-only ({r_budget:.4f}) > fraction ({r_frac:.4f}) on VAL by {r_budget - r_frac:+.4f} "
                "(budget-only overfit its finer train grid).")
    nest_ok = train_ok  # the bug-detector verdict is TRAIN nestedness

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [
        "Always-Stop", "Never-Stop", frac_name,
        "budget-only\n(regret)", "full\n(regret)",
        "budget-only\n(MSE+BCE)", "full\n(MSE+BCE)",
    ]
    keys = [
        "Always-Stop", "Never-Stop", frac_name,
        "budget-only (regret)", "full (regret)",
        "budget-only (MSE+BCE)", "full (MSE+BCE)",
    ]
    vals = [results[k]["average_regret"] for k in keys]
    colors = ["#9aa6c0", "#9aa6c0", "#4063A3",
              "#3a7d44", "#2e6b39", "#B5475B", "#8c2f42"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.axhline(r_frac, ls="--", lw=0.8, color="#4063A3", alpha=0.7)
    ax.set_ylabel("mean regret on VAL (lower = better)")
    ax.set_title(f"Minimal MC 2x2 — root-state value (regret col) = {headline:+.4f};  "
                 f"surrogate cost on 1-param = {surrogate_cost:+.4f}\n"
                 f"(trivial features, {n_trees_used} trees)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / run["figure_file"]
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\n  saved figure -> {fig_path}")

    # --- machine-readable results (provenance) --------------------------------
    out = {
        "config_path": str(Path(cfg["run"].get("_config_path", "minimal_mc.yaml"))),
        "seed": seed,
        "n_trees_used": n_trees_used,
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "fraction_f_star": best_f,
        "fraction_train_regret": ftrain,
        "fraction_val_regret": r_frac,
        "budget_only_threshold": best_t,
        "budget_only_train_regret": btrain,
        "full_train_regret": full_train,
        "regret_direct_provenance": full_prov,
        "surrogate_budget_only_provenance": bo_prov,
        "surrogate_full_provenance": full_s_prov,
        "results": results,
        "two_by_two_regret": {
            "budget-only": {"regret-direct": bo_reg, "MSE+BCE": bo_sur},
            "full": {"regret-direct": full_reg, "MSE+BCE": full_sur},
        },
        "read1_surrogate_cost_on_1param": surrogate_cost,
        "read1_budget_only_MSEBCE_beats_fstar": bool(bo_sur <= r_frac + 1e-3),
        "read2_root_state_value_regret_col": bo_reg - full_reg,
        "read2_root_state_value_msebce_col": bo_sur - full_sur,
        "headline_value_of_root_state": headline,
        "train_nestedness_ok": bool(train_ok),
        "val_nestedness_ok": bool(ok_floor and ok_frac),
    }
    res_path = out_dir / run["table_file"]
    res_path.write_text(json.dumps(out, indent=2))
    log(f"  saved results json -> {res_path}")
    log(f"  saved run log      -> {out_dir / run['log_file']}")
    log("\nDONE.")


if __name__ == "__main__":
    main()
