# eval.md — Part 3: alternative stop-policy models on the new oracle trees

**Status:** plan draft. Awaiting line-by-line answers on **[Q#]** before building.
Blocked by [[partition]] 1A (the train/test FEN split) and the MC packing step
below.

Goal: contextualize the (not-yet-trained) GNN+readout controller against simple
stop-policy baselines, evaluated identically. Deliverables: on the **test** set,
(1) a **Regret** plot per model, (2) a **fraction-of-episodes-where-stop == OSS**
plot per model (OSS = the DP-optimal stop step given budget+tree =
`oracle_stop_step`; replaces GSS per your [Q2] answer; still pluggable).

Related: [[partition]], [[gnn_train]], [[tree-set-and-gss]], [[diagnose_gain]].

---

## 1. The models (all as snapshot → {HALT, CONTINUE} policies)

Every model is the same object the MC readout head will be: given a tree snapshot
`T` and starting budget `B`, emit HALT or CONTINUE. The three baselines already
exist in [baselines.py](lmcos/analysis/_budgeted/baselines.py):

| model | rule | fit? | existing impl |
|-------|------|------|---------------|
| **Always Stop** | HALT at step 0 | none | `always_halt_0` |
| **Never Stop** | CONTINUE to the last step | none | `always_continue_to_end` |
| **Fraction of Budget** | HALT iff `expansions_so_far ≥ round(f·B)` | fit `f∈(0,1)` | `_fixed_fraction_stop_step` (currently a fixed ρ grid) |

Always/Never need no fitting (wired outputs). **Fraction of Budget** is the only
fitted model: a single scalar `f`, fit on **train**, frozen, evaluated on **test**.

### Fitting `f`
Two routes; I recommend running both for the parity practice you flagged:
- **(primary) 1-D objective minimization.** `f` lives in `(0,1)`; fit by a 1-D
  sweep / `scipy.optimize.minimize_scalar` over train episodes against the chosen
  objective (**[Q1]** below). Clean, exactly reproducible, matches the eval metric.
- **(parity) MC-style fit.** Frame it as the same per-snapshot stop-classification
  the controller trains on (HALT/CONTINUE target per snapshot, gradient steps,
  same early-stop / checkpoint hooks as `controller_train.py`). Slower, but it
  exercises the exact training+stopping machinery so we have parity for the GNN
  controller later. Treated as a stretch goal / parity check, not the headline.

---

## 2. Data path (MC packing — the new dependency)

The baselines consume **episodes** with `halt_rewards`, `tree_sizes`,
`time_budgets`, `oracle_value`, `oracle_stop_step`. These come from the budgeted
oracle ([oracle.py](lmcos/src/data/preprocess_mc/oracle.py)) at packing time —
**none exist for the new ysagiv trees yet** (confirmed: no packed episodes under
`/scratch/.../hl4291`). So:

1. **Split** the trees via [[partition]] 1A → `partition/train`, `partition/test`
   (DONE; manifests over `human_trees`).
2. **Pack** MC episodes from `human_trees` for each side separately
   ([materialize.py](lmcos/src/data/preprocess_mc/materialize.py) →
   [pack.py](lmcos/src/data/preprocess_mc/pack.py)) with
   `DEFAULT_BUDGET_BUCKETS` + the canonical `BudgetedOracleConfig`, writing to
   **`/scratch/gpfs/GRIFFITHS/hl4291/packed/mc/{train,test}/`**. Produces
   per-episode diagnostics consumable by `_evaluate_baseline`.
3. **Attach GSS** per episode (see §3) so the second metric is computable.
4. Fit `f` on train episodes; evaluate all three models on test episodes.

I will profile the packing time on a small slice first (smoke ≤ 2 min) before the
full run.

---

## 3. Metrics

- **Regret** (minimize): `oracle_value − return_for_stop_step(stop_step)`, exactly
  as `_evaluate_baseline` already computes it
  ([baselines.py:116](lmcos/analysis/_budgeted/baselines.py#L116)). Report mean ±
  CI over test episodes per model.
- **Proportion stop == OSS** (your [Q2] answer): fraction of test episodes where the
  model's chosen stop step **equals the episode's DP-optimal stop step** — the
  budget-aware `oracle_stop_step` / `optimal_stop_step` from the budgeted-oracle DP
  ([oracle.py:132-140](lmcos/src/data/preprocess_mc/oracle.py#L132-L140)). This is
  already in episode-step space (budget-aware), so the GSS↔step truncation issue of
  the old [Q2] disappears. `reference_stop_step` stays injectable so a
  filtered/GSS variant is a one-line swap.

**Fitting objective resolution (your [Q1]).** You asked to match the
meta-controller's loss (`advantage_mse + sign_loss_weight·sign_bce`, selected on
regret — [controller_train.py:232-234](lmcos/src/train/controller_train.py#L232-L234)).
A fraction-of-budget model has **no per-snapshot advantage prediction**, so
`advantage_mse` is undefined for it. The faithful threshold-model analogue: fit the
scalar `f` to minimize `mean_regret + sign_loss_weight·mean_sign_BCE`, where the
sign-BCE is computed on the **induced** per-snapshot halt/continue decisions
(continue for steps `< stop`, halt at `stop`) vs the oracle action signs. `f` is
1-D so this is a clean `minimize_scalar`. (If you'd rather fit on regret alone, that
is one term-weight away — flag it.)

### Open questions
- **[Q1] Objective for fitting `f`.** Regret is the headline "minimize" metric, so
  I propose **fit `f` to minimize mean train Regret**. Alternatives: maximize
  train fraction-stop==GSS, or the MC classification loss. Which is canonical?
  *Assumption:* minimize Regret.
  It should match the optimization loss function for the meta-controller; i.e. MC classification loss (which I believe is regret + BCE)

- **[Q2] GSS ↔ episode-step mapping.** `_tree_gss` is an **expansion count** on the
  full (96-expansion) tree; an episode is a snapshot sequence under a sampled
  starting budget `B` that may truncate before GSS. Do we (a) define episode-GSS =
  `min(GSS_tree, len(episode)−1)`, (b) drop episodes whose `B < GSS_tree`, or (c)
  compute GSS within the episode's own snapshot trace? *Assumption:* (a), clamp,
  and report the truncated fraction.
  - we should, rather than using GSS, use the DP optimal OSS given the budget and the tree. 

- **[Q3] `BudgetedOracleConfig` values.** Use the current canonical config
  (maintenance/time-cost constants, `DEFAULT_BUDGET_BUCKETS`, `samples_per_bucket`,
  seed) unchanged? *Assumption:* yes — frozen so baselines and the eventual GNN
  controller share one reward function.
  - yes.

- **[Q4] Plot form.** Two figures (`regret_by_model.png`,
  `stop_eq_gss_by_model.png`), bar-per-model with CIs. Include the ρ-grid Fraction
  variants as faint context bars behind the fitted `f`? *Assumption:* fitted `f`
  only, clean bars; grid in an appendix table.
  - grid in appendix

- **[Q5] Should the GNN/MC controller appear in these plots as a placeholder slot**
  now (greyed, "pending"), so adding it post-training is a drop-in? *Assumption:*
  yes, reserve the slot.
  - yes, reserve the slot.

---

## 4. Tests
- **`test_always_stop_is_step0` / `test_never_stop_is_last`** — wired rules return
  0 and `len−1` on synthetic episodes.
- **`test_fraction_stop_step`** — `round(f·B)` exactly; clamped into
  `[0, len−1]`; boundary `f→0`, `f→1`.
- **`test_regret_nonnegative_vs_oracle`** — every model's per-episode Regret ≥ 0
  (oracle is optimal by construction) within float tolerance.
- **`test_fit_f_recovers_planted_optimum`** — synthetic episodes whose regret is
  minimized at a known `f*`; the fitter must recover `f*`.
- **`test_gss_attach_matches_tree`** — episode-GSS equals `_tree_gss(source tree)`
  (clamped per [Q2]).
- **`test_train_test_disjoint`** — no FEN appears in both partition manifests
  (leakage guard; shared with [[partition]]).
- **`test_metric_reference_pluggable`** — swapping `reference_stop_step` from GSS
  to a stub OSS changes the metric without touching model code.
