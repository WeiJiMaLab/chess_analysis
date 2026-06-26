# mc_minimal_plan.md — Minimal MC pipeline & the tree-value question

**Purpose.** The single live plan for the meta-controller (MC) thread. Consolidates the four former
proposal reports (R-MC-READOUT / R-MC-COST / R-MC-SIGNAL / R-MC-DELIB) into one prioritized program.
The current code/data lineage (definitions, filters, cost/loss math) lives in [mc_pipeline.md](mc_pipeline.md); this file is
the **forward plan** for the MC thread (mc_pipeline.md documents the current code/data lineage;
the labnotebook holds history and resolved diagnoses).

---

## Results log (live)

**Agent status (2026-06-26)**

| Track | What | State | Output |
|---|---|---|---|
| P1 | cost-regime oracle relabel | ✅ merged `0ca7e7c` | `costsweep_relabel.py` + 11 tests; 7 regimes in `packed/mc_costsweep/` |
| P0 | Readout harness + D0 tiers 1–4 | ✅ merged `0ca7e7c` | `readout.py`, 5-model `barh` eval, 25 tests; tiers 1–4 real, GNN-z pending |
| VG | value-gain MC (train-once, sweep cost) | ⏳ running | redirected: trajectory + real-DP-at-eval (greedy threshold is misaligned) |
| P2 | Stockfish ladder + tiny GNN | ✅ merged `eacfa1a` | ladder 1350/1800/full; valid child-WDL; tiny GNN trains in 14s; crashes/OOM root-caused |

**P0 — D0 tiers 1–4 (smoke; regret, lower=better; tier 5 GNN-z = Wave 3)**

| tier | production `packed/mc` (2000 ep) | transition `lambda_1.0` (1200 ep) |
|---|---|---|
| 1 Always-stop | 0.481 | 0.552 |
| 2 Never-stop | 1.394 | 0.085 |
| 3 Fraction (budget-only) | **0.074** | 0.048 |
| 4 Tree-stats (height,width,n_nodes) | 0.108 | **0.035** |

*Headline (tier4 − tier3):* production **+0.034** (tree-stats *loses* — stopping is budget-determined
at production cost, confirms the objective diagnosis); `lambda_1.0` **−0.013** (tree-stats *beats*
budget-only). **First evidence the tree carries stopping value — but only once the convex cost is
relaxed, and already without the GNN.** Method: tier-4 threshold tuned on TRAIN regret directly (the
regret-direct fix); height/width
derived in-loader from the `depth` array (no pack change).

**P1 — cost sweep (smoke, 4000 ep/split; regret, lower=better; f\* fit on train, eval on val)**

| regime | thinking cost | stop-instantly | search-forever | fixed-fraction |
|---|---|---|---|---|
| linear_0.0 | ~free | 0.587 | **0.007** | 0.007 |
| lambda_0.1 | very cheap | 0.586 | **0.014** | 0.014 |
| lambda_1.0 | cheap *(transition)* | 0.581 | 0.080 | 0.038 |
| linear_0.003 | cheap *(transition)* | 0.570 | 0.080 | 0.049 |
| lambda_5.0 | moderate | 0.564 | 0.376 | 0.051 |
| **lambda_18.537** | **production** | 0.514 | 1.390 | 0.065 |
| linear_0.02 | expensive | 0.538 | 0.558 | 0.145 |

*Reads:* production row **reproduces the documented production baseline** (validates pipeline). Stop-instantly is bad in *every* row
(0.51–0.59) → search genuinely improves the move, **trees are not useless**. Cheap rows are won by
search-forever (trivial). **Open (=P0):** does the tree itself beat fixed-fraction? Test in the
transition rows. *Caveat:* P1 agent also wrote the relabeler into the shared checkout `src` —
reconcile on merge.

## Where we are — settled vs open

**Settled (objective diagnosis, 2026-06-26; archived in labnotebook).** The controller (regret 0.21–0.29) lost to a
1-parameter Fraction baseline (0.077) **because of the objective/decision-rule, not the harness,
encoder, `z_t`, materialization, or feature-scaling**:
- Decoy heads pushed through the controller's own greedy eval reproduce the standalone baselines to
  4 dp → the harness is fair and the MC class *contains* the 0.077 solution.
- Even a **one-parameter budget-only head** trained on the cluster recipe (MSE + 0.1·sign-BCE +
  fixed `adv≤0` threshold) blows up to regret 0.303 → the objective is the whole bug.
- **Fix:** train *and* select on **regret directly** (or tune the decision threshold on regret),
  with every model behind one eval entrypoint over one physical split.

**Open (the real question).** On *this* budgeted oracle, optimal stopping is ~budget-determined:
a 1-param budget-only head ≈ the full head when both are fit on regret. **The tree's value for
*when to stop*, properly measured, is ≈0 here.** Is that because of (a) the cost regime, (b) the
training signal, or (c) genuinely low deliberation value of strong-teacher trees? The experiments
below discriminate.

**New capability (2026-06-26 profiling).** Stockfish is **~1,155× faster than Lc0** on CPU MCTS
rollouts (10-expansion tree: 0.18 s vs 209 s); Elo-limited rollouts (UCI_Elo=1800, nodes=100) add
no overhead. `StockfishDirectEvalProvider` ([lmcos/src/core/providers/stockfish.py](lmcos/src/core/providers/stockfish.py))
is implemented. It sets `UCI_ShowWDL` and returns a WDL per node, so **child-WDL pretrain targets ARE
producible** — the `edge_wdl_targets` are the search-consolidated mean leaf WDL backed up over PUCT
visits ([teacher_targets.py:126](lmcos/src/data/preprocess_gnn/teacher_targets.py#L126)), and that
backup loop is provider-agnostic. The genuine differences vs lc0: WDL comes from Stockfish's internal
eval→WDL model (not a trained value head), and **priors are uniform** (no policy head) so PUCT
explores differently — a *weaker prior*, which is exactly what P3 (deliberation value) wants.

---

## Headline deliverable (D0) — the 5-model readout comparison

**The crucial output.** One figure (replacing `regret_by_model.png` + `stop_eq_oss_by_model.png`,
both **re-oriented horizontally** via `barh`) comparing five readout models on the same axis, all
trained+selected on **regret directly** and evaluated through **one entrypoint over one physical
split**:

| # | Readout | Input scope | Isolates |
|---|---------|-------------|----------|
| 1 | **Always-Stop** | none | floor (stop@0) |
| 2 | **Never-Stop** | none | floor (full budget) |
| 3 | **Fraction-Stop** | budget `B` only (1 scalar θ) | the budget-only baseline that beat the old MC |
| 4 | **Readout(tree-stats)** | hand-crafted root vector: **height, width, n_nodes** (+ `B`) | *does any tree info beat budget-only?* |
| 5 | **Readout(GNN-z)** | learned GNN root embedding `z` (+ `B`) | *does the learned rep add over hand-crafted stats?* |

The 4-vs-3 gap = "tree summary statistics carry stopping value"; the 5-vs-4 gap = "the learned GNN
representation adds beyond raw stats". Two metrics per model: **regret** (lower better) and
**P(stop==OSS)**. Figures: `figures/regret_by_model.png`, `figures/oss_by_model.png` (renamed from
`stop_eq_oss_by_model.png`), horizontal bars, models ordered top→bottom 1→5.

> **Build site:** the 5 readouts are the P0 `Readout` family + the complexity ladder; tier 5 consumes
> the P2 tiny-GNN encoder, so the *final* figure lands after P2. Tiers 1–4 land at end of P0 (GNN slot
> shown "pending", as the current figure already does).

## Procedures (checklists)

Each track has explicit steps, a command/artifact per step, and an acceptance check. `[x]` done ·
`[~]` in progress · `[ ]` not started.

### P0 — Readout harness, regret-direct (R-MC-READOUT + R-MC-SIGNAL) · ✅ merged `0ca7e7c`
- [x] `Readout(nn.Module)` base, single decision rule (continue iff `A>0`) → `lmcos/src/models/readout.py`
- [x] Subclasses `AlwaysStop`/`NeverStop`/`FractionStop`/`StatsReadout`/`GnnMetaController`; tiers 4&5 share one head (isolate representation, not capacity)
- [x] Train+select on **regret directly** (threshold tuned on TRAIN regret) — the regret-direct fix
- [x] `--packed-root`-parameterized 5-model `barh` eval → `alt_models_eval.py`; figs `regret_by_model.png` + `oss_by_model.png`
- [x] Derive `height/width/n_nodes` in-loader from `depth` array (no pack change)
- [x] Tests (25 pass: `test_readout.py` + `test_costsweep_relabel.py`)
- **Accept:** nested model never beats oracle; tiers 1–4 real on both `packed/mc` and `mc_costsweep/lambda_1.0`. ✅ (see results log)
- [ ] Tier 5 (GNN-z) — deferred to Wave 3 (needs an encoder)

### P1 — Cost-regime relabel (R-MC-COST) · ✅ merged `0ca7e7c`
- [x] Relabel DP oracle over cached traces, no tree-gen → `costsweep_relabel.py`
- [x] Linear/flat cost mode (`c̄` per step) + λ-sweep `{0.1,1.0,5.0,18.537}` → 7 regimes in `packed/mc_costsweep/`
- [x] Built-in Always/Never/Fraction regret check per regime (11 tests)
- **Accept:** production λ reproduces the documented production baseline. ✅ (results log)
- [ ] Reconcile shared-`src` write of relabeler on full merge (currently committed in main — done via file-copy)

### VG — Value-gain MC, train-once / sweep-many · [~] running (agent `ac626…`)
Replace the cost-baked advantage target with a **cost-free value-gain** target; bolt cost on at decision time.
- **Greedy vs DP — checked, DP needed.** `halt_reward(t) = oracle_final_root_q_values[bmi[t]]`
  = **depth-96 Q of the committed move** (pack.py:660), NOT a shallow estimate — so once the move locks
  in, gain is identically 0 (verified: 100% of trajectories flat after lock-in; converged move == max).
  But **38.1% of `packed/mc` curves still reverse, and 100% of those reversals are *pre-convergence***:
  before lock-in the best-so-far move hops among candidates whose final Q is non-monotone (e.g. committed
  move A→B→C with Q_96 .50→.45→.60). At the A→B step marginal value is negative → greedy stops at A while
  DP pushes to C. So greedy ≠ DP on the ~⅓ unconverged episodes → **trajectory prediction + real-DP-at-eval**.
- [~] Define cost-free target `value_gain(t) = continue_value_zerocost(t) − halt_reward(t) ≥ 0` (reuse zero-cost DP)
- [~] Verify `value_gain` is non-negative and **non-flat** (the point — must not collapse to ≈0 like cost-baked advantage)
- [~] Train value-gain head on tree-stats features (no GNN needed); add z-head if materialized cache loads cheaply
- [~] Eval: `advantage(t;λ)=pred_gain(t)−analytic_cost(t;λ,T)` reusing oracle's exact cost fn; greedy stop
- [~] **Sweep all 7 regimes with ONE trained model**; add `value-gain-MC` column to the cost-sweep table
- **Accept:** swept single model ≈ per-λ behavior (→search-forever as λ→0; production-ballpark at λ=18.5); report VG-vs-Fraction gap per regime
- [ ] Tests: target non-negativity, reduces to greedy/GSS at zero cost, eval matches relabeled-oracle stop at matching λ

### P2 — Stockfish strength ladder + tiny-GNN POC (engineering) · ✅ merged `eacfa1a`
Go/no-go infra milestone — **PASS.**
- [x] End-to-end Stockfish tree gen via `StockfishDirectEvalProvider`; `edge_wdl_targets` finite/sum-1/non-neg on every rung
- [x] **Crash root-caused (not swallowed):** SF14 SIGSEGVs on board-only (<4-field) FENs → added 4-field-FEN guard before engine; canonical `fens.txt`. Zero crashes after.
- [x] **OOM root-caused:** `_load_fens` slurped the 5.4 GB/110 M-line `fens.txt`; fix = pre-sampled subset → 334 MB peak
- [x] Strength ladder Elo **`{1350,1800,full}`** (1100 infeasible — SF14 `UCI_Elo` floor 1350); smoke 150/rung in 8m46s → `sf_trees/<rung>/`
- [x] Tiny-GNN (`d_embed=32`) child-WDL pretrain smoke: loss descends in **14 s**
- [ ] **Full-scale ladder gen** — documented sbatch (`generate_sf_ladder_array.slurm`), needs pre-sampled `sf_ladder_roots.txt`; NOT launched (awaiting go)
- **Frugality contract honored:** one sbatch at a time, footprint fixed not brute-forced, no swallowed crashes

### P3 — Deliberation-value axis (R-MC-DELIB) · [ ] blocked on P2
Pure measurement over P2's ladder; no new generation.
- [ ] **A. PUCT-unstable subset:** filter a set to step-1-move ≠ final-move; budget-only vs full regret
- [ ] **B. Strength-ladder regret-gap:** plot `Regret(budget-only) − Regret(full)` vs teacher strength; expect gap to open as strength drops
- **Accept:** measurable full-vs-budget lift on unstable subset and/or weaker rungs

---

## Execution order & parallelism

Two things gate everything: **P0 is the measurement harness** (every regret/OSS number flows through
it) and **P2-generation is background CPU** (independent of P0/P1).

- [x] **Wave 1 (parallel):** P0 (harness + D0 tiers 1–4) ✅ · P1 (relabel) ✅ · P2 (Stockfish ladder + tiny GNN smoke) ✅
- [~] **Wave 2:** VG (value-gain trajectory + DP-at-eval) running · P2 full-scale ladder gen pending go-ahead
- [ ] **Wave 3 (needs P0 + encoder):** D0 final (fill tier 5 GNN-z → headline figure) · P3 (deliberation-value over ladder)

**Critical path to first result:** P0 → P1 → VG (done/running). **Longest pole:** P2-gen → P2-gnn →
D0/P3. **Hard serialization:** nothing reports regret/OSS except through P0's one harness/one split —
else you reintroduce the harness/objective mismatch.

## Decision gates
- After **P0**: if the regret-direct full head still ≈ budget-only floor on lc0 trees → confirms the
  open finding; proceed to P1 (it's the environment) and P3 (it's the trees).
- After **P1**: linear/low-λ lift ⇒ **cost regime** was the ceiling (report + pick a working λ).
  No lift ⇒ environment exonerated; weight shifts to **P3**.
- After **P3-B**: weak-teacher lift ⇒ deliberation value is real but teacher-strength-gated ⇒
  the MC should be deployed selectively / trained on a tactically-unstable curriculum.

## File map
| what | path |
|---|---|
| current code/data lineage — definitions, filters, cost/loss math | [mc_pipeline.md](mc_pipeline.md) |
| Stockfish provider (Elo-limited) | [lmcos/src/core/providers/stockfish.py](lmcos/src/core/providers/stockfish.py) |
| DP oracle (cost model, advantage target) | [lmcos/src/data/preprocess_mc/oracle.py](lmcos/src/data/preprocess_mc/oracle.py) |
| `Readout` family (built) | `lmcos/src/models/readout.py` (legacy `lmcos/src/models/mc.py`) |
| cost-sweep relabeler | `lmcos/src/data/preprocess_mc/costsweep_relabel.py`; regimes in `packed/mc_costsweep/` |
| eval + D0 figure (extend to 5 models; flip to `barh`; rename `stop_eq_oss`→`oss`) | [lmcos/analysis/_budgeted/alt_models_eval.py](lmcos/analysis/_budgeted/alt_models_eval.py) (`_plot`, `main`) |
| D0 figures (output) | `figures/regret_by_model.png`, `figures/oss_by_model.png` |
| profiling scripts | `scratch/benchmark_cts_rollout.py`, `scratch/profile_elo_matched_stockfish.py` |

*Folds the former R-MC-READOUT / R-MC-COST / R-MC-SIGNAL / R-MC-DELIB proposal reports.*
