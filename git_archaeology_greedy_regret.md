# Git archaeology: the `fittedq_subtree_weighted_zt_tt.pt` "greedy regret 0.024" figure

Read-only investigation. Task: determine what cost model / regret definition / population
produced the `greedy regret 0.024` figure recorded in `labnotebook.md` for
`fittedq_subtree_weighted_zt_tt.pt` (producer 2026-05-21), and whether it is comparable to
current (`src/analysis/evaluate.py`) regret numbers on a like-for-like basis.

## 1. Branch divergence

```
git merge-base main jordan  →  7644d53dd53b9bc571c761c68a93abeb0ebbe9cd  (2026-05-28 15:33:15 -0400, "new plots")
git rev-parse main          →  7644d53d... (same commit — jordan branched off main's current tip)
```

The divergence point is **2026-05-28**, six days after the checkpoint in question was produced
(2026-05-21/22). This turned out to be a red herring for this investigation: the relevant code
lived entirely on `main`'s history *before* the divergence, under `lmcos/` (an old package layout
later restructured into `src/cts/` — the restructure, `41316e7`, is itself on `main`, 2026-05-13,
predating the checkpoint). No `jordan`-side commit is involved at all.

## 2. The producing commit

```
git log main --oneline --since=2026-05-15 --until=2026-05-25
```

surfaces `4bca905c7e3581d8893b732b720a0598dfa696f1` — **"lmcos: subtree-size-weighted encoder +
controller_inputs config"**, authored by Yotam Sagiv, committed 2026-05-22 10:30:48 -0400. Its
commit message literally states the headline: *"Headline result (lab notebook 2026-05-21):
subtree-weighted encoder + [z_t, T_t] reaches greedy regret 0.024 in 20 epochs of vanilla slw01
training — a 7x improvement over the prior 0.176 best, attributable to the encoder."* This commit
adds `lmcos/configs/train/controller_subtree_weighted_zt_tt.yaml`, whose
`output_checkpoint: .../checkpoints/fittedq_subtree_weighted_zt_tt.pt` matches the checkpoint path
in `labnotebook.md:622` exactly. The full narrative (materialization + 4 training runs + reference
grid) is in `labnotebook.md:3754-3857` under `### 2026-05-21`.

## 3. Exact cost model / regret definition in effect

### Config (baked into the packed corpus and echoed in the training config)

`git show 4bca905:lmcos/configs/data/preprocess_mc/pack.yaml` (the packing config for this
lineage's data) and `git show 4bca905:lmcos/configs/train/controller_subtree_weighted_zt_tt.yaml`:

- `time_lambda: 18.537`
- `time_p: 2.8`, `time_tau: 2.5`, `time_delta: 1` (power-law cost shape parameters)
- `maintenance_scale: 0.0` (no maintenance cost)
- `packed_train_data` / `packed_validation_data` → `controller_packed_combined_nomaint_no_xaba/`

`labnotebook.md:610` dates that packed directory to **"producer: pre-2026-04-29"** — i.e. the
0.024 result was trained/evaluated against a corpus packed roughly 3-4 weeks earlier under the
**same** `time_lambda=18.537` used continuously across "the λ=18.5 / λ=5 sweeps and Section 3
rerun controllers" (same line). There was a separate, deliberate `time_lambda=5` re-pack
experiment (`labnotebook.md:2967-3042`, 2026-04-30) on a *different* split
(`pretrain_split_oracle96_trace_filtered_rerun`), but the checkpoint in question used the
canonical `time_lambda=18.537` combined-trees corpus, not the λ=5 one.

### Cost shape

At commit `4bca905`, `lmcos/src/cts/data/preprocess_mc/oracle.py`'s `BudgetedOracleConfig`
dataclass has **no `time_mode` field at all** — `time_cost()` unconditionally implements a single
power-law shape:

```python
# lmcos/src/cts/data/preprocess_mc/oracle.py (commit 4bca905)
def time_cost(remaining_budget, config):
    left  = (remaining_budget - config.time_delta + config.time_tau) ** (-(config.time_p - 1.0))
    right = (remaining_budget + config.time_tau) ** (-(config.time_p - 1.0))
    return float(config.time_lambda * (left - right))
```

i.e. what the *current* codebase's `time_mode="power_law"` option is — the `"linear"`/`"quadratic"`
modes were added later (first appears with `time_mode: Literal["power_law", "linear"]` in commit
`bf54289`, after `4bca905`). This shape is backloaded: cost is tiny while `remaining_budget` is
large and spikes only as the budget nears exhaustion (hand-computed: at `remaining_budget=60`,
`continue_cost ≈ 0.00024`; at `remaining_budget=2`, `continue_cost ≈ 0.71`, comparable to the
reward's own [-1,1] range).

### Regret construction — structurally identical to today's `_regret_at`

`lmcos/src/cts/train/controller_train.py::_aggregate_greedy_rollout_metrics` (commit `4bca905`,
lines ~1235-1320):

- Walks each validation episode's predicted per-step advantages, halts at the **first step with
  predicted advantage ≤ 0** (greedy stopping rule).
- `predicted_return = return_for_stop_step(halt_rewards, tree_sizes, time_budgets, predicted_stop, oracle_config)`
  — halt reward at the stop step minus accumulated `continue_cost` over the steps taken
  (`oracle.py::return_for_stop_step`, `continue_cost = maintenance_cost + time_cost`).
- `regret = meta.oracle_value - predicted_return` (per episode); `average_regret` = mean over all
  validation episodes — **this is "greedy regret."**

This is the *same* mathematical construction as `src/analysis/evaluate.py::_regret_at` today:
`regret = max(curve) - curve[stop]` where `curve[s] = halt_reward[s] - accumulated continue_cost`
(`evaluate.py:176-193`), both built on a `BudgetedOracleConfig` with a `maintenance_cost +
time_cost` continue-cost term. **The metric definition itself has not changed in kind** — only the
calibration (`time_lambda`, `time_mode`), the population, and the underlying corpus have.

### Population

`BudgetBucket`s spanning starting budgets 1-120 expansions (`scramble`/`medium-small`/
`medium-large`/`large`/`very-large`), `samples_per_bucket=2`, **no argmax-filter** — the "thinking
helps" (`argmax_stop_step > 2`) subsetting that the *current* pipeline applies
(`config_minply15_maxply75.yaml` `argmax_filter` stage, `plan.md:41`) did not exist yet; that
stage was added 2026-07-08 (`labnotebook.md:41`). So the 0.024 population also includes whatever
fraction of trivial-to-stop trees the current pipeline now explicitly filters out.

### Reward/value scale

`value_feature: "value"` was already in effect for this lineage
(`lmcos/src/cts/data/preprocess_mc/pack.py:130`, commit `4bca905`), and
`value_features_from_wdl()` (`lmcos/src/cts/core/providers/parsers.py`) defines
`value = p_win - p_loss`, i.e. **already win-probability-scaled, bounded to [-1, 1]** — the same
"currency" the July 2026 investigation later diagnosed as incompatible with a large linear
`time_lambda`. This predates and is independent of the 2026-07-06/07 BeFS/PUCT
`cp_order`-vs-`value` scale confusion (`labnotebook.md:72,81`), which concerned a *different*,
much later tree-generation pipeline (`befs1cp_md36`/`puctvalue_md36`, poisoned and deleted
2026-07-06, `labnotebook.md:70-83`). The May corpus's reward scale is not implicated in that
specific poisoning bug, but it does share the same [-1,1] value currency that the July
recalibration work determined is incompatible with `time_lambda` values anywhere near double
digits under a *linear* cost.

## 4. Does `time_lambda=18.537` on a [-1,1]-scaled reward collapse the oracle? Direct evidence, not inference

Independent of the July linear-cost bug (which is about `time_mode="linear"`, not the power-law
mode in effect for the May corpus), **this repo's own currently-committed documentation states
directly that the same numeric default (`λ≈18.5`) collapses the oracle under the current,
still-power-law-capable code path**:

> `labnotebook.md:30` (§ Running state, 2026-06-18): *"Metric (GSS): greedy stopping step = first
> expansion the eventual-best move is found (the budgeted oracle's zero-cost stop) — **not** the
> cost-aware DP `optimal_stop_step` (default λ=18.5 bails at step 1)."*

This is a direct, current statement — not a guess — that the shipped default `time_lambda≈18.5`
(`BudgetedOracleConfig.time_lambda: float = 18.537`, still the dataclass default today in
`src/cts/data/preprocess_mc/oracle.py:71`) drives the DP oracle's optimal stop to step 1 in the
regime this repo actually uses it in. My own hand-computation of the power-law `time_cost` above
(§3) is *consistent* with this being budget-dependent rather than universal — cost is negligible
for large remaining budgets and only becomes reward-comparable near budget exhaustion — but the
`labnotebook.md:30` citation is a direct, first-party confirmation (not my inference) that the
same λ default produces near-immediate bailout in this codebase's typical operating regime.

Additionally, `labnotebook.md:2972` (2026-04-30) shows the *original team itself* already
suspected `time_lambda=18.537` might be "set high enough that time pressure dominates the optimal
stopping decision" and ran a dedicated `time_lambda=5` re-pack + retrain to test it — i.e., the
canonical λ was already under live suspicion of miscalibration one lab-notebook-day before the
0.024 checkpoint (2026-05-21) was trained under the *unmodified*, still-suspect λ=18.537. That
sweep (`labnotebook.md:2967-3042`) found the T_t-only floor and kitchen-sink-vs-floor gap moved in
the *opposite* direction from the miscalibration hypothesis (floor dropped 0.218→0.150 at λ=5, gap
shrank 0.042→0.018) and concluded the effect was confounded with a dataset change, not cleanly
attributable to λ alone — the sweep was inconclusive, not exculpatory. No further calibration work
on this lineage's λ is recorded after 2026-04-30; the 2026-05-21 run reused the original,
never-cleanly-validated λ=18.537.

## 5. Comparison to the current (`src/analysis/evaluate.py`) regret regime

| | May 2026 lineage (`fittedq_subtree_weighted_zt_tt.pt`, regret 0.024) | Current (`plan.md`/`config_minply15_maxply75.yaml`, regret ~0.11-0.12) |
|---|---|---|
| Regret construction | `oracle_value − return_for_stop_step(...)`, greedy stop at first `advantage ≤ 0` (`controller_train.py::_aggregate_greedy_rollout_metrics`) | `max(curve) − curve[stop]`, `curve = halt_reward − accumulated continue_cost` (`evaluate.py::_regret_at`) — same construction |
| `time_lambda` | 18.537 | 0.01 (calibrated 2026-07-07; see `labnotebook.md:54,59,65`) — **~1854x smaller** |
| `time_mode` | power-law only (field didn't exist yet); `time_p=2.8`, `time_tau=2.5` | `"linear"` in the live calibrated regime (`config_minply15_maxply75.yaml`) |
| `maintenance_scale` | 0.0 | 0.0 in the base regime; 0.1 in the "sweet spot" headline frontier numbers (`labnotebook.md:63`) |
| Reward/value scale | win-probability `value = p_win − p_loss` ∈ [-1,1] (already, since before this commit) | win-probability, same currency, resolved as canonical 2026-07-06/07 (`labnotebook.md:72,81`) |
| Population | All packed episodes, budget-bucket-stratified 1-120, **no argmax filter** | argmax-filtered ("thinking helps", `argmax_stop_step > 2`) validation subset, added 2026-07-08 (`labnotebook.md:41`) |
| Tree-generation pipeline / corpus | `controller_packed_combined_nomaint_no_xaba` (pre-2026-04-29, lc0-provider PUCT teacher search per `teacher_targets.py`) | `puctvalue_md36` corpus (PUCT-primary, 2026-07-08), a wholly different, much later generation pipeline that itself required a full from-scratch BeFS/PUCT rewrite after the 2026-07-06 poisoning fix |

**Assessment:** the regret *metric construction* is the same family of object in both eras (an
oracle-vs-achieved-return gap under a `BudgetedOracleConfig` cost model) — this is not a "genuinely
different metric," it is the same metric under (a) a ~1854x different cost scale, (b) a different
cost shape (power-law vs the later-calibrated linear), (c) an unfiltered vs argmax-filtered
population, and (d) an entirely different, non-overlapping tree corpus generated by different code.
Any one of (a)-(d) alone would be sufficient to make the two numbers non-comparable; here all four
differ simultaneously. The `time_lambda=18.537` power-law regime specifically has **never been
independently validated as non-degenerate** by this codebase's own later standards: the one
calibration attempt on this exact lineage (`time_lambda=5` re-pack, 2026-04-30) was inconclusive
and confounded, and the current codebase's own running-state note (`labnotebook.md:30`) states that
this same λ default drives the DP oracle to bail at step 1 in the regime the repo now actually
runs it in. This does not by itself *prove* 0.024 is a degenerate near-zero-signal number — the
power-law shape means the effective time pressure is budget-dependent, not obviously collapse-inducing
for every episode the way the later `time_mode="linear"`+λ=10 bug obviously was (`labnotebook.md:53`,
`curve[0] = [-0.534, -10.534, ...]`, argmax@step0 for 100% of episodes) — but no evidence exists
in the historical record that `time_lambda=18.537`/power-law was ever checked for the analogous
failure mode, and the one first-party 2026-06-18 note about this exact default is a direct
statement that it does collapse the oracle's optimal stop to step 1.

## 6. Bottom line

0.024 is not safely comparable to the current ~0.11-0.12 figures. It is the same *kind* of metric
(oracle-return-gap under a `BudgetedOracleConfig`), but computed on a different corpus, a
different (much larger, power-law rather than linear) cost scale that this repo's own later
documentation (`labnotebook.md:30`) says drives the oracle to bail at step 1 under this default,
an unfiltered population, and a cost-calibration attempt on this exact lineage
(`time_lambda=5`, 2026-04-30) that was run, found suggestive but confounded, and never followed up
on before the 0.024 checkpoint was trained nine days later under the original, unvalidated
`time_lambda=18.537`. Re-measuring the underlying encoder/controller lineage
(`tree_encoder_child_wdl_async_k1_subtree_weighted.pt`) under the current, calibrated
`time_lambda=0.01`/linear/argmax-filtered regime (the separate probe referenced in `plan.md:38`)
is the only way to get a number that means the same thing as today's ~0.11-0.12 figures.
