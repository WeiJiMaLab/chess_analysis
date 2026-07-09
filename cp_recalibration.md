# CP — value-scale recalibration probe (no retrain)

Record of the `CP` node in `plan.md`'s Next-phase DAG, gating `CP-RETRAIN`. Targets T3's
"Related discovery": 56% of all nodes / 89% of leaf nodes in the tree corpus have
`|value| >= 0.99` — the shipped `value` feature (Stockfish WDL win_prob − loss_prob)
saturates like a step function around ~300cp, while the underlying `cp_order` column is
continuous (median 268). This probe asks: if node values are relabeled with a gentler
saturation curve and PUCT's backup is replayed over the *existing* tree topology (no new
search, no retraining), does the regret/headroom story improve enough to justify actually
retraining the child-WDL encoder (`CP-RETRAIN`)?

**Status: code, sweep, and plot were already produced by a prior session that was
interrupted (killed externally) before it could write up the interpretation.** This
document verifies those artifacts against source and adds the interpretation + decision.
No code was changed and no re-run was needed — the sweep, the plot, and both test files
(`test_cp_recalibration.py`, `test_relabel_replay.py`, 12/12 tests) all check out.

## Methodology

**Relabeling.** `tanh_cp_value(cp_order, T) = tanh(cp_order / T)` (`src/analysis/relabel_replay.py`),
a bounded `[-1, 1]` relabeling of each node's own static value with a temperature `T`
controlling how sharply it saturates — small `T` saturates fast (close to the original WDL
step-function behavior), large `T` keeps `cp_order` in tanh's near-linear regime around 0
for typical cp magnitudes, which desaturates but also compresses everything toward 0.
Swept `T ∈ {100, 300, 600}` against the original WDL `value` column as baseline.

**Replay, not regeneration.** `replay_puct_backup` (`src/analysis/relabel_replay.py`)
re-executes PUCT's real backup rule — walk from each node to root on the causal event
order reconstructed from the saved CSR tree arrays (`is_expanded`/`is_terminal`/`child_ptr`),
accumulating sign-flipped value into `total_value`/`visit_count` per edge — using the
*new* relabeled value at each node while holding tree **topology fixed** (same nodes, same
edges, same expansion order). `build_trajectory` turns the replay into the
`halt_rewards`/`tree_sizes`/`heights`/`widths` trajectory shape `analysis.evaluate`'s
regret machinery expects, mirroring production's `_build_compact_trajectory` exactly
(trims to the root's own expansion, `halt_rewards[t] = final_root_q[best_move_index[t]]`).

**Explicit approximation already documented in `relabel_replay.py`'s module docstring —
repeated here because it is directly load-bearing for interpreting any result below:**
real PUCT *selection* during original tree generation was made under the OLD saturating
WDL value. This replay only asks "what would the halt-reward trajectory have looked like
with different value **labels** painted onto the SAME generated shape" — it does **not**
answer "what would search look like if it explored differently under better-calibrated
values," because which nodes got expanded at all was already decided under the old value
function before this probe ever sees the tree. A null or mixed result here therefore does
not rule out the saturation fix helping if search were actually rerun under corrected
values; a positive result here is still real evidence (motivates `CP-RETRAIN`), but the
ceiling of what this specific probe can show is capped below "full re-generation."

**Population and regime.** 16,000 raw trees loaded (seed 0), reduced to **1,123** after
applying production's exact "thinking helps" filter (`filter_argmax.py`'s criterion:
baseline-WDL oracle argmax step `> min_argmax=2`) — same filter S1/T2/T3 use, so these
regret numbers are directly comparable to the rest of the investigation. 70/30
fit/eval split (fixed `rng(seed=0)` boolean mask, shared across every value-labeling
variant so all four conditions are scored on the *same* held-out episodes): **768 fit /
355 eval**, identical across baseline and all three `T` values (confirmed — see table).
Cost regime: `BudgetedOracleConfig(time_mode="linear", time_lambda=0.01,
maintenance_scale=0.0, maintenance_exponent=1.0)` — the single validated,
cost-regime-independent point used throughout T2/T3/S1 (the `REGIME` multi-point sweep
hadn't landed yet when this ran; re-running CP across the full confirmed regime set once
`REGIME` exists is a natural follow-up, not required for this decision).

**Metrics**, all bootstrapped 95% CIs (`cts.stats.bootstrap_ci`, percentile, `n_boot=2000`
— never normal-theory, per repo convention):
- `saturation_all` / `saturation_leaf`: fraction of nodes / leaf-nodes with `|value| >= 0.99`, over the **full raw 16,000-tree sample** (T3-comparable population, not the argmax-filtered one).
- `noise_proxy`: mean `|Δhalt_reward|` between consecutive steps within a trajectory (T3's reward-noise proxy), averaged per-tree with a bootstrap CI over trees.
- `k_singlehalt`: fixed stopping step fit on the 768 fit-episodes (`fit_singlehalt_stop`, pure argmin over `_regret_at`).
- `regret_singlehalt`: `SingleHalt*`'s regret vs. the true per-episode oracle, on the 355 held-out eval episodes.
- `regret_always_stop`: the `k=0` "AlwaysStop" floor's regret — the denominator for headroom.
- `fraction_recovered = 1 − mean(regret_singlehalt) / mean(regret_always_stop)`: the scale-invariant headroom metric (same framing as labnotebook's "Retraining-free headroom sweep"). This is a **ratio of two same-labeling quantities**, so a uniform rescaling of the reward signal cancels out of it to first order — this is exactly why it, not raw regret, is the right cross-temperature comparison (see Part 2 below).
- `paired_fraction_recovered_delta`: paired-bootstrap significance test of `fraction_recovered(T) − fraction_recovered(baseline)` on the **same 355 held-out episodes** (shared resampling keeps the pairing intact) — strictly stronger than eyeballing whether two independent CIs overlap.

## Part 1 — Verified numbers

Pulled directly from `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/cp/cp_recalibration_sweep.json` (confirmed present, `n_trees_raw_loaded=16000`, `n_trees_filtered=1123`, `min_argmax=2`, `seed=0`, `temperatures=[100,300,600]`) and `outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/cp_recalibration.*` (confirmed non-empty: 27,846 / 362,166 bytes). 12/12 tests pass in `test_cp_recalibration.py` + `test_relabel_replay.py` — no bug found, nothing re-run.

| condition | n_fit / n_eval | saturation_all | saturation_leaf | noise_proxy [95% CI] | k* | regret_singlehalt [95% CI] | regret_always_stop | fraction_recovered [95% CI] |
|---|---|---|---|---|---|---|---|---|
| WDL baseline | 768 / 355 | 0.5633 | 0.5681 | 0.004190 [0.003927, 0.004470] | 8 | 0.10872 [0.09291, 0.12718] | 0.17755 | 0.3877 [0.2918, 0.4758] |
| T=100 | 768 / 355 | 0.6281 | 0.6334 | 0.004730 [0.004428, 0.005018] | 10 | 0.12689 [0.10890, 0.14733] | 0.24444 | 0.4809 [0.3902, 0.5613] |
| T=300 | 768 / 355 | 0.2557 | 0.2588 | 0.003606 [0.003392, 0.003829] | 8 | 0.09031 [0.07767, 0.10518] | 0.15330 | 0.4109 [0.3121, 0.4987] |
| T=600 | 768 / 355 | 0.0247 | 0.0250 | 0.002571 [0.002402, 0.002746] | 5 | 0.07447 [0.06333, 0.08634] | 0.09401 | 0.2079 [0.0607, 0.3357] |

Paired-bootstrap significance of `fraction_recovered(T) − fraction_recovered(baseline)`, same 355 episodes:

| T | Δ fraction_recovered [95% CI] | significant (CI excludes 0)? |
|---|---|---|
| 100 | +0.0932 [+0.0034, +0.1832] | **yes** |
| 300 | +0.0232 [−0.0234, +0.0757] | no |
| 600 | **−0.1798** [−0.2927, −0.0751] | **yes** |

The n_fit/n_eval split is identical (768/355) across baseline and all three T — confirms the shared `fit_mask` is working as intended (same held-out population scored under every labeling, exactly as `run_sweep`'s docstring claims). The headline summary numbers in the task prompt check out against the source JSON to full precision (only reporting was rounded there; table above carries the un-rounded values).

## Part 2 — Interpreting the pattern

The raw pattern is genuinely counter-intuitive: **desaturation is monotonic in T** (saturation_all: T100 0.628 > baseline 0.563 > T300 0.256 > T600 0.025 — T=100 is actually *more* saturated than the original WDL curve, T=600 essentially eliminates it), but **headroom is not monotonic the same way** — `fraction_recovered` peaks at T=100 (0.481, significantly above baseline) and is *worst* at T=600 (0.208, significantly below baseline), with T=300 statistically indistinguishable from baseline.

**Hypothesis tested (from the task): is this just a units/scale artifact?** As T grows, `tanh(cp_order/T)` for typical `cp_order` (median 268) increasingly sits in tanh's near-linear small-`|x|` regime, which compresses *all* values toward 0, not just the previously-saturated extremes. A uniform rescaling of the reward signal would mechanically shrink `regret_singlehalt` (smaller numbers throughout) without saying anything about real headroom.

This does **not** hold as the primary explanation, and it's checkable directly from the JSON:

1. **If it were pure uniform compression, `regret_singlehalt` and `regret_always_stop` would shrink by the same factor, and `fraction_recovered` (their ratio-derived quantity) would stay flat across T.** It doesn't: `fraction_recovered` moves significantly in both directions (+0.093 at T=100, −0.180 at T=600, both CIs excluding 0). Since `fraction_recovered` is constructed to be scale-invariant to first order (`1 − mean(regret_sh)/mean(regret_as)`, both computed under the same labeling), a significant *change* in it is direct evidence of a real headroom effect on top of whatever scale change is happening, not an artifact of scale alone.
2. **`regret_always_stop_mean` does not shrink proportionally with `regret_singlehalt_mean` — in fact at T=100 it doesn't shrink at all.** Relative to baseline: T=100 → regret_sh ×1.167, regret_as ×1.377 (both **grew**, not the "compress toward 0" story at all — this T actually widens the AlwaysStop floor's oracle gap more than SingleHalt*'s); T=300 → regret_sh ×0.831, regret_as ×0.863 (close, small movement); T=600 → regret_sh ×0.685, regret_as ×0.530 (regret_as shrinks noticeably *more* than regret_sh). The `regret_sh/regret_as` ratio (`= 1 − fraction_recovered`) is 0.612 (baseline) → 0.519 (T=100) → 0.589 (T=300) → 0.792 (T=600) — clearly not constant, which is exactly what "uniform compression" predicts it should be.
3. **Where compression is real (T=600), it's not enough on its own to explain the headroom loss.** T=600 does show genuine dynamic-range shrinkage (`noise_proxy` drops to 0.614× baseline, `regret_as` to 0.530× baseline) — some compression is really happening at high T, as expected from the near-linear-tanh argument. But `regret_sh` shrinks *less* (0.685×) than `regret_as` does, so SingleHalt* is capturing a larger relative share of the (now smaller) achievable value — that's a real shift in headroom, riding on top of, not explained away by, the scale change.

**So: the "just compression" explanation is insufficient — confirmed, not just asserted — there is a real headroom effect.** What is the mechanism? A pattern that *does* hold cleanly across the data: `saturation_all` and `noise_proxy` and `fraction_recovered` all rank in the **same order** across all four conditions (T100 > baseline > T300 > T600, for all three quantities simultaneously — noise_proxy: 0.00473 > 0.00419 > 0.00361 > 0.00257; fraction_recovered: 0.481 > (0.411 ≈) 0.388 > 0.208). The plausible reading: saturation to `|value| ≈ 1` doesn't just clip individual values, it makes the halt-reward trajectory *jumpier* — a node's backed-up value can snap between near-±1 plateaus as `cp_order` crosses the effective saturation threshold, producing large step-like `Δhalt_reward` swings within a single episode (that's exactly what `noise_proxy` measures). A single fixed stopping step `k` (`SingleHalt*`) cannot track that per-episode volatility — it's blind to which episodes are about to jump — so a jumpier reward landscape leaves *more* residual regret for an adaptive/input-conditioned controller to recover, i.e. more headroom. Desaturating (raising T) smooths the trajectory into something closer to monotonic-in-search-effort, which is precisely the shape a single fixed `k` is *good* at approximating — so `SingleHalt*` closes more of the gap to the oracle on its own, leaving less for anything smarter to gain. This reading is **consistent with, but not independently proven by**, the data checked here — it explains the co-movement of saturation/noise/headroom cleanly, but this probe wasn't designed to isolate "volatility" from "saturation" as separate causal variables (they're both driven by the same T knob here), so treat it as the best-supported hypothesis rather than a confirmed mechanism.

**Caveat this interpretation must carry (from `relabel_replay.py`'s docstring, restated per this repo's "no ad-hoc mechanism explanations without checking against data" convention):** every one of these four conditions is scored on the exact same tree *shapes*, generated under the original saturating WDL value. PUCT's real expansion decisions already happened under the old value function before any of this relabeling occurs — this probe only recolors the existing trajectory, it cannot show what a rerun search would explore differently under a desaturated value. So even T=100's significant headroom gain is evidence *within this constrained "same shape, new labels" test*, not evidence that regenerating trees under `tanh(cp_order/100)` would produce this same gain (selection itself would change, which could help or hurt in ways this replay cannot see).

## Part 3 — Recommendation on `CP-RETRAIN`

**None of the three temperatures makes a clean, unambiguous case for `CP-RETRAIN` on this evidence, and the strongest desaturation setting (T=600, the one most directly aimed at "fixing" T3's saturation finding) is the worst performer, not the best. Recommendation: do not launch `CP-RETRAIN` off this sweep as currently framed. Route to `Z-EXT` on the current (frozen-shape) encoder, per the DAG's `(no change)` branch.**

Reasoning:
- T3's finding was framed as "value saturation is bad, desaturate it." This probe shows that's too simple: the temperature that most fully removes saturation (T=600, 2.5% saturated vs. 56% baseline, and the largest noise reduction) has *significantly worse* headroom than baseline (−0.180, CI excludes 0) — the opposite of the DAG's stated trigger condition ("promising" branch). Retraining the encoder to target T=600-style labels would very plausibly train away exactly the reward volatility that gives a learned controller room to add value over a fixed threshold.
- T=100 is the one condition with a significant, positive headroom gain (+0.093, CI excludes 0) — but it is not a desaturation fix at all; it's *more* saturated than the shipped WDL curve (0.628 vs 0.563). Retraining the child-WDL encoder to reproduce T=100-style labels would mean deliberately choosing a harsher, more saturating value curve than what's already shipped — a legitimate-looking lever in isolation, but not what T3's "fix the saturation" motivation was asking for, and not yet validated beyond this shape-frozen replay (see caveat above — this is the condition most exposed to the "same shape, new labels only" limitation, since it pushes furthest from what the tree was actually grown under).
- T=300 (the "middle" temperature, closest to a naive prior for "somewhat gentler than WDL") lands in between and is **not** statistically distinguishable from baseline (Δ = +0.023, CI includes 0) — a null result on the metric that matters.
- No condition in the grid is unambiguously good: the metric plan.md and this module both designed as the correct like-for-like comparator (`fraction_recovered`, scale-invariant, paired-tested) is non-monotonic and only one point clears significance in the "helpful" direction, while a different point clears significance in the "harmful" direction just as clearly. That's a mixed/negative pattern, not a promising one, under the DAG's own "(promising) vs (no change)" fork.
- The one genuinely promising thread here (T=100's significant gain) is confounded with the shape-frozen-replay caveat in the most acute way of all four conditions, since it moves furthest from the value function the tree topology was actually grown under — acting on it via a full encoder retrain (which *would* change downstream selection, unlike this replay) is a bigger, harder-to-reverse commitment than the evidence currently supports.

**If this lever is revisited later**, the more informative next step would not be "retrain on T=100 labels" directly, but either (a) re-run this same replay-only probe across the full `REGIME` regime set once that lands (this sweep predates `REGIME` and only covers the single `m=0` point), to check whether T=100's gain is regime-specific or general, or (b) a real (non-replay) small-scale re-generation at T=100 to test whether the caveat actually bites — i.e. whether letting PUCT selection itself respond to the new, more-saturating value changes the picture materially, before spending a full retrain on it. Neither is proposed as part of this decision; per the DAG, the recommended path from here is straight to `Z-EXT` on the current encoder.
