# When ought one think? — calibrating the budgeted-oracle stop step

**Ref:** `R-HALT-CALIB` · [Index](reference.md)

> **Status:** 📝 active. Metric reframed to **step\* (optimal stop step) vs human RT** (the regret-vs-RT
> framing is retired — see *Why step\*, not regret*). Interim findings on filtered elo2000 (n≈13k moves);
> a definitive ~61k re-run and the SF-1350 weak rung are in progress. Sibling of
> [(R-TREESEARCH)](treesearch.md), which changes the *mechanism*; this report only *calibrates* it.

## What is the normative "when to think," and what should it be graded against?

The budgeted oracle's normative output is the **optimal stop step, step\* = argmax_s [V(s) − cost(s)]** —
the number of expansions it is worth doing before committing to a move. The human analog is **response
time**. So the only question this report asks is whether **step\* tracks human RT**, and how the cost
(shape, scale) and engine strength change that correspondence.

> **Decision:** the correspondence metric is **Spearman(step\*, human log-RT)** with bootstrap 95% CIs
> ([[bootstrap-cis-always]]). Not regret, not value — step\*.

## Why step\*, and not regret? (the correction)

We previously also correlated **regret** (the cost-aware value-of-thinking, `oracle_value − return@stop0`)
with RT. That was uninformative *by construction*, for a concrete structural reason:

- **Our cost is position-independent.** `cts.data.preprocess_mc.oracle` charges `time_cost = f(remaining
  budget)` with `maintenance_scale = 0` — i.e. the *same* per-step cost schedule for every board.
- **So regret is a near-monotone transform of the cost-free Gain.** Subtracting an identical per-step
  curve from every position shifts all the value-of-thinking curves down near-uniformly and preserves
  their cross-position ranking. Measured: ρ(cost-free Gain, cost-aware regret) = **0.98** (power-law),
  0.90–0.96 (linear/quadratic). The RT correlation is therefore unchanged by the cost: +0.107 cost-aware
  ≈ +0.108 cost-free, flat across all four cost shapes. A position-independent cost has no
  position-specific content — it can neither *help* the RT correlation nor *add noise* to it.

**step\* is the part regret throws away.** It is an `argmax` over the reward-curve × cost *interaction*:
the cost moves *where* the maximum lands, so two positions with identical Gain stop at different steps
depending on their convergence shape and the cost regime. step\* is genuinely cost-dependent; regret is not.

> **Clarification:** regret-vs-RT couldn't discriminate cost regimes — the cost can't re-rank a
> position-independent shift. step\* is the only oracle output on which the cost actually does anything,
> so it is the correct correlate of RT.

## Is the stop step degenerate, and what controls it? (A1)

step\*=0 ("never think") is the degeneracy that flattens the signal. Measured per cost regime on filtered
elo2000 (per-tree, B=96): **%step\*=0 ranges 6% (low-cost power-law) → 52% (high-cost linear)**. The cost
**scale** is the lever — not engine strength alone.

> **Result:** cost scale drives degeneracy: %step\*=0 ≈ 0.06–0.20 at ×0.25, up to 0.52 at ×4. Calibrate
> the scale *low* to keep step\* graded (and distinct from Gain — ρ(step\*, Gain) is 0.25–0.5 at ×0.25 vs
> 0.7–0.77 at ×4). *(Supersedes the earlier episode-level "~47% OSS=0" figure, which was measured on
> sampled-budget mc-pack episodes, not per-tree at fixed B.)*

## Does cost shape/scale change step\*↔RT? (A2/A3)

Yes — this is the payoff of the reframe. Across the 12 cost regimes, **step\*↔RT swings from +0.006 to
+0.119** (where regret was flat at +0.107). Best: **low-cost (×0.25) quadratic / linear (+0.119 / +0.111)**
and power-law-p2.8 ×4 (+0.111); worst: quadratic ×1 (+0.006). step\*↔legal-moves reaches **+0.30** under the
same low-cost regimes — recovering the full decision-width magnitude.

> **Result:** under the best cost regime step\*↔RT ≈ **+0.12**, finally edging the value signals (+0.107) —
> but still only *half* the raw legal-moves effect (+0.248). step\* is a **lossy proxy for width**: it
> correlates with legal-moves at +0.30 yet predicts RT at only +0.12. *(figure: `oss_rt_costsweep`)*

## Does a weaker engine sharpen step\*? (A4)

A weaker engine converges slower ⇒ more expansions carry value ⇒ less 0-spiked step\*. The **SF-1350** rung
is generating (Stockfish's UCI_Elo floor is 1350; sub-1350 segfaults, so the originally-planned SF-1000 is
infeasible). Re-test A1–A3 on it and compare the step\* distribution + step\*↔RT against elo2000.

## Explicitly not recommended

Training/evaluating only on step\*>0 boards — it selects on the outcome, prejudices against Always-Stop,
and biases the human-correspondence estimate. Fix the *generative* process (cost scale, engine strength),
not the label set.

## Shared evaluation

Every variant reports (a) the **step\* distribution** (granularity / degeneracy) and (b) **Spearman(step\*,
human log-RT)** with bootstrap 95% CIs on held-out moves, plus step\*↔legal-moves as the width bridge.

## Caveat

Calibration sharpens the *value-convergence* mechanism; it does **not** add the missing **uncertainty**
mechanism (the softmax VOC of [(R-TREESEARCH)](treesearch.md), which at calibrated temperature also
plateaus at ρ≈+0.12 on RT). step\*↔RT topping out near +0.12 — half the width effect — says calibration
alone will not close the convergence-vs-width gap; it makes the cost knob *meaningful*, not *sufficient*.
