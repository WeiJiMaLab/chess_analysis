# Uncertainty-aware metareasoning: value of computation under a softmax policy

**Ref:** `R-METAREASON` · [Index](reference.md)

> **Status:** 📝 proposal (analyses M1–M3). This is the report where the model's *mechanism* changes;
> its sibling [(R-HALT-CALIB)](halt_calibration.md) only calibrates the existing one.

## Question

The budgeted oracle has **no notion of uncertainty**: it commits to the argmax move and treats the
search's own value as ground truth, so "value of computation" collapses to *value-gap-to-convergence*.
But humans deliberate when **uncertain** — decision-width (legal moves) is the robust RT driver
([(R-MOVETIME-MODEL)](engine.md)). **Does an uncertainty-aware value of computation — a softmax policy
graded against a supervisor — make the normative *when-to-think* correspond to human deliberation?**

## Background: this is rational metareasoning

Value of computation (VOC) = the expected improvement in **decision quality** from more computation,
**under uncertainty about which option is best** (the rational-metareasoning / VOC tradition — Russell
& Wefald; Hay & Russell; and on the human side Callaway, Lieder et al. on resource-rational planning).
Our current oracle is the degenerate, certainty-assuming special case (argmax, self-graded). Making the
policy a *distribution* and grading it against a *supervisor* recovers the real VOC — and, we
hypothesize, an uncertainty-driven "when to think" that looks like the human one.

## The model

At search step `t`, with root-child values `v_t`:

- **Policy = softmax(v_t / τ)** — a distribution over the move you'd play (not argmax); `τ` is the
  one new knob.
- **Halt value = E_{softmax}[ V_sup(child) ]** — the expected supervisor value of the move the policy
  would pick. **Closed-form** (`Σ softmax·V_sup`), no sampling.
- Early search ⇒ flat `v_t` ⇒ near-uniform softmax ⇒ averages good and bad moves ⇒ low value; search
  **sharpens** the softmax ⇒ value rises. **The benefit of thinking = the sharpening = uncertainty
  reduction.** That is the term the current oracle lacks.

**Supervisor** (breaks the "search grades itself" circularity):

- **First cut — free:** the **deep (96-expansion) tree value**. Grade the step-`t` softmax policy on
  the converged tree's `final_Q`. Computable from existing `oracle_root_q_trace` / `final_q` — **no
  re-generation** (a softmax generalization of the engine-eval "Gain/VOC").
- **Refinement:** an **independent, stronger** Stockfish (deeper / higher-Elo) to remove the
  self-grading circularity (requires re-generation).

Keep the GNN encoder **unchanged**: uncertainty enters the **oracle reward**, not the representation.
(Retraining the encoder on `E[V] + entropy` adds confounds — defer until the reward-side change is shown
to matter.)

## Analyses

| # | analysis | feasibility |
|---|---|---|
| **M1** | Deep-tree softmax-VOC oracle: `halt_reward[t] = E_softmax_τ[V_deep(child)]` at mc-pack; sweep `τ`. Eval: OSS distribution + Spearman(VOC/OSS, human RT) **vs** the value-convergence oracle. | **medium** (pack change; no re-gen) |
| **M2** | Bridge to the human driver: does the uncertainty-aware VOC correlate with **decision-width (legal moves)** — the known RT driver? Direct test of "uncertainty ⇒ deliberation" as the shared mechanism. | **cheap** (correlation) |
| **M3** | Independent supervisor: replace deep-tree with a stronger/deeper Stockfish; re-test M1/M2 without self-grading. | **medium-high** (re-gen) |

## Predictions

If uncertainty is the missing mechanism: (i) the softmax-VOC OSS is no longer 0-spiked the same way
(uncertain positions ⇒ think); (ii) VOC/OSS **correlates with legal moves** (M2); (iii)
Spearman(VOC, RT) **exceeds** the value-convergence oracle's. A *null* result is also informative — it
would say the human width-effect is **not** captured by normative VOC, which is itself a finding.

## Caveats / degrees of freedom

`τ` and the supervisor are researcher DOF. Judge on **held-out** human-RT correspondence with bootstrap
95% CIs ([[bootstrap-cis-always]]); pre-commit the metric before sweeping. Calibration of the existing
mechanism (degeneracy, cost) is the separate [(R-HALT-CALIB)](halt_calibration.md) — run that first so
M1's comparison is against a well-calibrated baseline rather than a degenerate one.
