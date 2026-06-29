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

## Where did M1–M3 actually land? (interim, filtered elo2000, n≈13k)

- **M1 (τ-sweep).** τ=1 was degenerate (softmax ≈ uniform at win-prob scale ⇒ VOC inverts; it anti-correlated
  with RT and legal-moves). At a calibrated τ≈0.1 the softmax-VOC flips positive and **recovers** the argmax
  value-of-thinking (ρ≈+0.12 vs RT) but does **not exceed** it. (figure `voc_tau_sweep`.)
- **M2 (width bridge).** The VOC couples to legal-moves at sharp τ (ρ≈+0.29) but is a **lossy proxy**: it
  tracks width yet predicts RT at only +0.12, while raw width predicts RT at +0.25.
- **Partial correlations (the decisive test).** Control for legal-moves and *every* value/VOC/step\* signal
  collapses (+0.11 → **+0.04**); control for any of them and legal-moves barely moves (+0.225). So the value
  signals are **proxies for legal-moves**, not independent deliberation signals.
- **Causal vs hindsight (the halter check).** A causal tree-stats halter (sees only the tree-so-far) does
  **not** beat the hindsight oracle on RT — both are weak (halter ≈0, oracle +0.06), both ≪ legal-moves
  (+0.26). So the information-asymmetry story is not the explanation. *(n≈2.5k val — provisional; re-run on
  the 61k set.)*

> **Result:** legal-moves is a near-orthogonal, **dominant** RT driver (+0.26) that no flavor of
> value-of-computation or stopping-time model (cost-tuned, τ-calibrated, causal, or hindsight) captures. The
> normative *deliberation* residual beyond legal-moves is only ≈+0.04. *(figures: `voc_rt_costsweep`,
> `oss_rt_costsweep`, `voc_tau_sweep`)*

## So what *is* the legal-moves effect — enumeration, Hick's law, or planning?

The pure-enumeration reading (cost ∝ raw move count) is too dumb, and three live objections sharpen what to
do next. **These are open hypotheses, not settled** — recorded here so the next pass tests them rather than
re-deriving them.

### It shouldn't be raw count — it should be prior-weighted consideration

People don't evaluate every legal move; a **policy prior** (pattern recognition) focuses attention on a few
candidates, and search proceeds from there. So the right cost is something like *(size of the prior-plausible
top set) × (stakes of getting it wrong)* — a decision-relevant VOC restricted to the candidate set, not all
`n`. **The structural problem:** in our data Stockfish is doing double duty — it supplies the leaf *values*
(the oracle) **and** stands in for the *search policy*, while having **no learned policy prior** (SF priors
are uniform ⇒ H(π)=log(legal moves), degenerate [[engine-eval-sf2000-repoint]]). So the prior that should
*focus* consideration is absent, and "consideration" defaults to ~all `n`. To model prior-guided
consideration we need an explicit policy prior (e.g. an lc0 policy head guiding expansion order) — SF alone
cannot separate "which moves to consider" from "how good they are."

### It looks like Hick's law — which is supposed to be perceptual, not planning

`RT ∝ (log) n` is Hick's law, traditionally a perceptual / response-selection effect — i.e. *not* planning.
That sits badly two ways: treating the dominant effect as perceptual nuisance to residualize out concedes the
most robust phenomenon; folding it into planning conflates perception with deliberation.

> **Clarification:** in chess, legal-move count **conflates** two co-varying things — the perceptual
> choice-set size (Hick's) and the planning branching factor (more branches genuinely = more to search). They
> can't be separated by observation alone. The partial correlation (planning residual after removing `n` ≈
> +0.04) is the coarse separation; the clean separation is whether a *human-like search's* node count tracks
> RT after controlling for raw `n` (next section). If it does, the effect is planning mediated by search
> breadth, and Hick's is just the floor; if not, it's mostly perceptual + a thin planning residual.

### Maybe our search is simply too smart — make the generator human-like

MCTS/UCT is good: principled selection, non-greedy exploration, intermediate/deep rollouts. That smartness
**prunes away exactly the breadth that drives human RT**, so the tree-stats carry a *machine's* consideration,
not a human's. Two levers, cheap → expensive:

1. **Shorten rollouts / cap depth** — a parameter change; makes the search myopic and bushier, closer to human
   look-ahead horizons.
2. **Swap MCTS → optimistic Best-First-Search** — BeFS is *deliberately dumber*: expand the currently
   most-promising node under an **optimistic** value heuristic. Optimistic null/unknown values get explored,
   then **discarded as disappointments** — producing a **bushy** tree whose breadth reflects how many
   candidates *looked* worth calculating. That "consider → calculate → reject, but you spent the time"
   dynamic is far more human-like than UCT's, and its expansion count is a natural consideration-cost that
   could track RT where the value-VOC can't.

> **Decision (next pass), cheapest first:**
> (a) **SF-1350 weak rung** — a dumber engine prunes less ⇒ tree-width → raw `n` ⇒ tree-stats should recover
>     the legal-moves signal and track RT better than SF-2000 (direct, falsifiable; data already generating);
> (b) **shorten rollouts / cap depth** on SF and re-test tree-stats↔RT (parameter sweep);
> (c) if (a)/(b) move the needle, implement **optimistic BeFS** as the tree generator and compare its
>     bushiness/expansion-count↔RT against UCT;
> (d) inject a **policy prior** (lc0 policy head as expansion guide, SF as evaluator — a hybrid) to model
>     prior-focused consideration and test the *(top-k contested set × stakes)* cost rather than raw `n`.

> **Caveat:** (b)–(d) change the *generative model* of the tree, so they re-open the strength-laddering and
> faithfulness contracts ([(R-DATA)](reference.md)). Treat as a v2 generator, gated on (a) showing that a
> dumber search recovers the width effect at all.
