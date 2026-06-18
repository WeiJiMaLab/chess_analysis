# Branching and resource-rational deliberation

**Ref:** `R-BRANCH` (draft / proposal) · [Index](README.md)

## Why does decision *width* drive human think time?

The strongest single predictor of how long a human thinks is **branching** — the number of
legal moves / candidate options at the position (Spearman ρ(branching, log RT) ≈ **+0.35**),
far above any engine value-search quantity (Gain ≈ +0.07, GSS ≈ +0.12, MQ ≈ −0.15). A
normative search model whose oracle halts on **value-convergence** (stop once the value gap is
decided) reproduces the *direction* of the value effects but **ignores branching**, which is
what actually paces human deliberation.

> **Result:** Humans deliberate in proportion to the **width of the decision** (how many moves
> they must weigh), not in proportion to the realized value-of-computation. Our value-convergence
> model misses this because it never represents decision width.

This looks paradoxical only if "value of computation" is read as the **realized, ex-post** gain
that search actually produced (our `Gain` metric — zero in ~⅔ of positions, because the 1-ply
move was already best). The resolution below is that a resource-rational agent acts on the
**ex-ante expected** value of computation, which scales with branching.

## A resource-rational account: deliberate to resolve *which move is best*

Frame move choice as a **metalevel decision problem** (Russell & Wefald; Hay; Lieder & Griffiths):
the agent holds a belief over which action is best, each unit of computation (an expansion /
candidate evaluation) is a noisy observation that sharpens that belief, and the agent should keep
computing while the **expected** value of another computation exceeds its cost.

The key quantity is **prior uncertainty over the argmax** — how unsure the agent is about which
move is best *before* searching. With more plausible candidates (higher branching, flatter
policy `π`), that uncertainty is larger, so:

- **Expected** value of computation is high (a yet-unevaluated move is more likely to overturn
  the current best) → the resource-rational agent keeps searching.
- As candidates are ruled out and the posterior over the argmax concentrates, expected VOC falls
  below cost → stop.

So the resource-rational **stopping time increases with prior argmax-uncertainty ≈ policy
entropy H(π) ≈ branching.** Realized `Gain` is the *outcome* of this process (usually small,
because the prior-favored move usually survives); it is **not** the driver. This dissociation —
RT tracks *expected* VOC (uncertainty) while being weakly related to *realized* VOC — is exactly
what the data show.

> **Modeling proposal (◆):** Replace "stop when the *value* is decided" with "stop when the
> *argmax* is decided." Deliberation is resource-rational evidence accumulation about **which
> move is best**; its optimal duration grows with the number/uncertainty of competing candidates.

This is a chess instantiation of **Hick's law** (choice RT increases with the number of
alternatives): branching is the candidate count, and a metalevel/sequential-sampling agent
choosing among `N` near-tied options crosses its confidence threshold later as `N` grows. It also
reframes the **lc0-too-strong** problem: lc0 collapses most positions to a decided *value*
(|final Q| ≥ 0.9 in ~79%), but the *argmax* among near-equivalent moves can still be wide — and
it is argmax-uncertainty, not value-uncertainty, that humans appear to spend time on.

## How the tree could operate on branching

Each idea is an instantiation of the same principle — allocate computation to resolve
argmax-uncertainty — differing in where branching enters the lc0 search:

1. **Width-gated stop (most direct, lc0-native).** Stop when the **root visit distribution
   concentrates** (top-move visit share > θ, or entropy of visits < τ). Wide decisions start
   diffuse and take many expansions to concentrate → search longer, by construction. This *is*
   the resource-rational "stop when the posterior over the argmax is peaked" rule.
2. **Candidate-coverage search.** Require each plausible candidate to be visited ≥ k times before
   committing → search effort ∝ effective branching. Branching sets the *width* of required
   exploration; value-convergence sets the *depth*. (A fixed cost-per-candidate-evaluated makes
   total deliberation ∝ branching, principled rather than ad hoc.)
3. **Expected-VOC / metalevel cost.** Make the controller's continue-value depend on E[VOC] =
   probability an unevaluated move overturns the current best, which is monotone in H(π). The
   normative *target* then rewards searching wide positions longer — so a controller trained on it
   learns branching-sensitive stopping (the GNN already *encodes* branching; the current
   value-convergence target just never rewards using it).
4. **We already have an implicit branching channel — and discard it.** GSS (expansions-to-find-best)
   is partly branching-mediated (wide → more expansions), which is *why* GSS↔RT > Gain↔RT. The
   value-convergence stop halts early and throws that width signal away; a width-gated stop simply
   stops discarding it.

## Predictions & experiments

- **P1 (entropy ≈ branching):** root policy entropy H(π) / effective branching predicts log RT
  about as well as raw branching (ρ ≈ +0.3+), and better than any value metric. *Cheap:* the
  prior is already in `node_features`.
- **P2 (width-gated > value-convergence):** a halt-on-visit-concentration stopping rule predicts
  human RT better than the value-convergence oracle. Slots into the budgeted-baselines harness.
- **P3 (expected ≫ realized VOC):** an *ex-ante* uncertainty signal (prior argmax entropy)
  dominates realized `Gain` as an RT predictor, and adds unique variance over branching.
- **P4 (scale vs modulation):** within fixed branching, residual RT tracks `Gain`/value — i.e.
  branching sets the *scale* of deliberation, value-of-computation *modulates* it.
- **P5 (Hick form):** check whether RT scales with `branching` or with `log(branching)` /
  H(π) — the functional form discriminates "enumerate all moves" (linear) from
  "evidence accumulation among alternatives" (log / entropy).

> **Clarification:** This does not discard value-of-computation — it subordinates *realized* Gain
> to *expected* Gain (argmax-uncertainty). The claim is that the resource-rational **driver** of
> deliberation is uncertainty about which move is best (≈ branching/entropy), with value gain as
> the modulating, ex-post quantity.

## Preliminary evidence: Gain vs GSS (tests P3–P4)

A first pass on the 112.8K-FEN subset (one observation per FEN) already fits the account —
**branching is the scale, value-of-search the modulation**:

- **Branching dominates.** Alone it explains R² ≈ **0.12** of log RT; both tree metrics *together*
  add ΔR² ≤ 0.01. The width channel is the story; the value channel is a small correction.
- **GSS is branching-mediated effort.** GSS is the metric most collinear with branching
  (ρ ≈ +0.19 vs Gain +0.14), wins the *linear* comparison (Pearson +0.114 vs +0.073), and its
  unique signal sits in *decisive* positions ("effort to confirm the obvious best").
- **Gain is the orthogonal value-of-search signal (P3/P4 supported, on ranks).** Gain is 0 in ~60%
  of positions, so Pearson understates it; on **ranks** it is the stronger predictor (Spearman
  +0.138 vs GSS +0.123) and adds **more unique rank variance over branching** (ΔR² +0.009 vs
  +0.0056). Its signal is **concentrated in non-decisive positions** (~21% of cases, where lc0 is
  genuinely unsure): there Gain ≈ +0.13 while GSS ≈ **0** — exactly where value-of-computation
  should bite.

> **Result:** Branching sets the **scale** of deliberation (P4) and realized Gain **modulates** it
> only where the engine is genuinely uncertain (non-decisive positions) — matching the
> resource-rational picture (uncertainty over the best move is the driver; value-of-search is a
> second-order, ex-post correction). It also explains why naive *realized* Gain under-predicts RT
> marginally: 60% zeros, swamped by the width channel. Still open: the *expected*-VOC / entropy
> signal of P1–P3 (not yet computed) — the hypothesis is that it, not branching per se, is the true
> driver and that branching is its observable proxy.

## Open questions

- The cleanest formalization: a **Bayesian metalevel MDP** over root move-selection, or a simpler
  **sequential-sampling / Bayesian sequential test** over moves whose threshold-crossing time
  grows with the number of competing alternatives. Which is identifiable from the RT data?
- Is the right uncertainty signal the **policy** entropy (prior), the **value-head WDL** spread
  over candidates, or the evolving **visit** entropy? P1/P3 adjudicate.
- Does a **strength-matched (SF-2000)** engine widen the argmax distribution on human-hard
  positions (vs lc0's collapse), making expected-VOC track human RT more cleanly? (Ties to the
  gated SF-2000 follow-up in the move-time model report.)

*Draft proposal — captures the branching mechanisms and a resource-rational framing for the
branching effect; experiments P1–P5 are not yet run. Not yet wired into the analysis roadmap.*
