# Uncertainty-aware metareasoning → what actually drives human deliberation time

**Ref:** `R-METAREASON` · [Index](reference.md)

> **Status:** 📝 active. The original mechanism question — *does an uncertainty-aware value of computation
> explain when-to-think?* — is essentially **answered: no.** No value-of-computation signal tracks human RT
> beyond a ≈+0.04 residual once decision-width is accounted for. The inquiry has pivoted to (1) a **positive
> model** — RT as *decision difficulty* — and (2) a **generative-search hypothesis** — our search is too smart
> to be human. Interim results on filtered elo2000 (n≈6–13k moves, bootstrap 95% CIs); a powered 61k re-run, a
> strength-ladder, and a dumber-generator sweep are queued (see **Moving parts & overnight plan**).
> Sibling [(R-HALT-CALIB)](halt_calibration.md) calibrates the value-convergence oracle (step\*↔RT).

## The question, and where it went

We started from rational metareasoning (Russell & Wefald; Hay & Russell; Callaway/Lieder on resource-rational
planning): the budgeted oracle is the degenerate, certainty-assuming special case (argmax, self-graded), and
making the policy a softmax graded against a supervisor should recover a real value of computation (VOC) that —
we hypothesized — would make normative "when to think" match human deliberation. The model spec (softmax-VOC,
supervisor, τ) is in the **Appendix**.

The first pass answered it: **the uncertainty-aware VOC does not explain human RT.** So the live question became
*what does, and why does the VOC frame miss it?*

## What we found (established — filtered elo2000, bootstrap 95% CIs)

| signal vs human log-RT | ρ | note |
|---|---|---|
| **legal moves** | **+0.26** | dominant, robust driver |
| cost-free Gain / regret-alwaysstop | +0.11 | → **+0.04** controlling for legal-moves |
| τ-calibrated softmax-VOC (τ≈0.1) | +0.12 | recovers but doesn't beat argmax Gain (τ=1 degenerate) |
| step\* (OSS, best cost regime) | +0.12 | cost-dependent (regret is not — see R-HALT-CALIB) |
| causal tree-stats halter | ≈0.00 | no better than hindsight oracle (+0.06); n≈2.5k val, provisional |
| **# good moves** (≤ε below best) | **−0.12 … −0.20** | the sign-flip — see below |

- **Partial correlations (decisive):** control for legal-moves and *every* value/VOC/step\* signal collapses
  (+0.11 → +0.04); control for any of them and legal-moves barely moves (+0.225). The value signals are
  **proxies for legal-moves**, not independent deliberation signals.
- **Causal ≠ the issue:** removing the oracle's hindsight (a causal halter) did not recover RT signal.

> **Result:** no value-of-computation or stopping-time formulation (cost-tuned, τ-calibrated, causal, or
> hindsight) captures human RT beyond ≈+0.04. *(figures: `voc_rt_costsweep`, `oss_rt_costsweep`,
> `voc_tau_sweep`)*

## The emerging model: RT is decision *difficulty* (size − forgiveness + sharpness)

The positive finding. Human RT decomposes into oppositely-signed decision-difficulty terms, each far larger
than any VOC signal:

| term | operationalization | ρ vs RT | reading |
|---|---|---|---|
| **size (+)** | # legal moves | **+0.26** | more options ⇒ more to scan |
| **forgiveness (−)** | # / fraction good moves (≤ε of best) | **−0.12 … −0.26** | many acceptable ⇒ easy ⇒ faster |
| **sharpness (+)** | action gap (top-1 − top-2) | **+0.19** | a critical move to *find* ⇒ calculate |

The forgiveness term **survives partialling on n_legal** (−0.05 to −0.10), so it is not just the inverse of
size. `action_gap` is *positive* (not the naïve "clear best ⇒ easy"): in chess a large gap is usually a
sharp/tactical position whose one critical move must be calculated — stakes, not ease.

> **Result:** the driver is decision *difficulty*, not raw option count and not the VOC. `RT ≈ size(+) −
> forgiveness(−) + sharpness(+)`, each ≈±0.2–0.26. *(figure: `good_moves_rt`)*

### The sign-flip is also an engine-strength litmus test

`# good moves` is defined by the *engine's* values, so the flip appears **only if engine-good = human-good.**
No flip ⇒ a strength mismatch — e.g. a superhuman engine whose move-goodness humans can't track (a human likes
a move; the far-stronger engine disagrees ⇒ `# good` carries no human signal). The flip is clean at **SF-2000**
(the dataset's players are ≥2000 Elo — exactly the correspondence we want), which *validates* that rung.

> **Decision:** run the flip across the strength ladder (SF-1350 vs SF-2000 …). The strength where the flip is
> **strongest** is the regime where engine-goodness = human-goodness — the precondition for every other
> analysis here. If the flip *weakens* at higher strength, the engine is outrunning the humans and we should
> grade against a population-matched strength, not maximal SF.

## Why the VOC frame misses it: the generator is too smart

The tree we analyze is a **model artifact**. A strong MCTS/UCT + strong-Stockfish search prunes away exactly
the breadth that drives human RT, so the tree-stats carry a *machine's* consideration, not a human's. Three
coupled issues:

- **Prior.** People don't evaluate every move; a policy prior focuses attention, so the cost should be
  *(prior-plausible top set) × stakes*, not raw `n`. But SF does double duty — leaf *values* (oracle) **and**
  stand-in *search policy* — with **no learned prior** (uniform priors ⇒ H(π)=log n, degenerate
  [[engine-eval-sf2000-repoint]]). So consideration defaults to ~all `n`.
- **Hick's law.** `RT ∝ (log) n` is classically *perceptual*. In chess, legal-move count **conflates**
  perceptual choice-set size with the planning branching factor; they can't be separated by observation. The
  clean separation is whether a *human-like search's* node count tracks RT after controlling for raw `n`.
- **Too-smart search.** UCT's principled selection prunes the breadth; a *dumber* generator should produce
  bushier, more human-like trees whose stats carry the decision-difficulty signal.

## Moving parts & overnight plan

Cheapest → most invasive. (a)–(b) are pure config/continuations and run **overnight on the cluster**; (c)–(e)
are gated on them.

| # | experiment | tests | cost | gate |
|---|---|---|---|---|
| **P0** | **61k powered re-run** of the whole suite (cost sweep, τ, step\*, halter, partials, sign-flip) on the 250k→~61k filtered set | tightens every CI above; confirms the halter (was n≈2.5k) | sbatch, **overnight** | 250k elo2000 filter |
| **P1** | **SF-1350 vs SF-2000**: sign-flip strength + width↔RT + `# good`↔RT | dumber-engine width recovery **and** the strength litmus test | sbatch, **overnight** | both 250k filters |
| **P2** | **`sf_search_limit_nodes` 100→1** generation (noisy-myopic leaf values), then the suite | isolates dumb *values* from a weak *engine*; "shorten rollouts" lever | gen (hrs), **overnight** | new config |
| **P3** | **optimistic Best-First-Search** generator vs UCT (bushiness / expansion-count ↔ RT) | does a deliberately dumber search recover the human breadth? | v2 generator | P1/P2 positive |
| **P4** | **policy-prior hybrid** (lc0 policy guides expansion, SF evaluates) | prior-focused consideration; *(top-k × stakes)* cost vs raw `n` | v2 generator | P3 |

> **Decision (tonight):** launch P0 (61k suite) and P2 (`sf_search_limit_nodes=1` generation) overnight; P1 is
> already in motion (SF-1350 250k generating). Light RT-join analyses run on completion. P3/P4 are gated on
> P1/P2 showing a dumber search recovers width↔RT at all — and they re-open the strength-ladder + faithfulness
> contracts ([(R-DATA)](reference.md)), so they are explicit v2-generator work.

---

## Appendix — the softmax-VOC model (original M1–M3 spec)

At search step `t` with root-child values `v_t`: **policy = softmax(v_t/τ)**; **halt value = E_softmax[V_sup]**
(closed form `Σ softmax·V_sup`, no sampling). Early ⇒ flat `v_t` ⇒ near-uniform ⇒ averages good and bad ⇒ low;
search sharpens ⇒ value rises ⇒ benefit of thinking = uncertainty reduction. **Supervisor:** the deep
(96-expansion) tree `final_Q` (free, from `oracle_root_q_trace`), or an independent stronger Stockfish (M3,
re-gen). Encoder unchanged — uncertainty enters the oracle reward, not the representation.

- **M1** deep-tree softmax-VOC, sweep τ → *done*: τ=1 degenerate, τ≈0.1 recovers but doesn't beat argmax Gain.
- **M2** width bridge → *done*: VOC↔legal-moves +0.29 at sharp τ, but lossy (predicts RT only +0.12).
- **M3** independent stronger supervisor → folded into P1 (the strength ladder / litmus test).

DOF: τ and supervisor are researcher choices — judged on held-out RT correspondence with bootstrap CIs
([[bootstrap-cis-always]]); metric pre-committed before sweeping.
