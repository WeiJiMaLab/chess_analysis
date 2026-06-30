# Value of information: do people think to RESOLVE which move is best?

**Ref:** `R-VOI` · [Index](reference.md) · parent thread [(R-TREESEARCH)](treesearch.md)

> **Status:** 📝 proposal — the hypothesis, the operationalization off the *existing* traces, and the one
> discriminating test. The thesis: the normative quantity that times deliberation may be **how much
> uncertainty over the *argmax* a search resolves** — a *value-of-information* (VoI) account about
> **belief dynamics over which move is best** — and this is sharply distinct from value-of-computation
> (Gain), from the already-tried softmax-VOC, and from the static satisfaction term. Built entirely on the
> 250K `n1md36` trees (no regeneration); join to RT by 4-field FEN; bootstrap 95% CIs ([[bootstrap-cis-always]]).

---

## 1 · The question — and why "value of information" is not "value of computation"

The project's robust finding is `RT ≈ size(+0.31) − satisfaction(−0.31) + sharpness(+0.24)` — *satisficed
decision difficulty*, dominated by decision **width**. Every **value-of-computation** account collapsed to a
legal-moves proxy (≤ +0.04 partialled): Gain (how much the chosen move's **value** improves with search),
step\*/regret, value-pruning, construal.

The user's seed reframes the normative object. People may not think to make the move **better** (value of
computation); they may think to find out **which move is best** — to **resolve uncertainty over the argmax**.
The quantity is then the **information gain** of planning: how much a search collapses the belief about *which
action wins*. This is a **value-of-information** account, and its object is **belief dynamics**, not value
levels.

> **The one-line claim.** RT tracks **how much uncertainty over the best move a search must resolve** — you
> think when (and as long as) you are *unsure which move is best* and the search keeps re-ranking it — *not*
> how much value the search adds, and *not* the static count of good moves.

### Why this is genuinely a different axis (the make-or-break)

There are **three** incumbents this must be distinct from. The distinction is not rhetorical — it is
measurable on the trace, and §6 reports the actual collinearities. The summary up front:

| Incumbent | Its object | What it uses | Why VoI is different |
|---|---|---|---|
| **Value-of-computation** (Gain, step\*, regret) | how much the move's **VALUE** improves with search | `final_Q(best@96) − final_Q(best@1)`; the **value level** of the halt | VoI ignores value level entirely; it measures **belief over the *identity* of the argmax** and how it shifts. Two positions with identical Gain can have very different argmax-uncertainty trajectories (a slow grind to a known-best vs a late reversal). |
| **softmax-VOC** ([(R-TREESEARCH)](treesearch.md) H2) | the **sharpening of a softmax policy's expected value** | `Σ_c softmax(q_trace[s]/τ)·final_Q[c]`, DP'd to a halt value | softmax-VOC re-grades the **value** of the softmaxed policy (a real-number *reward*), then asks the value-of-computation DP "how much does that reward improve." It **recovered the argmax value (~+0.16) and never beat it** because it is *still a value quantity*. VoI uses the **entropy of the belief and its trajectory** — an information quantity in **bits**, never multiplied by `final_Q`. It never touches the value scale. |
| **satisfaction / sharpness** (the static terms) | **# near-best moves** / top1−top2 **value gap** at convergence | counts/gaps on `final_Q` only | These are **static, end-of-search** counts on the *final* values. VoI is about the **trajectory**: the *initial* uncertainty before search and the *path* by which it collapses (the stabilization step, the entropy-reduction, the early→late KL). A position can be ambiguous-at-the-end (low satisfaction) yet resolve **instantly** (argmax never moves), or decisive-at-the-end yet only after a **late reversal**. Those dissociate the dynamics from the static count. |

> **Decision:** VoI is operationalized strictly as a property of the **belief trajectory over the argmax**
> (entropy *reduction*, argmax *stabilization step*, early→late *KL*), and graded **in bits / step-counts,
> never re-multiplied by `final_Q`**. The moment a quantity touches the value scale it has become
> value-of-computation or softmax-VOC; we keep it on the information scale on purpose.

The crisp NEW-vs-old, restated for the record:
- **vs Gain:** Gain is `Δ value of the best move`. VoI is `Δ entropy of which move is best`. Orthogonal scales.
- **vs softmax-VOC:** softmax-VOC took the *final softmax value*; VoI takes the *entropy of the softmax
  policy and its trajectory*. softmax-VOC asked "how much better is the softmaxed value after searching"; VoI
  asks "how much did the search *concentrate* the policy." Same `q_trace` input, different functional — value
  vs information.
- **vs satisfaction:** satisfaction is a **static count** on `final_Q`; VoI is the **dynamics** of the
  argmax belief (initial uncertainty + how fast it collapses). The discriminator is whether the **trajectory**
  quantities (stabilization step, entropy reduction) survive a partial on the static count — §6 shows the
  *trajectory* ones largely do, while the *final-belief* one does not.

---

## 2 · What the trace actually contains (verified on `n1md36`)

Read off three payload arrays per tree (verified `000000`–`000800`):

- `oracle_root_q_trace` — shape `(96, L)`, `L` = # legal root moves. Row `s` = each root child's backed-up Q
  after `s+1` expansions, **from the mover's perspective** (win-prob in `[−1, 1]`).
- `oracle_final_root_q_values` — shape `(L,)` — the converged Q (== last finite trace row).
- `oracle_best_move_index` — shape `(96,)` — the argmax child index at each step. **This is exactly
  `argmax_c q_trace[s, c]`** (verified), so the argmax-stability signal is pre-computed for us.
- `oracle_trace_expansion_counts` — `[1, 2, …, 96]` (the step axis).

**Two structural facts that the operationalization MUST respect** (both verified, both genuine traps):

1. **`oracle_root_visits_trace` does NOT exist** in any set (`n1md36`, `n100md36`, …). The prompt named it,
   but it is absent. **Substitute: visit breadth = the count of non-zero entries in `q_trace[s]`.** A child's
   Q is exactly `0.0` until it is first visited, then becomes a real (usually negative) win-prob. So
   `(q_trace[s] != 0).sum()` is the size of the **visited consideration set** at step `s` — the live
   contenders the search has actually evaluated. Verified: it rises `1, 2, 3, …` then **saturates early**
   (median final breadth ≈ 20 of ≈32 legal; p10/p90 = 1/41).

2. **`q_trace[0]` is all-zeros, and unvisited children sit at `Q = 0` throughout** — *above* the typical
   visited value (positions are often losing, Q ≈ −1). So a **naïve softmax/entropy over a raw early row is
   dominated by the unvisited Q=0 slots, not by a real belief**, and `argmax(q_trace[s])` early just walks the
   visit order (0,1,2,…) — an artifact, not a decision. **Handling (locked):**
   - For any **entropy/KL**, restrict the softmax to the **visited set** (`q_trace[s] != 0`), or equivalently
     mask Q=0 slots to `−∞` before softmax. Never softmax the raw padded row.
   - Define the **initial** belief not at step 0 but at the **first informative step** — the first step at
     which ≥ `min(L, k)` children are visited (take `k`≈ the breadth-saturation point, ≈ 6–10), or simply
     `s = L−1` (every child seen once). Document the choice; sweep it as a robustness check.
   - For **argmax stabilization**, the pre-computed `oracle_best_move_index` is the right object *as long as
     we use the last-change-step* (below): the early lockstep walk is automatically superseded once a visited
     child overtakes the Q=0 baseline.

---

## 3 · The signals — exact quantities and formulas

All per-position, from the three arrays above. `L` = legal moves; `vis(s) = {c : q_trace[s,c] ≠ 0}`;
`p_τ(s) = softmax(q_trace[s, vis(s)] / τ)` over the **visited** set; `s0` = first-informative step
(§2); `sf` = last finite step (== convergence). Pick `τ` on the win-prob scale (gaps ≈ 0.01–0.1, so
`τ ≈ 0.05–0.1`, matching the softmax-VOC τ-sweep finding that τ=1 is far too hot).

**A — Initial argmax uncertainty (static, *before* the search resolves).** How unsure you are *which* move is
best at the outset.
- `H_init = H(p_τ(s0))` — entropy of the early visited-belief (bits).
- `n_live_init = |{c ∈ vis(s0) : final_Q[c] ≥ max(final_Q) − δ}|` — # early contenders that end up plausible.
- *Prediction:* `RT ∝ H_init` (you think when you start out unsure which move wins).

**B — Information gain / uncertainty *reduction* (the VoI core).** How much the search *resolves*.
- `ΔH = H_init − H_fin`, `H_fin = H(p_τ(sf))` — entropy collapse over the search (bits resolved).
- `KL_early→late = KL(p_τ(sf) ‖ p_τ(s0))` — how far the belief moved (a directed, sharper resolution measure).
- *Prediction:* `RT ∝ ΔH` (or `KL`) — you keep thinking while the search is still *resolving* the argmax,
  and stop once it has converged. **This is the signal that is most orthogonal to width (§6).**

**C — Resolvability / argmax-stabilization step (the dynamic stopping-time, the headline new object).** *When*
the belief locks on. This is the **entropy/argmax analog of GSS/step\*** — but GSS/step\* are **value-based**
(when the *value* peaks), this is **argmax-based** (when the *identity* of the best move stops changing).
- `t_stab = 1 + max{ s : best_move_index[s] ≠ best_move_index[s+1] }` — the **last** step the argmax changed
  (0 if it never changes after the early walk). The number of expansions until the best move stabilizes.
- `t_stab_eps` — a tolerant variant: last step the **top-`δ` set** changed (robust to ties flickering).
- `n_flips = #{ s : best_move_index[s] ≠ best_move_index[s−1] }` restricted to `s ≥ s0` — how *churny* the
  argmax is (a position that keeps re-ranking is hard to resolve).
- *Prediction:* `RT ∝ t_stab` and `RT ∝ n_flips` — positions whose best move only settles late (or keeps
  flipping) are the ones people deliberate over; positions that lock in immediately are fast. Verified spread:
  `t_stab` p10/50/90 = **0 / 34 / 90** — graded, but **bimodal** (15% never-change, 21% late-reversal ≥ 80),
  which §7 flags as a risk.

**D — Live-contender breadth dynamics (a width-adjacent control, included to *expose* overlap).**
- `breadth_fin = |vis(sf)|` (visited set size). *Expected to track legal-moves* (verified +0.42) — included
  precisely so we can show it is **not** the carrier and partial it out.

> **Decision:** the **headline VoI signals are B (`ΔH`/`KL`) and C (`t_stab`/`n_flips`)** — the *trajectory*
> quantities. A (`H_init`) and D (`breadth`) are static/width-adjacent and serve as controls and as the
> "initial uncertainty" leg of the prediction, not as the discriminating test.

---

## 4 · Operationalization plan (concrete, reuses existing infrastructure)

A single trace-analysis script, modeled on `compute_voc_signals.py` (same `ProcessPoolExecutor`, same
`torch.load → fen → arrays` skeleton, same 4-field-FEN key), emitting one parquet keyed by FEN:

1. Per tree, load `oracle_root_q_trace`, `oracle_final_root_q_values`, `oracle_best_move_index`. Skip if
   `L < 2` or all-flat (handle below).
2. Compute the **visit-breadth mask** `q_trace != 0`; derive `s0`, `vis(s)`.
3. Compute **A/B/C/D** per the formulas in §3, at `τ ∈ {0.05, 0.1}` and `δ ∈ {0.05, 0.1}` (small sweep for
   robustness; one chosen pair is the headline).
4. Write `[fen, L, H_init, n_live_init, H_fin, dH, kl, t_stab, t_stab_eps, n_flips, breadth_fin]`.
5. **Light follow-up (the actual analysis):** join to RT (`processed_moves_nonzero`, `move_time`, log) by
   4-field FEN; **also join the existing** `sf_analysis/n1md36/voc_signals.parquet` (it already has
   `legal_moves`, `n_good_{0.02..0.5}`, `action_gap`) so satisfaction/sharpness are free, no recompute. Then:
   - marginal Spearman ρ(signal, log-RT) with bootstrap CIs;
   - **partial ρ controlling for legal-moves** (the width floor) — the H4 test;
   - **partial ρ controlling for satisfaction (`n_good`) AND sharpness (`action_gap`)** — the distinctness test;
   - the **joint** partial controlling for **all three** (legal, satisfaction, sharpness) — the make-or-break.

> **Decision:** reuse the FEN-keyed parquet pattern and the *already-computed* satisfaction/sharpness columns;
> the only new compute is the trace functional, which is cheap (§5).

---

## 5 · Feasibility — cheap, no regeneration

This is a pure **read-only trace analysis on the existing 250K `n1md36` trees**. No tree-building, no DP, no
GPU.

| quantity | number | note |
|---|---|---|
| trees | **250K** on disk (`sf_trees/n1md36`) | already generated; reused as-is |
| per-tree work | load + 3 arrays + O(96·L) numpy | strictly cheaper than `compute_voc_signals` (no budgeted-oracle DP, no 12-cost grid) |
| precedent runtime | `voc_signals`/`voc_tau_sweep` already ran on the full 250K | this is a strict subset of that work → **comparable or faster**, one CPU sbatch |
| RT join | `voc_signals.parquet` is already 250K rows keyed by FEN | identical join already validated (262K human moves matched in R-PRUNING) |
| satisfaction/sharpness | **already in `voc_signals.parquet`** | zero extra compute |
| new code | one worker fn + one analysis notebook | mirrors `compute_voc_signals.py` line-for-line |

> **Result (feasibility):** a **single CPU job**, strictly lighter than the VOC step that has already run
> twice on this exact tree set. No regeneration, no new trees, no storage pressure. This is the cheapest probe
> in the project.

---

## 6 · Predictions & the ONE discriminating test

**Predictions** (directional, to be confirmed/refuted):
- `RT ∝ H_init` (+): unsure-at-the-outset → think more.
- `RT ∝ ΔH` / `KL` (+): more uncertainty *resolved* by search → longer think (the VoI core).
- `RT ∝ t_stab` / `n_flips` (+): later/churnier argmax stabilization → longer think.
- `breadth_fin` tracks legal-moves and should **not** survive the width partial (control).

**The discriminating test (make-or-break), in one sentence:** a VoI signal must **(i) beat or match the
+0.34 legal-moves floor, AND (ii) retain a non-trivial partial ρ in a JOINT control on legal-moves +
satisfaction + sharpness.** (i) alone is a re-description of width; (ii) alone could be a faint reindex of
satisfaction. Only **both** establish a new axis.

**Preliminary collinearity evidence (n=793 trees, computed for this proposal)** — this is the honest read on
which signals can possibly pass:

| VoI signal | ρ vs **legal** | ρ vs **satisfaction** (`n_good0.1`) | ρ vs **sharpness** (`action_gap`) | verdict |
|---|---|---|---|---|
| `H_fin` (final-belief entropy) | +0.01 | **+0.47** | **−0.87** | ✗ **a reindex of satisfaction/sharpness** — NOT new |
| `breadth_fin` | **+0.42** | +0.03 | — | ✗ a width reindex (control) |
| `H_init` (initial entropy) | +0.09 | +0.19 | — | ~ weak ties; the static "unsure" leg |
| **`ΔH` (info gain)** | **−0.01** | **−0.20** | — | ✓ **orthogonal to width**, partly distinct from satisfaction |
| **`t_stab` (argmax stabilization)** | **−0.01** | +0.30 | — | ✓ **orthogonal to width & sharpness**; a *dynamic* axis, modest satisfaction tie |

> **Result (pre-registered expectation):** the **trajectory** signals (`ΔH`, `t_stab`) are the live
> candidates — they are essentially **uncorrelated with legal-moves**, so they are *not* the width effect, and
> they are only *modestly* tied to satisfaction (|ρ| ≤ 0.30), so a partial can separate them. The
> **final-belief entropy `H_fin` is dead on arrival** — it is −0.87 with sharpness and +0.47 with
> satisfaction, i.e. *the static decomposition wearing an information costume*. **We must NOT headline
> `H_fin`**; doing so would re-discover satisfaction/sharpness and call it VoI.

> **Open question (the test itself, not yet run):** does `ΔH` or `t_stab` clear **both** bars — beat ≈+0.34
> marginally on RT **and** keep partial ρ ≳ +0.1 under the joint legal+satisfaction+sharpness control? If
> yes, VoI is a new axis and the first thing in the project to beat the floor on its own dynamics. If a
> trajectory signal survives the joint partial but is *small* marginally, that is still a **publishable
> dissociation** (a belief-dynamics term the static decomposition misses), even if it does not dethrone width.

---

## 7 · Honest risks (and how each is handled)

1. **VoI collapses into satisfaction/sharpness.** The central risk. The §6 table already shows the *final*
   belief (`H_fin`) is exactly this collapse. **Mitigation:** headline only the **trajectory** signals
   (`ΔH`, `t_stab`, `n_flips`), which §6 shows are decoupled from sharpness and only ~0.3 with satisfaction;
   gate every claim on the **joint partial** on legal+satisfaction+sharpness. If even `ΔH`/`t_stab` die under
   the partial, report the **null honestly** — "belief dynamics add nothing beyond the static satisfied-width
   decomposition" is a real, citable result that tightens Act 3.

2. **The `q_trace[0]` / Q=0 unvisited trap.** A naïve entropy over the padded row measures the
   *un*-evaluated children, and the raw early argmax just walks the visit order. **Mitigation (locked, §2):**
   mask Q=0 to the visited set before any softmax; define `s0` at the first-informative step; use the
   *last-change* step for `t_stab` (immune to the early walk). Validate by confirming `argmax` over the masked
   visited row matches `oracle_best_move_index` only after `s0`.

3. **`t_stab` is bimodal and partly an artifact of late reversals.** 15% never change (`t_stab=0`), 21%
   reverse late (≥80). The late-reversal mass risks being driven by **hindsight**: the oracle's backed-up Q
   can flip the root late on a deep line a human never sees — exactly the reversal/hindsight asymmetry flagged
   in [report.md](../report.md) Act 4. **Mitigation:** (a) report `t_stab` both raw and **capped/winsorized**;
   (b) split easy/late-reversal strata and check the RT correlation **within** the non-reversal stratum (where
   `t_stab` is a clean forward-resolution signal); (c) note that if `t_stab` tracks RT *only* via reversals,
   that is itself evidence *for* the hindsight story, not VoI — keep the two interpretations separate.

4. **`τ`/`δ` dependence.** Entropy and KL depend on temperature; the win-prob scale (gaps 0.01–0.1) makes
   τ=1 degenerate (the softmax-VOC lesson). **Mitigation:** sweep `τ ∈ {0.05, 0.1}`, `δ ∈ {0.05, 0.1}`;
   report the headline at one pair and show the conclusion is stable across the sweep (as the τ-sweep report
   did for softmax-VOC).

5. **Resolution-cost confound.** A position that resolves slowly (`t_stab` high) may *also* be one with many
   legal moves to churn through. §6 says no (`t_stab` ρ −0.01 with legal), but verify at full N and partial
   anyway — `t_stab` must beat the floor *net of* width.

6. **`oracle_root_visits_trace` is missing.** The plan named it; it does not exist. **Mitigation:** the
   non-zero-count of `q_trace` is a faithful visit-breadth substitute (a child's Q is 0 iff unvisited);
   documented and used throughout. No regeneration needed to recover visits.

---

## 8 · Self-assessment — sub-ideas ranked

Ranked by **LIFT** (chance of beating the +0.34 floor *and* surviving the joint partial — a genuinely new
axis), **FEASIBILITY**, **LIKELIHOOD** (that the directional prediction holds at all). All signals are equally
cheap (one shared trace script), so feasibility differs only in interpretability/cleanliness.

| Rank | Sub-idea | LIFT | FEAS | LIKELIHOOD | One-line rationale |
|---|---|---|---|---|---|
| **1** | **`t_stab` — argmax stabilization step** (C) | **High** | High (but bimodal → needs strata) | Medium | The cleanest NEW object: a *dynamic* belief-convergence stopping time, ρ≈0 with both width and sharpness, only ~0.3 with satisfaction → best shot at surviving the joint partial. Risk: late-reversal/hindsight mass. |
| **2** | **`ΔH` / `KL` — information gain** (B) | **High** | High | Medium | The literal VoI quantity, orthogonal to width (ρ−0.01) and only −0.20 with satisfaction. Most defensible as "uncertainty resolved." Risk: may be small marginally even if it survives the partial. |
| 3 | **`n_flips` — argmax churn** (C) | Medium | High | Medium | A robustness sibling of `t_stab`, less hindsight-coupled (counts churn, not the last flip). Good corroborator if `t_stab` passes. |
| 4 | **`H_init` — initial argmax uncertainty** (A) | Medium-low | High | Medium | The "you think when unsure at the outset" leg; static, weakly tied to satisfaction (+0.19). Useful as the *predictor* half of the story, unlikely to beat the floor alone. |
| 5 | **`H_fin` — final-belief entropy** | **Near-zero** | High | High (direction) | **Do not headline.** −0.87 with sharpness, +0.47 with satisfaction → it *is* the static decomposition. Included only to *prove* the trajectory signals are not this. |
| 6 | **`breadth_fin` — live-contender breadth** (D) | Near-zero | High | High | +0.42 with legal — a width reindex. A control, not a candidate. |

**Bottom line.** The honest, evidence-based expectation is that **`t_stab` and `ΔH` are the only two sub-ideas
that can possibly clear both bars**, because they are the only ones that the preliminary collinearities show
are decoupled from *both* width and sharpness. The probe's most likely outcome is one of: **(a)** `t_stab`/`ΔH`
survive the joint partial at a modest ρ (≈ +0.1–0.15) — a publishable belief-dynamics dissociation the static
decomposition misses, even if it does not dethrone width; or **(b)** they too collapse under the partial — a
clean null that strengthens "RT = satisficed width" by ruling out the last plausible value/information rival.
Either is worth the (very cheap) run. The trap to avoid at all costs is reporting `H_fin` as a VoI win: it is
satisfaction/sharpness relabeled, and §6 says so quantitatively.
