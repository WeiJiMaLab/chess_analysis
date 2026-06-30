# When are people actually planning? Conditioning the analysis on genuine deliberation

**Ref:** `R-PLANNING-ENGAGEMENT` · [Index](reference.md) · parents [(R-TREESEARCH)](treesearch.md) / [(R-MOVETIME-MODEL)](engine.md)

> **Status:** 📝 proposal — the plan + the open decisions. **The thesis:** the project's headline number — RT
> tracks decision width (ρ(legal, RT) ≈ +0.34) and every engine value-of-computation (VOC) signal collapses to a
> legal-moves proxy — is computed by **averaging over a mixture**: moves where the player is genuinely
> *deliberating* (building a search tree) and moves where they are merely *acting on a plan formed earlier* (a
> recapture, a forced reply, executing a known sequence). If the engine VOC signals look weak because they are
> diluted by the habitual majority, then **on the deliberation subset they should strengthen** — and the
> legal-moves floor should weaken. This report defines 2–3 **non-circular** engagement proxies (derivable from the
> board / FEN sequence, never from RT), specifies the stratified re-analysis on the **existing** data, and pins the
> one discriminating test against the +0.34 floor. It is honest up front that two of the three proxies look likely
> to collapse, and says which one carries the live shot.

---

## Why this is distinct from the three ruled-out accounts

Three generative "effort" stories have already collapsed to decision-width, all by the same diagnostic
(partial-controlling-for-legal-moves ≤ +0.04): **value-of-computation** ([(R-TREESEARCH)](treesearch.md) H1–H5),
**value-pruning the cost profile** ([(R-PRUNING)](pruning.md)), and **construal complexity**
([(R-CONSTRUAL)](construal.md)). Each tried to find a *better cost or value signal* on the **same population of
moves**. This inquiry does **not** propose a new signal. It proposes a **new partition of the moves**, and asks
whether the *already-computed* signals behave differently inside the deliberation stratum.

> **Decision:** this is a **conditioning / selection** experiment, not a new-signal experiment. We reuse the
> existing VOC parquet (`sf_analysis/<set>/voc_signals.parquet`) and the existing RT decomposition verbatim; the
> only new object is a per-move **engagement label** computed from the game's FEN sequence. If conditioning is the
> missing ingredient, the *existing* signals strengthen on the engaged subset — no regeneration, no new model.

The sharp scientific claim it could earn: **"people plan, but only on the engaged moves, and averaging over the
habitual majority masked it."** That converts "RT re-describes decision width" into "decision width dominates
*because most moves are not deliberated*; where they are, normative search signals re-emerge." That is a genuine
reframe, not another collapse — *if* the prediction holds.

> **Clarification (the circularity trap, stated once, up front).** "Deliberation" must **not** be defined by RT or
> by anything monotone in RT (move-quality, the engine's stop-step, "the player thought hard here"). If engagement
> were defined as "long RT," then re-running ρ(legal, RT) inside "high engagement" is conditioning on the outcome
> — guaranteed to distort the floor for trivial reasons. **Every proxy below is computed from the board / move
> sequence only.** This is the load-bearing constraint of the whole inquiry.

---

## The core hypothesis

> **H-ENGAGE.** Human RT is a mixture over two move-types: **deliberated** moves (a search tree is built; RT
> reflects the search) and **executed** moves (a plan/heuristic supplies the move; RT reflects motor + recognition
> latency, near-floor). The aggregate ρ(VOC, RT) ≈ +0.04-over-legal and ρ(legal, RT) ≈ +0.34 are **mixture
> averages**. On the deliberated subset, the engine VOC signals (Gain, OSS/step\*, action-gap, softmax-VOC)
> **strengthen**, and the legal-moves floor **weakens** (the move set is no longer the dominant driver once you
> condition on positions where the *value landscape*, not the bare count, is what is being weighed).

The mechanism, in the project's own umbrella terms ([(R-TREESEARCH)](treesearch.md) §8): the legal-moves floor is
the *enumeration cost of the consideration set*. On an **executed** move you do not enumerate — you already know
the move — so RT is near-constant and the leaf-count cannot drive it; the +0.34 is built almost entirely from the
**deliberated** moves where enumeration actually happens. The interesting and **non-obvious** part is the second
half: does conditioning *also* let the *value* signals (which the oracle uses, but humans appear to ignore in
aggregate) re-emerge? That is the part that could beat the floor.

---

## Operationalization on the existing data

All proxies are computed from `processed_moves_nonzero` ordered by `(gid, move_ply)` — no trees, no RT. The
position-level engine signals are joined from the existing VOC parquet by 4-field FEN. **No regeneration.**

### The data we have to work with (verified)

`processed_moves_nonzero` carries, per move: `gid`, `move_ply`, `n_possible_moves` (= legal moves, the floor
variable), `n_pieces_on_board_inc_pawns` (piece count), `fen` (4-field, the join key), `move_time`, `ply_tertiles`,
clocks, piece-subset counts. There is **no `move_uci` / SAN column** in this table — so capture/check/only-move
flags must be **derived from the FEN sequence**, not read off a move string. The two clean derivations:

- **Capture on the prior ply** ⇒ the piece count *dropped* from ply *p−1* to *p*:
  `n_pieces[p] < n_pieces[p−1]` (a `lag` window over `(gid, ORDER BY move_ply)`). Verified on a 6K-game sample:
  **22%** of moves follow a capture; the flag is exact (piece count is monotone non-increasing within a game except
  for the capture drop). *Promotions and en-passant are rare edge cases that perturb the count by ≤1 and do not
  affect the stratum at scale.*
- **Forcedness / only-move** ⇒ `n_possible_moves` is small (the player has few legal replies). A check that leaves
  one legal escape is the extreme; `legal ≤ 2` is "essentially forced," `legal ≤ 8` "constrained." This is read
  directly off the existing column.
- **Moves since last capture (quiet-stretch length)** ⇒ run-length within a game since the last piece-count drop:
  `row_number()` within a "block" id that increments on each capture. Verified computable in one DuckDB pass;
  quartiles of the run length are [1, 3, 7, 13].

> **Decision:** all three engagement proxies are **pure SQL window functions** over the existing table. No FEN
> re-parsing with `python-chess` is required for the headline proxies (capture-delta + legal-count + run-length),
> which keeps the whole thing a sub-minute DuckDB query on the full 135M rows. (Checks/recaptures *as such* would
> need FEN parsing — deferred to a refinement, see Risks.)

### The three principled proxies (and how each maps to acting-vs-planning)

| # | Proxy | "Executed / acting" (low engagement) | "Deliberated / planning" (high engagement) | Derivation |
|---|---|---|---|---|
| **P-CAP** | prior-ply capture | the prior move was a capture ⇒ this move is often a **recapture / forced response** | quiet position, no capture to answer ⇒ **free choice**, must construct a plan | `n_pieces[p] < n_pieces[p−1]` |
| **P-QUIET** | quiet-stretch length | mid-exchange / tactical flurry (short run since last capture) ⇒ riding a sequence | long quiet stretch (large run) ⇒ **maneuvering / staging**, the planning regime | run-length since last capture |
| **P-FORCED** | reply width (forcedness) | few legal moves ⇒ **executing** the only reasonable reply | many legal moves ⇒ a genuine branch point to weigh | `n_possible_moves` (the floor var itself) |

> **Decision (which proxies are principled, which are kept only as a foil).** **P-CAP is the cleanest** non-circular
> deliberation marker: it is causally about the *prior* ply (so it cannot be a re-description of the *current*
> position's width), and empirically it barely re-indexes legal-moves (mean legal 29.4 post-capture vs 30.4
> elsewhere — see Risks). **P-QUIET** is the staging intuition and is principled but its signal looks thin (below).
> **P-FORCED is deliberately included as a foil, not a clean proxy** — it is *defined on the floor variable
> itself*, so stratifying RT on it and then measuring ρ(legal, RT) is mechanically circular (restricting `legal`'s
> range kills its correlation by construction). We keep P-FORCED only to *demonstrate* that circularity and to
> separate it from the real result; **the headline test rides on P-CAP.**

### The exact stratified re-analysis

For a chosen binary engagement label `E ∈ {executed, deliberated}` (start with **P-CAP**: deliberated = prior ply
was *not* a capture):

1. **Join** the per-move table (with `E`, `legal`, `log move_time`) to `voc_signals.parquet` on 4-field FEN — the
   same join `prune_regen_analyze.py` / `construal_proxy.py` already do, so the harness is in hand.
2. **Within each stratum** compute, with percentile-bootstrap 95% CIs ([[bootstrap-cis-always]]):
   - **the floor:** ρ(legal_moves, RT);
   - **the VOC signals:** ρ(Gain `gain_costfree`, RT), ρ(OSS / step\* `oss__…`, RT), ρ(action_gap, RT),
     ρ(softmax_voc, RT) — and crucially each one's **partial-controlling-for-legal-moves** (the H4 diagnostic that
     killed them in aggregate);
   - **the satisficing decomposition:** ρ(`n_good_*`, RT) (satisfaction), ρ(action_gap, RT) (sharpness), so we can
     re-read `size − satisfaction + sharpness` within each stratum.
3. **The contrast that is the result:** does ρ(VOC | legal) rise from ≈ +0.04 (aggregate) to something material in
   the **deliberated** stratum, while ρ(legal, RT) falls? Report the deliberated-minus-executed difference in each
   correlation, with a bootstrap CI on the **difference** (paired resample within the joined table).

> **Decision:** the primary endpoint is the **partial-controlling-for-legal-moves** of the VOC signals *inside the
> deliberated stratum*. The aggregate value of that partial is the ≈ +0.04 that defined the null. Beating it
> *within* a non-circular stratum (P-CAP) is the only thing that counts as "planning re-emerged."

This is a faithful copy of the existing `compute_voc_signals` → RT-correlation pipeline with one extra `GROUP BY E`.
A new script `lmcos_tiny/analysis/engagement_strata.py` (modeled on `construal_proxy.py`'s `analyze()` +
`prune_regen_analyze.py`'s DuckDB join and bootstrap helpers) is the whole implementation.

---

## Feasibility

| quantity | number | note |
|---|---|---|
| engagement labels | **1 DuckDB pass**, window functions | sub-minute on 135M rows; verified on a 6K-game sample (407K moves) in ~2 s |
| VOC join | reuse existing `voc_signals.parquet` (250K FENs) | same FEN join as `construal_proxy.py`; ~260K matched human moves |
| new code | **one script** (`engagement_strata.py`) | bootstrap + partial helpers copied from existing analysis files |
| regeneration | **none** | no trees built, no model trained — pure analysis on existing parquet + DB |
| compute | **a single sbatch CPU task** (DuckDB join wants memory, per the prune script's note) | minutes, not hours |

> **Result (feasibility):** this is the **cheapest** experiment in the deliberation thread — strictly a DuckDB +
> existing-parquet re-analysis, one new script, no generation. It can be run and decided in an afternoon.

---

## Predictions and the one discriminating test

**If H-ENGAGE is true** (planning was masked by mixing):

- **P1 (the discriminating test).** In the **P-CAP deliberated** stratum (prior ply not a capture), the VOC
  partial-|-legal rises materially above the aggregate +0.04 — and the deliberated-minus-executed difference in
  ρ(VOC | legal) has a bootstrap CI excluding 0. **This is the one test that matters.** Everything else is
  supporting.
- **P2.** ρ(legal, RT) is **lower** in the deliberated stratum than in the executed stratum — the floor weakens
  where value enters.
- **P3.** The satisficing decomposition sharpens in the deliberated stratum: satisfaction (`n_good`) and sharpness
  (action_gap) carry *more* of RT there.

**If H-ENGAGE is false** (the project's modal expectation, stated honestly): the VOC partial stays ≈ +0.04 in
*both* strata; ρ(legal, RT) is roughly flat across P-CAP strata (the 6K-game pilot already shows only +0.26 vs
+0.22 — a real but small shift); and the only stratum where the floor "collapses" is P-FORCED, where it collapses
**mechanically** (range restriction on `legal`), not because planning re-emerged.

> **The single discriminating contrast vs the +0.34 floor:** does the VOC **partial-|-legal** clear, say, +0.10 in
> the **P-CAP-deliberated** stratum (a >2× lift over the +0.04 null), with a CI excluding the executed-stratum
> value? Yes ⇒ "planning re-emerges on engaged moves" is a real result. No ⇒ the mixture explanation is ruled out
> too, and decision-width-dominance is confirmed as *not* a mixing artifact — itself a clean, publishable negative
> that closes a live objection to the headline finding.

---

## Honest risks

- **Circularity with RT (the existential risk).** Any engagement proxy that is even partly a function of RT
  poisons the test. P-CAP, P-QUIET are RT-free by construction (board sequence only). **P-FORCED is reported only
  as the cautionary foil** — it is the floor variable itself, so its "collapse" is mechanical range-restriction,
  not evidence (pilot: ρ(legal, RT) = +0.02 in `legal ≤ 8`, exactly because `legal` barely varies there). We must
  not let P-FORCED's collapse be mistaken for the real prediction.
- **Engagement re-indexing legal-moves / phase (the collapse risk).** If "deliberated" just means "more legal
  moves" or "middlegame," then stratifying is stratifying on the floor and we learn nothing. **Pilot evidence is
  reassuring for P-CAP:** mean legal is 29.4 (post-capture) vs 30.4 (else) — essentially no re-indexing of width —
  and the capture flag is about the *prior* ply, so it is not a current-width restatement. **But** captures
  cluster in the middlegame, so the stratum partly co-varies with `ply_tertiles`; the analysis must **also report
  the VOC partials controlling for *both* legal-moves and ply** to be safe.
- **P-QUIET looks thin (an honest near-null in hand).** Pilot: long quiet stretches (≥4 since capture) move
  ρ(legal, RT) only +0.26→+0.25 and barely shift RT — the "staging ⇒ planning" intuition shows little leverage at
  the proxy level. Keep P-QUIET as a secondary, low-prior probe; do not headline it.
- **The motor-floor confound cuts the *right* way but must be acknowledged.** Post-capture moves are faster (median
  3 vs 5 ms in the pilot), consistent with "executing." But "faster" could be pure motor latency, not "less
  search." That is fine for P1 (P1 is about the *VOC correlation*, not the RT *level*) — but any claim about RT
  *levels* across strata must flag it.
- **Definitional reductiveness of "deliberation."** A prior-ply capture is a *proxy* for executing-a-plan, not a
  measurement of it; a player may deliberate hard over a recapture (which piece to take with). The proxy will have
  substantial label noise, which **biases the stratum contrast toward null** — so a *positive* P1 result would be
  conservative (real effect ≥ measured), while a null is only suggestive, not decisive. State this asymmetry.
- **Off-book / novelty proxy is deferred.** "Leaving theory" is a strong engagement marker but needs an opening
  book to score; not derivable from the existing table. Flagged as a future refinement, not in the cheap core.

---

## Open question

> **Open question (not a result):** is decision-width-dominance a **mixture artifact** (planning is real but
> diluted by habitual moves) or a **genuine population fact** (width dominates even where people deliberate)? This
> inquiry answers it with one non-circular stratification (P-CAP) and the VOC-partial-|-legal contrast. The pilot
> already hints the effect is **small** — so the most likely outcome is a *clean negative that defends the headline
> finding against the mixing objection*, with a residual chance that P-CAP's deliberated stratum shows the first
> material VOC lift in the project. Either way it is cheap, non-circular, and decisive on a question the synthesis
> paper currently leaves implicit.
