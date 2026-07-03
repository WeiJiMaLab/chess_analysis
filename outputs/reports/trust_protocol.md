# Trust protocol — pre-registered tests, plot standards, and go/no-go rules

**Format: protocol / engineering doc.** **Run: `minply15_maxply75`.** This document governs every
analysis in [plan_final_push.md](plan_final_push.md) (P0–P3). It is written *before* the analyses
run. The rule is simple: **no curve is reported, and no sign enters a conclusion table, unless it
has passed the checks below.** Fails are findings — a signal that dissolves under an invariance
check is a *result about the signal*, and gets written up as such, not dropped.

**Why this exists.** This project was burned repeatedly by plausible-looking artifacts: a single
transposition FEN (1,111 replayed instances) loading >10% of one quantile bin and carving a fake
dip; signals with 15–58% of their mass on a single value making quantile bins jitter; a pooled
RT curve that was jagged across legal-move strata and flat within them; mean trends that wiggled
where the median was flat; "structure" that appeared only under a log x-axis; bin spikes no larger
than the bin CI; and pwin saturation (±1 rails at |cp|≳400, ~46% of root children) manufacturing
zero-inflation downstream. Every rule below traces to one of these.

**Conventions assumed throughout.** RT analyses are on **log RT** (think time is log-normal —
bankable). All CIs are **95% percentile-bootstrap** — never normal-theory. Spearman *and* Pearson
are always reported together. Engine metrics are side-burnered unless they pass P1/P2.

---

## What must every RT-vs-x figure show before we believe it?

The unit of trust is the *figure*, because that is where the artifacts fooled us. Every reported
RT-vs-signal figure must satisfy all of the following. A figure missing any item is a draft, not
evidence.

**The checklist:**

1. **Marginal histogram of x** directly under (or beside) the trend panel, on the *same x-axis*.
   You cannot judge a trend without seeing where the data lives — the log-x wiggle artifact and
   the point-mass jitter were both invisible without the marginal.
2. **Point-mass annotation.** If any single value of x holds >5% of the mass, the figure prints
   the value and its share (e.g. "voc==0: 31%"). If the share exceeds 10%, the hurdle rule
   (below) applies before the trend is interpreted at all.
3. **Binning scheme stated in the caption**, and appropriate to the variable's type:
   - integer / discrete / point-massed signals → **integer (or fixed-width) bins with
     `min_bin_count = 300`** (the established house scheme); a bin never straddles a point-mass;
   - genuinely continuous signals → quantile bins are acceptable, but the invariance battery
     (below) requires an integer/fixed-width cross-check anyway.
4. **Median trend primary, mean trend shown too** (mean may go to an appendix panel). RT is
   right-skewed; the mean-wiggle/median-flat episode is disqualifying for mean-only plots. If
   the two disagree qualitatively, the figure says so and the median is what gets quoted.
5. **Bootstrap CI bands on every binned trend** (percentile; see the clustering rule below), and
   **n per bin printed** (or encoded as marker size with the mapping stated). A "spike" whose
   height is within the neighboring bins' CI width is noise by definition — we measured across-bin
   spans of 0.2–0.6 log-RT against per-bin CI widths of ~0.065, so most visual texture is noise.
6. **Instance and FEN counts in the caption**: total instances, unique FENs, and the max
   single-FEN share of any bin (see invariance check (d)).
7. **No claims from monotone transforms of x.** Log/logit axes are display choices, stated in the
   caption. All *claims* are made on bin statistics (which are invariant to monotone x-transforms)
   — curvature that "appears" only after stretching the axis was never in the bins.

> **Decision:** A figure is reportable iff it carries the marginal histogram, point-mass
> annotations, a stated type-appropriate binning scheme, median+mean trends with percentile-bootstrap
> CI bands, per-bin n, and instance/FEN counts. Transformed axes are display-only.

---

## What does a curve have to survive before we call it real?

The invariance battery. Each check targets a failure mode we actually hit. Every check is cheap
(a re-aggregation or a re-run on cached joins — minutes, not hours) and is run for **every**
signal whose sign or shape we intend to report. Results go in a small per-signal table
(check × pass/fail) in the report appendix.

### (a) Per-instance vs per-FEN aggregation *(targets pseudoreplication)*

Recompute the statistic (correlation, partial, or binned trend) with each FEN collapsed to one
row (FEN-level x, FEN-median log RT), alongside the per-instance version.

> **Decision:** PASS iff the sign agrees between units **and** the per-FEN CI excludes zero.
> Per-FEN is the conservative unit: an effect that exists only per-instance is, until shown
> otherwise, a replay-weighting artifact. (A per-FEN-only effect is also suspicious — diagnose
> before reporting either.)

### (b) Binning-scheme change *(targets point-mass/quantile jitter)*

Recompute the binned trend under the other scheme (quantile ↔ integer/fixed-width with
`min_bin_count = 300`).

> **Decision:** PASS iff the qualitative reading is unchanged: the sign of the bin-mean trend
> (Spearman over bin medians) agrees, and for shape claims the peak bin moves by at most one bin.
> Wiggle that survives only under one scheme is binning artifact, not signal.

### (c) Split-half replication by game id *(targets overfitting to one sample's noise)*

Split games into two halves by a hash of game id (fixed seed, recorded); rerun the statistic in
each half independently.

> **Decision:** PASS iff both halves give the same sign **and** the full-sample CI excludes zero.
> If the halves disagree in sign, the effect is unreplicated — report as null regardless of the
> pooled p-value. (We do not require each half's CI to exclude zero — halving n costs power and
> that stricter rule would fail true small effects.)

### (d) Top-FEN influence *(targets the 1,111-instance transposition FEN)*

For every bin of every reported binned trend, compute the max share of the bin's instances
contributed by a single FEN. Additionally, for the headline statistics, recompute leave-one-FEN-out
over the top-10 FENs by instance count.

> **Decision:** PASS iff (i) no bin has any single FEN contributing ≥5% of its instances, and
> (ii) no top-10 leave-one-FEN-out recomputation flips the sign or moves the estimate by more than
> half its CI width. If (i) fails, the figure must be rebuilt per-FEN (or with FENs down-weighted
> to one vote) before any interpretation.

### (e) Within-strata consistency *(targets Simpson/pooling across legal-move levels)*

Recompute the RT relationship inside low/mid/high `n_legal_moves` tertiles.

> **Decision:** PASS iff the sign is the same in all three tertiles. A tertile whose CI covers
> zero is tolerated (power); a tertile with the *opposite* sign and a CI excluding zero is an
> automatic FAIL — the pooled curve is then a mixture artifact and only the stratified result may
> be reported. (This is exactly how the jagged pooled curve / flat strata episode should have
> been caught.)

> **Clarification:** The battery is conjunctive: a signal must pass **all five** to be reported
> as a positive finding. A signal failing any check is still reported — as the corresponding
> artifact diagnosis. That is a methods finding, and those have been among our most useful
> results (e.g. pwin saturation).

---

## How do we report statistics without fooling ourselves?

**Bootstrap CIs, clustered by FEN.** All CIs are percentile-bootstrap. Because instances of the
same FEN are not independent (pseudoreplication), the default bootstrap **resamples FENs, not
instances** (cluster bootstrap: draw FENs with replacement, keep all their instances). B = 1,000
for any number that appears in a report; B = 200 is acceptable while iterating. *(Judgment call:
the FEN-cluster default is new — instance-level bootstrap CIs are anti-conservative under
replay clustering and are what let the fake dip look significant.)*

**Spearman and Pearson, always both, with CIs.** No global swap. When they disagree — operationally,
when their CIs are disjoint or their signs differ — neither may be quoted until a **mandatory
diagnosis** is run and recorded:

1. *Ties:* report the fraction of tied x values. If >10% of mass sits on single values, Spearman
   is degraded by construction (rank collapse) — weigh Pearson-on-transformed or the binned trend.
2. *Tail leverage:* recompute Pearson with x and log RT winsorized at the 1st/99th percentiles.
   If winsorizing moves Pearson toward Spearman, the raw Pearson was tail-driven — quote the
   winsorized value with that label.

The divergence itself is diagnostic and gets one sentence in the report.

**Partials vs `n_legal_moves` by default.** Legal moves is the dominant RT predictor (ρ≈+0.26,
bankable) and sits *mechanically* inside several signals (frac_good = #good/#legal). Every new
signal reports its raw correlation **and** its partial vs `n_legal_moves` side by side. A sign
flip under partialling is not an error — it is the finding (as with frac_good) — but the raw sign
alone is never quoted.

**Hurdle decomposition for point-massed signals.** Any signal with >10% of mass on a single value
gets a mandatory two-part treatment before its trend is interpreted:

- *Part 1 (hurdle):* is `x == mass-value` associated with RT? (binned median log RT for the mass
  group vs the rest, with CIs; or equivalently a dummy regressor.)
- *Part 2 (continuous):* the RT trend on the off-mass instances only.

A pooled trend over a point-massed signal conflates these two effects (the voc==0 / decided-position
episode: the zero-mass was 6.8× enriched in decided positions — the "trend" was mostly the hurdle).

**Effect sizes as ΔR², not raw ρ.** Every candidate signal's headline effect size is the
**increment in R² for log RT over the baseline model `n_legal_moves + ply + clock`** (OLS,
per-instance for power, per-FEN as the confirmatory unit). Raw correlations flatter signals that
are mobility in disguise — the engine-value battery had respectable raw ρ and ΔR² ≤ 0.0003.

> **Decision:** Thresholds (for the go/no-go table): **ΔR² ≥ 0.005** = a real independent
> contribution (action_gap's ≈0.014 comfortably clears it); **0.001–0.005** = marginal, report
> but do not build on; **< 0.001** = collapsed onto the baseline, reported as such. *(Judgment
> call — these cut points bracket our observed cases but the user should ratify them.)*

---

## How do we confirm a pre-registered shape (the ∩ for material_imbalance)?

The P0 star hypothesis is that RT peaks near `material_imbalance == 0` and declines in
|imbalance| (engine-free decidedness). "It looks like a hump" is not a criterion — here is the
one we commit to. Imbalance is integer-valued, so this all runs on integer bins
(`min_bin_count = 300`) with FEN-cluster bootstrap CIs on bin medians of log RT.

**Test 1 — binned separation (the primary test).**
Define the *peak region* as imbalance ∈ {−1, 0, +1} and the *tail regions* as imbalance ≤ −3 and
≥ +3 (bins between are a buffer, not scored).

> **Decision:** The ∩ is **confirmed** iff the peak-region bins' CI *lower* bounds lie above the
> CI *upper* bounds of at least one bin in **each** tail region — i.e., the peak is credibly above
> both the ahead side and the behind side. Above only one tail ⇒ classified **monotone** (in the
> direction of the surviving decline), not ∩. Neither ⇒ **null/flat**.

**Test 2 — quadratic-vs-linear model comparison (the confirmatory scalar).**
Fit log RT ~ imbalance + imbalance² (per-instance, with the same FEN-cluster bootstrap on the
coefficients).

> **Decision:** Supports the ∩ iff β(imbalance²) < 0 with its 95% bootstrap CI excluding zero
> **and** the quadratic term adds ΔR² ≥ 0.001 over the linear fit. (The ΔR² floor stops us
> declaring a "shape" that is statistically nonzero but visually and practically flat.)

**Verdict rule.** Both tests pass ⇒ **∩ confirmed** (report language: "confirmed, pre-registered").
Exactly one passes ⇒ **suggestive** — reported with the failing test shown, never upgraded by
narrative. Neither ⇒ the pre-registered hypothesis is disconfirmed; the observed classification
(monotone/null from Test 1) is the finding. The shape verdict is only valid if the underlying
figure also passed the invariance battery — a ∩ carved by one transposition FEN is check (d)'s
job to catch, not Test 1's.

The same two-test template applies to any future shape hypothesis: pre-register the peak/tail
regions and the polynomial contrast *before* looking.

---

## What exactly must the P1 sign-mirror show?

The engine-metric trust criterion, formalized as pre-registered pass/fail. Rationale: `action_gap`
(gap between best and second-best move value) and `frac_good` (fraction of moves near the best)
are antithetical *by construction* — a decisive position has a large gap and few good moves. If
the engine's values carry real RT signal, their partial correlations with RT net of
`n_legal_moves` must mirror. Current evidence (+0.04 frac_good vs −0.05 action_gap, per-instance,
pwin units) is consistent but provisional.

> **Decision (P1 trust criterion):** Engine metrics graduate off the side burner iff, on the
> windowed [15,75] join, **all four** hold:
> 1. partial(action_gap, log RT | n_legal_moves) < 0 and partial(frac_good, log RT | n_legal_moves) > 0;
> 2. the two partials' 95% FEN-cluster bootstrap CIs each exclude zero **and do not overlap each
>    other** (mirrored with credible separation);
> 3. (1)–(2) hold in **both** aggregation units: per-instance and per-FEN;
> 4. the mirror survives the within-tertile check (e): no legal-move tertile shows a
>    CI-excluding-zero sign reversal for either metric.
>
> PASS ⇒ action_gap (and the decisiveness construct) enters P3's sign table as *trusted*;
> frac_acceptable and cp-unit variants (P2) are then worth computing as refinements.
> FAIL ⇒ the entire engine-value section stays honest-negative ("engine values collapse onto move
> count"), P2's *interpretive* payoff is downgraded to a methods comparison (pwin vs cp point-mass
> fractions), and P3 runs on board features only. Either way we ship a definite statement.

`frac_acceptable` (P1.2) is evaluated by the same standard as any new signal: distribution check
first (point-mass share → hurdle rule if >10%), then partial vs legal moves, then the invariance
battery, then ΔR² against the baseline.

---

## The go/no-go table — one row per planned analysis

Every planned test, its pass rule, and what each outcome *means*. This table is the contract:
after the run, each row gets a verdict column and nothing else changes.

| # | Analysis (plan ref) | Test / statistic | PASS rule (pre-registered) | On PASS conclude | On FAIL conclude |
|---|---|---|---|---|---|
| G1 | P0 `in_check` | raw + partial vs legal moves | battery (a)–(e); ΔR² tier per thresholds | partial ≈0 ⇒ size account complete; partial + ⇒ stakes add beyond size | raw sign wrong or battery fail ⇒ check effect is artifact/mobility |
| G2 | P0 `n_captures_avail` | partial vs legal moves > 0, CI excl. 0 | battery + ΔR² ≥ 0.001 | inclusion is salience-weighted (composition matters) | composition null: consideration set is size-only |
| G3 | P0 `n_checks_avail` | partial vs legal moves > 0, CI excl. 0 | battery + ΔR² ≥ 0.001 | same, stronger (forcing moves prioritized) | same null; if G2 passes and G3 fails, diagnose before narrating |
| G4 | P0 `self_material` | partial vs legal moves > 0 | battery + ΔR² ≥ 0.001 | depth component beyond root width | width-dominated RT — itself informative, report as such |
| G5 | P0 `material_imbalance` ∩ | shape Tests 1+2 above | both tests + battery | engine-free decidedness/VOC confirmed — the battery's star | monotone ⇒ being-ahead/behind story; null ⇒ decidedness not board-readable; either is reportable |
| G6 | P0 `n_own_pieces_attacked` | partial vs legal moves, sign | battery; + = pre-registered | threat-verification (response-set inflation) | − with CI excl. 0 ⇒ forced-narrowing (arbitrate with G1's mediation); 0 ⇒ null |
| G7 | P0 `prev_move_was_capture/check` | raw sign − | battery + CI excl. 0 | forcing prior move constrains the reply set | prior-move context doesn't reach RT |
| G8 | P1 sign-mirror | the 4-part criterion above | all 4 clauses | engine values trusted; decisiveness row in P3 unlocked; P2 interpretive | engine section stays honest-negative; P3 board-only; P2 = methods comparison |
| G9 | P1 `frac_acceptable` | distribution → hurdle if needed → partial | battery + ΔR² tier | absolute-aspiration satisficing signal exists, distinct from ε-from-best | satisficing not measurable in these values (or eaten by saturation → P2) |
| G10 | P2 cp re-run of G8 | same 4-part criterion, cp units | as G8; plus predicted point-mass drop (action_gap zeros 19%→~2%) | pwin verdict robust to units; saturation was noise not distortion | if cp flips the G8 verdict, cp wins (native units) and the pwin result is re-labeled saturation artifact |
| G11 | P3 size row (`n_legal_moves`) | model stop-step sign vs human + | sign match, both CIs excl. 0 | model replicates size effect (Medium tier support) | model fails the *easiest* row ⇒ Medium tier not reached; report at Minimum |
| G12 | P3 satisfaction row (mq) | model stop-step sign vs human − | sign match, both CIs excl. 0 | model replicates satisfaction | as G11 |
| G13 | P3 decisiveness row (action_gap) | only runs if G8 (or G10) PASSED | sign match − in model and human | model replicates decisiveness — strongest Medium evidence | row reported "provisional/blocked" — acceptable per plan |
| G14 | P3 board-correlate rows (from P0 passes) | model stop-step sign vs human, per passing feature | sign match per row | each match strengthens the qualitative-replication claim | mismatches are the *open-reconciliation* content — listed, not hidden |

**Reading the table.** "Battery" always means all five invariance checks. Row verdicts are
per-row: a G5 fail does not contaminate G2. The Medium-tier claim in the final report requires
G11 **and** G12 to pass; G13/G14 modulate its strength. No row's FAIL is ever silently dropped —
the fail-column text is the sentence that goes in the report.

> **Decision:** This table is frozen as of 2026-07-02, before P0 execution. Any added analysis
> gets a new row *before* it runs; no row's pass rule is edited after its data has been seen.

---

## Appendix — the thresholds in one place (ratify these)

| Parameter | Value | Basis |
|---|---|---|
| CI method | 95% percentile bootstrap, **clustered by FEN**, B=1,000 (200 while iterating) | pseudoreplication; house rule + clustering extension |
| Point-mass annotation | >5% at one value | figure legibility |
| Hurdle trigger | >10% at one value | observed 15–58% masses all far above it |
| Discrete binning | integer/fixed-width, `min_bin_count = 300` | established house scheme |
| Top-FEN bin share | <5% per bin | the fake-dip FEN was >10% |
| Leave-one-FEN-out | top-10 FENs; no sign flip; shift < ½ CI width | cheap influence audit |
| ΔR² baseline | `n_legal_moves + ply + clock` on log RT | legal moves is the confound to beat |
| ΔR² tiers | ≥0.005 real / 0.001–0.005 marginal / <0.001 collapsed | brackets action_gap (0.014) vs collapsed battery (≤0.0003) |
| Spearman–Pearson disagreement | disjoint CIs or sign difference → mandatory ties+winsorize diagnosis | divergence is diagnostic |
| Winsorization (diagnosis only) | 1st/99th percentiles | tail-leverage probe, never the headline |
| ∩ peak / tails | peak {−1,0,+1}; tails ≤−3 and ≥+3; quadratic ΔR² ≥ 0.001 | simple, defensible, stated in advance |
| Split-half | by game-id hash, fixed recorded seed | replication without new data |

Everything above runs on cached joins and existing figures code; nothing requires new data
collection. Estimated overhead per signal: one re-aggregation, one re-binning, one split, one
influence scan, one stratified rerun — minutes each.
