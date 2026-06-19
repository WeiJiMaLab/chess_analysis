# diagnose_gain.md — Part 2: the Gain≈1.0 spike, the narrow-CI puzzle, and CI method audit

**Status:** DONE. The diagnosis is complete and its conclusions are baked into the production
figure standard (`utils/analysis.py`: LOWESS + bootstrap band for continuous predictors,
native-integer for discrete) and the kept tests (`tests/test_gain.py`, `tests/test_jaggedness.py`).
The one-off **diagnostic drivers this doc references have been removed** now that their findings
are captured here and in the figure standard — `gain_dip_diagnostic.py`, `gain_ci_audit.py`,
`gain_binning_diagnostic.py`, `jaggedness_diagnosis.py`, and `utils/ci.py` / `tests/test_ci.py`
(the analytic-with-guards CI was superseded by the LOWESS bootstrap band, so it was never wired
into production). The narrative below is preserved verbatim as the record of findings; its
links to those scripts are historical.

Related: [[partition]], [[tree-set-and-gss]], [[report-deck-format-convention]].

---

## 0. The history of the incorrect Gain definition (confirmed against code + git)

Gain (column `voc`, displayed **"Gain"**) is meant to be the value a deep search
adds over a shallow one at the root: `final_Q(deep best) − final_Q(shallow best)`,
both root side-to-move Qs ∈ [−1, +1].

**The two buggy definitions (pre-`c903646`).** Both read *both* terms from
`oracle_final_root_q_values` — the converged, full-budget (96-expansion) root-Q
vector:
- `a_deep = argmax(final_Q)` — the deep winner. Fine.
- `a_shallow` = the 1-ply value-head pick, `argmax(−child.value)`. **Bug:** lc0
  pours its entire 96-expansion budget into the 1–2 winning root moves, so the
  shallow pick is frequently a root move the deep search **never visited**. Its
  slot in `oracle_final_root_q_values` is the **uninitialized 0.0**, not a real Q.
- Result: `Gain = final_Q(deep ≈ +1, a proven win) − 0.0 ≈ +1.0`. A spurious
  point-mass at **Gain ≈ 1.0**, concentrated in saturated / endgame positions
  (one move is a proven win; the rest are abandoned).

This is exactly the failure you described — "T_shallow picks `a_shallow` that
T_deep never expanded → defaults to 0 → `1 − 0 = 1`." Mechanically it was a
*single*-tree lookup of an unexpanded move's 0.0 (not literally two trees), but
functionally identical. The earlier of the two definitions additionally read an
all-zero step-0 trace; both routes produced the same ≈1.0 artifact. The dedicated
diagnostic [gain_dip_diagnostic.py](human_analytics/gain_dip_diagnostic.py)
documents the saturation signature (`deep ≥ 0.99 AND |shallow| ≤ 0.05`) and the
`shallow_q_is_unvisited_zero` root cause.

**The fix (`c903646`, "redefine Gain (growing-tree")**, in
[_tree_voc_and_gap](human_analytics/tree_values_analysis.py#L75): read from **one
growing tree** —
- `a_shallow = argmin(first_visit)` — the **first move the search actually
  expanded**, hence guaranteed visited with a real converged `final_Q`.
- `a_deep = argmax(final_Q)`.
- `Gain = final_Q[a_deep] − final_Q[a_shallow] ≥ 0` by construction.

This is your `sn_96(a_96) − sn_96(a_0)`: both moves scored under the *same* final
(96-expansion) Q; `a_0` = the snapshot-0 first expansion, `a_96` = the converged
best. The ≈1.0 mass disappears; the labnotebook reports `r = +0.085`, monotone, no
filtering needed.

---

## 1. The narrow-CI puzzle — what we think is going on

The plot CI is the **analytic normal SEM**: `1.96 · stddev / √n`, computed in
DuckDB as `stddev(y)` and `count(*)` per quantile bin
([analysis.py:366–386](human_analytics/utils/analysis.py#L366-L386)).

Your worry: a *spurious/random* effect should make the CI **blossom**, not narrow
to a point. The narrowing therefore implies either (1) those positions are genuinely
special, or (2) the CI method is flimsy. **Our read is a third thing, and it
reconciles both:**

- The tie-safe top bin **lumped the entire artifact point-mass at Gain≈1.0 into a
  single bin**, so its `n` was enormous → SEM → tiny. The mass was also low-RT
  (proven wins / endgames are fast), so the bin **mean was pulled down** *with* a
  tight band.
- So the CI was **not** wrong as a formula — it faithfully reported the precision
  of a **contaminated, artifactually huge-`n` bin**. The averaging *did* smooth;
  it just smoothed toward the wrong value because the bin was full of a degenerate
  0.0-lookup mass. The dip + tight CI were both symptoms of the Gain bug, not of
  the SEM estimator.
- Your size-invariance intuition is right in spirit: SEM legitimately shrinks with
  n, but a *real* heterogeneous effect would still show bin-to-bin scatter; a
  point-mass shows none. The tell is **n-per-bin** and **within-bin RT variance**,
  not the CI width alone.

The diagnosis below **tests** this read rather than asserting it, and audits
whether the analytic SEM is trustworthy going forward (now that the contamination
is removed).

---

## 2. Diagnosis tasks (for the subagent)

1. **Confirm the spike is gone on the fixed Gain**, at full n (ties into
   [[partition]] 1B). Reproduce the fine histogram near 1.0 and the saturation
   signature counts from `gain_dip_diagnostic.py`; expect the ≈1.0 point-mass to
   be gone and `a_shallow` always a visited move.
2. **Analytic vs bootstrap CI, head-to-head**, for every binned figure that
   carries error bars (`gain_vs_rt`, `mq_vs_rt`, `gss_vs_rt`, `actiongap_vs_rt`).
   For each bin compute: analytic `1.96·std/√n`, and a percentile bootstrap CI
   (B≈2000 resamples of the bin's `y`). Report half-width ratio
   (bootstrap/analytic) and coverage agreement. Expectation on fixed data: they
   agree closely (ratios ≈ 1.0) — confirming the estimator is sound and the old
   narrow band was contamination, not a broken formula.
3. **Per-bin contamination diagnostics:** print n-per-bin, within-bin `y`
   variance, and a point-mass score (max single-value frequency) so any future
   "tight CI" bin can be checked for a degenerate mass. Flag bins where one value
   is > X% of the mass.
4. **(Decision, [Q1])** Should we *change* the plotting CI? Options: (a) keep
   analytic SEM (cheap, and validated by task 2); (b) switch all error bars to
   bootstrap; (c) keep analytic but add a guard — min-n per bin and a point-mass
   warning. *Recommendation:* (c). Leave the call to you.

### Open questions
- **[Q1]** Which CI for the published figures going forward — keep analytic,
  switch to bootstrap, or analytic-with-guards (recommended)?
- **[Q2]** Bootstrap on the **raw** per-move RT (`rt`) or on **logT** (the plotted
  transform)? The figures plot mean RT in seconds but bin/threshold on logT.
  *Assumption:* bootstrap on the same transformed quantity the panel displays.
- **[Q3]** Scope: just `gain_vs_rt`, or all four error-bar figures (recommended)?

---

## 3. Tests to guarantee correctness

Unit tests (pytest, alongside `human_analytics/tests/`):
- **`test_gain_shallow_is_visited`** — on a hand-built tiny tree where the 1-ply
  value-head best is an **unexpanded** root move, the new `_tree_voc_and_gap` must
  pick `a_shallow` = first *visited* move and must **not** return ≈1.0 from a 0.0
  lookup. (Regression test pinning the exact bug.)
- **`test_gain_nonnegative`** — `Gain ≥ 0` for any tree (since `a_deep =
  argmax(final_Q)`); NaN only on degenerate trees (no expansions).
- **`test_gain_matches_growing_tree_identity`** — synthetic tree with known
  `final_Q`, `first_visit`: assert `Gain == final_Q[argmax] − final_Q[first
  expanded]` exactly.
- **`test_no_unvisited_zero_leak`** — assert no sampled real tree yields
  `shallow_q == 0.0 AND that move ∉ visited set` under the new definition.
- **CI tests:**
  - **`test_analytic_vs_bootstrap_agree`** — on synthetic Gaussian bins, analytic
    `1.96·std/√n` and percentile bootstrap half-widths agree within tolerance
    (e.g. < 5% for n ≥ 500).
  - **`test_pointmass_bin_flagged`** — a bin that is 90% a single value triggers
    the point-mass guard.
  - **`test_ci_shrinks_with_n`** — sanity: half-width ∝ 1/√n on synthetic data
    (the "size-invariance" intuition made concrete — variability per-sample is
    fixed, the *mean's* CI shrinks as expected, and that is the correct behavior).

### Open question
- **[Q4]** Where should these live — extend `human_analytics/tests/`, or a new
  `test_gain.py` / `test_ci.py`? *Assumption:* new files under
  `human_analytics/tests/`.
  - extend human_analytics/tests/

---

## Results (Part 2 execution)

**Cache used:** `vals_human_trees_200000_7.parquet` (+ `rootmoves_human_trees_200000_7.parquet`)
— the largest full-n, current-format (non-`_96`) human_trees cache. No Part-1B
sentinel/full cache was present at run time; the diagnostic auto-prefers one if it appears.
**Code (new only):** `human_analytics/gain_ci_audit.py`, `human_analytics/utils/ci.py`,
`human_analytics/tests/test_gain.py`, `human_analytics/tests/test_ci.py`. Run with
`PYTHONPATH=human_analytics python human_analytics/gain_ci_audit.py`.

### Task 1 — the ≈1.0 point-mass is GONE on the fixed Gain
- n = 207,912 moves; Gain ∈ [0, 2], **Gain<0 count = 0** (a_deep = argmax ⇒ ≥0 holds).
- mass in **[0.99, 1.01] = 0.19%** (n=399), exactly 1.0 = 0.08% — a thin, smooth shoulder,
  **not** the old degenerate spike. Interior (Gain>0.05) point-mass score = **0.005** (no
  single value dominates). The fine 0.90–1.06 histogram rises gently to ~200/bin at 0.99–1.01
  and falls off — no tower. `a_shallow` is read from the q-trace (always a visited move);
  the unvisited-0.0 lookup is impossible by construction (pinned by `test_gain_shallow_is_visited`).

### Task 2 — analytic vs percentile-bootstrap CI (B=2000), headline
Half-width ratio = bootstrap / analytic on the **displayed mean** (mean logT for
gain/actiongap/gss; mean MQ for mq). Bins honor the figures' `min_bin_count=100`.

| figure          | bins | median ratio | mean | max|ratio−1| |
|-----------------|-----:|-------------:|-----:|------------:|
| gain_vs_rt      |    9 | **1.001**    | 0.999| 0.020       |
| actiongap_vs_rt |    9 | 0.996        | 0.993| 0.052       |
| gss_vs_rt       |   20 | 0.995        | 0.993| 0.052       |
| mq_vs_rt        |   20 | 0.995        | 0.997| 0.047       |
| **ALL**         |   58 | **0.996**    | 0.995| **0.052**   |

**100% of bins agree within 10%** (in fact within ~5%). The analytic SEM is sound — the old
narrow CI was a contaminated-bin artifact, not a broken estimator.

### Task 3 — per-bin contamination
- gain/actiongap/gss bins: within-bin logT variance ≈ 0.85–1.34, point-mass score 0.13–0.27
  (RT is continuous; no degenerate mass). **No bins flagged.**
- mq_vs_rt: **7 bins flagged POINTMASS** (score >0.5) — these are the **genuine** MQ masses at
  0.0 (engine-best) and −1.0 (blunder), which the figure deliberately handles by binning on
  the well-behaved logT x-axis. Even there analytic≈bootstrap (ratios 0.99–1.05), so the SEM
  is still trustworthy; the flag is the guard correctly surfacing a real point-mass, not the
  Gain artifact. (The Gain≈1.0 contamination, by contrast, no longer exists — see Task 1.)

### Task 4 / [Q1] — analytic-with-guards helper
`utils/ci.py`: `analytic_ci_half` (the figures' `1.96·std/√n`), `bootstrap_ci_half`
(percentile bootstrap of the displayed mean), `pointmass_score`, `bin_contamination`, and
`analytic_ci_with_guards(min_n=100, pointmass_threshold=0.5)` → keeps the analytic half-width
and adds `low_n` / `pointmass` warning flags. Reusable, no figure-driver edits.

### Hypothesis verdict
**HELD.** On the fixed data the analytic SEM matches the bootstrap everywhere (median ratio
0.996, max |ratio−1| 0.052), and the Gain≈1.0 degenerate mass that previously inflated one
tie-safe bin's n (→ artificially tight SEM around a pulled-down mean) is gone. The narrow CI
was contamination, not a flimsy formula. Recommendation: keep analytic SEM + guards ([Q1]=c).

### Tests (pytest, all green: 10 new / 39 suite total)
`test_gain.py`: `test_gain_shallow_is_visited` (pins the unvisited-0.0 bug),
`test_gain_nonnegative`, `test_gain_matches_growing_tree_identity`,
`test_gain_nan_on_degenerate_tree`.
`test_ci.py`: `test_analytic_vs_bootstrap_agree` (<5% for n≥500), `test_ci_shrinks_with_n`
(½-width ∝ 1/√n), `test_pointmass_bin_flagged`, `test_min_n_guard`,
`test_bin_contamination_triple`, `test_degenerate_inputs`. Built on tiny synthetic
trees/arrays (no real-tree load).

---

## Jaggedness diagnosis (gain_vs_rt & gss_vs_rt)

**The puzzle.** Both curves are visibly JAGGED bin-to-bin, yet the per-bin SEM bars are
razor-thin. Naively that's paradoxical: a jagged curve "should" come with either fat bars or a
smooth shape. We resolved it procedurally on the FULL-n cache.

**Code (new only):** `human_analytics/utils/jaggedness.py` (estimators + curve-level
bootstrap), `human_analytics/jaggedness_diagnosis.py` (driver), `human_analytics/tests/
test_jaggedness.py`. Figure: `figures/jaggedness_diagnosis.png`. Run with
`PYTHONPATH=human_analytics python human_analytics/jaggedness_diagnosis.py`.
**Cache:** `vals_human_trees_10000000_7.parquet` joined to `processed_moves_nonzero` by `fen`
(450,751 moves; 444,161 with non-NaN Gain). Confounders pulled from `processed_moves_nonzero`:
`n_possible_moves` (legal moves) and `n_self_pieces_exc_pawns` (own-material proxy).

**Crucial framing point** (corrects a latent assumption): the figures do NOT plot arithmetic
mean RT. `Variable(column="move_time", is_log=True)` means the binned/plotted point is **mean
logT = ⟨ln move_time⟩** (a geometric-mean RT after `exp`), and the SEM bar is on mean logT
(analysis.py:369, 466). So the heavy-tail-of-the-arithmetic-mean story is already partly
defused by the figures themselves.

### The resolution in one line
The thin bars are **correct** and the jaggedness is **mostly real conditional structure**, not
a sampling-noise artifact. The per-bin SEM is conditional on bin membership — but here a
curve-level (shape) bootstrap that re-computes the quantile edges, re-bins, and recomputes
every bin's mean produces a band **the same width** as the per-bin SEM (ratio ≈ 1.00). So
bootstrapping does NOT make the jaggedness go away; it confirms each bin's mean is pinned to
±0.01–0.04 in logT. The wiggles are larger than that because **adjacent bins genuinely differ
in confounders** (ply, legal-move count, material), i.e. the curve has fine structure, not
because the bars understate sampling error. The estimator was never the problem — at n≈50k–140k
per bin the SEM is legitimately tiny.

### Mechanism-by-mechanism (headline numbers)

**M1 — Conditional-SEM blind spot / curve-level bootstrap.** The crux test. Per-bin SEM vs a
B=1000 curve bootstrap that re-derives the edges and re-bins each resample:

| figure | binning | median per-bin SEM ½w | median curve-boot ½w | band/SEM | RMS adj. jump | jumps > SEM | jumps > boot band |
|--------|---------|----------------------:|---------------------:|---------:|--------------:|------------:|------------------:|
| gain_vs_rt | quantile-8 | 0.0088 | 0.0088 | **1.00** | 0.225 | **100%** | **100%** |
| gss_vs_rt | native-int | 0.0394 | 0.0392 | **1.00** | 0.101 | 31% | 29% |

The curve-level band does **not** widen vs the per-bin SEM (edge-jitter + reassignment add
nothing at this n), and the jaggedness sits OUTSIDE both bands (every gain jump, ~30% of gss
jumps exceed the combined two-bin band). ⇒ jaggedness is **not** sampling noise and **not**
dissolved by bootstrapping.

**M2 — Bin-count robustness.** Roughness (RMS adjacent |Δmean logT|) vs K, quantile vs
equal-width:

| K | gain quantile | gain eq-width | gss quantile | gss eq-width |
|---|--------------:|--------------:|-------------:|-------------:|
| 5  | 0.317 | 0.405 | 0.119 | 0.110 |
| 8  | 0.225 | 0.326 | 0.102 | 0.082 |
| 12 | 0.214 | 0.359 | 0.086 | 0.062 |
| 20 | 0.155 | 0.365 | 0.182 | 0.054 |
| 40 | 0.118 | 0.260 | 0.138 | 0.064 |

The **broad monotone trend (RT rises with Gain / with GSS) is stable across every K** — see the
multi-K overlays in the figure: they stack on top of one another. Roughness **falls** as K rises
for gain quantile (more bins → smaller per-bin steps of a smooth underlying trend), i.e. the
wiggle is bounded, trend-consistent fine structure, not random reshuffling. GSS quantile
roughness is non-monotone in K precisely because quantile cuts of an integer predictor land on
different integer sets at different K (see M3).

**M3 — Predictor discreteness (dominates GSS).** GSS is integer-valued, 0–95, with a massive
0/1 spike (gss=1 alone is 142,936 of 450,751 moves; see the histogram panel). The 8-quantile
interior edges are `[1,1,2,10,27,45,70]` — **two coincident edges at 1.0 and one at 2.0**:
quantile bins straddle wildly different integer sets, so their mean-RT jumps for a purely
mechanical reason. Native-integer binning (one point per GSS value + CI) gives a clean,
near-monotone curve at the SAME roughness (0.1011 native vs 0.1024 quantile) but with
**interpretable x-positions** and no straddle artifact — this is the right way to plot GSS.
Gain is the analogous mass case: it is heavily **zero-inflated** (median Gain = 0; the 8-quantile
edges collapse to `[0,0,0,0,0,0.02,0.08]` — five coincident zero edges), which is exactly why
`gain_vs_rt` already uses the **zero-inflated + tie-safe** mode (zero-lump as its own point).

**M4 — Estimator choice.** Normalized roughness (RMS jump / std of curve) for three centers:

| figure | arith mean RT | geo mean = mean logT (the FIGURE) | median RT |
|--------|--------------:|----------------------------------:|----------:|
| gain_vs_rt | 1.052 | 1.100 | 1.201 |
| gss_vs_rt  | 0.650 | 0.700 | 0.853 |

A robust center does **not** smooth these curves — arithmetic mean, geometric mean and median
are within ~15% of each other in normalized roughness, and the figure's geometric mean is
already mid-pack. So heavy-tailed RT is **not** the source of the jaggedness here (it would be
if the figures plotted arithmetic mean RT in raw seconds, but they plot mean logT). Estimator
choice is a non-issue.

**M5 — Confound spot-check (decisive).** Composition of the two adjacent bins straddling the
biggest jump:

- **gain_vs_rt**, Gain≈0.000 vs Gain≈0.006 (Δmean logT = 0.41): ply 84→68, legal_moves 21→26,
  own_material 3.1→4.1.
- **gss_vs_rt**, GSS=0 vs GSS=1 (Δmean logT = 0.64): ply 109→65, legal_moves 10.8→28.5,
  own_material 2.4→4.4.

Adjacent bins differ **systematically and monotonically** in ply, legal-move count and material.
The GSS=0 bucket is dominated by late-game, few-legal-move, low-material positions (fast moves);
GSS=1 by mid-game, many-legal-move positions (slower). **The jaggedness is REAL conditional
structure — the bars are "right".** It is not noise to be averaged away; it is the curve telling
you the predictor is entangled with phase/branching/material.

### Why thin bars coexist with jaggedness — the procedural answer
1. At n≈50k–140k per bin, the SEM on a bin mean is genuinely 0.01–0.04 in logT — correctly tiny.
   (`bar ∝ 1/√n`; nothing is broken.)
2. The bar measures uncertainty in **that bin's mean given its membership**. It does NOT measure
   how much the mean *should* move to the next bin. The curve-level bootstrap confirms even the
   shape-level uncertainty (edges + reassignment) is the same tiny size (M1, ratio 1.00).
3. The adjacent-bin jumps are therefore **real**: driven by (a) discrete/zero-inflated predictor
   edges straddling different value sets (M3, dominant for GSS), and (b) confounders that shift
   sharply across bins (M5, dominant for both). A jagged-but-tight curve = many precisely-located
   points of a wiggly underlying function, not imprecise points of a smooth one.

### Which mechanism dominates each figure
- **gss_vs_rt:** dominated by **M3 (integer-predictor discreteness)** + **M5 (confounds)**. The
  jagged quantile version is an artifact of quantile-cutting an integer with a huge 0/1 spike;
  native-integer binning fixes the x-axis. The residual wiggle (29% of jumps > band) is real
  confounded structure.
- **gain_vs_rt:** dominated by **M5 (confounds)** on top of the **zero-inflation** the figure
  already lumps. Every adjacent jump exceeds the band — the interior Gain bins carve a continuum
  that is monotonically entangled with ply/branching/material. M1/M4 contribute nothing (band ==
  SEM; center choice irrelevant).

### Recommendation for how to plot these two figures
1. **gss_vs_rt → bin on GSS's NATIVE integer values** (one point + analytic SEM per GSS value),
   not quantiles. Same precision, honest x-positions, removes the straddle artifact. If too many
   sparse high-GSS values clutter, keep `integer_bin_width` but choose a width that does not split
   the 0/1 spike, and apply `min_bin_count`.
2. **gain_vs_rt → keep the zero-inflated + tie-safe scheme** (it already isolates the zero mass)
   and **keep the analytic SEM bars** — both are validated. Do NOT switch to bootstrap (no width
   change) and do NOT switch the center (no roughness change).
3. **For both:** keep the geometric-mean (mean logT) center — already robust. Optionally annotate
   that the remaining bin-to-bin wiggle is real conditional structure (confounded with phase /
   branching / material), e.g. by also showing the ply-tertile-segmented panel (already produced)
   so the confound is explicit rather than read as noise. A LOESS/spline smoother + the
   curve-bootstrap band is a fair *summary* overlay, but the honest message is that the wiggles
   are signal, so do not over-smooth them away.

### Tests (pytest, 5 new; full suite green: 20 total)
`test_jaggedness.py`: `test_curve_bootstrap_brackets_true_curve_and_widens_vs_sem`,
`test_integer_binning_cleaner_than_quantile_on_integer_predictor`,
`test_robust_center_reduces_roughness_on_heavy_tailed_rt`,
`test_frac_jumps_and_band_consistency`, `test_roughness_decreases_with_smoothing`.
Synthetic arrays only (no DB / parquet load).

---

## Gain binning (K-vs-n)

**The question.** Why is the *continuous* Gain still jagged in `gain_vs_rt.png`, and is its
TRUE Gain→RT relation smooth? Framing (correct): fixed-K quantile binning does **not** converge
to the smooth regression function as n→∞ — the resolution is frozen by K. To smooth you must
either **grow K with n** or use a **bandwidth smoother**. This subsection tests both on the
FULL-n cache.

**Code (new only):** `human_analytics/gain_binning_diagnostic.py`,
`utils/jaggedness.py::lowess_curve` / `lowess_bootstrap` (binning-free LOWESS + curve-level
bootstrap band), `human_analytics/tests/test_jaggedness.py` (+2 tests). Figure:
`figures/gain_binning_diagnostic.png`. The canonical `gain_vs_rt.png` is **unchanged** (this
only diagnoses + recommends). Run with
`PYTHONPATH=human_analytics python human_analytics/gain_binning_diagnostic.py`.
**Cache:** `vals_human_trees_10000000_7.parquet` ⋈ `processed_moves_nonzero` by `fen` (444,161
moves with non-NaN Gain); lc0-Q panel samples 3,000 trees from `human_trees`.

### 1 — value concentration & lc0-Q quantization (the cause)
- **Gain is extremely concentrated:** **frac ≤ 0.05 = 0.835**, frac ≤ 0.01 = 0.722, and
  **frac == 0 = 0.554** (a literal 55% point-mass at exactly 0 — "search adds nothing"). Median
  Gain = 0.
- **The underlying lc0 root-Q is heavily QUANTIZED** (Gain is a difference of these): **|Q|≈0
  (draw) ≈ 0.53** (exactly 0.0 alone ≈ 0.51 — a network "draw" attractor), and **|Q|≈1
  (saturated win/loss) ≈ 0.28** (−1 ≈ 0.27 ≫ +1 ≈ 0.02). So the predictor's density is a few
  towering spikes (0, ±1) with a thin continuum between — exactly the regime where a *fixed* K
  cannot resolve the bulk.

### 2 — K-scaling: the jag shrinks and the curve converges as K grows
Quantile bins on `gain_vs_rt`, roughness = RMS adjacent |Δ mean logT| (`min_bin_count=100`):

| K   | #bins kept | RMS adj jump (logT) | #distinct interior edges |
|-----|-----------:|--------------------:|-------------------------:|
| 8   | 5          | **0.225**           | 4 (of 7 — zero-mass collapses 3) |
| 20  | 10         | 0.155               | 9  |
| 50  | 24         | 0.119               | 23 |
| 100 | 46         | **0.106**           | 45 |

Roughness **falls monotonically** as K grows (0.225 → 0.106), and the curve **converges**: RMS
difference K=50-vs-K=100 (0.0318) < K=8-vs-K=20 (0.0332). The figure's `K=8` keeps only **5**
points because the 56% zero-mass collapses 3 of the 7 interior quantile edges — that is precisely
the "frozen resolution over a concentrated density" pathology. The broad monotone trend (RT rises
with Gain) is stable across every K (overlays stack); only the *resolution* of the fine structure
improves with K.

### 3 — binning-free LOWESS smoother + curve-bootstrap band
LOWESS of logT on Gain (statsmodels, frac=0.5, 15k-subsampled interior, B=150 curve-level
bootstrap). The 55% Gain==0 lump is **isolated as its own point** (mean logT = **1.416**, a fast
move) exactly as `gain_vs_rt.png` does; the smoother runs on the continuous interior (Gain>0):

- **Smooth & monotone-increasing:** **26/31 core steps up**, worst ex-boundary step
  **−0.009 logT** (sampling noise); normalized roughness (RMS 2nd-diff / std) = **0.088** (≈0).
  The single material down-step (−0.07) is the LOWESS boundary point immediately above the
  zero-spike, not curve wiggle.
- The interior fit rises **1.39 → 1.93 logT (Δ ≈ 0.55)**, monotone, with a tight curve-bootstrap
  band (median half-width **0.055** logT). The smoother is itself smooth.

### 4 — Verdict
**The TRUE Gain→RT relation is SMOOTH and monotone-increasing.** The bin-to-bin jag in
`gain_vs_rt` is a **coarse-fixed-K artifact over a concentrated, quantized density** (83.5% of
Gain ≤ 0.05; lc0 Q clustered at 0 and ±1) — **not** a genuinely wiggly relationship. Evidence:
roughness shrinks monotonically with K and the K-curves converge (fixed K just under-resolves);
the binning-free LOWESS is smooth, monotone and tightly banded.

**Recommendation for `gain_vs_rt` (do NOT change the canonical figure yet — recommend only):**
1. **Best:** plot the **binning-free LOWESS + curve-bootstrap band** as the canonical summary,
   with the **Gain==0 lump isolated** as its own point (55% of moves, distinctly faster).
2. **If keeping bins:** replace the frozen **K=8** with a **K-grows-with-n** rule (K ∝ n^(1/3) ⇒
   ≈50–100 here), still **zero-inflated + tie-safe** so the 0-mass is its own point. The jag
   visibly shrinks (roughness 0.225→0.11) without manufacturing structure.
3. Either way keep the **geometric-mean (mean logT)** center and **analytic SEM** (both validated
   earlier in this doc). The residual fine structure is real conditional structure (see the
   Jaggedness diagnosis above), so a smoother is a fair *summary*, not a way to erase signal.

### Tests (pytest, +2; `test_jaggedness.py`, full suite green: 46 total)
`test_lowess_recovers_smooth_monotone_trend_on_concentrated_predictor` (LOWESS recovers a smooth
monotone truth on a 55%-zero-inflated predictor — the Gain density),
`test_lowess_bootstrap_band_brackets_fit_and_is_tight_at_large_n` (curve-level LOWESS band
brackets the fit and is tight at large n). Synthetic arrays only (no DB / parquet load).

---

## Estimator-by-type plotting standard (IMPLEMENTED 2026-06-19)

We promoted the recommendation above into a project-wide rule: **pick the curve estimator by the
predictor's nature; standardize the uncertainty on a bootstrap-style band.** Implemented in
`utils/analysis.py` (`Analyzer.plot_lowess` + `plot_lowess_tertile_segmented`, dispatched by
`save_dashboard(estimator="lowess", mass_values=[...])`) reusing `utils/jaggedness.py`
(`lowess_curve`/`lowess_bootstrap`, given a backward-compatible `delta` large-n speedup).

| predictor type | examples | estimator |
|---|---|---|
| continuous | **Gain, Action Gap** (RT vs x); **MQ, Player Clock** (x = log RT) | LOWESS + curve-bootstrap band; value point-masses isolated |
| discrete / integer | **GSS, # legal moves, own non-pawn pieces, ply** | native-integer (one point + SEM per value) |
| point-mass / zero-inflated | Gain's 55%-at-0 (MQ's 0/−1 are on the Y axis → handled by averaging) | isolate the mass as its own labeled point, smooth the rest |

Mechanics: LOWESS fits in the **displayed** x-coordinate (log RT where the axis is log); the
band is a curve-level bootstrap (resample → refit → per-grid percentiles); a faint **binned-means
overlay** is drawn behind the smoother as a sanity check; fits are capped at 500k rows
(curve/band unchanged, keeps the multi-million-row movetime panels tractable). Regenerated figures:
`gain_vs_rt`, `actiongap_vs_rt`, `mq_vs_rt` (LOWESS); `gss_vs_rt`, `legal_moves_vs_movetime`,
`own_material_vs_movetime`, `ply_vs_movetime` (native-int); `clock_vs_movetime` (LOWESS).
`gain_vs_rt` is now a smooth monotone rise with the 0-lump split out — the staircase is gone.
See [[report-deck-format-convention]].
