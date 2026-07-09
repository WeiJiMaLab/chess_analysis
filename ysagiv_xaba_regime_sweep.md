# ysagiv `human_trees` (xaba branch) — `time_lambda` sweep: is z_t's win a ceiling artifact?

Follow-up to `ysagiv_trees.md`'s xaba-branch result (this file does not edit that one — additive,
new file). At exactly one of the three regimes Agent 1 tested (`time_lambda=0.0005,
maintenance_scale=0.0`), a learned z_t-Controller significantly beat both SingleHalt* and
Stats-Controller — the strongest positive z_t result anywhere in this investigation. The catch:
at that same regime, SingleHalt*'s own fitted optimal fixed stop `k*=95` out of a 96-expansion
ceiling — i.e. SingleHalt* is pinned almost at "never stop early," a right-censored/near-
degenerate point. Open question: is z_t's win a real, general per-episode-adaptivity advantage,
or an artifact of comparing against a baseline forced into a coarse, nearly-trivial global policy
at that one boundary point? This document sweeps `time_lambda` (at `maintenance_scale=0.0` fixed)
on the SAME already-packed/materialized xaba population to answer that directly.

## Method

Reused, unmodified, everything Agent 1 built — no repacking, no retraining, no GPU:
- Population: `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba/mc_packed` (packed root),
  `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba/materialized/validation_cache.pt`
  (materialized validation cache), both resolved via `config_ysagiv_xaba.yaml`'s `eval.*` fields.
- **Phase 1 — map k\* vs `time_lambda`**: `analysis.regime_select.sweep()`, the exact script and
  auto-criterion used to build `regime_select.md` on our own corpus, run unmodified against this
  population's own `mc_packed` root over its full grid (22 `time_lambda` values, 5e-05…0.3,
  log-spaced, × 16 `maintenance_scale` values). Only the `maintenance_scale=0.0` slice is used
  below (matching the winning regime's `m`). Criterion: a point is **meaningful** iff
  `1 < k* < ceiling - 1`, where `ceiling = max(len(curve) for curve in curves)` is derived
  directly from the data (not assumed) — confirmed **96** here, matching the corpus's own
  96-root-expansion `oracle_trace_expansion_counts` (same as `regime_select.md`'s finding on our
  own corpus, not assumed a priori).
- **Phase 2 — pick a representative spread**: an automatic point-selector (no hand-picking) chose
  the continuity point (`0.0005`), the already-known boundary point (`0.01`), three points spread
  across `k*/ceiling ≈ 0.5 / 0.3(interior-actual 0.095) / 0.15`(nearest available), and the
  smallest-`k*` meaningful point, from the meaningful `m=0` band.
- **Phase 3 — full 3-way paired significance** at each chosen point plus the known nonzero-
  maintenance point (`λ=0.001, m=0.001`, re-run fresh for consistency): `analysis.ysagiv_sig`,
  same paired-bootstrap-CI-on-per-episode-regret-difference methodology as the rest of this
  investigation (`cts.stats.bootstrap_ci`, percentile, `n_boot=2000`, seed=0 — fully
  deterministic; the two points re-run here (`λ=0.01,m=0` and `λ=0.001,m=0.001`) reproduce
  `ysagiv_trees.md`'s original numbers exactly, confirming determinism). Same `n_fit=1,465 /
  n_eval=628` fit/eval split (70/30 of the 2,093 validation episodes) reused at every regime, as
  in the original xaba branch.
- Jobs: `slurm/pipeline/ysagiv_xaba_regime_sweep.slurm` (job `10846482`, qos=test/cpu, completed
  in 9m19s covering the full 352-point Phase-1 grid + 6 Phase-3 significance calls) and a small
  follow-up `slurm/pipeline/ysagiv_xaba_regime_sweep_extra.slurm` (job `10847064`) adding two more
  points (`λ=0.005, 0.007`) to bracket the win→tie transition more precisely.

## Result 1: `k*` vs `time_lambda` (the "loss spread" the user asked about)

`maintenance_scale=0.0` slice, ceiling=96 (data-derived):

| `time_lambda` | `k*` | `k*/ceiling` | meaningful? |
|---|---|---|---|
| 5e-05 | 95 | 1.000 | **no — right-censored (pinned at ceiling-1)** |
| 0.0001 | 95 | 1.000 | no — right-censored |
| **0.0005** | **95** | **1.000** | **no — right-censored** (the original "winning" regime) |
| 0.001 | 69 | 0.726 | yes |
| **0.0015** | **47** | **0.495** | **yes — dead-center mid-band** |
| 0.002 | 9 (7 on the 1,465-ep fit-split used by `ysagiv_sig`†) | 0.095 | yes |
| 0.0025 | 6 | 0.063 | yes |
| 0.003 | 6 | 0.063 | yes |
| 0.005 | 4 | 0.042 | yes |
| 0.007 | 3 | 0.032 | yes |
| **0.01** | **3** | **0.032** | **yes** (the original default/every-other-eval regime) |
| 0.011 | 3 | 0.032 | yes |
| **0.012** | **2** | **0.021** | **yes — smallest meaningful `k*` in the whole grid** |
| 0.013–0.02 | 2 | 0.021 | yes |
| 0.03–0.3 | 1 (0 at 0.3×0.3 maint) | ≤0.011 | no — collapsed (`k*≤1`) |

†`regime_select.sweep()` fits `k*` on the full 2,093-episode validation split; `ysagiv_sig` fits
on only the 1,465-episode FIT sub-split (same 70/30 split used for the significance test itself).
The two agree exactly at every other point tested (95, 47, 3, 2, and both known points) and are
close at `λ=0.002` (9 vs 7) — a minor fit-sample-size effect, not a discrepancy in the underlying
curve.

**Meaningful band on this population, `m=0`: `time_lambda ∈ [0.001, 0.02]`** (16/22 grid points).
Below `0.001`, SingleHalt* is right-censored at `k*=95` (pinned essentially at the 96-expansion
ceiling — cheap search makes continuing always worth it). Above `0.02`, it collapses to `k*≤1`
(expensive search makes stopping immediately always worth it). **`time_lambda=0.0005` — the
single regime where z_t won in the original xaba-branch result — sits just OUTSIDE this
meaningful band**, exactly the near-ceiling pileup point the user flagged as suspicious. This
alone confirms the concern was well-founded methodologically: that regime is not one of REGIME's
confirmed "meaningful" points on this population.

Full grid (352 points, all `maintenance_scale` values): `outputs/figures/ysagiv/xaba/regime_sweep/
regime_select_results.json`; heatmap: `outputs/figures/ysagiv/xaba/regime_sweep/{pdf,png}/
regime_select.png`.

## Result 2: 3-way paired significance across the spread

`n_fit=1,465 / n_eval=628` at every point (same episodes reused, only the cost function changes).
Diffs are `a − b`; positive + CI excluding 0 means `b` wins (lower regret).

| regime | `k*` | meaningful? | SingleHalt\* | Stats | z_t | SH−Stats | SH−z_t | Stats−z_t |
|---|---|---|---|---|---|---|---|---|
| λ=0.0005, m=0 | 95 | **no (ceiling)** | 0.0396 | 0.0360 | **0.0217** | +0.0036 [+0.0001,+0.0060] Stats wins | **+0.0179 [+0.0148,+0.0202] z_t WINS** | **+0.0143 [+0.0104,+0.0185] z_t WINS** |
| λ=0.0015, m=0 | 47 | **yes (mid-band)** | 0.1115 | 0.0925 | **0.0443** | +0.0190 [+0.0129,+0.0247] Stats wins | **+0.0672 [+0.0513,+0.0837] z_t WINS** | **+0.0482 [+0.0331,+0.0646] z_t WINS** |
| λ=0.002, m=0 | 7 | **yes (interior)** | 0.1041 | 0.0820 | **0.0577** | +0.0221 [+0.0089,+0.0357] Stats wins | **+0.0464 [+0.0299,+0.0638] z_t WINS** | **+0.0243 [+0.0078,+0.0414] z_t WINS** |
| λ=0.005, m=0 | 4 | yes | 0.0961 | 0.0860 | 0.0936 | +0.0101 [−0.0005,+0.0215] ns (barely) | +0.0026 [−0.0068,+0.0120] ns | −0.0076 [−0.0192,+0.0042] ns |
| λ=0.007, m=0 | 3 | yes | 0.0921 | 0.0844 | 0.0896 | +0.0076 [−0.0019,+0.0179] ns | +0.0025 [−0.0053,+0.0110] ns | −0.0051 [−0.0157,+0.0054] ns |
| λ=0.01, m=0 (known) | 3 | yes | 0.0834 | 0.0718 | 0.0805 | +0.0116 [+0.0035,+0.0210] Stats wins | +0.0029 [−0.0037,+0.0108] ns | −0.0087 [−0.0183,+0.0004] ns |
| λ=0.012, m=0 | 2 | **yes (near-1 floor)** | 0.0767 | 0.0709 | 0.0744 | +0.0058 [−0.0024,+0.0150] ns | +0.0023 [−0.0068,+0.0124] ns | −0.0035 [−0.0103,+0.0046] ns |
| λ=0.001, m=0.001 (known) | 4 | — | 0.0533 | 0.0462 | 0.0515 | +0.0072 [−0.0026,+0.0175] ns | +0.0019 [−0.0031,+0.0078] ns | −0.0053 [−0.0147,+0.0042] ns |

Raw JSON: `outputs/figures/ysagiv/xaba/ysagiv_sig_xswp_lambda{0p0005,0p0015,0p002,0p005,0p007,
0p01,0p012}_m0.json`, `ysagiv_sig_xswp_lambda0p001_m0p001.json`.

**All 8 points are now landed** (job `10846482` for the first 6 + the sweep, job `10847064` for
the two bracketing points at `λ=0.005`/`0.007`). The `λ=0.005`/`0.007` results **confirm the
predicted "tie" zone** and, combined with `λ=0.002`'s clear win, narrow the win→tie transition to
somewhere in `time_lambda ∈ (0.002, 0.005)` (`k*` crossing from 7 down to 4) — a transition that
happens well inside the meaningful band, far from either the ceiling or the `k*≤1` collapse
boundary.

## Answering the core question

**z_t's advantage does NOT vanish as `k*` moves away from the ceiling — it persists, and is in
fact STRONGEST at a point squarely in the interior of REGIME's meaningful band, not at the
ceiling.** The pattern across the sweep is unambiguous:

- At **λ=0.0005** (`k*=95`, ceiling, NOT meaningful): z_t wins, `+0.0179` over SingleHalt*.
- At **λ=0.0015** (`k*=47`, `k_frac=0.495`, dead-center of the meaningful band): z_t wins by
  **more**, `+0.0672` over SingleHalt* — nearly 4x the margin at the ceiling point.
- At **λ=0.002** (`k*=7`, well into the interior, `k_frac≈0.095`): z_t still wins clearly,
  `+0.0464` over SingleHalt*.
- At **λ=0.005** (`k*=4`), **λ=0.007** (`k*=3`), **λ=0.01** (`k*=3`), **λ=0.012** (`k*=2`, the
  smallest meaningful `k*` in the whole grid), and **λ=0.001,m=0.001** (`k*=4`): z_t is
  statistically indistinguishable from both baselines — no win, but also no loss.

So the boundary between "z_t wins" and "z_t ties" is **not** "near ceiling vs. away from
ceiling" — `λ=0.0015` and `λ=0.002` are both comfortably inside the meaningful band, nowhere near
`k*→ceiling` pileup, and z_t wins there just as clearly (more clearly, even) as at the boundary
point. The transition instead falls **between `λ=0.002` (`k*=7`, win) and `λ=0.005` (`k*=4`,
tie)** — i.e. the real dividing line in this sweep is **low `time_lambda` (≲0.002-0.005,
cheap-search regimes, `k*` ranging from very large down to high-single-digits) vs. higher
`time_lambda` (≥0.005, `k*≤4`, expensive-search regimes)** — not proximity to the 96-expansion
ceiling specifically. Notably this transition band (`k*` crossing roughly 7→4) is itself well
inside REGIME's meaningful range, nowhere near either the ceiling or the `k*≤1` collapse edge.
This directly answers the disambiguation the user asked for: **the win is not an artifact of
SingleHalt* being right-censored at one boundary point.** It generalizes across a genuine span of
distinct, non-degenerate `k*` values (95 → 47 → 7), which rules out the narrowest version of the
"coincidence of a near-degenerate baseline" concern. A companion, still-open question the sweep
surfaces on its own: what actually separates the "z_t wins" low-λ band from the "z_t ties"
high-λ band is worth its own follow-up (candidate mechanisms — larger `k*` giving z_t more
per-episode trajectory to condition on before committing, vs. the cost function itself changing
which structure is decision-relevant — are plausible but not adjudicated by this sweep; flagged
as a hypothesis, not a finding, per this investigation's no-ad-hoc-mechanism-claims convention).

## Bottom line

1. This population's own ceiling is 96 (data-derived, matches the assumed `search_budget=96`).
2. The meaningful `time_lambda` band at `maintenance_scale=0.0` is `[0.001, 0.02]`; the original
   winning regime (`λ=0.0005`) sits just outside it (right-censored, `k*=95=ceiling-1`) — the
   user's suspicion that this was a boundary artifact was methodologically well-founded.
3. **But the suspicion doesn't hold as the explanation for the result**: z_t significantly beats
   both SingleHalt* and Stats-Controller not just at that boundary point but at two more points
   squarely inside the meaningful band (`λ=0.0015`, `k*=47`; `λ=0.002`, `k*=7`), with the largest
   margin of all three at the interior mid-band point, not at the ceiling. z_t only ties (not
   loses) once `time_lambda` rises to `≥0.005` and `k*` drops to `4` or below (confirmed at 5 of
   the 8 points tested: `λ∈{0.005, 0.007, 0.01, 0.012}` at `m=0`, plus `λ=0.001,m=0.001`).
4. **Verdict: z_t's win on the xaba-filtered ysagiv population is a real, cheap-search-regime
   effect, not an artifact of comparing against a near-degenerate right-censored baseline.** It
   generalizes across a >10x range of `k*` (95 down to 7) as long as `time_lambda` stays in the
   low/cheap-search part of the meaningful band (`≲0.002-0.005`); it disappears (not reverses)
   once `time_lambda` pushes `k*` down to `≤4`. The win→tie transition sits well inside the
   meaningful band (`k*` crossing ~7→4, `λ` crossing (0.002, 0.005)), far from both the ceiling
   pileup and the `k*≤1` collapse edge — reinforcing that this is a property of the cost regime's
   cheapness, not an edge-of-the-grid artifact in either direction.
