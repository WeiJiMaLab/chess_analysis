---
theme: default
title: Oracle stop step vs human RT
info: Does a normative DP oracle's stop step track human think time?
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: text-left
mdc: true
math: katex
---

# Oracle stop step vs human RT

<div class="mt-4 text-lg opacity-80">Does a normative DP oracle's stop step track human think time?</div>

<div class="mt-6 text-sm opacity-60 max-w-2xl">
Source report: <code>reports/oracle-stop-vs-human-rt.md</code> (R-ORACLE-RT)
</div>

---
layout: default
---

# The validation tiers

<div class="mt-6 text-base opacity-90">

| Tier | Claim | Requires |
|---|---|---|
| **A** | Same features predict oracle stop & RT directionally | LMCOS trees |
| **B** | Same FEN: oracle stop ↔ human RT | Human FEN trees |
| **C** | Trained controller ↔ human RT | Controller on human positions |

</div>

<div class="mt-8 text-sm opacity-80">

Headline so far: **Tier A — 4/4 feature directions match**; **Tier B smoke — r(oss, log RT) =
+0.091** (significant at n=497). The oracle stops on *value-landscape* features; humans deliberate
on *structural complexity*.

</div>

---
layout: default
---

# Tier A — same features drive oracle stop & human RT

<div class="text-sm opacity-90 mt-2">

n = 39,668 LMCOS trees (budget 96):

| Feature | r(oracle_stop_step) | r(human RT) | dir |
|---|---|---|---|
| branching | +0.014 | +0.195 | ✓ tiny |
| material | +0.011 | +0.039 | ✓ tiny |
| gain_depth (VOC) | **+0.233** | +0.096 | ✓ |
| toptwo | **−0.290** | −0.064 | ✓ |

</div>

<div class="mt-4 text-sm opacity-80">

**oracle_stop_step: 4/4 correct directions.** Branching/material ≈ 0 for the oracle is *expected*:
stopping is driven by **value convergence** (toptwo, gain_depth), not board structural complexity.
lc0 handles branching natively via its PUCT prior.

</div>

<div class="mt-3 text-xs opacity-50">
Figure: <code>lmcos/analysis/figures/oracle_stop_step_vs_human_rt.png</code>
</div>

---
layout: default
---

# Tier B — same FEN: oracle stop ↔ human RT

<div class="text-sm opacity-90 mt-2">

Smoke, n = 497 clean human-FEN trees:

| Metric | Value |
|---|---|
| **r(oracle_stop_step, log RT)** | **+0.091** ✓ (sig at n=497) |
| r(gain_depth, log RT) / oracle | +0.020 / **+0.463** |
| r(branching, log RT) / oracle | +0.208 / +0.166 |
| r(toptwo, log RT) / oracle | −0.128 / −0.030 |

</div>

<div class="mt-4 text-sm opacity-80">

**The split sharpens:** gain_depth strongly predicts the *oracle's* halt (+0.463) but is near-zero
for *humans* (+0.020). Humans don't natively compute VOC — their RT tracks structural complexity.
10K trees needed for full statistical power.

</div>

<div class="mt-3 text-xs opacity-50">
Figures: <code>lmcos/analysis/figures/human_oracle_rt_comparison.png</code>,
<code>…/human_oracle_features_comparison.png</code>
</div>

---
layout: default
---

# Follow-up & status

<div class="mt-4 text-sm opacity-90">

**SF-2000 (strength-matched, on hold).** Hypothesis: LC0 (~3000+ ELO) "sees through" positions a
≥2000-ELO human pool finds hard → a Stockfish ELO=2000 oracle on the *same* FENs may align
deliberation better. Gated on the 10K matched-position r. If SF-2000 *also* fails → engine VOC
fundamentally misses human deliberation (not a strength artifact).

</div>

<div class="mt-6 text-sm opacity-80">

| Step | Status |
|---|---|
| Tier A features + converged_expansions + 72 tests | ✅ done |
| Tier B: 10K FENs as 20 shards | ✅ submitted |
| Tier B: oracle + join + plots A/B/C | ⬜ after jobs |
| SF-2000 oracle | ⬜ gated on Tier B |

</div>
