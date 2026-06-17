---
theme: default
title: Human VOC / MQ vs move time
info: Engine value-of-computation and move-quality vs human think time.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: text-left
mdc: true
math: katex
---

# Human VOC / MQ vs move time

<div class="mt-4 text-lg opacity-80">Does engine value-of-computation track human deliberation?</div>

<div class="mt-6 text-sm opacity-60 max-w-2xl">
Source report: <code>reports/human-voc-mq.md</code> (R-VOC-MQ)
</div>

---
layout: default
---

# The question

<div class="text-lg opacity-80 mt-4">

Do engine-defined **value-of-computation (VOC)** and **move-quality (MQ)** track
how long humans think on the same positions?

</div>

<div class="mt-8 grid grid-cols-2 gap-6 text-sm">
<div>

**Headline**

- VOC ↔ log move time: **weak but real positive** (r ≈ +0.10)
- MQ ↔ move time: **negative** — but a *difficulty confound*, not a sign bug

</div>
<div>

**Two load-bearing caveats**

- The kept **MQ figure is Stockfish depth-5, not LC0**
- MQ subset (1M) ≠ tree subset (OSS/VOC) population

</div>
</div>

---
layout: default
---

# MQ vs move time — the confound

<img src="/figures/mq_vs_rt.png" class="h-72 mx-auto rounded" />

<div class="mt-4 text-sm opacity-80">

- **MQ ≤ 0** by construction; large mass at 0 (human played engine-best move).
- **r(MQ, clock) = −0.098**, negative within *every* ply tertile.
- **Not a sign bug — a difficulty confound** (N=1M): Spearman **ρ = −0.119**, survives all
  controls/de-meaning. Best-move rate falls **0.62 → 0.49** across move-time deciles.
- MQ × MT × branching interaction **β ≈ −0.007 (t=−7)** — branching amplifies, magnitude small.

</div>

---
layout: default
---

# VOC vs move time

<img src="/figures/voc_vs_rt.png" class="h-72 mx-auto rounded" />

<div class="mt-4 text-sm opacity-80">

- **r(log RT, VOC) = +0.097** — more remaining value-of-computation → longer thinks.
- Direction predicted by Russek-style value-of-computation.
- VOC near-zero-inflated: **VOC > 0.005 in ~33%** of positions (mean +0.100, median 0).

</div>

---
layout: default
---

# Companion tree values — OSS & Action Gap

<div class="grid grid-cols-2 gap-4">
<img src="/figures/oss_vs_rt.png" class="h-64 rounded" />
<img src="/figures/actiongap_vs_rt.png" class="h-64 rounded" />
</div>

<div class="mt-4 text-sm opacity-80">

Oracle stop step (left) and Action Gap (right) are the lc0-tree "generated values" on the same RT
subset — quantile-binned globally and by ply tertile (canonical Analyzer 1×2 dashboard).

</div>

---
layout: default
---

# Methods & caveats

<div class="text-sm opacity-80 mt-4">

| Quantity | Definition | Engine |
| :-- | :-- | :-- |
| **MQ** (kept fig) | `e_win_taken − e_win_best` (≤0) | **Stockfish d5** (not LC0) |
| **MQ** (LC0) | `final_Q(played) − final_Q(best)` | LC0 tree |
| **VOC** | `V_deep(a_deep) − V_deep(a_shallow)` | SF d5/d1 or LC0 tree |

- **100K run:** Stockfish d5/1, ply 15–75, opp clock ≥ 60s.
- **MQ "FULL" plot** is a 1M-row subset; ply-tertile cuts (27/56) from the whole-dataset
  distribution → "Early" panel has no openings.
- **Fix recommended:** recompute MQ from **LC0** + report difficulty-residualized partials.

</div>
