---
theme: default
title: Minimal model & budgeted baselines
info: How much controller stopping skill survives stripping it down.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: text-left
mdc: true
math: katex
---

# Minimal model & budgeted baselines

<div class="mt-4 text-lg opacity-80">How much of the controller's stopping skill survives a strip-down?</div>

<div class="mt-6 text-sm opacity-60 max-w-2xl">
Source report: <code>reports/minimal-model-and-baselines.md</code> (R-MINMODEL)
</div>

---
layout: default
---

# Two inquiries

<div class="mt-6 grid grid-cols-2 gap-8 text-sm">
<div>

**(1) Minimal model**

Can raw scalar tree-stat features (no GNN) predict the oracle's halt/continue?

→ **No.** 4-feature MLP ≈ chance.

</div>
<div>

**(2) Hand-written baselines**

How much lift does the trained controller have over simple stop rules?

→ **Controller dominates** (5.6× lower regret).

</div>
</div>

---
layout: default
---

# Minimal model: 4 features can't replace the GNN

<div class="mt-4 text-base opacity-90">

| Model | Val sign acc | Val exact stop | r(pred, oracle) |
|---|---|---|---|
| GNN+MC | **90.1%** | — | — |
| Minimal MLP (4 feat) | **54.6%** | 5.2% | +0.291 |

</div>

<div class="mt-6 text-sm opacity-80">

- Near-chance sign accuracy → the 4 raw scalars (`best_q`, `wdl_var`, `t_norm`, `budget_rem_norm`)
  carry little halt/continue information.
- MSE loss falls monotonically but **sign accuracy oscillates** (0.54→0.41→0.68→0.52): the model
  fits the regression target, but advantage zero-crossings are unstable.
- **The GNN encoding of the full Q-value landscape is essential.**

</div>

---
layout: default
---

# Controller vs hand-written stopping rules

<img src="/figures/archive/u3_baselines/controller_vs_baselines.png" class="h-56 mx-auto rounded" />

<div class="mt-3 text-xs opacity-80">

| Stop rule | avg regret ↓ | exact-stop | 
|---|---|---|
| **GNN+MC controller** | **+0.026** | **0.63** |
| fixed-fraction 0.25 | +0.069 | 0.19 |
| gain-depth ≤ 0.05 | +0.148 | 0.10 |
| always-stop / never-stop | +0.52 / +1.37 | 0.23 / 0.10 |

Controller ~2.6× lower regret than the best baseline; **fixed-fraction-of-budget is the best
*hand-written* rule**, beating every value-based rule.

</div>

---
layout: default
---

# Why fixed-fraction competes: budget vs value

<img src="/figures/archive/u3_baselines/oracle_stop_cost_bimodal.png" class="h-56 mx-auto rounded" />

<div class="mt-3 text-sm opacity-80">

1. **Stops are bimodal:** ~49% value-driven (cost < 0.01) vs ~28% cost-forced (budget depleted).
2. **The oracle stops early:** median **3.8% of budget** (corr(stop, budget) = +0.41) → a small
   fixed fraction approximates it *without value info*.
3. **Regret is a weak discriminator** among good rules → **exact-stop** separates them far better
   (0.63 vs 0.19 vs 0.10).

</div>

---
layout: default
---

# Takeaways

<div class="mt-6 text-base opacity-90">

- The **GNN encoder is necessary** — scalar features are near-chance.
- The **controller dominates** every baseline on regret *and* exact-stop.
- `time_lambda` (cost weight) and the budget distribution are **levers**: ~28% of stops are
  budget-forced, masking the value signal.
- **Report exact-stop alongside regret** — regret alone undersells the value-based controller.

</div>

<div class="mt-8 text-xs opacity-50">
Minimal-MC diagnostic figures (sign accuracy, stop scatter, weights) live in
<code>lmcos/analysis/figures/</code>.
</div>
