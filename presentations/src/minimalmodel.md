---
theme: default
title: Minimal model & budgeted baselines
info: How much controller stopping skill survives stripping it down.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search</span>

# Minimal model & budgeted baselines

<div class="mt-2 text-lg opacity-80">How much of the controller's stopping skill survives a strip-down?</div>

<div class="mt-4 text-sm opacity-50">Source: <code>reports/minimal-model-and-baselines.md</code> (R-MINMODEL)</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Overview</span>Two inquiries</div>

<div class="mdl-content">
  <div class="mdl-two">
    <div class="mdl-card mdl-card--neutral">
      <p class="mdl-card-h">1 · Minimal model</p>
      <p class="body">Can raw scalar tree-stat features (no GNN) predict the oracle's halt / continue?</p>
      <div class="mdl-found danger">→ <strong>No.</strong> A 4-feature MLP performs at chance.</div>
    </div>
    <div class="mdl-card mdl-card--neutral">
      <p class="mdl-card-h">2 · Hand-written baselines</p>
      <p class="body">How much lift does the trained controller have over simple stop rules?</p>
      <div class="mdl-found success">→ <strong>Controller dominates</strong> — 5.6× lower regret.</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Inquiry 1</span>Four features can't replace the GNN</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Model</th><th class="num">Val sign acc</th><th class="num">Exact stop</th><th class="num">r(pred, oracle)</th></tr>
      </thead>
      <tbody>
        <tr class="hl"><td>GNN + MC</td><td class="num">90.1%</td><td class="num">—</td><td class="num">—</td></tr>
        <tr><td>Minimal MLP (4 feat)</td><td class="num">54.6%</td><td class="num">5.2%</td><td class="num">+0.291</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <ul>
        <li>Near-chance sign accuracy → the 4 raw scalars (<code>best_q</code>, <code>wdl_var</code>, <code>t_norm</code>, <code>budget_rem_norm</code>) carry little halt/continue signal.</li>
        <li>MSE loss falls monotonically, but <strong>sign accuracy oscillates</strong> (0.54 → 0.41 → 0.68 → 0.52): advantage zero-crossings are unstable.</li>
        <li><strong>The GNN encoding of the full Q-value landscape is essential.</strong></li>
      </ul>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig mdl-titlefig--wide">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Inquiry 2</span>Controller vs hand-written stop rules</div>
      <table class="mdl-table mdl-table--sm mdl-table--wide">
        <thead>
          <tr><th>Stop rule</th><th class="num">avg regret ↓</th><th class="num">exact-stop</th></tr>
        </thead>
        <tbody>
          <tr class="hl"><td>GNN + MC controller</td><td class="num">+0.026</td><td class="num">0.63</td></tr>
          <tr><td>fixed-fraction 0.25</td><td class="num">+0.069</td><td class="num">0.19</td></tr>
          <tr><td>gain-depth ≤ 0.05</td><td class="num">+0.148</td><td class="num">0.10</td></tr>
          <tr class="dim"><td>always / never-stop</td><td class="num">+0.52 / +1.37</td><td class="num">0.23 / 0.10</td></tr>
        </tbody>
      </table>
      <div class="mdl-text" style="margin-top:0.7rem">Controller ~2.6× lower regret than the best baseline; <strong>fixed-fraction-of-budget</strong> is the best <em>hand-written</em> rule, beating every value-based rule.</div>
    </div>
    <div class="mdl-figbox">
      <img src="../public/figures/archive/u3_baselines/controller_vs_baselines.png" alt="Controller vs baselines" />
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Why it competes</span>Budget vs value</div>
      <div class="mdl-text">
        <ol style="padding-left:1.1em; margin:0">
          <li><strong>Stops are bimodal:</strong> ~49% value-driven (cost &lt; 0.01) vs ~28% cost-forced (budget depleted).</li>
          <li><strong>The oracle stops early:</strong> median <strong>3.8% of budget</strong> (corr = +0.41) → a small fixed fraction approximates it <em>without value info</em>.</li>
          <li><strong>Regret is a weak discriminator</strong> among good rules → <strong>exact-stop</strong> separates them far better (0.63 vs 0.19 vs 0.10).</li>
        </ol>
      </div>
    </div>
    <div class="mdl-figbox">
      <img src="../public/figures/archive/u3_baselines/oracle_stop_cost_bimodal.png" alt="Oracle stop cost bimodal" />
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Wrap-up</span>Takeaways</div>

<div class="mdl-content mdl-content--top">
  <ol class="mdl-list">
    <li>The <strong>GNN encoder is necessary</strong> — scalar features are near-chance.</li>
    <li>The <strong>controller dominates</strong> every baseline on regret <em>and</em> exact-stop.</li>
    <li><code>time_lambda</code> and the budget distribution are <strong>levers</strong>: ~28% of stops are budget-forced, masking the value signal.</li>
    <li><strong>Report exact-stop alongside regret</strong> — regret alone undersells the value-based controller.</li>
  </ol>
</div>

<div class="text-xs opacity-50 mt-4">Minimal-MC diagnostic figures (sign accuracy, stop scatter, weights) live in <code>lmcos/analysis/figures/</code>.</div>
