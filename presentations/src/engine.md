---
theme: default
title: Human move time — does a normative model match it?
info: Do lc0-search quantities (Gain, MQ, GSS, action gap) track human think time?
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search · human move time</span>

# Does a normative search model match human think time?

<div class="mt-2 text-lg opacity-80">Do lc0-search quantities on the same position track how long humans think?</div>

<div class="mt-4 text-sm opacity-50">~199K lc0 search trees on 2023 human-game FENs → ~211K joined moves.</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">The metrics</span>Four lc0-tree quantities vs human RT</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide" style="margin-top:0.2rem">
    <thead>
      <tr><th>Metric</th><th>Definition (lc0 tree)</th><th class="num">r with log RT</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>GSS</strong> — greedy stopping step</td><td>first expansion the eventual-best move is found (greedy, zero-cost)</td><td class="num">+0.110</td></tr>
      <tr><td><strong>Gain</strong> — value of computation</td><td><code>final_Q(best @ 96 exp.) − final_Q(best @ 1 exp.)</code> ≥ 0</td><td class="num">+0.085</td></tr>
      <tr><td><strong>MQ</strong> — move quality</td><td><code>final_Q(played) − final_Q(best)</code> ≤ 0</td><td class="num">−0.151</td></tr>
      <tr><td><strong>Action gap</strong></td><td>top1 − top2 of children's 1-ply value-head backup</td><td class="num">−0.058</td></tr>
    </tbody>
  </table>
  <div class="mdl-found">All four value-search quantities are weak — they track human deliberation only faintly, and MQ runs the "wrong" way. The strong tree-derived signal is instead a <em>structural</em> one — the policy entropy H(π) (next slides). <span class="opacity-50">199K trees → ~211K joined moves; 98% match.</span></div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Value of computation</span>Gain vs think time</div>
      <div class="mdl-text">
        <ul>
          <li><strong>Gain = final_Q(best @ 96 exp.) − final_Q(best @ 1 exp.) ≥ 0</strong> — both scored on the converged 96-expansion Q, read from <em>one growing oracle tree</em>.</li>
          <li><strong>r = +0.085</strong>: more value-of-computation → longer thinks, monotone. Direction matches Russek-style accounts.</li>
          <li>Zero-inflated: Gain = 0 in ~⅔ of positions (the first expansion already lands on the search-best move).</li>
          <li><strong>Definition fix:</strong> the 1-expansion tree is a <em>subset</em> of the 96-expansion tree, so the moves line up. This kills a spurious <code>Gain ≈ 1.0</code> spike (+ RT dip) that two earlier definitions hit by reading an <em>unvisited</em> <code>final_Q = 0.0</code>.</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/gain_vs_rt.png" alt="Gain vs RT" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Move quality</span>MQ vs think time — worse moves on longer thinks</div>
      <div class="mdl-text">
        <ul>
          <li><strong>MQ = final_Q(played) − final_Q(best) ≤ 0</strong> (0 = engine-best played); plotted as the <em>outcome</em> of think time.</li>
          <li><strong>r(MQ, log RT) = −0.151</strong> — longer thinks → worse moves.</li>
          <li>Same relationship, segmented three ways: <strong>global · by ply tertile · by GSS difficulty stratum</strong>. The standing read was a pure <strong>difficulty confound</strong>; the third panel tests it → <em>next slide</em>.</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/mq_vs_rt.png" alt="MQ vs RT — global, by ply, by GSS stratum" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Is it just difficulty?</span>MQ↔RT survives difficulty controls</div>

<div class="mdl-content">
  <div class="mdl-text">
    <ul>
      <li>From the <strong>third panel</strong> (previous slide): segmenting MQ↔RT <strong>within GSS difficulty strata</strong>, the negative slope holds in <strong>every</strong> stratum (ρ −0.19 to −0.24 vs pooled −0.23).</li>
      <li><strong>partial ρ | GSS = −0.213</strong> — conditioning on GSS removes only ~7%. GSS is even a <em>poor</em> difficulty proxy (its "easy" GSS 0–1 stratum has the worst MQ).</li>
      <li>Branching does more (<strong>partial | branching = −0.160</strong>), but <strong>partial | GSS + branching = −0.150</strong> — <strong>~⅔ of the effect survives</strong>.</li>
      <li>So it is <strong>not</strong> a pure difficulty confound: a residual points to <em>selection / uncertainty</em> (people think long precisely when unsure).</li>
    </ul>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Companion values</span>Greedy stop step &amp; action gap</div>

<div class="mdl-content">
  <div class="grid grid-cols-2 gap-6 items-center" style="flex:1; min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/gss_vs_rt.png" alt="greedy stop step vs RT" /></div>
    <div class="mdl-figbox"><img src="../public/figures/actiongap_vs_rt.png" alt="Action gap vs RT" /></div>
  </div>
  <div class="mdl-text" style="text-align:center; margin-top:0.5rem">Greedy stop step (left, groups of 5) rises monotonically with RT (r = +0.110, full range — no cap); action gap (right) is flat-to-weak. Same lc0-tree RT subset, quantile-binned globally and by ply tertile (sparse bins n &lt; 100 dropped).</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Where they diverge</span>The oracle stops on value; humans on structure</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Feature</th><th class="num">r(oracle stop)</th><th class="num">r(human RT)</th></tr>
      </thead>
      <tbody>
        <tr><td>branching</td><td class="num">+0.014</td><td class="num">+0.195</td></tr>
        <tr><td>material</td><td class="num">+0.011</td><td class="num">+0.039</td></tr>
        <tr class="hl"><td>gain_depth (value gained)</td><td class="num">+0.233</td><td class="num">+0.096</td></tr>
        <tr class="hl"><td>action gap (toptwo)</td><td class="num">−0.290</td><td class="num">−0.064</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <p class="mdl-lead">Same features, opposite emphasis.</p>
      <ul>
        <li>The oracle halts on <strong>value-landscape</strong> features (gain_depth, action gap) — it doesn't care about branching.</li>
        <li>Humans deliberate on <strong>structural complexity</strong> (branching dominates) — and barely track Gain.</li>
        <li>4/4 directions agree, but the <em>dominant driver differs</em> → the model captures direction, not mechanism.</li>
      </ul>
      <div class="text-xs opacity-50 mt-3">Tier A: n = 39,668 LMCOS trees · Tier B smoke: r(oracle stop, log RT) = +0.091, n = 497 human-FEN trees</div>
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
      <div class="mdl-title"><span class="mdl-kicker">Relationships</span>How do the model variables relate?</div>
      <div class="mdl-text">
        <ul>
          <li><strong>Branching ↔ log RT is the strongest RT tie</strong> — stronger than any engine metric.</li>
          <li><strong>H(π)</strong> (policy-prior entropy) is the strongest <em>tree-derived</em> RT predictor (<strong>+0.24</strong>, ~3× any value metric) — but it's a <em>width</em> signal (ρ +0.59 with branching) that <strong>raw branching subsumes</strong> (partial ρ | branching = +0.05). It survives Gain/GSS, so it's distinct from the value cluster.</li>
          <li><strong>MQ ↔ log RT ≈ −0.2</strong>: only ~⅓ difficulty; a real residual survives (prev. slide).</li>
          <li>GSS ties to Gain and action gap (the value-convergence cluster), not to branching.</li>
        </ul>
        <div class="text-xs opacity-50 mt-3">Spearman because Gain / MQ / action gap are zero-inflated &amp; monotone-nonlinear (Pearson understates / can flip sign).</div>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/correlation_matrix.png" alt="Spearman correlation matrix — lc0 metrics" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Takeaways</span>What can we conclude?</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide">
    <thead>
      <tr><th>Claim</th><th>Evidence</th></tr>
    </thead>
    <tbody>
      <tr class="hl"><td><strong>Decision width drives human deliberation</strong> — more than engine value-of-computation</td><td>branching r ≈ +0.20/+0.30 ≫ Gain; policy entropy H(π) +0.24 (subsumed by raw branching); oracle ignores branching, humans don't</td></tr>
      <tr><td>The normative model captures <strong>direction, not the dominant driver</strong></td><td>4/4 feature directions agree, but oracle halts on value-convergence while humans track structure</td></tr>
      <tr><td>Engine value-of-computation tracks RT, but <strong>weakly</strong></td><td>Gain r = +0.085 (growing-tree def: best @ 96 vs @ 1 expansion); GSS r = +0.110</td></tr>
      <tr><td>"More time → worse moves" is <strong>not just</strong> a difficulty confound</td><td>MQ r = −0.151; ~⅔ survives partialling GSS+branching (ρ = −0.150)</td></tr>
    </tbody>
  </table>
  <div class="mdl-found">Humans look <em>resource-rational about the width of the decision</em>; the value-search model explains the easy direction but misses what most strongly paces human thought. <span class="opacity-50">Open: selection-vs-blunder split of the MQ residual; strength-matched (SF-2000) oracle on the 10K matched-FEN run.</span></div>
</div>
