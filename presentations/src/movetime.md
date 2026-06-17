---
theme: default
title: Human move time — empirical & LC0 analysis
info: How humans allocate think time, and whether engine-derived metrics track it.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search</span>

# Human move time — empirical &amp; LC0 analysis

<div class="mt-2 text-lg opacity-80">What predicts how long humans think, and do engine-derived metrics track it?</div>

<div class="mt-4 text-sm opacity-50">Empirical (board features) → model-derived (LC0 search trees). One deck, three sections.</div>

---
layout: default
class: mdl-section
---

<span class="mdl-kicker">Section I</span>

# Data filtering

<div class="mdl-section-sub">How the human dataset is selected, and the two analysis subsets that follow.</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Data</span>Dataset &amp; filtering</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Filter</th><th>Choice</th></tr>
      </thead>
      <tbody>
        <tr><td>Time control</td><td><strong>60+0</strong> (10-min), no berserk</td></tr>
        <tr><td>Skill tranche</td><td>fixed Elo <strong>≥ 2000</strong></td></tr>
        <tr><td>Move time</td><td><code>move_time &gt; 0</code> (drop premove/instant)</td></tr>
        <tr><td>Scale</td><td>~1.97M games → <strong>135M</strong> scored moves</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <p class="mdl-lead">One control removes the time-control &amp; skill confounds; analyses then run model-free first.</p>
      <ul>
        <li><strong>Empirical subset</strong> — the full 135M moves; RT vs board features (no engine).</li>
        <li><strong>LC0-tree subset</strong> — positions that have a generated lc0 search tree (~100K), joined to human RT on the 4-field FEN; for MQ, matched to the human's <em>played</em> move.</li>
        <li>Think time is heavy-tailed → all analyses use <strong>log RT</strong>.</li>
      </ul>
    </div>
  </div>
</div>

---
layout: default
class: mdl-section
---

<span class="mdl-kicker">Section II</span>

# Analysis (empirical)

<div class="mdl-section-sub">Zero-parameter findings: what board features predict human think time, with no model at all.</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Distribution</span>Think time is log-normal — Weber's Law</div>
      <div class="mdl-text">
        <ul>
          <li>Think time is approximately <strong>log-normal</strong>, not exponential or uniform.</li>
          <li>Players scale thinking time <strong>multiplicatively</strong> with difficulty — consistent with Weber's Law.</li>
          <li>Justifies working in <strong>log(RT)</strong> throughout.</li>
        </ul>
        <div class="text-xs opacity-50 mt-3">n = 135M non-zero-time moves from 1.97M games</div>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/log_movetime_histogram.png" alt="log move-time histogram" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Predictors</span>What predicts when humans think longer?</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Feature (from position)</th><th class="num">r with log(RT)</th><th>Intuition</th></tr>
      </thead>
      <tbody>
        <tr class="hl"><td>branching factor</td><td class="num">+0.20</td><td>More candidates → more uncertainty</td></tr>
        <tr><td>gain_depth (ΔUC@5)</td><td class="num">+0.10</td><td>Deeper search finds a better move</td></tr>
        <tr><td>own material</td><td class="num">+0.04</td><td>More pieces → more interactions</td></tr>
        <tr><td>toptwo (action gap)</td><td class="num">−0.06</td><td>One move clearly best → less to weigh</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <p class="mdl-lead">Four board features — all computed from the position, none from RT.</p>
      <ul>
        <li>Replicates Russek et al. (2022): r(log RT, ΔUC) = +0.096 at depth 5 (vs their depth 15).</li>
        <li>All effects hold <strong>within ply tertiles</strong>.</li>
      </ul>
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
      <div class="mdl-title"><span class="mdl-kicker">Joint structure</span>Correlation matrix — the full picture</div>
      <div class="mdl-text">
        <ul>
          <li>Ply ↔ own material: <strong>−0.81</strong> (strong, expected).</li>
          <li>gain_depth ↔ MQ: <strong>−0.42</strong>; toptwo ↔ MQ: <strong>−0.30</strong>.</li>
          <li>branching ↔ log RT: <strong>+0.20</strong>; gain_depth ↔ log RT: <strong>+0.10</strong>.</li>
          <li>MQ ↔ log RT: <strong>−0.12</strong>; toptwo ↔ log RT: <strong>−0.06</strong>.</li>
        </ul>
        <div class="text-xs opacity-50 mt-3">n = 1,000,000 · Pearson r</div>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/correlation_matrix.png" alt="correlation matrix" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Takeaways</span>Three counterintuitive findings</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide">
    <thead>
      <tr><th>#</th><th>Finding</th><th>Reading</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>1</strong></td><td>More clock → worse moves (r(MQ, clock) = −0.091, every ply tertile)</td><td>A difficulty confound: more time was spent on low-VOC positions.</td></tr>
      <tr><td><strong>2</strong></td><td>67% of positions have VOC = 0 at depth 5</td><td>Depth-1 already finds the move — yet humans deliberate.</td></tr>
      <tr class="hl"><td><strong>3</strong></td><td>Branching beats VOC as an RT predictor (+0.20 vs +0.10)</td><td>Width of the decision drives deliberation more than depth value.</td></tr>
    </tbody>
  </table>
  <div class="mdl-found">Candidate-set <em>uncertainty</em> (branching), not realized search-depth value, looks like the dominant driver of human think time.</div>
</div>

---
layout: default
class: mdl-section
---

<span class="mdl-kicker">Section III</span>

# Analysis (LC0)

<div class="mdl-section-sub">Model-derived metrics read from the lc0 search trees on the human FENs: OSS, VOC, MQ, action gap — vs human RT.</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Overview</span>The question &amp; the metrics</div>

<div class="mdl-content">
  <p class="mdl-lead" style="max-width:48rem">Do engine-defined search quantities — read from the lc0 tree on the <em>same</em> position — track how long humans think?</p>
  <table class="mdl-table mdl-table--wide" style="margin-top:0.4rem">
    <thead>
      <tr><th>Metric</th><th>Definition (LC0 tree)</th><th class="num">r with log RT</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>OSS</strong> — oracle stop step</td><td>budgeted DP oracle's optimal halt step (budget 96)</td><td class="num">+0.119</td></tr>
      <tr><td><strong>VOC</strong> — value of computation</td><td><code>final_Q(deep best) − final_Q(1-ply best)</code> ≥ 0</td><td class="num">+0.075</td></tr>
      <tr><td><strong>MQ</strong> — move quality</td><td><code>final_Q(played) − final_Q(best)</code> ≤ 0</td><td class="num">−0.152</td></tr>
      <tr><td><strong>Action gap</strong></td><td>top1 − top2 of the children's 1-ply value-head backup</td><td class="num">−0.067</td></tr>
    </tbody>
  </table>
  <div class="mdl-text" style="margin-top:0.5rem">VOC and action gap share the <strong>same 1-ply value-head lookahead</strong> basis; MQ is the only per-played-move quantity. <span class="opacity-50">100K lc0 trees → ~108K joined human moves; 98.1% of moves match a root move.</span></div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Move quality</span>MQ vs think time</div>
      <div class="mdl-text">
        <ul>
          <li><strong>MQ = final_Q(played) − final_Q(best) ≤ 0</strong> (0 = human played the engine-best move); large mass at 0.</li>
          <li>Plotted as the <strong>outcome of think time</strong>: mean MQ vs log RT.</li>
          <li><strong>r(MQ, log RT) = −0.152</strong> — longer thinks associate with worse moves: a <em>difficulty confound</em> (hard positions take longer <em>and</em> yield worse moves), not a sign bug.</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/mq_vs_rt.png" alt="MQ vs RT" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Value of computation</span>VOC vs think time</div>
      <div class="mdl-text">
        <ul>
          <li><strong>VOC = final_Q(deep best) − final_Q(1-ply best) ≥ 0</strong> — how much deep search improves on the shallow 1-ply value-head pick.</li>
          <li><strong>Weak but real positive</strong> with log RT (r = +0.075): more value-of-computation → longer thinks (Russek-style direction).</li>
          <li>Zero-inflated: VOC = 0 in ~⅔ of positions (search confirms the 1-ply choice).</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/voc_vs_rt.png" alt="VOC vs RT" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Companion values</span>Oracle stop step &amp; action gap</div>

<div class="mdl-content">
  <div class="grid grid-cols-2 gap-6 items-center" style="flex:1; min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/oss_vs_rt.png" alt="OSS vs RT" /></div>
    <div class="mdl-figbox"><img src="../public/figures/actiongap_vs_rt.png" alt="Action gap vs RT" /></div>
  </div>
  <div class="mdl-text" style="text-align:center; margin-top:0.5rem">Oracle stop step (left, capped at 64; binned in groups of 5) and action gap (right) on the same lc0-tree RT subset — quantile-binned globally and by ply tertile (canonical Analyzer 1×2 dashboard, sparse bins n &lt; 100 dropped).</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Oracle stop · Tier A</span>Same features drive oracle stop &amp; human RT</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Feature</th><th class="num">r(oracle stop)</th><th class="num">r(human RT)</th><th>dir</th></tr>
      </thead>
      <tbody>
        <tr><td>branching</td><td class="num">+0.014</td><td class="num">+0.195</td><td>✓ tiny</td></tr>
        <tr><td>material</td><td class="num">+0.011</td><td class="num">+0.039</td><td>✓ tiny</td></tr>
        <tr class="hl"><td>gain_depth (VOC)</td><td class="num">+0.233</td><td class="num">+0.096</td><td>✓</td></tr>
        <tr class="hl"><td>toptwo</td><td class="num">−0.290</td><td class="num">−0.064</td><td>✓</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <p class="mdl-lead"><strong>4/4 correct directions</strong> for oracle_stop_step.</p>
      <ul>
        <li>Branching / material ≈ 0 for the oracle is <em>expected</em> — stopping is driven by <strong>value convergence</strong> (toptwo, gain_depth), not board structure.</li>
        <li>lc0 handles branching natively via its PUCT prior.</li>
      </ul>
      <div class="text-xs opacity-50 mt-3">n = 39,668 LMCOS trees (budget 96)</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Oracle stop · Tier B</span>Same FEN: oracle stop ↔ human RT</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Metric (corr with log RT / oracle)</th><th class="num">value</th></tr>
      </thead>
      <tbody>
        <tr class="hl"><td>r(oracle_stop_step, log RT)</td><td class="num">+0.091 ✓</td></tr>
        <tr><td>gain_depth — human / oracle</td><td class="num">+0.020 / +0.463</td></tr>
        <tr><td>branching — human / oracle</td><td class="num">+0.208 / +0.166</td></tr>
        <tr><td>toptwo — human / oracle</td><td class="num">−0.128 / −0.030</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <p class="mdl-lead">The split sharpens.</p>
      <ul>
        <li>gain_depth strongly predicts the <strong>oracle's</strong> halt (+0.463) but is near-zero for <strong>humans</strong> (+0.020).</li>
        <li>Humans don't natively compute VOC — their RT tracks structural complexity.</li>
        <li>10K trees needed for full statistical power.</li>
      </ul>
      <div class="text-xs opacity-50 mt-3">smoke, n = 497 clean human-FEN trees</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Methods</span>Definitions &amp; caveats</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide" style="margin-bottom:1rem">
    <thead>
      <tr><th>Quantity</th><th>Definition</th><th>Source</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>MQ</strong></td><td><code>final_Q(played) − final_Q(best)</code> (≤ 0)</td><td>LC0 tree (was Stockfish d5 — retired)</td></tr>
      <tr><td><strong>VOC</strong></td><td><code>final_Q(deep best) − final_Q(1-ply best)</code> (≥ 0)</td><td>LC0 tree, 1-ply value-head shallow</td></tr>
      <tr><td><strong>OSS</strong></td><td>budgeted DP oracle optimal_stop_step</td><td>LC0 tree (budget 96)</td></tr>
      <tr><td><strong>Action gap</strong></td><td>top1 − top2 of 1-ply value-head backup</td><td>LC0 tree (root children)</td></tr>
    </tbody>
  </table>
  <div class="mdl-text">
    <ul>
      <li><strong>VOC fix:</strong> the shallow choice is the 1-ply value-head best, <em>not</em> the search trace at step 0 (all-zero pre-search → previously inflated VOC to a spurious ~1.0 mass in losing positions).</li>
      <li>All four read from the lc0 tree on the human FEN; MQ is matched to the human's played move (≈ 98% of moves match a root move).</li>
      <li><strong>Open:</strong> difficulty-residualized partials for MQ; SF-2000 strength-matched oracle (gated on the 10K matched-position run).</li>
    </ul>
  </div>
</div>
