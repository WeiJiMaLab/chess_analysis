---
theme: default
title: Oracle stop step vs human RT
info: Does a normative DP oracle's stop step track human think time?
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search</span>

# Oracle stop step vs human RT

<div class="mt-2 text-lg opacity-80">Does a normative DP oracle's stop step track human think time?</div>

<div class="mt-4 text-sm opacity-50">Source: <code>reports/oracle-stop-vs-human-rt.md</code> (R-ORACLE-RT)</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Overview</span>The validation tiers</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide" style="margin-bottom:1.1rem">
    <thead>
      <tr><th>Tier</th><th>Claim</th><th>Requires</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>A</strong></td><td>Same features predict oracle stop &amp; RT directionally</td><td>LMCOS trees</td></tr>
      <tr><td><strong>B</strong></td><td>Same FEN: oracle stop ↔ human RT</td><td>Human FEN trees</td></tr>
      <tr><td><strong>C</strong></td><td>Trained controller ↔ human RT</td><td>Controller on human positions</td></tr>
    </tbody>
  </table>
  <div class="mdl-found">Headline: <strong>Tier A — 4/4 feature directions match</strong>; <strong>Tier B smoke — r(oss, log RT) = +0.091</strong> (sig at n=497). The oracle stops on <em>value-landscape</em> features; humans deliberate on <em>structural complexity</em>.</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Tier A</span>Same features drive oracle stop &amp; human RT</div>

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

<div class="mdl-title"><span class="mdl-kicker">Tier B</span>Same FEN: oracle stop ↔ human RT</div>

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

<div class="mdl-title"><span class="mdl-kicker">Status</span>Follow-up &amp; next steps</div>

<div class="mdl-content">
  <div class="mdl-figrow mdl-figrow--wide-left">
    <div class="mdl-text">
      <p class="mdl-lead">SF-2000 (strength-matched, on hold).</p>
      <ul>
        <li>LC0 (~3000+ ELO) may "see through" positions a ≥2000-ELO human pool finds hard.</li>
        <li>A Stockfish ELO=2000 oracle on the <em>same</em> FENs may align deliberation better.</li>
        <li>Gated on the 10K matched-position r. If SF-2000 <em>also</em> fails → engine VOC fundamentally misses human deliberation (not a strength artifact).</li>
      </ul>
    </div>
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Step</th><th>Status</th></tr>
      </thead>
      <tbody>
        <tr><td>Tier A features + converged_expansions + 72 tests</td><td>✅ done</td></tr>
        <tr><td>Tier B: 10K FENs as 20 shards</td><td>✅ submitted</td></tr>
        <tr class="dim"><td>Tier B: oracle + join + plots A/B/C</td><td>⬜ after jobs</td></tr>
        <tr class="dim"><td>SF-2000 oracle</td><td>⬜ gated on Tier B</td></tr>
      </tbody>
    </table>
  </div>
</div>
