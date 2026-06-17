---
theme: default
title: Human VOC / MQ vs move time
info: Engine value-of-computation and move-quality vs human think time.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search</span>

# Human VOC / MQ vs move time

<div class="mt-2 text-lg opacity-80">Does engine value-of-computation track human deliberation?</div>

<div class="mt-4 text-sm opacity-50">Source: <code>reports/human-voc-mq.md</code> (R-VOC-MQ)</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Overview</span>The question</div>

<div class="mdl-content">
  <p class="mdl-lead" style="max-width:48rem">Do engine-defined <strong>value-of-computation (VOC)</strong> and <strong>move-quality (MQ)</strong> track how long humans think on the same positions?</p>
  <div class="mdl-two" style="height:auto; margin-top:0.6rem">
    <div class="mdl-card mdl-card--neutral">
      <p class="mdl-card-h">Headline</p>
      <div class="body">
        <ul style="padding-left:1.1em; margin:0">
          <li>VOC ↔ log move time: <strong>weak but real positive</strong> (r ≈ +0.10)</li>
          <li>MQ ↔ move time: <strong>negative</strong> — a <em>difficulty confound</em>, not a sign bug</li>
        </ul>
      </div>
    </div>
    <div class="mdl-card mdl-card--neutral">
      <p class="mdl-card-h">Two load-bearing caveats</p>
      <div class="body">
        <ul style="padding-left:1.1em; margin:0">
          <li>The kept <strong>MQ figure is Stockfish depth-5, not LC0</strong></li>
          <li>MQ subset (1M) ≠ tree subset (OSS/VOC) population</li>
        </ul>
      </div>
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
      <div class="mdl-title"><span class="mdl-kicker">Move quality</span>MQ vs move time — the confound</div>
      <div class="mdl-text">
        <ul>
          <li><strong>MQ ≤ 0</strong> by construction; large mass at 0 (human played engine-best move).</li>
          <li><strong>r(MQ, clock) = −0.098</strong>, negative within <em>every</em> ply tertile.</li>
          <li><strong>A difficulty confound, not a sign bug</strong> (N = 1M): Spearman <strong>ρ = −0.119</strong>, survives all controls. Best-move rate falls <strong>0.62 → 0.49</strong> across move-time deciles.</li>
          <li>MQ × MT × branching interaction <strong>β ≈ −0.007 (t = −7)</strong> — branching amplifies, magnitude small.</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox">
      <img src="../public/figures/mq_vs_rt.png" alt="MQ vs RT" />
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
      <div class="mdl-title"><span class="mdl-kicker">Value of computation</span>VOC vs move time</div>
      <div class="mdl-text">
        <ul>
          <li><strong>r(log RT, VOC) = +0.097</strong> — more remaining value-of-computation → longer thinks.</li>
          <li>Direction predicted by Russek-style value-of-computation.</li>
          <li>VOC near-zero-inflated: <strong>VOC &gt; 0.005 in ~33%</strong> of positions (mean +0.100, median 0).</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox">
      <img src="../public/figures/voc_vs_rt.png" alt="VOC vs RT" />
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Companion values</span>OSS &amp; Action Gap</div>

<div class="mdl-content">
  <div class="grid grid-cols-2 gap-6 items-center" style="flex:1; min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/oss_vs_rt.png" alt="OSS vs RT" /></div>
    <div class="mdl-figbox"><img src="../public/figures/actiongap_vs_rt.png" alt="Action gap vs RT" /></div>
  </div>
  <div class="mdl-text" style="text-align:center; margin-top:0.5rem">Oracle stop step (left) and Action Gap (right) — the lc0-tree generated values on the same RT subset, quantile-binned globally and by ply tertile (canonical Analyzer 1×2 dashboard).</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Methods</span>Definitions &amp; caveats</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide" style="margin-bottom:1rem">
    <thead>
      <tr><th>Quantity</th><th>Definition</th><th>Engine</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>MQ</strong> (kept fig)</td><td><code>e_win_taken − e_win_best</code> (≤0)</td><td>Stockfish d5 (not LC0)</td></tr>
      <tr><td><strong>MQ</strong> (LC0)</td><td><code>final_Q(played) − final_Q(best)</code></td><td>LC0 tree</td></tr>
      <tr><td><strong>VOC</strong></td><td><code>V_deep(a_deep) − V_deep(a_shallow)</code></td><td>SF d5/d1 or LC0 tree</td></tr>
    </tbody>
  </table>
  <div class="mdl-text">
    <ul>
      <li><strong>100K run:</strong> Stockfish d5/1, ply 15–75, opp clock ≥ 60s.</li>
      <li>MQ "FULL" plot is a 1M-row subset; ply-tertile cuts (27/56) from the whole-dataset distribution → "Early" panel has no openings.</li>
      <li><strong>Fix recommended:</strong> recompute MQ from <strong>LC0</strong> + report difficulty-residualized partials.</li>
    </ul>
  </div>
</div>
