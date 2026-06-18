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

<div class="mt-2 text-lg opacity-80">Read quantities off an lc0 search tree on the same position — do they pace how long a human thinks?</div>

<div class="mt-4 text-sm opacity-50">~199K lc0 search trees on 2023 human-game FENs → ~211K joined moves.</div>

<!--
Source: reports/engine.md (R-MOVETIME-MODEL). Numbers verbatim (run 2026-06-18, ~199K trees / ~211K joined).
Figures: ../public/figures/ (gain_vs_rt, mq_vs_rt [3-panel: global·ply·GSS], gss_vs_rt, actiongap_vs_rt, correlation_matrix).
Arc: setup (4 readouts) -> what *is* Gain (concept SVG) -> Gain result -> MQ result -> "is it difficulty?" ->
companions (GSS/gap) -> oracle vs human -> how the variables relate -> takeaways (3 cards).
-->

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">The setup</span>Four quantities read off the same search tree</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">For each human position we run an lc0 search and read four numbers off the tree — then ask which (if any) tracks the human's think time on that move.</p>
  <table class="mdl-table mdl-table--wide" style="margin-top:0.2rem">
    <thead>
      <tr><th>Metric</th><th>What it measures (lc0 tree)</th><th class="num">r with log RT</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>GSS</strong> — greedy stop step</td><td>how many expansions until the search locks onto its best move</td><td class="num">+0.110</td></tr>
      <tr><td><strong>Gain</strong> — value of computation</td><td>how much full search beats its own first guess (next slide)</td><td class="num">+0.085</td></tr>
      <tr><td><strong>MQ</strong> — move quality</td><td>how far the human's <em>played</em> move sits below engine-best</td><td class="num">−0.151</td></tr>
      <tr><td><strong>Action gap</strong></td><td>1-ply value-head margin between the best and 2nd-best move</td><td class="num">−0.058</td></tr>
    </tbody>
  </table>
  <p class="mdl-tree-cut">Every value-search quantity is <strong>weak</strong> (|r| ≲ 0.11), and MQ runs the "wrong" way. The strong tree-derived signal turns out to be <em>structural</em> — the policy entropy H(π) (last slides). <span class="opacity-50">98% UCI match.</span></p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Concept · value of computation</span>What is "Gain"?</div>
      <p class="mdl-lead">Gain asks: <strong>how much value does the full search find beyond its very first guess?</strong> Read from one growing tree, at two budgets:</p>
      <div class="mdl-text">
        <ol>
          <li>The <strong>shallow tree</strong> (after 1 expansion) names a <strong>shallow action a₁</strong> — the first move the search expands.</li>
          <li>The <strong>deep tree</strong> (after 96) names the <strong>deep action a₉₆</strong> = <code>argmax final_Q</code>.</li>
          <li>Score <em>both</em> on the deep tree's converged Q and subtract.</li>
        </ol>
      </div>
      <p class="mdl-design"><span class="lbl">Gain</span> Q₉₆(a₉₆) − Q₉₆(a₁) &nbsp;≥ 0</p>
      <p class="mdl-tree-cut">a₁ is read <em>from the shallow tree</em>, so it is always in the deep tree too — <strong>no unvisited-0 to read, no filtering</strong>. (Two earlier definitions took a₁ from elsewhere and hit a spurious <code>Gain≈1.0</code> spike + RT dip.)</p>
    </div>
    <div class="mdl-figbox">
      <svg viewBox="0 0 380 212" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto">
        <text x="92" y="13" text-anchor="middle" font-size="11" font-weight="700" fill="#475569">shallow tree · 1 expansion</text>
        <text x="290" y="13" text-anchor="middle" font-size="11" font-weight="700" fill="#475569">deep tree · 96 expansions</text>
        <!-- LEFT: after 1 expansion -->
        <circle cx="92" cy="34" r="8" fill="#0f172a"/>
        <g stroke="#cbd5e1" stroke-width="1.5" fill="none">
          <line x1="92" y1="42" x2="42" y2="74"/>
          <line x1="92" y1="42" x2="110" y2="74"/>
          <line x1="92" y1="42" x2="142" y2="74"/>
        </g>
        <line x1="92" y1="42" x2="74" y2="74" stroke="#6366f1" stroke-width="2.6"/>
        <g fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5">
          <circle cx="42" cy="82" r="7"/>
          <circle cx="110" cy="82" r="7"/>
          <circle cx="142" cy="82" r="7"/>
        </g>
        <circle cx="74" cy="82" r="8.5" fill="#6366f1"/>
        <text x="74" y="104" text-anchor="middle" font-size="10" fill="#6366f1" font-weight="700">a₁</text>
        <text x="74" y="116" text-anchor="middle" font-size="8" fill="#475569">first move expanded</text>
        <!-- RIGHT: after 96 expansions -->
        <circle cx="290" cy="34" r="8" fill="#0f172a"/>
        <g stroke="#94a3b8" stroke-width="1.5" fill="none">
          <line x1="290" y1="42" x2="240" y2="74"/>
          <line x1="290" y1="42" x2="272" y2="74"/>
          <line x1="290" y1="42" x2="340" y2="74"/>
        </g>
        <line x1="290" y1="42" x2="308" y2="74" stroke="#6366f1" stroke-width="2.6"/>
        <g fill="#94a3b8">
          <circle cx="240" cy="82" r="7"/>
          <circle cx="340" cy="82" r="7"/>
        </g>
        <circle cx="272" cy="82" r="7" fill="#e2e8f0" stroke="#6366f1" stroke-width="2" stroke-dasharray="2.5 1.5"/>
        <circle cx="308" cy="82" r="8.5" fill="#6366f1"/>
        <g stroke="#c7d2fe" stroke-width="1.3" fill="none">
          <line x1="308" y1="90" x2="293" y2="118"/>
          <line x1="308" y1="90" x2="308" y2="118"/>
          <line x1="308" y1="90" x2="323" y2="118"/>
        </g>
        <g fill="#a5b4fc">
          <circle cx="293" cy="124" r="5"/>
          <circle cx="308" cy="124" r="5"/>
          <circle cx="323" cy="124" r="5"/>
        </g>
        <text x="272" y="104" text-anchor="middle" font-size="10" fill="#6366f1" font-weight="700">a₁</text>
        <text x="308" y="146" text-anchor="middle" font-size="10" fill="#6366f1" font-weight="700">a₉₆</text>
        <text x="308" y="158" text-anchor="middle" font-size="8" fill="#475569">deep best</text>
        <!-- bottom formula -->
        <line x1="20" y1="176" x2="360" y2="176" stroke="#e2e8f0" stroke-width="1"/>
        <text x="190" y="196" text-anchor="middle" font-size="12" fill="#0f172a">Gain = Q₉₆(a₉₆) − Q₉₆(a₁)</text>
        <text x="190" y="208" text-anchor="middle" font-size="8.5" fill="#475569">both scored on the 96-expansion converged Q</text>
      </svg>
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
      <div class="mdl-title"><span class="mdl-kicker">Result · Gain</span>More value-of-search → longer thinks</div>
      <p class="mdl-lead">Does the value the search adds over its first guess track human think time? Plot mean RT against Gain.</p>
      <div class="mdl-text">
        <ul>
          <li><strong>r = +0.085</strong>, positive and <strong>monotone in every ply tertile</strong> — the direction Russek-style accounts predict.</li>
          <li>Gain = 0 in ~⅔ of positions: the first expansion already lands on the search-best move.</li>
          <li>Real but <strong>weak</strong> — value-of-computation is a minor pacing signal.</li>
        </ul>
      </div>
      <p class="mdl-tree-cut">The old pathological dip at the top bin is gone — it was an artifact of reading an unvisited 0 (prev. slide), not a property of human deliberation.</p>
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
      <div class="mdl-title"><span class="mdl-kicker">Result · move quality</span>Longer thinks → <em>worse</em> moves</div>
      <p class="mdl-lead">MQ is how far the human's <em>played</em> move sits below engine-best (≤ 0). Plotted as the <strong>outcome</strong> of think time — the same relationship, segmented three ways.</p>
      <div class="mdl-text">
        <ul>
          <li><strong>r(MQ, log RT) = −0.151</strong> — the longer the think, the worse the move.</li>
          <li>Engine-best rate falls <strong>0.62 → 0.49</strong> across move-time deciles.</li>
          <li>Three panels: <strong>global · by ply tertile · by GSS difficulty stratum</strong> — same plot, different grouping.</li>
        </ul>
      </div>
      <p class="mdl-tree-cut">A paradox? The standing read is a pure <strong>difficulty confound</strong> — hard positions take longer <em>and</em> get worse moves. The third panel tests it → next slide.</p>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/mq_vs_rt.png" alt="MQ vs RT — global, by ply, by GSS stratum" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Is it just difficulty?</span>The negative slope survives difficulty controls</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">If "more time → worse" were <em>only</em> difficulty, the slope should vanish once we condition on a difficulty proxy. It doesn't.</p>
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Conditioning</th><th class="num">Spearman ρ(MQ, log RT)</th></tr>
      </thead>
      <tbody>
        <tr><td>overall (pooled)</td><td class="num">−0.232</td></tr>
        <tr><td>within GSS strata (easy / med / hard)</td><td class="num">−0.235 / −0.202 / −0.188</td></tr>
        <tr><td>partial | GSS</td><td class="num">−0.213</td></tr>
        <tr><td>partial | branching</td><td class="num">−0.160</td></tr>
        <tr class="hl"><td>partial | GSS + branching</td><td class="num">−0.150</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <ul>
        <li>Negative inside <strong>every</strong> GSS stratum; conditioning on GSS removes only ~7%.</li>
        <li>GSS is even a <em>poor</em> difficulty proxy — its "easy" stratum has the <strong>worst</strong> MQ.</li>
        <li>Branching does more, but the joint partial still leaves <strong>~⅔ of the effect</strong>.</li>
      </ul>
    </div>
  </div>
  <p class="mdl-tree-cut">Not a pure confound: a real residual survives both controls → <strong>selection / uncertainty</strong> (people deliberate precisely when unsure, and uncertainty predicts errors beyond objective difficulty).</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Companion values</span>Search effort up, decisiveness flat</div>

<div class="mdl-content">
  <p class="mdl-lead">The other two readouts: GSS (effort to find the best move) and the action gap (how decided the position is at 1-ply).</p>
  <div class="grid grid-cols-2 gap-6 items-center" style="flex:1; min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/gss_vs_rt.png" alt="greedy stop step vs RT" /></div>
    <div class="mdl-figbox"><img src="../public/figures/actiongap_vs_rt.png" alt="action gap vs RT" /></div>
  </div>
  <p class="mdl-tree-cut">GSS rises monotonically with RT (r = +0.110, full range — no cap); the action gap is flat-to-weak (r = −0.058). More search effort, longer thinks — but the value margin barely matters.</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Where they diverge</span>The oracle stops on value; humans on structure</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">Same four features point the <strong>same direction</strong> for the normative oracle's stop step and for human RT — but the <em>dominant driver</em> is opposite.</p>
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Feature</th><th class="num">r(oracle stop)</th><th class="num">r(human RT)</th></tr>
      </thead>
      <tbody>
        <tr class="hl"><td>branching (decision width)</td><td class="num">+0.014</td><td class="num">+0.195</td></tr>
        <tr><td>material</td><td class="num">+0.011</td><td class="num">+0.039</td></tr>
        <tr class="hl"><td>gain_depth (value gained)</td><td class="num">+0.233</td><td class="num">+0.096</td></tr>
        <tr><td>action gap (toptwo)</td><td class="num">−0.290</td><td class="num">−0.064</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <ul>
        <li>The oracle halts on the <strong>value landscape</strong> (gain_depth, action gap) — it ignores branching.</li>
        <li>Humans deliberate on <strong>structural width</strong> (branching dominates) — and barely track Gain.</li>
        <li>4/4 directions agree → the model gets <em>direction</em>, not <em>mechanism</em>.</li>
      </ul>
      <div class="text-xs opacity-50 mt-2">Tier A: n = 39,668 LMCOS trees · Tier B smoke: r(oracle stop, log RT) = +0.091, n = 497.</div>
    </div>
  </div>
  <p class="mdl-tree-cut">The oracle stops when the value gap is <em>decided</em>; the human deliberates when the move set is <em>wide</em>.</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">How the variables relate</span>Width is the connective tissue</div>
      <p class="mdl-lead">A rank-correlation map of every tree metric, board structure, and log RT. What clusters with human think time?</p>
      <div class="mdl-text">
        <ul>
          <li><strong>Branching ↔ RT</strong> is the strongest tie — stronger than any engine metric.</li>
          <li><strong>H(π)</strong> (policy-prior entropy, no search) is the best <em>tree-derived</em> predictor (+0.24) — but it's a width signal (ρ +0.59 with branching) that the <strong>raw legal-move count subsumes</strong> (partial | branching ≈ +0.05).</li>
          <li>GSS / Gain / action gap form a separate, weak <em>value-convergence</em> cluster.</li>
        </ul>
      </div>
      <p class="mdl-tree-cut">No policy-weighted "effective width" beats the raw legal-move count — branching <em>is</em> the operative width variable.</p>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/correlation_matrix.png" alt="Spearman correlation matrix — lc0 metrics" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Wrap-up</span>Takeaways — what paces human thought</div>

<div class="mdl-content">
  <div class="mdl-three">
    <div class="mdl-issue">
      <span class="mdl-issue-n">Finding 1</span>
      <h3 class="mdl-issue-h">Width &gt; value-of-search</h3>
      <p class="mdl-issue-found">Decision <strong>width</strong> (branching) paces deliberation far more than any realized value-of-computation.</p>
      <p class="mdl-issue-label">Evidence</p>
      <ul class="mdl-issue-list">
        <li>branching r ≈ +0.20 / +0.33 ≫ Gain +0.085, GSS +0.110</li>
        <li>H(π) +0.24 but subsumed by the raw count</li>
      </ul>
    </div>
    <div class="mdl-issue">
      <span class="mdl-issue-n">Finding 2</span>
      <h3 class="mdl-issue-h">Direction, not mechanism</h3>
      <p class="mdl-issue-found">The normative model gets the sign of every effect right but the <strong>dominant driver</strong> wrong.</p>
      <p class="mdl-issue-label">Evidence</p>
      <ul class="mdl-issue-list">
        <li>4/4 feature directions agree</li>
        <li>oracle halts on value-convergence; humans on structure</li>
      </ul>
    </div>
    <div class="mdl-issue">
      <span class="mdl-issue-n">Finding 3</span>
      <h3 class="mdl-issue-h">Not just difficulty</h3>
      <p class="mdl-issue-found">"More time → worse moves" is <strong>not</strong> a pure difficulty confound — a real residual survives.</p>
      <p class="mdl-issue-label">Evidence</p>
      <ul class="mdl-issue-list">
        <li>MQ r = −0.151; partial | GSS+branching = −0.150</li>
        <li>~⅔ of the effect intact → selection / uncertainty</li>
      </ul>
    </div>
  </div>
  <p class="mdl-tree-cut">Humans look <strong>resource-rational about the width of the decision</strong>. <span class="opacity-50">Open: selection-vs-blunder split of the MQ residual; strength-matched SF-2000 oracle on the 10K matched-FEN run.</span></p>
</div>
