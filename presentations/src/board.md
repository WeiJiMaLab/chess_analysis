---
theme: default
title: Human move time — what board features predict it
info: Model-free: which board features predict how long humans think.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search · human move time</span>

# What board features predict how long humans think?

<div class="mt-2 text-lg opacity-80">Model-free — no engine. Which features of the position alone predict think time?</div>

<div class="mt-4 text-sm opacity-50">1.97M Lichess games (60+0, Elo ≥ 2000) → 135M non-zero-time moves.</div>

<!--
Source: reports/board.md (R-MOVETIME-BOARD). Numbers verbatim. Figures: ../public/figures/.
Arc: how is RT distributed (log-normal) -> per-feature dashboards -> which predicts (width wins) ->
how the features relate (branching not reducible; raw count is the operative width).
-->

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Distribution</span>Think time is log-normal — Weber's Law</div>
      <p class="mdl-lead">Before asking <em>what</em> predicts think time, fix the right scale. How is move time distributed across 135M moves?</p>
      <div class="mdl-text">
        <ul>
          <li>log(move time) is approximately <strong>normal</strong> → move time is <strong>log-normal</strong>, not exponential or uniform.</li>
          <li>Players scale thinking <strong>multiplicatively</strong> with difficulty (Weber's Law).</li>
        </ul>
      </div>
      <p class="mdl-tree-cut">So we work in <strong>log(RT)</strong> throughout — the natural scale for a multiplicative process.</p>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/movetime_logmt_qq.png" alt="log move-time histogram + QQ" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Per feature</span>Which board features move with think time?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">Each candidate feature gets the same quantile-bin dashboard (global + by ply tertile). Eyeball which bends RT.</p>
  <div class="grid grid-cols-2 gap-3 items-center" style="flex:1; min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/clock_vs_movetime.png" alt="player clock vs move time" /></div>
    <div class="mdl-figbox"><img src="../public/figures/legal_moves_vs_movetime.png" alt="legal moves vs move time" /></div>
    <div class="mdl-figbox"><img src="../public/figures/own_material_vs_movetime.png" alt="own material vs move time" /></div>
    <div class="mdl-figbox"><img src="../public/figures/ply_vs_movetime.png" alt="ply vs move time" /></div>
  </div>
  <p class="mdl-tree-cut">Player clock · legal moves · own non-pawn material · game stage (ply). The steepest, cleanest rise is <strong>legal moves</strong> — the next slide quantifies it.</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Predictors</span>Decision <em>width</em> is the strongest predictor</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">"Width" = the number of legal moves: how many options the player must weigh. More options → more to evaluate → longer think.</p>
  <div class="mdl-figrow mdl-figrow--wide-left">
    <table class="mdl-table mdl-table--wide">
      <thead>
        <tr><th>Feature (from position)</th><th class="num">r with log(RT)</th><th>Intuition</th></tr>
      </thead>
      <tbody>
        <tr class="hl"><td>legal moves (decision width)</td><td class="num">+0.20</td><td>More candidates → more to weigh</td></tr>
        <tr><td>Gain (ΔUC, depth 5)</td><td class="num">+0.10</td><td>Deeper search finds a better move</td></tr>
        <tr><td>own material</td><td class="num">+0.04</td><td>More pieces → more interactions</td></tr>
        <tr><td>action gap (toptwo)</td><td class="num">−0.06</td><td>One move clearly best → less to weigh</td></tr>
      </tbody>
    </table>
    <div class="mdl-figbox">
      <svg viewBox="0 0 240 150" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto">
        <text x="60" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#475569">narrow</text>
        <text x="180" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#475569">wide</text>
        <!-- narrow decision: few options -->
        <circle cx="60" cy="34" r="8" fill="#0f172a"/>
        <g stroke="#cbd5e1" stroke-width="2" fill="none">
          <line x1="60" y1="42" x2="38" y2="86"/>
          <line x1="60" y1="42" x2="60" y2="86"/>
          <line x1="60" y1="42" x2="82" y2="86"/>
        </g>
        <g fill="#cbd5e1"><circle cx="38" cy="92" r="5"/><circle cx="60" cy="92" r="5"/><circle cx="82" cy="92" r="5"/></g>
        <text x="60" y="118" text-anchor="middle" font-size="9" fill="#475569">few legal moves</text>
        <text x="60" y="131" text-anchor="middle" font-size="9" fill="#475569" font-weight="700">quick</text>
        <!-- wide decision: many options -->
        <circle cx="180" cy="34" r="8" fill="#0f172a"/>
        <g stroke="#a5b4fc" stroke-width="2" fill="none">
          <line x1="180" y1="42" x2="132" y2="86"/>
          <line x1="180" y1="42" x2="148" y2="86"/>
          <line x1="180" y1="42" x2="164" y2="86"/>
          <line x1="180" y1="42" x2="180" y2="86"/>
          <line x1="180" y1="42" x2="196" y2="86"/>
          <line x1="180" y1="42" x2="212" y2="86"/>
          <line x1="180" y1="42" x2="228" y2="86"/>
        </g>
        <g fill="#6366f1"><circle cx="132" cy="92" r="5"/><circle cx="148" cy="92" r="5"/><circle cx="164" cy="92" r="5"/><circle cx="180" cy="92" r="5"/><circle cx="196" cy="92" r="5"/><circle cx="212" cy="92" r="5"/><circle cx="228" cy="92" r="5"/></g>
        <text x="180" y="118" text-anchor="middle" font-size="9" fill="#475569">many legal moves</text>
        <text x="180" y="131" text-anchor="middle" font-size="9" fill="#6366f1" font-weight="700">long think</text>
      </svg>
    </div>
  </div>
  <p class="mdl-tree-cut">All four are computed from the position, none from RT. Replicates Russek et al. (2022): r(log RT, ΔUC) = +0.096 at depth 5. Every effect holds <strong>within ply tertiles</strong>.</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">How the features relate</span>Width isn't reducible to game stage</div>
      <p class="mdl-lead">Could "width" just be a proxy for the game-stage complex (material/clock fall as games go on)? The rank-correlation map says no.</p>
      <div class="mdl-text">
        <ul>
          <li><strong>Legal moves ↔ log RT is the strongest tie</strong> (Spearman ρ ≈ +0.26).</li>
          <li>Material, clock, and ply move together — largely redundant with game stage — yet width's RT coupling is <strong>not</strong> reducible to that complex.</li>
          <li><strong>Raw count is the operative width:</strong> lc0's policy entropy H(π) — a "smarter" effective width — predicts RT <em>worse</em> and is subsumed by the raw count (partial ρ ≈ +0.05).</li>
        </ul>
      </div>
      <p class="mdl-tree-cut">The <strong>width of the decision</strong> is the strongest board-feature predictor of think time — and no refinement of it beats the raw legal-move count.</p>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/board_feature_corr.png" alt="board-feature Spearman correlation matrix" /></div>
  </div>
</div>
