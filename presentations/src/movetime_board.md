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

<div class="mt-2 text-lg opacity-80">Model-free: which features of the position predict think time?</div>

<div class="mt-4 text-sm opacity-50">1.97M Lichess games (60+0, Elo ≥ 2000) → 135M non-zero-time moves.</div>

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
          <li>log(move time) is approximately <strong>normal</strong> → move time is <strong>log-normal</strong>, not exponential or uniform.</li>
          <li>Players scale thinking time <strong>multiplicatively</strong> with difficulty (Weber's Law).</li>
          <li>Justifies working in <strong>log(RT)</strong> throughout.</li>
        </ul>
        <div class="text-xs opacity-50 mt-3">n = 135M non-zero-time moves from 1.97M games</div>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/movetime_logmt_qq.png" alt="log move-time histogram + QQ" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Per feature</span>Which board features move with think time?</div>

<div class="mdl-content">
  <div class="grid grid-cols-2 gap-3 items-center" style="flex:1; min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/clock_vs_movetime.png" alt="player clock vs move time" /></div>
    <div class="mdl-figbox"><img src="../public/figures/npossiblemoves_vs_movetime.png" alt="branching vs move time" /></div>
    <div class="mdl-figbox"><img src="../public/figures/self_pieces_exc_pawns_vs_movetime.png" alt="own material vs move time" /></div>
    <div class="mdl-figbox"><img src="../public/figures/ply_vs_movetime.png" alt="ply vs move time" /></div>
  </div>
  <div class="mdl-text" style="text-align:center; margin-top:0.4rem">Player clock · branching · own non-pawn material · game stage (ply) — each the canonical quantile-bin dashboard (global + by ply tertile).</div>
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
        <tr><td>action gap (toptwo)</td><td class="num">−0.06</td><td>One move clearly best → less to weigh</td></tr>
      </tbody>
    </table>
    <div class="mdl-text">
      <p class="mdl-lead">All computed from the position, none from RT.</p>
      <ul>
        <li><strong>Branching is the strongest single predictor</strong> — the <em>width</em> of the decision.</li>
        <li>Replicates Russek et al. (2022): r(log RT, ΔUC) = +0.096 at depth 5.</li>
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
      <div class="mdl-title"><span class="mdl-kicker">Relationships</span>How do the board features relate?</div>
      <div class="mdl-text">
        <ul>
          <li><strong>Branching ↔ log RT is the strongest tie</strong> (Spearman ρ ≈ +0.26).</li>
          <li>Material, clock, and ply move together — material/clock fall as games progress, so they're largely redundant with game stage.</li>
          <li>Branching's RT coupling is <strong>not</strong> reducible to that ply/material complex.</li>
        </ul>
        <div class="text-xs opacity-50 mt-3">Spearman (rank): structural features are skewed/bounded. n = 1M reservoir sample.</div>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/board_feature_corr.png" alt="board-feature Spearman correlation matrix" /></div>
  </div>
  <div class="mdl-found">The <strong>width of the decision</strong> (branching) is the strongest board-feature predictor of human think time — stronger than realized value-of-search, and not reducible to the ply/material/clock complex.</div>
</div>
