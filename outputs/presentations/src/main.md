---
theme: default
title: Learned Meta-control of Tree Search
info: Project status — human behavioral analysis and normative DP agent.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: text-left
mdc: true
math: katex
---

<ChessBackground />

<div class="absolute bottom-12 left-14 z-10 text-white">
  <h1 class="m-0" style="line-height: 1.0; font-size: 3rem; color: white !important;">
    Learned Meta-control<br>of Tree Search
  </h1>
  <div class="mt-2 text-lg opacity-80">Yotam Sagiv & Jordan Lei</div>
  <div class="mt-6 text-[10px] font-bold uppercase tracking-widest opacity-40">
    The Great Reunification — sprint plan · June 2026
  </div>
</div>

<!--
Map beats are CLICK-driven (one persistent <ThoughtMap seq=.../> per slide): the camera PANS, children REVEAL,
questions RESOLVE, the next one BLINKS — all on click. map↔content slide changes use a simple fade
transition. Content slides are normal. Set `clicks:` = (sequence length − 1). Sequences live in
ThoughtMap.vue. Source: reports/board.md, allply.md, treesearch.md, engine.md.

Structure (three parts under one root):
  Part 1  How do people actually think?  →  (a) game/board features [ply, legal, clock]
                                             (b) how do we model planning? [engine, gain, action-gap, frac-good]
  Part 2  What is thinking worth?        →  when to stop · optimal stopping step (OSS/GSS) · meta-controller
  Part 3  Do human & normative agree?    →  empty for now (pivot only)

Correlation numbers are final from the allply run (board matrix n=1M; engine matrix n=109,434). The
Part 3 pivot still cites the separate treesearch analysis (n≈65k). Each content slide has THREE parts:
  (a) .mdl-proc      — what we did      (b) .mdl-intuition — how to read it      (c) .mdl-conv — takeaway
-->

---
layout: default
class: mdl-slide
transition: fade
clicks: 5
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="build" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · Game / board features</span>Ply — how far into the game?</div>
      <div class="mdl-proc"><strong>Model-free</strong> — board feature vs human log(RT), all plies. This leaf: <strong>ply</strong> = game stage.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> Game stage — a candidate difficulty proxy.</div>
      <div class="mdl-conv"><span class="lbl">Near-null</span> Ply ↔ log(RT): <strong>+0.08</strong>. Weak — game stage, not decision width.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/board/ply.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="ply" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · Game / board features</span>Legal moves — the width of the decision</div>
      <div class="mdl-proc"><strong># legal moves</strong> vs log(RT) — how many options the player must weigh.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> More options to weigh → longer think.</div>
      <div class="mdl-conv"><span class="lbl">Strongest board feature</span> Legal moves ↔ log(RT): <strong>+0.26</strong>. Holds within ply tertiles.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/board/legal_moves.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="legal" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · Game / board features</span>Clock left — how much time remains?</div>
      <div class="mdl-proc"><strong>Clock (time left)</strong> vs log(RT).</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> Clock falls as the game runs on — tracks game stage.</div>
      <div class="mdl-conv"><span class="lbl">Game-stage complex</span> Clock ↔ log(RT): <strong>−0.16</strong>. Moves with ply — yet width isn't reducible to it.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/board/clock.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="clock" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · Game / board features</span>How the board features relate</div>
      <div class="mdl-proc"><strong>Spearman matrix</strong> over the board features and log(RT), n = 1M.</div>
      <div class="mdl-intuition"><span class="lbl">Key parts</span> <strong>Legal moves +0.26</strong> is the top row's strongest RT tie; ply / clock / game-fraction move together (the game-stage complex) and each is weaker.</div>
      <div class="mdl-conv"><span class="lbl">Width wins</span> Decision width is not reducible to the game-stage complex — no board feature beats the raw legal-move count.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/board/board_feature_corr.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="feats" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · How do we model planning?</span>Model planning as a search tree</div>
      <div class="mdl-proc"><strong>AlphaZero-style PUCT tree</strong>, no rollouts: a heuristic values nodes, a selector expands. Value features are read off the tree.</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> Prior weights <em>which</em> child; value is backed up from leaves. Best-first → narrow &amp; deep; UCB → broad.</div>
      <div class="mdl-conv"><span class="lbl">Now we can ask</span> Uniform priors ⇒ PUCT = <strong>UCB</strong>. With a tree in hand, what does an engine's <em>value</em> say about think-time?</div>
    </div>
    <TreeGrowth />
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="plan" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · How do we model planning?</span>Which engine do we use?</div>
      <div class="mdl-proc"><strong>lc0 → Stockfish</strong> for ~1000× CPU speedup. Cost: prior → uniform, value → N-node SF search (WDL).</div>
      <div class="mdl-intuition"><span class="lbl">Three knobs, kept distinct</span> <strong>N</strong> = leaf-eval nodes · <strong>M</strong>=96 = planning budget · <strong>UCI_Elo</strong> = play handicap.</div>
      <div class="mdl-conv"><span class="lbl">Safe swap</span> RT correlations stable across the change. Only <strong>N</strong> moves the eval; <strong>UCI_Elo is a no-op</strong> (SF-1350 ≡ SF-2000).</div>
    </div>
    <div style="display:flex; flex-direction:column; justify-content:center; gap:1.2rem">
      <div class="mdl-card mdl-card--neutral"><div class="mdl-card-h">lc0 (before)</div><div class="body"><strong>prior:</strong> trained policy head<br><strong>value:</strong> trained value head (WDL)<br><span style="opacity:.6; font-size:.85em">slow on CPU · requires GPU</span></div></div>
      <div class="mdl-card mdl-card--accent"><div class="mdl-card-h">Stockfish (after)</div><div class="body"><strong>prior:</strong> uniform — α-β has no policy head<br><strong>value:</strong> eval→WDL from an N-node search<br><span style="opacity:.6; font-size:.85em">~1000× faster on CPU</span></div></div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="engine" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · How do we model planning?</span>Gain — value of computation</div>
      <div class="mdl-proc"><strong>Gain</strong> = value the full search finds beyond its own first guess. vs log(RT).</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> People should think where thinking pays off.</div>
      <div class="mdl-conv"><span class="lbl">As predicted, but weaker</span> Gain ↔ RT: <strong>+0.08</strong> — below the legal-moves width effect.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/engine/gain.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="gain" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · How do we model planning?</span>Action gap — decisiveness</div>
      <div class="mdl-proc"><strong>Greedy action-gap</strong> = top-1 − top-2 of the root values, pre-search. vs RT.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> One move clearly best → less to weigh → faster.</div>
      <div class="mdl-conv"><span class="lbl">Null</span> Action gap ↔ RT: <strong>+0.01</strong> — no signal.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/engine/action_gap.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="agap" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · How do we model planning?</span>Frac-Good — how many good moves?</div>
      <div class="mdl-proc"><strong>Greedy frac-good</strong> = fraction of moves within 0.1 of best, pre-search. vs RT.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> More good-enough options → satisfice → faster.</div>
      <div class="mdl-conv"><span class="lbl">Satisficing signature</span> Frac-good ↔ RT: <strong>−0.10</strong> — more good moves, less time.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/engine/frac_good.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="fracgood" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · Optimal stopping step</span>Does the stop step track think-time?</div>
      <div class="mdl-proc"><strong>OSS</strong> (optimal stop step) vs human RT.</div>
      <div class="mdl-intuition"><span class="lbl">Expected</span> If people meta-control, they think longer when the oracle stops later.</div>
      <div class="mdl-conv"><span class="lbl">As predicted, but weak</span> OSS ↔ RT: <strong>+0.06</strong> — far below the width effect.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/engine/oss.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="oss" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · Meta-controller</span>How to train a meta-controller?</div>
      <div class="mdl-proc"><strong>RL readout (policy gradient)</strong> halts each step on the <strong>advantage</strong> (continue − halt value). Compare halt rules on regret.</div>
      <div class="mdl-intuition"><span class="lbl">The models</span> blind: always · never · fraction-θ* · learned: GNN-z (embedding) · tree-stats (raw [h, w, n]).</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> <strong>tree-stats ≻ GNN-z ≻ fraction ≻ always/never.</strong> Raw structure wins — but low regret is self-consistency, not a match to people.</div>
    </div>
    <div class="mdl-figbox"><img src="../archived_plots/regret_by_model.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="meta" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · What is thinking worth?</span>The engine signals, all at once</div>
      <div class="mdl-proc"><strong>Spearman matrix</strong> — engine value / stop signals, board structure, RT (SF-1 n1md36, n = 109k).</div>
      <div class="mdl-intuition"><span class="lbl">Key parts</span> The top row (vs RT) is uniformly faint — every engine signal <strong>|ρ| ≲ 0.17</strong>. GSS↔OSS <strong>+0.91</strong> (near-identical), and each signal ties to <strong>legal moves</strong> more than to RT.</div>
      <div class="mdl-conv warn"><span class="lbl">Thinking's value is faint</span> No value-of-computation signal approaches the +0.26 width effect — they collapse toward the move count.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/allply/engine/correlation_matrix.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="q1a" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 3 · the pivot</span>Do the human and normative model agree?</div>
      <div class="mdl-proc"><strong>Every model signal</strong> — regret, step*, VOC — vs human RT (Spearman, n≈65k, bootstrap CIs).</div>
      <div class="mdl-intuition"><span class="lbl">Expected</span> If people meta-control like the model, VOC signals should lead.</div>
      <div class="mdl-conv warn"><span class="lbl">Answer — NO</span> Structural drivers win: legal moves <strong>+0.31</strong>, good-move fraction <strong>−0.31</strong>. Every VOC signal only +0.12…+0.16. Our measure is mis-specified.</div>
    </div>
    <div class="mdl-figbox"><img src="../archived_plots/rt_headline.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 1
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="qmatch" /></div>
</div>
