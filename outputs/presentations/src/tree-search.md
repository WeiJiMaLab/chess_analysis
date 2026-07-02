---
theme: default
title: Learned Meta-Control of Search — Analysis
info: From a chess tree-search model to a meta-rational account of human deliberation time.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search · human deliberation time</span>

# Learned Meta-Control of Search — Analysis

<div class="mt-3 text-lg opacity-80">One root question, three parts: how do people actually think, what is thinking worth, and do the two agree?</div>

<div class="mt-4 text-sm opacity-50">SF leaf-eval search trees on 2023 human-game FENs · ~63k positions · ~65k human moves · board analysis on 1.97M Lichess games → 135M non-zero-time moves.</div>

<!--
Map beats are CLICK-driven (one persistent <ThoughtMap seq=.../> per slide): the camera PANS, children REVEAL,
questions RESOLVE, the next one BLINKS — all on click. map↔content slide changes use a simple fade
transition. Content slides are normal. Set `clicks:` = (sequence length − 1). Sequences live in
ThoughtMap.vue. Source: reports/board.md, treesearch.md, engine.md, lmcos_small.md.

Structure: Part 1 = model-free board analysis · Part 2 = normative model · Part 3 = human-vs-model.
Hindsight section (why the mismatch: cost shape / cost type / evaluator strength) is temporarily CUT —
qmatch's children were removed from ThoughtMap.vue and the slides truncated; restore together from git.

Each content slide has THREE parts:
  (a) .mdl-proc      — what we did (procedure line)
  (b) .mdl-intuition — what we expected / how to read the slide
  (c) .mdl-conv      — the bottom takeaway
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
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · How do people actually think?</span>Decision width — how many legal moves?</div>
      <div class="mdl-proc"><strong>Model-free</strong> — board feature vs human log(RT), 135M moves. This leaf: <strong># legal moves</strong> = decision width.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> More options to weigh → longer think.</div>
      <div class="mdl-conv"><span class="lbl">Strongest board feature</span> Legal moves ↔ log(RT): <strong>+0.20</strong>. Holds within ply tertiles.</div>
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
  <div class="mdl-map"><ThoughtMap seq="b_legal" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · board feature</span>Realized value of search — Gain (ΔUC)</div>
      <div class="mdl-proc"><strong>Gain = ΔUC at depth 5</strong> — does deeper search improve the eval? vs log(RT).</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> People should think where thinking pays off.</div>
      <div class="mdl-conv"><span class="lbl">Weaker than width</span> Gain ↔ log(RT): <strong>+0.10</strong> — half the legal-moves effect. (Russek 2022: +0.096.)</div>
    </div>
    <div class="mdl-figbox"><img src="../archived_plots/board_gain.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="b_gain" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · board feature</span>Own material — how many pieces?</div>
      <div class="mdl-proc"><strong>Own non-pawn material</strong> vs log(RT). Falls with game stage.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> More pieces → more interactions — or just a game-stage proxy?</div>
      <div class="mdl-conv"><span class="lbl">Weak</span> Material ↔ log(RT): <strong>+0.04</strong> — near the floor, redundant with ply/clock.</div>
    </div>
    <div class="mdl-figbox"><img src="../archived_plots/board_material.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="b_material" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 1 · back to the root</span>Action gap → the correlation summary</div>
      <div class="mdl-proc"><strong>Action gap</strong> (best − 2nd-best) vs log(RT), then back to the root: the feature-correlation map.</div>
      <div class="mdl-intuition"><span class="lbl">Intuition</span> Big gap = one clear move = less to weigh → negative tie.</div>
      <div class="mdl-conv"><span class="lbl">Width wins — not reducible</span> Gap <strong>−0.06</strong>. Legal-moves is the strongest tie (ρ ≈ +0.26), not reducible to ply/material/clock. Even lc0's H(π) is subsumed (partial ρ ≈ +0.05).</div>
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
  <div class="mdl-map"><ThoughtMap seq="b_gap" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · Question 2</span>How do we model planning?</div>
      <div class="mdl-proc"><strong>AlphaZero-style PUCT tree</strong>, no rollouts: a heuristic values nodes, a selector picks what to expand.</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> Prior weights <em>which</em> child; value is backed up from leaves. Best-first → narrow &amp; deep; UCB → broad.</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> Uniform priors ⇒ PUCT = <strong>UCB</strong>. Already a heuristic best-first search; the lever is selector + cost.</div>
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
  <div class="mdl-map"><ThoughtMap seq="q2" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · Question 3</span>Which engine do we use?</div>
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
  <div class="mdl-map"><ThoughtMap seq="q3" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · Question 4</span>When should the search stop?</div>
      <div class="mdl-proc"><strong>step* = argmax<sub>s</sub>(V(s) − cost(s))</strong> from a budgeted-oracle DP. RL readout halts on the <strong>advantage</strong> (continue − halt value).</div>
      <div class="mdl-intuition"><span class="lbl">Beat the blind rules</span> always-stop · never-stop · fixed fraction-θ*.</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> Yes — a <strong>tree-stats readout</strong> [height, width, n_nodes] wins on regret at less compute. Stopping is solvable.</div>
    </div>
    <div class="mdl-figbox"><img src="../archived_plots/regret_vs_compute.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · Question 4 · the readout zoo</span>Which stop model wins?</div>
      <div class="mdl-proc"><strong>Every halt rule's regret</strong> on held-out trees.</div>
      <div class="mdl-intuition"><span class="lbl">The models</span> always · never · fraction-θ* · GNN-z (learned embedding) · tree-stats (raw [h, w, n]).</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> <strong>tree-stats ≻ GNN-z ≻ fraction ≻ always/never.</strong> Raw structure beats the embedding — but low regret is self-consistency, not a match to people.</div>
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
  <div class="mdl-map"><ThoughtMap seq="q4" /></div>
</div>

---
layout: default
class: mdl-slide
transition: fade
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Part 2 · stopping · a hindsight check on step*</span>Does step*'s backward DP inflate it?</div>
      <div class="mdl-proc"><strong>step* uses backward DP</strong> — it "knows the future." Swap in a <strong>causal halter</strong> (tree-so-far only); compare both to legal-moves.</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> If hindsight inflated step*, the oracle would sit far above the causal halter.</div>
      <div class="mdl-conv"><span class="lbl">No inflation</span> Causal halter ≈ hindsight oracle, and <strong>both ≪ legal-moves</strong>. The gap to people is real.</div>
    </div>
    <div class="mdl-figbox"><img src="../archived_plots/hindsight_halter.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: fade
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="h7" /></div>
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
