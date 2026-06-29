---
theme: default
title: Tree search & human deliberation — when (and how) ought one to think?
info: From a chess tree-search model to a meta-rational account of human deliberation time.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search · human deliberation time</span>

# When — and how — ought one to think?

<div class="mt-3 text-lg opacity-80">A branching set of questions: build a chess search, ask what thinking is worth — then ask whether people actually think that way.</div>

<div class="mt-4 text-sm opacity-50">SF-2000 search trees on 2023 human-game FENs · ~63k positions · ~65k human moves.</div>

<!--
Map beats are CLICK-driven (one persistent <ThoughtMap seq=.../> per slide): the camera PANS, children REVEAL,
questions RESOLVE, the next one BLINKS — all on click. map↔content slide changes use the zoom-in/zoom-out
transitions (style.css). Content slides are normal. Set `clicks:` = (sequence length − 1). Sequences live in
ThoughtMap.vue. Source: reports/treesearch.md, engine.md, lmcos_tiny.md.
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
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Question 2</span>How do we model "searching"?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">An <strong>AlphaZero-style PUCT tree</strong>, no rollouts. The selector decides what to expand — compare three.</p>
  <TreeGrowth />
  <div class="mdl-conv"><span class="lbl">Answer</span> <strong>Best-First is greedy</strong> (narrow &amp; deep). Our <strong>uniform-prior PUCT = UCB</strong> doesn't search greedily — it spreads visits, so it's already a best-first search we read off the existing trace.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-q2" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Question 3</span>Which engine evaluates the tree?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">The tree needs a <strong>prior</strong> (which child to try) and a <strong>value</strong> (what a node is worth) — re-sourced for ~1000× CPU speedup.</p>
  <div class="mdl-two" style="height:auto; flex:0 0 auto; margin:.5rem 0">
    <div class="mdl-card mdl-card--neutral"><div class="mdl-card-h">lc0 (before)</div><div class="body"><strong>prior:</strong> trained policy head<br><strong>value:</strong> trained value head (WDL)</div></div>
    <div class="mdl-card mdl-card--accent"><div class="mdl-card-h">Stockfish (after)</div><div class="body"><strong>prior:</strong> uniform — α-β has no policy head<br><strong>value:</strong> eval→WDL from an N=100-node search</div></div>
  </div>
  <div class="mdl-conv"><span class="lbl">Answer</span> The swap <strong>didn't change the science</strong> — every metric keeps its sign &amp; magnitude. Only <strong>N</strong> (search nodes) moves the eval; <strong>UCI_Elo is a no-op</strong>.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-q3" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Question 4</span>When should the search stop?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">A budgeted-oracle DP gives the optimal stop step <strong>step* = argmax(V − cost)</strong>. Can a learned readout beat blind rules?</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/regret_vs_compute.png" /></div>
  <div class="mdl-conv"><span class="lbl">Answer</span> Yes — a <strong>tree-stats readout</strong> [height, width, n_nodes] wins: regret <strong>0.10</strong> vs 0.20 (fraction), 0.50–1.4 (always/never), at less compute. Raw structure beats a learned embedding.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-q4" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">The pivot</span>Do the model's signals match human think-time?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">Correlate every signal — regret, step*, softmax-VOC — with human log response time (n≈65k).</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/rt_headline.png" /></div>
  <div class="mdl-conv warn"><span class="lbl">Answer — NO</span> The drivers are <strong>structural</strong>: # legal moves <strong>+0.31</strong>, good-move fraction <strong>−0.31</strong>. Every VOC signal is only +0.12…+0.16. Not "people are irrational" — <strong>our measure is mis-specified</strong>.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 4
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-qmatch" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Hypothesis 1 of 5</span>Is it the wrong cost shape?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">Sweep the cost function — linear, quadratic, power-law × scales.</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/cost_sweep_rt.png" /></div>
  <div class="mdl-conv warn"><span class="lbl">No</span> Regret is flat (a monotone transform of cost-free Gain, ρ=0.98). step* responds, but tops out at <strong>+0.12</strong> — half the legal-moves effect.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-h5" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Hypothesis 2 of 5</span>Is uncertainty missing?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">Re-grade the halt value as a <strong>softmax</strong> over root values at temperature τ — the value of <em>sharpening</em>.</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/voc_tau_sweep.png" /></div>
  <div class="mdl-conv warn"><span class="lbl">No</span> At calibrated τ it <strong>recovers</strong> the argmax value (~+0.16) — but never <em>exceeds</em> it. Uncertainty-reduction is real, but it's not the missing driver.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-h6" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Hypothesis 3 of 5</span>Is it hindsight asymmetry?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">step* is solved by backward DP — it "knows the future". Replace it with a <strong>causal halter</strong> that sees only the tree-so-far.</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/regret_by_model.png" /></div>
  <div class="mdl-conv warn"><span class="lbl">No</span> The causal halter <strong>≈ 0</strong> — no better than the oracle, both far below legal-moves. Information asymmetry isn't it.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-h7" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Hypothesis 4 of 5 · the promising one</span>Is it just a legal-moves proxy?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">The decisive test: <strong>partial out the legal-move count</strong> from every value / VOC / step* signal.</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/rt_partials.png" /></div>
  <div class="mdl-conv"><span class="lbl">YES — this is the one</span> Every VOC signal collapses to <strong>≈ +0.04</strong>; legal-moves <em>survives</em> the reverse control at +0.22. The "value of thinking" signal was a <strong>decision-width</strong> proxy all along.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-h8" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">Hypothesis 5 of 5</span>Is the evaluator too strong?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc">Vary the leaf-eval node budget N (n1 vs n100); test UCI_Elo.</p>
  <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/litmus_strength.png" /></div>
  <div class="mdl-conv warn"><span class="lbl">N matters; Elo doesn't</span> A noisier "gut" (n1 vs n100) genuinely differs (ρ≈0.93), but <strong>UCI_Elo is a no-op</strong>. Open thread, not the main driver.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-h9" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">The finding</span>So what is think-time, really?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-proc"><strong>RT ≈ size (+0.31) − satisfaction (−0.31) + sharpness (+0.24)</strong> — more options slow you down, more <em>good</em> options speed you up.</p>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.2rem; flex:1; min-height:0">
    <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/good_moves_signflip.png" /></div>
    <div class="mdl-fig"><img src="../public/figures/lmcos_tiny/rt_vs_n_concavity.png" /></div>
  </div>
  <div class="mdl-conv"><span class="lbl">The satisficing signature</span> Satisfaction <strong>reshapes the whole RT-vs-n curve</strong>: high-satisfaction positions plateau low (stop once good-enough); low stay steep. Stop-when-good-enough — not bare problem size.</div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="after-finding" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-title"><span class="mdl-kicker">The plan</span>A meta-rational reward − cost model</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">Grow a consideration set of candidate moves while marginal VOC &gt; c, else <strong>satisfice</strong>. Reward on stop = value(chosen) − c·#included. <strong>~1–2 parameters</strong> (cost c, prior noise).</p>
  <div class="mdl-two" style="height:auto; flex:0 0 auto; margin:.4rem 0">
    <div class="mdl-card mdl-card--accent"><div class="mdl-card-h">The test (P-FIT)</div><div class="body">Fit c to RT on the breadth-inclusion trace. How much of size − satisfaction + sharpness emerges from 1–2 params vs the ±0.31 ceiling? <em>(now: +0.16.)</em></div></div>
    <div class="mdl-card mdl-card--neutral"><div class="mdl-card-h">Open question</div><div class="body">Does the search add value? It reorders heavily (rank-corr ≈0.33) — but the value-add lives in <em>depth</em>, while RT is <em>breadth</em>-driven.</div></div>
  </div>
  <div class="mdl-conv"><span class="lbl">Root question, answered</span> People <em>do</em> meta-control — resource-rationally over <strong>decision width</strong>, satisficing once good-enough — not over engine value-of-computation.</div>
</div>
