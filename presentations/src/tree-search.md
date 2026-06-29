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

<div class="mt-2 text-lg opacity-80">We build a chess tree search, ask when it should stop, then ask whether that explains <em>when people actually think</em>. It doesn't — and chasing why yields a meta-rational account.</div>

<div class="mt-4 text-sm opacity-50">SF-2000 search trees on 2023 human-game FENs · ~63k filtered positions · ~65k joined human moves.</div>

<!--
Source: reports/treesearch.md (R-TREESEARCH). Linear narrative: question -> planning box (PUCT, lc0->SF) ->
meta-control (regret; tree-stats > GNN-z) -> pivot (regret != RT) -> 5 hypotheses -> finding (satisficed
decision difficulty) -> model + Q1/Q2 + plan. Figures: /figures/lmcos_tiny/. Tree-growth animation: <TreeGrowth/>.
-->

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">The map</span>What we tried, found, and followed up on</div>

<div class="mdl-content mdl-content--top">

```mermaid {scale: 0.62}
flowchart TD
  Q["<b>Q</b> When should one search deeper?<br/>Do people think where an engine would?"] --> TS
  TS["<b>Procedure</b> AlphaZero-style PUCT tree<br/>lc0 → Stockfish · M=96 · depth 4"] --> META
  META["<b>Procedure</b> Budgeted oracle: step* = argmax(V−cost)<br/>learn a readout to minimize regret"] --> R1
  R1["<b>Result</b> tree-stats ≻ GNN-z ≻ fraction-halt"] --> PIV
  PIV{"<b>Q</b> Is minimizing regret the same as<br/>matching human response time?"} --> HYP
  HYP["<b>Procedure</b> 5 hypotheses:<br/>cost shape · softmax-VOC · hindsight · legal-moves · evaluator-strength"] --> R2
  R2["<b>Result</b> NO — every VOC signal is a<br/>legal-moves proxy (collapse to +0.04)"] --> FIND
  FIND["<b>Result</b> RT = satisficed decision difficulty<br/>size(+) − satisfaction(−) + sharpness(+)"] --> PLAN
  PLAN["<b>Plan</b> meta-rational reward−cost model<br/>fit c to RT · breadth trace · does the search add value?"]
```

  <p class="mdl-tree-cut">Each <strong>Question</strong> node opens a junction; a <strong>Procedure</strong> resolves it; a <strong>Result</strong> either routes onward or closes a branch. The next slides zoom into each.</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Junction 1 · the planning box</span>How do we model "searching"?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">We grow an <strong>AlphaZero-style PUCT tree</strong> — <strong>no rollouts</strong>; a heuristic values each node, and a selector chooses what to expand. How does the selector shape the tree?</p>

  <TreeGrowth />

  <div class="mdl-design" style="margin-top:.6rem"><span class="lbl">Two heads</span> <strong>prior</strong> P(child) weights only <em>which child to visit</em> (exploration); <strong>value</strong> is backed up from leaves (what a node is worth). They are different things.</div>
  <p class="mdl-tree-cut">With <strong>uniform priors</strong> PUCT reduces to <strong>UCB</strong> — so we are already running a heuristic best-first search with a UCT selector, breadth-leaning early. <span class="opacity-50">(Q1 later: we may not need a new generator at all.)</span></p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Junction 2 · the evaluator</span>We swapped lc0 → Stockfish</div>
      <p class="mdl-lead">Choosing lc0's heads as the evaluator was somewhat arbitrary. Stockfish is ~1000× faster on CPU — so we swapped it in. What changed?</p>
      <table class="mdl-table mdl-table--sm" style="margin-top:.3rem">
        <thead><tr><th></th><th>prior (exploration)</th><th>value (node)</th></tr></thead>
        <tbody>
          <tr><td><strong>lc0</strong></td><td>policy head (informative)</td><td>value head (WDL)</td></tr>
          <tr><td><strong>Stockfish</strong></td><td><strong>uniform</strong> (α-β has no policy head)</td><td><strong>WDL</strong> from N=100-node search</td></tr>
        </tbody>
      </table>
      <div class="mdl-design" style="margin-top:.4rem"><span class="lbl">Three knobs</span> <strong>N</strong>=leaf eval nodes (the "gut" — <em>matters</em>, n1≠n100) · <strong>M</strong>=96 expansions (the planning) · <strong>UCI_Elo</strong> (<em>no-op</em> — we read the eval, not the played move).</div>
      <p class="mdl-tree-cut">Materially: the value is now <strong>WDL/win-prob</strong> (not centipawns), and the prior went <strong>uniform</strong> — which is exactly why the search became breadth-leaning UCB.</p>
    </div>
    <div class="mdl-figwrap">
      <LeelaSearchLoop />
      <div class="mdl-figcaption">The base planner (loop): board → search → value/policy. We replace the engine inside, not the loop.</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Junction 3 · meta-control</span>When should the search stop?</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">A budgeted-oracle DP gives the <strong>optimal stop step</strong> step* = argmax<sub>s</sub>[V(s) − cost(s)], and <strong>regret = reward − cost</strong>. Blind rules anchor the scale; a learned readout should beat them:</p>

  <div class="mdl-steps">
    <div class="mdl-step"><span class="mdl-step-n">A</span><div><div class="mdl-step-h">Always-stop</div><div class="mdl-step-t">commit at step 0 — never think</div></div></div>
    <div class="mdl-step"><span class="mdl-step-n">N</span><div><div class="mdl-step-h">Never-stop</div><div class="mdl-step-t">use the full budget M — always think to the end</div></div></div>
    <div class="mdl-step"><span class="mdl-step-n">θ</span><div><div class="mdl-step-h">Fraction-θ*</div><div class="mdl-step-t">stop at a fixed fraction θ·M (one parameter)</div></div></div>
    <div class="mdl-step"><span class="mdl-step-n">★</span><div><div class="mdl-step-h">Learned readout</div><div class="mdl-step-t">GNN-z (embedding) vs tree-stats [height, width, n_nodes]</div></div></div>
  </div>

  <div class="mdl-figrow" style="margin-top:.5rem">
    <div class="mdl-figbox"><img src="/figures/lmcos_tiny/regret_vs_compute.png" /><div class="mdl-figcaption">regret vs compute — the readouts beat the blind rules</div></div>
    <div class="mdl-figbox"><img src="/figures/lmcos_tiny/regret_by_model.png" /><div class="mdl-figcaption">tree-stats PG ≻ GNN-z — raw structure beats the learned embedding</div></div>
  </div>
  <p class="mdl-tree-cut"><strong>Result:</strong> the meta-control problem is solvable — but low regret is <em>self-consistency</em>, not a match to people. So: is regret even the right target?</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">The pivot</span>Does any of this match human think time?</div>
      <p class="mdl-lead">We correlate every signal with human log response time. The landscape is blunt:</p>
      <div class="mdl-text">
        <ul>
          <li><strong># legal moves (size): +0.31</strong> — the dominant driver.</li>
          <li><strong>fraction of good moves (satisfaction): −0.31</strong> — equal and opposite.</li>
          <li>every value-of-computation signal (regret, step*, softmax-VOC): <strong>+0.12 … +0.16</strong>.</li>
          <li>the causal controller: <strong>≈ 0</strong>.</li>
        </ul>
      </div>
      <p class="mdl-tree-cut"><strong>Result:</strong> our <em>measure</em> of optimal stop-time doesn't match human RT. Not "normativity fails" — the model is mis-specified. The next junction enumerates <em>why</em>.</p>
    </div>
    <div class="mdl-figwrap">
      <img src="/figures/lmcos_tiny/rt_headline.png" />
      <div class="mdl-figcaption">What predicts human RT (n≈65k). blue = problem size · red = satisfaction · grey = value-of-computation.</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Junction 4 · the hypotheses</span>Why doesn't value-of-computation track RT?</div>

<div class="mdl-content mdl-content--top">
  <div class="mdl-issue-list">
    <div class="mdl-issue"><span class="mdl-issue-n">4a</span><div><div class="mdl-issue-h">Wrong cost shape?</div><div>Sweep it → <strong>no</strong>. Regret is a monotone transform of Gain (position-independent cost); step* swings but tops at +0.12.</div></div></div>
    <div class="mdl-issue"><span class="mdl-issue-n">4b</span><div><div class="mdl-issue-h">No uncertainty? (softmax-VOC)</div><div>Re-grade a softmax-τ policy on deep values: early flat ⇒ averages good+bad ⇒ low; search <em>sharpens</em> ⇒ value rises. → <strong>recovers</strong> the argmax value (~+0.16), doesn't beat it.</div></div></div>
    <div class="mdl-issue"><span class="mdl-issue-n">4c</span><div><div class="mdl-issue-h">Hindsight asymmetry?</div><div>step* knows the future; a <em>causal</em> halter doesn't. → <strong>no</strong> (halter ≈ 0, not better than the oracle).</div></div></div>
    <div class="mdl-issue mdl-issue-found"><span class="mdl-issue-n">4d</span><div><div class="mdl-issue-h">Just a legal-moves proxy?</div><div><strong>YES.</strong> Partial out legal-moves ⇒ every value/VOC/step* signal collapses to ≈+0.04; legal-moves survives.</div></div></div>
    <div class="mdl-issue"><span class="mdl-issue-n">4e</span><div><div class="mdl-issue-h">Evaluator too strong (squashed)?</div><div>Vary N → <strong>N matters</strong> (n1≠n100); separately UCI_Elo is a no-op ⇒ vary the <em>search</em>, not UCI_Elo.</div></div></div>
  </div>

  <div class="mdl-figrow" style="margin-top:.4rem">
    <div class="mdl-figbox"><img src="/figures/lmcos_tiny/cost_sweep_rt.png" /><div class="mdl-figcaption">4a · cost sweep</div></div>
    <div class="mdl-figbox"><img src="/figures/lmcos_tiny/voc_tau_sweep.png" /><div class="mdl-figcaption">4b · softmax-VOC by τ</div></div>
    <div class="mdl-figbox"><img src="/figures/lmcos_tiny/rt_partials.png" /><div class="mdl-figcaption">4d · the collapse</div></div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">The finding</span>RT is satisficed decision difficulty</div>
      <p class="mdl-lead">The pure-size reading is too dumb. The <strong>good</strong>-moves term flips the sign of the <strong>all</strong>-moves term:</p>
      <p class="mdl-design"><span class="lbl">RT</span> size(+0.31) − satisfaction(−0.31) + sharpness(+0.24)</p>
      <p class="mdl-text">More <em>options</em> ⇒ slower; more <em>good</em> options ⇒ faster. And — the clincher — <strong>satisfaction reshapes the whole RT-vs-n curve</strong>: high-satisfaction positions <em>plateau low</em> (stop once good-enough), low-satisfaction stay steep.</p>
      <p class="mdl-tree-cut"><strong>That is satisficing</strong>, made visible — not bare problem-size (Hick's law), but a stop-when-good-enough rule, which is exactly what a meta-rational model predicts.</p>
    </div>
    <div class="mdl-figwrap">
      <img src="/figures/lmcos_tiny/good_moves_signflip.png" />
      <div class="mdl-figcaption">The sign flip.</div>
      <img src="/figures/lmcos_tiny/rt_vs_n_concavity.png" style="margin-top:.4rem" />
      <div class="mdl-figcaption">The plateau-shift = the satisficing signature.</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">The model & the plan</span>A meta-rational account, and two open questions</div>

<div class="mdl-content mdl-content--top">
  <p class="mdl-lead">Deliberation time = <strong>optimal compute under a cost</strong>: grow a consideration set while marginal VOC &gt; c, else <strong>satisfice</strong>. Objective = reward − cost = −regret; <strong>~1–2 parameters</strong> (c, prior noise). Fit c to RT; how much of the structure does it reproduce vs the ±0.31 ceiling? <span class="opacity-60">(current 1-param: +0.16.)</span></p>

  <div class="mdl-issue-list" style="margin-top:.3rem">
    <div class="mdl-issue"><span class="mdl-issue-n">Q1</span><div><div class="mdl-issue-h">Are we already doing best-first search?</div><div><strong>Yes</strong> — uniform-prior PUCT ≈ UCB, breadth-leaning. Read the construal off the <em>existing</em> trace; lever = selector + per-move cost, not a new generator.</div></div></div>
    <div class="mdl-issue mdl-pending"><span class="mdl-issue-n">Q2</span><div><div class="mdl-issue-h">Does the tree search add value?</div><div><em>Open.</em> We stressed the prior (N) but not the search (M). Provisional: it reorders heavily (rank-corr ≈0.33, argmax settles ~step 78/95) — but the value-add is in <em>depth</em>, while RT is <em>breadth</em>-driven.</div></div></div>
  </div>

  <div class="mdl-steps" style="margin-top:.3rem">
    <div class="mdl-step"><span class="mdl-step-n">1</span><div><div class="mdl-step-h">P-FIT</div><div class="mdl-step-t">fit c to RT on the breadth-inclusion trace + satisficing stop — the headline number</div></div></div>
    <div class="mdl-step"><span class="mdl-step-n">2</span><div><div class="mdl-step-h">Q2-check</div><div class="mdl-step-t">1-ply heuristic vs M-converged: is the tree-building the weak link?</div></div></div>
    <div class="mdl-step"><span class="mdl-step-n">3</span><div><div class="mdl-step-h">selector / prior</div><div class="mdl-step-t">swap UCT→optimism; add a real policy prior — both on existing trees</div></div></div>
  </div>
  <p class="mdl-tree-cut">If a 1–2-parameter meta-RL reaches the ±0.31 ceiling with the size/satisfaction/sharpness signatures emerging, we have a parsimonious meta-rational account of <em>when people think</em>.</p>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Definitions</span>The vocabulary in one place</div>

<div class="mdl-content mdl-content--top">
  <table class="mdl-table mdl-table--wide">
    <tbody>
      <tr><td><strong>step* / regret</strong></td><td>step* = argmax<sub>s</sub>[V(s) − cost(s)]; regret = oracle_value − return(stop)</td></tr>
      <tr><td><strong>tree-stats</strong></td><td>per-step [height (deepest node), width (max nodes at a depth), n_nodes (total expanded)]</td></tr>
      <tr><td><strong>softmax-VOC</strong></td><td>halt value = Σ<sub>c</sub> softmax(q<sub>s,c</sub>/τ)·V<sub>deep</sub>(c); benefit of thinking = the sharpening</td></tr>
      <tr><td><strong>size / satisfaction / sharpness</strong></td><td># legal moves / fraction within ε of best / top-1 − top-2 action gap</td></tr>
      <tr><td><strong>N / M / UCI_Elo</strong></td><td>leaf-eval nodes (heuristic) / PUCT expansions (planning) / play handicap (no-op for the eval)</td></tr>
    </tbody>
  </table>
  <p class="mdl-tree-cut">Source of truth: <code>reports/treesearch.md</code> (R-TREESEARCH). Figures: <code>lmcos_tiny/analysis/make_rt_figures.py</code> · ρ = Spearman with percentile-bootstrap 95% CIs.</p>
</div>
