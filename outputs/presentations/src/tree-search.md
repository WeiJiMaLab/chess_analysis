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

<div class="mt-3 text-lg opacity-80">A branching set of questions: build a chess search, ask what thinking is worth — then ask whether people actually think that way.</div>

<div class="mt-4 text-sm opacity-50">SF leaf-eval search trees on 2023 human-game FENs · ~63k positions · ~65k human moves.</div>

<!--
Map beats are CLICK-driven (one persistent <ThoughtMap seq=.../> per slide): the camera PANS, children REVEAL,
questions RESOLVE, the next one BLINKS — all on click. map↔content slide changes use the zoom-in/zoom-out
transitions (style.css). Content slides are normal. Set `clicks:` = (sequence length − 1). Sequences live in
ThoughtMap.vue. Source: reports/treesearch.md, engine.md, lmcos_small.md.

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
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Question 2</span>How do we model planning?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Grow an <strong>AlphaZero-style PUCT tree</strong>, no rollouts: a heuristic values each node, a selector picks what to expand. Compare three selectors below.</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> The <strong>selector</strong> shapes the tree. We expected Best-First to run <em>narrow &amp; deep</em>; UCB to spread visits <em>broad</em>. Two heads not to conflate: the <strong>prior</strong> only weights <em>which</em> child to visit; the <strong>value</strong> is backed up from leaves.</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> With <strong>uniform priors</strong>, PUCT reduces to <strong>UCB</strong> — so we are already running a heuristic best-first search with a UCT selector, breadth-leaning early. The construal can be read off the <em>existing</em> trace; the lever is the selector + cost.</div>
    </div>
    <TreeGrowth />
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="q2" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Question 3</span>Which engine do we use?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Swapped <strong>lc0 → Stockfish</strong> for a pragmatic reason: ~1000× CPU speedup at scale, not a scientific claim. The swap forces two approximations: prior → <strong>uniform</strong> (α-β has no policy head); value → a short <strong>N-node SF search</strong> (WDL win-prob).</div>
      <div class="mdl-intuition"><span class="lbl">We expected</span> A model swap to be risky — so we checked it. We kept <strong>three knobs distinct</strong>: <strong>N</strong> (leaf-eval nodes, the heuristic), <strong>M</strong>=96 (the planning budget), and <strong>UCI_Elo</strong> (a play handicap).</div>
      <div class="mdl-conv"><span class="lbl">Validated safe swap</span> Correlations with human RT are <strong>stable across the change</strong> — signs and magnitudes hold. Only <strong>N</strong> moves the eval; <strong>UCI_Elo is a no-op</strong> (SF-1350 ≡ SF-2000, bit-identical). Engineering decision, not a scientific one.</div>
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
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="q3" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Question 4</span>When should the search stop?</div>
      <div class="mdl-proc"><strong>What we did.</strong> A budgeted-oracle DP gives the optimal stop step <strong>step* = argmax<sub>s</sub>(V(s) − cost(s))</strong>. We train a <strong>meta-control readout via RL (policy gradient)</strong> that decides, each step, to continue or halt — keyed on the <strong>advantage</strong> = continue-value − halt-value.</div>
      <div class="mdl-intuition"><span class="lbl">We expected</span> A learned readout to beat three <strong>blind baselines</strong>: <strong>always-stop</strong> (step 0), <strong>never-stop</strong> (full budget), and <strong>fraction-θ*</strong> (a fixed fraction of the budget). If structure carries any signal, the readout should stop later only when advantage stays positive.</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> Yes — a <strong>tree-stats readout</strong> [height, width, n_nodes] wins on regret at less compute. A learned stopping rule beats the blind rules — the stopping problem is solvable.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/regret_vs_compute.png" /></div>
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
      <div class="mdl-title"><span class="mdl-kicker">Question 4 · the readout zoo</span>Which stop model wins?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Compare every halt rule's regret on held-out trees. The contestants:</div>
      <div class="mdl-intuition"><span class="lbl">The models</span> <strong>always</strong> = stop at step 0 · <strong>never</strong> = run the full M=96 budget · <strong>fraction-θ*</strong> = a fixed fraction of the budget · <strong>GNN-z</strong> = a learned graph-embedding of the tree · <strong>tree-stats</strong> = the raw triple [height, width, n_nodes]. All RL-trained readouts decide on the <strong>advantage</strong>.</div>
      <div class="mdl-conv"><span class="lbl">Answer</span> <strong>tree-stats ≻ GNN-z ≻ fraction ≻ always/never.</strong> Raw structure beats the learned embedding. But low regret is <em>self-consistency</em> with the oracle — not yet a match to people.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/regret_by_model.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="q4" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Stopping · a hindsight check</span>Does step*'s backward DP inflate it?</div>
      <div class="mdl-proc"><strong>What we did.</strong> step* is solved by <strong>backward DP</strong> — it "knows the future." Before trusting it, we replace it with a <strong>causal halter</strong> that sees only the tree-so-far, and compare both against the legal-moves driver.</div>
      <div class="mdl-intuition"><span class="lbl">We expected</span> If "knowing the future" inflated step*, the <em>hindsight oracle</em> would sit far above the <em>causal halter</em>. So this is part of the <strong>stopping/oracle</strong> story, not a peer hypothesis — it validates the oracle before we lean on it.</div>
      <div class="mdl-conv"><span class="lbl">No inflation</span> The <strong>causal halter ≈ the hindsight oracle</strong> — and <strong>both ≪ legal-moves</strong>. Hindsight is not buying step* anything; the oracle is safe to use, and the gap to people is real, not an artifact.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/hindsight_halter.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="h7" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">The pivot</span>Do the model's signals match human think-time?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Correlate every signal — regret, step*, VOC(softmax_policy) — with human <strong>RT</strong> (Spearman, n≈65k, bootstrap 95% CIs).</div>
      <div class="mdl-intuition"><span class="lbl">We expected</span> If people meta-control like the model, the <strong>value-of-computation</strong> signals should lead. Read the bars as: which feature best predicts how long a person actually thought?</div>
      <div class="mdl-conv warn"><span class="lbl">Answer — NO</span> The drivers are <strong>structural</strong>: # legal moves <strong>+0.31</strong>, good-move fraction <strong>−0.31</strong>. Every VOC signal is only +0.12…+0.16. Not "people are irrational" — <strong>our measure is mis-specified</strong>.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/rt_headline.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 4
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="qmatch" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Hypothesis 1 of 4</span>Is it the wrong cost shape?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Sweep the <strong>cost curve</strong> — linear, quadratic, power-law — across scales, and re-derive step* / regret each time. Correlate <strong>step* and regret with RT</strong>, per cost config.</div>
      <div class="mdl-intuition"><span class="lbl">Why we expected this to matter</span> A different cost <em>shape</em> ⇒ a different optimal stop ⇒ a different step*. If RT tracks a particular shape, the right curve should pop the correlation up. Read each point as one (shape × scale) config.</div>
      <div class="mdl-conv warn"><span class="lbl">No</span> Regret is flat (a monotone transform of cost-free Gain, ρ=0.98). step* responds to cost shape, but tops out at <strong>+0.12</strong> — half the legal-moves effect.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/cost_sweep_rt.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="h5" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Hypothesis 2 of 4</span>Is uncertainty missing?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Re-grade the halt value under a <strong>softmax policy over the root-move values</strong> at temperature τ — <strong>VOC(softmax_policy)</strong> — and sweep τ.</div>
      <div class="mdl-intuition"><span class="lbl">Why a softmax</span> The policy is a <strong>distribution</strong> over root moves, not the argmax. Softmaxing the root values gives that distribution; as search proceeds the softmax <strong>sharpens</strong>, and that sharpening = uncertainty reduction = a candidate "value of thinking." Read it as: does grading the <em>distribution</em> instead of the best move recover a missing signal?</div>
      <div class="mdl-conv warn"><span class="lbl">No</span> At calibrated τ it <strong>recovers</strong> the argmax value (~+0.16) — but never <em>exceeds</em> it. Uncertainty-reduction is real, but not the missing driver.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/voc_tau_sweep.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="h6" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Hypothesis 3 of 4</span>Is the evaluator too strong?</div>
      <div class="mdl-proc"><strong>What we did.</strong> The realization: <strong>UCI_Elo is a gameplay parameter that doesn't change the evaluation</strong> — the only knob that does is <strong>N</strong> (leaf-eval nodes). So we compare <strong>SF-1 vs SF-100</strong> (N=1 vs N=100) and correlate each with RT.</div>
      <div class="mdl-intuition"><span class="lbl">We expected</span> A weaker, noisier "gut" (fewer leaf-eval nodes) might think more like a person. Read it as: does a 1-node eval (SF-1) shift the RT correlation versus a 100-node eval (SF-100)?</div>
      <div class="mdl-conv warn"><span class="lbl">No — the evaluator isn't it</span> <strong>SF-1 ≈ SF-100 on every RT-correlation</strong> (legal +0.30/+0.31, gain +0.16/+0.16). The per-position values <em>do</em> differ (ρ≈0.93), but the RT story doesn't move — and UCI_Elo changes nothing. Evaluator strength is not the missing driver.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/sf_n1_vs_n100.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="h9" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Hypothesis 4 of 4 · the key finding</span>Is it just a legal-moves proxy?</div>
      <div class="mdl-proc"><strong>What we did.</strong> The decisive test: <strong>partial out the legal-move count</strong> from every value / VOC / step* signal, and run the reverse control (partial out the signal from legal-moves).</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> If a signal is "real," it survives controlling for # legal moves. If it's a <strong>decision-width proxy</strong>, it collapses to ≈0 while legal-moves survives. Each bar pair = a signal before/after partialling.</div>
      <div class="mdl-conv"><span class="lbl">YES — this is the one</span> Every VOC signal collapses to <strong>≈ +0.04</strong>; legal-moves <em>survives</em> the reverse control at +0.22. The "value of thinking" signal was a <strong>decision-width</strong> proxy all along.</div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/normative/rt_partials.png" /></div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="h8" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">The finding</span>So what is think-time, really?</div>
      <div class="mdl-proc"><strong>What we did.</strong> Decompose RT structurally: <strong>RT ≈ size (+0.31) − satisfaction (−0.31) + sharpness (+0.24)</strong>, and trace how satisfaction reshapes the RT-vs-n curve.</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> More options <em>slow you down</em> (size, +); more <em>good</em> options <em>speed you up</em> (satisfaction, −). If people <strong>satisfice</strong>, high-satisfaction positions should plateau low and low-satisfaction ones stay steep.</div>
      <div class="mdl-conv"><span class="lbl">The satisficing signature</span> Satisfaction <strong>reshapes the whole RT-vs-n curve</strong>: high-satisfaction positions plateau low (stop once good-enough); low stay steep. Stop-when-good-enough — not bare problem size.</div>
    </div>
    <div style="display:grid; grid-template-rows:1fr 1fr; gap:1rem; min-height:0">
      <div class="mdl-figbox"><img src="../public/figures/normative/good_moves_signflip.png" /></div>
      <div class="mdl-figbox"><img src="../public/figures/normative/rt_vs_n_concavity.png" /></div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 2
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="finding" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">★ The umbrella</span>It was the cost of the leaves all along</div>
      <div class="mdl-proc"><strong>What we did.</strong> Asked what our cost actually <em>charged</em> for. When the tree expands a node it <strong>enumerates all its legal children as leaves</strong>; the root alone lists all <em>n</em> legal moves. We compared three candidate costs against RT and against legal-moves.</div>
      <div class="mdl-intuition"><span class="lbl">Read it as</span> <strong>n_expanded</strong> = the internal nodes the oracle paid for (= n_steps). <strong>n_total / n_leaf</strong> = the enumerated candidates we <em>omitted</em> (maintenance_scale = 0). If RT is decision width, the omitted leaf-cost — not the expansions — should carry it.</div>
      <div class="mdl-conv"><span class="lbl">The convergence</span> We charged for <strong>expansions</strong> (≈constant at M=96 → ρ(RT) = <strong>+0.018</strong>, noise); the omitted <strong>leaf</strong> cost gives ρ(RT) = <strong>+0.323</strong> — the strongest single signal, <strong>0.857-collinear with legal-moves</strong>. The legal-moves effect <em>is</em> the enumeration floor of a leaf-cost. The five hypotheses were all riding a near-constant.</div>
    </div>
    <div style="display:flex; flex-direction:column; justify-content:center; gap:0.6rem">
      <table class="mdl-table mdl-table--sm mdl-table--wide">
        <thead><tr><th>cost ∝</th><th>charges for</th><th>ρ vs RT</th><th>ρ vs legal-moves</th></tr></thead>
        <tbody>
          <tr><td><code>n_expanded</code> (= n_steps)</td><td>expansions (what the oracle used)</td><td><strong>+0.018</strong></td><td>+0.022</td></tr>
          <tr><td><code>n_total</code></td><td>all nodes + enumerated leaves</td><td><strong>+0.323</strong></td><td><strong>+0.857</strong></td></tr>
          <tr><td><code>n_leaf</code></td><td>frontier leaves only</td><td>+0.323</td><td>+0.857</td></tr>
          <tr><td><em>reference</em></td><td>legal moves</td><td>+0.292</td><td>—</td></tr>
        </tbody>
      </table>
      <div class="mdl-found">We spent the project measuring <strong>internal expansions</strong> — a constant the budget fixes — while the actual lever, the <strong>leaves</strong>, sat at <strong>+0.32</strong>, zeroed out.</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-out
clicks: 1
---

<div class="mdl-content mdl-content--top">
  <div class="mdl-map"><ThoughtMap seq="plan" /></div>
</div>

---
layout: default
class: mdl-slide
transition: zoom-in
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">The plan</span>A meta-rational leaf-cost + satisficing model</div>
      <div class="mdl-proc"><strong>What we'll do.</strong> Make the model pay for the <strong>leaves it brings into consideration</strong>: enumerate candidate moves while marginal VOC &gt; c, else <strong>satisfice</strong>. Reward on stop = value(chosen) − c·#included. <strong>~1–2 parameters</strong> (cost c, prior noise).</div>
      <div class="mdl-intuition"><span class="lbl">We expect</span> The <strong>floor</strong> — enumerate the root = <em>n</em> leaves — gives the +0.31 legal-moves effect <em>for free</em>. The model's job is whether <strong>satisficing + value-pruning</strong> shape the leaf-cost to carry <em>satisfaction</em> and <em>sharpness</em> beyond the bare floor. Pruning needs <strong>regeneration</strong> (it changes the tree's shape).</div>
      <div class="mdl-conv"><span class="lbl">Root question — working conclusion</span> RT tracks <strong>decision width</strong> with a <strong>satisficing signature</strong>, and VOC adds nothing beyond width. Whether that is genuinely <em>meta-rational</em> — a reward−cost optimum a 1–2-param model reproduces — is the <strong>open test</strong>, not a settled result. The cost-profile (pruning) experiment is how we'd earn it.</div>
    </div>
    <div style="display:flex; flex-direction:column; justify-content:center; gap:1.2rem">
      <div class="mdl-card mdl-card--accent"><div class="mdl-card-h">The test (P-FIT)</div><div class="body">Fit the stop/prune threshold to RT on the leaf-inclusion trace. How much of <em>size − satisfaction + sharpness</em> emerges from 1–2 params vs the ±0.31 ceiling? <em>(now: +0.16.)</em></div></div>
      <div class="mdl-card mdl-card--neutral"><div class="mdl-card-h">Prune by the prior, not hindsight</div><div class="body">Count only the <em>plausible</em> leaves — value-prune by the early/n1 prior, relative to best, and sweep the threshold. Regenerate the tree so the freed budget drives deeper.</div></div>
    </div>
  </div>
</div>
