---
theme: default
title: Learned Meta-control of Tree Search
info: CMC architecture overview — Control Flow, Representation, Decision, Training, Human Validation.
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
    Mechanistic Interpretability · Resource Rationality · Chess Search
  </div>
</div>

---

# Paper Sketch

<div class="h-full flex flex-col justify-start mt-2">
  <PaperSketch />
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Background</div>
    <h1 class="text-4xl">Related Work</h1>
  </div>
</div>

---

# Why meta-control?

<div class="text-sm leading-relaxed max-w-3xl space-y-3 mt-4">
  <ul class="list-disc pl-5 space-y-3">
    <li>Inference cost (tokens, MCTS nodes) is <b>budget</b> you want to spend where it helps — not uniformly across all decisions.</li>
    <li>A <b>meta-controller</b> that decides <em>when to stop</em> keeps an outside view; baking the same choice into the planner entangles "when" with "how."</li>
    <li>Mirrors <b>fast vs. slow</b> reasoning in cognitive science (Kahneman, 2011; MB/MF RL — Daw et al., 2005).</li>
  </ul>
</div>

---

# Adaptive Computation Time (ACT)

<div class="text-xs opacity-70 mb-3">Graves, NeurIPS 2016 (arXiv:1603.08983)</div>

<div class="grid grid-cols-2 gap-5 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-52" src="/figures/lit/act-fig1-rnn.png" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Standard RNN — fixed computation per step.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-52" src="/figures/lit/act-fig2-act.png" />
    <figcaption class="mt-1 opacity-60 text-[11px]">ACT — learned halting unit adds variable ponder steps.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  Sigmoidal halting unit + ponder cost $\tau$ in loss. Depth-per-input is learned end-to-end. Operates on <b>sequential</b> traces, not tree topology.
</p>

---

# Imagination-Based Planner (IBP)

<div class="text-xs opacity-70 mb-3">Pascanu et al., arXiv:1707.06170</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/ibp-fig1.png" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Manager imagines vs. acts; memory aggregates the trace.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/ibp-fig2.png" />
    <figcaption class="mt-1 opacity-60 text-[11px]">1-step / n-step / tree imagination strategies.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  Explicit <em>when-to-imagine</em>, end-to-end training. Weak on explicit tree topology in the representation — our gap.
</p>

---

# Thinker

<div class="text-xs opacity-70 mb-3">Chung, Anokhin & Krueger, arXiv:2307.14993</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/thinker-fig2-stage.svg" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Stage: K−1 imaginary steps, then one real env step.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/thinker-fig3-tree.svg" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Rollout/reset dynamics under stage budget.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  Planning and acting separated via augmented actions. Fixed stage geometry — not an independent halting policy over arbitrary search trees.
</p>

---

# Gap → this work

<div class="grid grid-cols-2 gap-6 mt-4 text-sm leading-relaxed">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider">Prior meta-control</b>
    <p class="mt-1">Good on <i>when</i> to stop; often sequential summaries, no tree structure.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary">
    <b class="text-secondary text-xs uppercase tracking-wider">Prior tree search</b>
    <p class="mt-1">Good on <i>how</i> to search; budget usually fixed externally.</p>
  </div>
</div>

<p class="mt-5 text-sm max-w-3xl">
  <b>Our approach:</b> a structural view of the search tree via GNN, trainable alongside a fixed engine — validated against human timing data as an existence proof for efficient stopping.
</p>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Architecture</div>
    <h1 class="text-4xl">The CMC Loop</h1>
  </div>
</div>

---

# Control flow

<div class="h-full flex flex-col items-center justify-center">
  <LeelaSearchLoop />
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Human Validation</div>
    <h1 class="text-4xl">Do humans allocate compute like a meta-controller?</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      If the VOC framework is right, think time should track the value of thinking — not clock pressure alone.
    </div>
  </div>
</div>

---

# Dataset

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-2">
  <ul class="list-disc pl-5 space-y-2">
    <li><b>Source:</b> Lichess, Oct–Dec 2023 · 10+0 · both players ≥ 2000 Elo</li>
    <li><b>Scale:</b> 1.97M games · 145M moves · <b>135M non-zero-time moves</b></li>
    <li><b>Engine analysis subset:</b> ply 15–75, opponent clock ≥ 60s → 88M positions; <b>1M sampled</b> at Stockfish depth 5/1</li>
  </ul>
</div>

---

# Move time distribution

<div class="grid grid-cols-2 gap-6 mt-4 items-center">
  <img class="w-full object-contain" src="/figures/movetime_histogram.png" />
  <img class="w-full object-contain" src="/figures/log_movetime_histogram.png" />
</div>

<div class="takeaway border-secondary bg-neutral-soft text-sm py-3 mt-3">
  Heavy right skew; approximately log-normal in log space — consistent with Weber's Law. Most moves are fast; deliberation has a long tail.
</div>

---

# Think time: game stage and clock

<div class="grid grid-cols-2 gap-4 mt-2 items-start">
  <div>
    <div class="text-[11px] opacity-60 mb-1">Game stage (ply)</div>
    <img class="w-full object-contain" src="/figures/ply_vs_movetime.png" />
  </div>
  <div>
    <div class="text-[11px] opacity-60 mb-1">Player clock remaining</div>
    <img class="w-full object-contain" src="/figures/clock_vs_movetime.png" />
  </div>
</div>

<div class="takeaway border-accent bg-accent-soft text-sm py-3 mt-2">
  <b>Ply:</b> non-monotone arc — fast openings, slower middlegame, faster endgame. &nbsp;·&nbsp; <b>Clock:</b> more time remaining → longer thinks; consistent across all game stages.
</div>

---

# Think time: opponent and branching

<div class="grid grid-cols-2 gap-4 mt-2 items-start">
  <div>
    <div class="text-[11px] opacity-60 mb-1">Opponent clock remaining</div>
    <img class="w-full object-contain" src="/figures/clock_opp_vs_movetime.png" />
  </div>
  <div>
    <div class="text-[11px] opacity-60 mb-1">Number of legal moves</div>
    <img class="w-full object-contain" src="/figures/npossiblemoves_vs_movetime.png" />
  </div>
</div>

<div class="takeaway border-success bg-success-soft text-sm py-3 mt-2">
  <b>Opp clock:</b> weaker but real — players think less when opponent is also pressed. &nbsp;·&nbsp; <b>Branching:</b> more legal moves → more thinking; decision complexity drives deliberation.
</div>

---

# Think time: material

<div class="grid grid-cols-2 gap-4 mt-2 items-start">
  <div>
    <div class="text-[11px] opacity-60 mb-1">Non-pawn pieces on board</div>
    <img class="w-full object-contain" src="/figures/pieces_exc_pawns_vs_movetime.png" />
  </div>
  <div>
    <div class="text-[11px] opacity-60 mb-1">Own non-pawn pieces</div>
    <img class="w-full object-contain" src="/figures/self_pieces_exc_pawns_vs_movetime.png" />
  </div>
</div>

<div class="takeaway border-sky-500 bg-sky-50/90 text-sm py-3 mt-2">
  Richer material → more thinking. Effect compresses in the endgame tertile — fewer pieces simplify the candidate set.
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Engine Analysis</div>
    <h1 class="text-4xl">Value of Computation & Move Quality</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      Do humans think longer on positions where thinking pays off?
    </div>
  </div>
</div>

---

# What the engine gives us

<table class="text-xs w-full mt-4 border-collapse">
  <thead>
    <tr class="border-b border-gray-300">
      <th class="text-left py-2 pr-4 font-semibold">Quantity</th>
      <th class="text-left py-2 pr-4 font-semibold">Formula</th>
      <th class="text-left py-2 font-semibold">Interpretation</th>
    </tr>
  </thead>
  <tbody class="opacity-90 text-[11px]">
    <tr><td class="py-1.5 pr-4 font-mono">e_win_best</td><td class="pr-4">V<sub>deep</sub>(a<sub>deep</sub>)</td><td>Win prob of objectively best move</td></tr>
    <tr><td class="py-1.5 pr-4 font-mono">e_win_second_best</td><td class="pr-4">V<sub>deep</sub>(rank-2)</td><td>Win prob of second-best</td></tr>
    <tr><td class="py-1.5 pr-4 font-mono">e_win_taken</td><td class="pr-4">V<sub>deep</sub>(move played)</td><td>Win prob of the actual move</td></tr>
    <tr class="border-t border-gray-200 font-semibold text-accent">
      <td class="py-1.5 pr-4 font-mono">VOC</td><td class="pr-4">e_win_best − V<sub>deep</sub>(a<sub>shallow</sub>)</td><td>Gain from deep vs. shallow best  (≥ 0)</td>
    </tr>
    <tr class="font-semibold text-accent">
      <td class="py-1.5 pr-4 font-mono">MQ</td><td class="pr-4">e_win_taken − e_win_best</td><td>How suboptimal was the move played  (≤ 0)</td>
    </tr>
    <tr class="font-semibold text-accent">
      <td class="py-1.5 pr-4 font-mono">toptwo</td><td class="pr-4">e_win_best − e_win_second_best</td><td>How decisive is the best move  (≥ 0)</td>
    </tr>
  </tbody>
</table>

<p class="mt-3 opacity-60 text-[11px]">Stockfish depth 5 (deep) / 1 (shallow). Win probabilities from engine WDL, side-to-move perspective. VOC = ΔUC from Russek et al. (2022).</p>

---

# VOC and MQ distributions

<div class="grid grid-cols-2 gap-4 mt-2 items-start">
  <div>
    <div class="text-[11px] opacity-60 mb-1">VOC  (n = 1M)</div>
    <img class="w-full object-contain" src="/figures/voc_histogram.png" />
  </div>
  <div>
    <div class="text-[11px] opacity-60 mb-1">MQ  (n = 1M)</div>
    <img class="w-full object-contain" src="/figures/mq_histogram.png" />
  </div>
</div>

<div class="takeaway border-accent bg-accent-soft text-sm py-3 mt-2">
  <b>VOC:</b> 67% zero (depth-1 already optimal), long right tail for tactical positions. &nbsp;·&nbsp; <b>MQ:</b> 62% near-optimal, long left tail from blunders. Both heavy-spike at 0.
</div>

---

# VOC vs. log(RT): humans think longer when it pays

<div class="grid grid-cols-[65fr_35fr] gap-8 mt-2 items-start">
  <img class="w-full object-contain" src="/figures/voc_vs_movetime.png" />
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">Key result</b><br><br>
    r(log RT, VOC) = <b>+0.096</b>  (n = 1M)<br><br>
    Positive: higher VOC → more think time. Holds across all ply tertiles. Consistent with Russek et al. (2022) at depth 15.
  </div>
</div>

---

# MQ vs. clock: a counterintuitive finding

<div class="grid grid-cols-[65fr_35fr] gap-8 mt-2 items-start">
  <img class="w-full object-contain" src="/figures/mq_vs_clock.png" />
  <div class="takeaway border-amber-500 bg-amber-50/90 text-sm py-4">
    <b class="text-amber-900 uppercase tracking-wider text-xs">Surprising</b><br><br>
    r(MQ, clock) = <b>−0.091</b>  (n = 1M)<br><br>
    More clock → <i>worse</i> MQ, even within ply tertiles. Players with more time have played quickly through low-VOC positions; depth-5 penalises those strategic choices.
  </div>
</div>

---

# Summary

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-4">
  <div class="grid grid-cols-2 gap-6">
    <div class="p-3 bg-neutral-soft border-l-2 border-accent">
      <b class="text-accent text-xs uppercase tracking-wider">Confirmed (Russek et al.)</b>
      <ul class="mt-2 list-disc pl-4 space-y-1 text-xs">
        <li>log(RT) ∝ VOC  — r = +0.096 ✓</li>
        <li>VOC spike at 0, right-skewed tail ✓</li>
        <li>Effect holds across game stages ✓</li>
      </ul>
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-amber-500">
      <b class="text-amber-700 text-xs uppercase tracking-wider">Open questions</b>
      <ul class="mt-2 list-disc pl-4 space-y-1 text-xs">
        <li>MQ–clock is negative (within all tertiles)</li>
        <li>67% zero-VOC at depth=5 → need depth=15</li>
        <li>E[ΔUC] (expected VOC) not yet computed</li>
        <li>Scale to full 88M-position dataset</li>
      </ul>
    </div>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Appendix</div>
    <h1 class="text-4xl">Architecture Details</h1>
  </div>
</div>

---

# Representation & architecture

<div class="mt-4">
  <MetaControllerZoom />
</div>

---

# GNN: bidirectional sweeps

<div class="mt-4">
  <GnnTwoSweeps />
</div>

---

# Halt controller

We frame stopping as a learnable policy $\pi_\theta$ over the current root embedding.

$$R(k) = \mathbb{E}\!\left[\text{Value}(T_k)\right] - C \cdot k$$

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">
  <div>
    <b>Input:</b> root hidden state $h_\text{root}$<br>
    <b>Output:</b> $P(\text{halt} \mid h_\text{root})$ via MLP<br>
    <b>Training:</b> policy gradients against DP Oracle
  </div>
  <div class="opacity-80">
    The agent learns the inflection point of the "thinking curve" — where additional computation stops paying off.
  </div>
</div>

---

# Pretraining: GNN representation

<div class="mt-4">
  <GnnPretrainDiagram />
</div>

---

# Pretraining: ChildWDL

<div class="flex flex-col items-center justify-start mt-4">
  <div class="w-4/5">
    <ChildWdlDiagram />
  </div>
</div>

---

# Training stage 2: DP Oracle

<div class="mt-4">
  <PolicyPretrainDiagram />
</div>

---

# DP Oracle intuition

<div class="mt-2">
  <DpOracleDiagram />
</div>

<div class="grid grid-cols-3 gap-6 mt-4 text-sm px-4">
  <div><b class="text-accent block mb-1">Why DP?</b>Optimal stopping is recursive — reason backwards from leaf outcomes.</div>
  <div><b class="text-accent block mb-1">Solution</b>$V^*(s) = \max(V,\, \mathbb{E}[V^*_\text{child}] - C)$</div>
  <div><b class="text-accent block mb-1">Supervision</b>DP result as ground truth for the GNN halt head.</div>
</div>
