---
theme: default
title: Learned Meta-control of Tree Search
info: Human behavior validation — VOC/MQ analysis and architecture overview.
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

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section I</div>
    <h1 class="text-4xl">Human Behavior</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      1.97M games · 145M moves · 10+0 · both players ≥ 2000 Elo · Lichess Oct–Dec 2023
    </div>
  </div>
</div>

---

# When do players think?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Move time (seconds)</div>
    <div><span class="label">y</span> Move count</div>
    <div><span class="label">Expect</span> Right-skewed; most moves are fast</div>
    <div class="finding"><span class="label">Finding</span> Heavy right skew — mode near 2–5 s, long tail up to 600 s. Deliberation is rare but extreme.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/movetime_histogram.png" />
</div>

---

# Is thinking time log-normal?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> log(Move Time)</div>
    <div><span class="label">y</span> Move count</div>
    <div><span class="label">Expect</span> Near-symmetric if Weber's Law holds</div>
    <div class="finding"><span class="label">Finding</span> Approximately normal in log space — consistent with Weber's Law on deliberation.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/log_movetime_histogram.png" />
</div>

---

# Do players think more in the middlegame?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Move ply (game stage)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Less thinking in the opening (book), more in middlegame</div>
    <div class="finding"><span class="label">Finding</span> Non-monotone arc — peaks around ply 30–60; opening and endgame both faster.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/ply_vs_movetime.png" />
</div>

---

# Do players with more time think longer?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Player clock remaining (s)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> More time available → longer deliberation</div>
    <div class="finding"><span class="label">Finding</span> Positive relationship; holds within all ply tertiles. Players budget time proportional to what they have left.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/clock_vs_movetime.png" />
</div>

---

# Does the opponent's time pressure affect thinking?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Opponent clock remaining (s)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Weaker than own clock; some social sensitivity</div>
    <div class="finding"><span class="label">Finding</span> Real but weaker effect — players think less when opponent is also pressed.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/clock_opp_vs_movetime.png" />
</div>

---

# Do more legal moves mean more thinking?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Number of legal moves</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> More options → more deliberation</div>
    <div class="finding"><span class="label">Finding</span> Positive; effect is strongest in the middlegame tertile where tactical complexity is highest.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/npossiblemoves_vs_movetime.png" />
</div>

---

# Does your own material count matter?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Own non-pawn pieces</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Similar to total material — own army size drives candidate set</div>
    <div class="finding"><span class="label">Finding</span> Same pattern as total material; isolates the own-side effect from opponent material.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/self_pieces_exc_pawns_vs_movetime.png" />
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl">Engine Analysis</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      Stockfish depth 5/1 · ply 15–75 · opponent clock ≥ 60s · <b>1M positions</b>
    </div>
  </div>
</div>

---

# What the engine gives us

<table class="text-xs w-full mt-6 border-collapse">
  <thead>
    <tr class="border-b border-gray-300">
      <th class="text-left py-2 pr-6 font-semibold">Quantity</th>
      <th class="text-left py-2 pr-6 font-semibold">Formula</th>
      <th class="text-left py-2 font-semibold">Interpretation</th>
    </tr>
  </thead>
  <tbody class="text-[11px] opacity-90">
    <tr><td class="py-1.5 pr-6 font-mono">e_win_best</td><td class="pr-6">V<sub>deep</sub>(a<sub>deep</sub>)</td><td>Win prob of objectively best move</td></tr>
    <tr><td class="py-1.5 pr-6 font-mono">e_win_second_best</td><td class="pr-6">V<sub>deep</sub>(rank-2)</td><td>Win prob of second-best move</td></tr>
    <tr><td class="py-1.5 pr-6 font-mono">e_win_taken</td><td class="pr-6">V<sub>deep</sub>(move played)</td><td>Win prob of the actual move</td></tr>
    <tr class="border-t border-gray-200 text-accent font-semibold">
      <td class="py-1.5 pr-6 font-mono">VOC</td><td class="pr-6">e_win_best − V<sub>deep</sub>(a<sub>shallow</sub>)</td><td>Gain from deep over shallow best &nbsp;(≥ 0)</td>
    </tr>
    <tr class="text-accent font-semibold">
      <td class="py-1.5 pr-6 font-mono">MQ</td><td class="pr-6">e_win_taken − e_win_best</td><td>How suboptimal was the move played &nbsp;(≤ 0)</td>
    </tr>
    <tr class="text-accent font-semibold">
      <td class="py-1.5 pr-6 font-mono">toptwo</td><td class="pr-6">e_win_best − e_win_second_best</td><td>How decisive is the best move &nbsp;(≥ 0)</td>
    </tr>
  </tbody>
</table>

<p class="mt-4 opacity-50 text-[10px]">VOC = ΔUC from Russek et al. (2022). Win probabilities from engine WDL, side-to-move perspective before the move is made.</p>

---

# How often does deeper search find a better move?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> VOC (win-prob gain, deep over shallow)</div>
    <div><span class="label">y</span> Count</div>
    <div><span class="label">Expect</span> Mass at 0 (depth-1 often correct), right-skewed tail</div>
    <div class="finding"><span class="label">Finding</span> 67% zero — depth-1 is already optimal. Right tail represents tactical positions where deep search matters.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/voc_histogram.png" />
</div>

---

# How optimal are human moves?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> MQ = e_win_taken − e_win_best &nbsp;(≤ 0)</div>
    <div><span class="label">y</span> Count</div>
    <div><span class="label">Expect</span> Spike at 0 (best move played), left tail for blunders</div>
    <div class="finding"><span class="label">Finding</span> 62% near-optimal (MQ ≥ −0.005). Long left tail from rare but large blunders.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/mq_histogram.png" />
</div>

---

# Do humans think longer when it pays?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> VOC (value of deep computation)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Higher VOC → more thinking (Russek et al., 2022)</div>
    <div class="finding"><span class="label">Finding</span> <b>r = +0.096</b> (n = 1M). Positive and consistent across all game stages — humans allocate compute where it helps.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/voc_vs_movetime.png" />
</div>

---

# Does a decisive best move mean more or less thinking?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> toptwo = e_win_best − e_win_second_best &nbsp;(≥ 0)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Ambiguous — decisive positions may need more search to find, or may be quickly recognised</div>
    <div class="finding"><span class="label">Finding</span> <b>r = −0.064</b> (n = 1M) — negative. When one move is clearly superior, players deliberate less. Contrast: VOC is positive, as players think more when computation could flip the best move.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/toptwo_vs_movetime.png" />
</div>

---

# Does more time mean better moves?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Player clock remaining (s)</div>
    <div><span class="label">y</span> MQ (move quality ≤ 0)</div>
    <div><span class="label">Expect</span> More clock → more deliberation → MQ closer to 0</div>
    <div class="finding"><span class="label">Finding</span> <b>r = −0.091</b> (n = 1M) — counterintuitive. More clock → worse MQ, even within ply tertiles. Mediated by VOC: fast-played book positions hurt depth-5 MQ.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/mq_vs_clock.png" />
</div>

---

# Correlation matrix

<div class="grid grid-cols-[62fr_38fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <img class="w-full object-contain max-h-[72vh]" src="/figures/correlation_matrix.png" />
  <div class="space-y-3 text-xs leading-relaxed">
    <div class="font-semibold text-sm mb-2">n = 1,000,000 · Pearson r</div>
    <div class="p-2 bg-neutral-soft rounded text-[11px]"><b>VOC ↔ MQ</b> r = −0.42 — positions where deep search helps are also ones where humans blunder more</div>
    <div class="p-2 bg-neutral-soft rounded text-[11px]"><b>toptwo ↔ MQ</b> r = −0.30 — decisive positions penalise errors more heavily</div>
    <div class="p-2 bg-neutral-soft rounded text-[11px]"><b>Ply ↔ own material</b> r = −0.81 — as expected: pieces leave the board over the game</div>
    <div class="p-2 bg-neutral-soft rounded text-[11px]"><b>log(RT)</b> correlates positively with branching (0.20) and VOC (0.10); negatively with toptwo (−0.06) and MQ (−0.12)</div>
  </div>
</div>

---

# Summary

<div class="mt-6 max-w-3xl space-y-4">
  <div class="grid grid-cols-2 gap-6 text-sm">
    <div class="p-3 bg-neutral-soft border-l-2 border-accent">
      <b class="text-accent text-xs uppercase tracking-wider">Confirmed</b>
      <ul class="mt-2 list-disc pl-4 space-y-1 text-xs">
        <li>log(RT) ∝ VOC &nbsp;—&nbsp; r = +0.096 ✓</li>
        <li>VOC spike at 0, right-skewed tail ✓</li>
        <li>More clock → more thinking ✓</li>
        <li>Branching & material drive deliberation ✓</li>
      </ul>
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-amber-500">
      <b class="text-amber-700 text-xs uppercase tracking-wider">Open</b>
      <ul class="mt-2 list-disc pl-4 space-y-1 text-xs">
        <li>MQ–clock is negative within all ply tertiles</li>
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
    <h1 class="text-4xl">Architecture & Related Work</h1>
  </div>
</div>

---

# Paper Sketch

<div class="h-full flex flex-col justify-start mt-2">
  <PaperSketch />
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

<p class="mt-3 text-sm">Sigmoidal halting unit + ponder cost $\tau$. Depth-per-input is learned end-to-end. Operates on sequential traces, not tree topology.</p>

---

# Imagination-Based Planner (IBP)

<div class="text-xs opacity-70 mb-3">Pascanu et al., arXiv:1707.06170</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/ibp-fig1.png" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Manager imagines vs. acts.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/ibp-fig2.png" />
    <figcaption class="mt-1 opacity-60 text-[11px]">1-step / n-step / tree strategies.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm">End-to-end, explicit when-to-imagine. Weak on tree topology in the representation.</p>

---

# Thinker

<div class="text-xs opacity-70 mb-3">Chung, Anokhin & Krueger, arXiv:2307.14993</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/thinker-fig2-stage.svg" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Stage: K−1 imaginary steps, then real step.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-60" src="/figures/lit/thinker-fig3-tree.svg" />
    <figcaption class="mt-1 opacity-60 text-[11px]">Rollout/reset under stage budget.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm">Fixed stage geometry — not an independent halting policy over arbitrary search trees.</p>

---

# Gap → this work

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider">Prior meta-control</b>
    <p class="mt-1">Good on when to stop; sequential summaries, no tree structure.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary">
    <b class="text-secondary text-xs uppercase tracking-wider">Prior tree search</b>
    <p class="mt-1">Good on how to search; budget usually fixed externally.</p>
  </div>
</div>

<p class="mt-5 text-sm max-w-3xl">
  <b>Our approach:</b> structural view of the search tree via GNN, trainable alongside a fixed engine — validated against human timing as an existence proof for efficient stopping.
</p>

---

# Control flow

<div class="h-full flex flex-col items-center justify-center">
  <LeelaSearchLoop />
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

$$R(k) = \mathbb{E}\!\left[\text{Value}(T_k)\right] - C \cdot k$$

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">
  <div>
    <b>Input:</b> root hidden state $h_\text{root}$<br>
    <b>Output:</b> $P(\text{halt} \mid h_\text{root})$ via MLP<br>
    <b>Training:</b> policy gradients against DP Oracle
  </div>
  <div class="opacity-80">Learns the inflection of the "thinking curve" — where additional computation stops paying off.</div>
</div>

---

# Pretraining: GNN representation

<div class="mt-4">
  <GnnPretrainDiagram />
</div>

---

# Pretraining: ChildWDL

<div class="flex flex-col items-center mt-4">
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
