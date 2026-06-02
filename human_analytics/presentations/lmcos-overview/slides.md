---
theme: default
title: Learned Meta-control of Tree Search
info: |
  A didactic breakdown of the CMC architecture: Control Flow · Representation · GNN Mechanics · Decision · Training.
css: ./style.css
class: text-left
mdc: true
math: katex
---

<ChessBackground />

<div class="absolute bottom-12 left-14 z-10 text-white">
  <h1 class="m-0" style="line-height: 1.0; font-size: 3rem; color: white !important;">
    Learned Meta-control<br>
    of Tree Search
  </h1>
  
  <div class="mt-2 text-lg opacity-80">
    Yotam Sagiv & Jordan Lei
  </div>

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
    <h1 class="text-4xl">Related work</h1>
    <div class="mt-3 text-sm opacity-60 max-w-xl mx-auto">
      Adaptive "when to think," imagination-based control, and learned tree search — with citations on-slide.
    </div>
  </div>
</div>

---

# Why meta-control (in one minute)

<div class="text-sm leading-relaxed max-w-3xl space-y-3 mt-1">
  <ul class="list-disc pl-5 space-y-2">
    <li>Inference cost (tokens, rollouts, MCTS nodes) is <b>budget</b> you want to spend where it helps.</li>
    <li>A <b>meta-controller</b> that only decides whether to continue can keep an <b>outside view</b>; baking the same choice into the planner often <b>entangles</b> "when" with "how."</li>
    <li>The same tradeoff is <b>fast vs. slow</b> reasoning: model-free heuristics vs. model-based lookahead (e.g. Kahneman, 2011; Daw, Niv &amp; Dayan, 2005 on MB/MF RL).</li>
  </ul>
</div>

---

# Adaptive Computation Time (ACT)

<div class="text-xs opacity-80 mb-2">Alex Graves — <i>Adaptive Computation Time for Recurrent Neural Networks</i>, NeurIPS 2016; arXiv:1603.08983. Figures reproduced from the paper (Fig. 1–2).</div>

<div class="grid grid-cols-2 gap-5 items-start mt-1">
  <figure class="m-0">
    <img class="w-full object-contain max-h-52" src="/figures/lit/act-fig1-rnn.png" alt="Standard RNN computation graph" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 1 — Standard RNN (two input steps).</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-52" src="/figures/lit/act-fig2-act.png" alt="RNN with ACT" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 2 — Same graph with variable intermediate "ponder" steps and halting.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  <b>Idea:</b> sigmoidal <b>halting unit</b> + <b>ponder cost</b> $\tau$ in the loss so depth-per-input is learned. Precursors: self-delimiting nets (SLIMs) with hard thresholds.
</p>

---

# Imagination-Based Planner (IBP)

<div class="text-xs opacity-80 mb-2">R. Pascanu, Y. Li, O. Vinyals, N. Heess, L. Buesing, S. Racanière, D. Reichert, T. Weber, D. Wierstra &amp; P. Battaglia — <i>Learning model-based planning from scratch</i>, arXiv:1707.06170. Figures reproduced from the paper (Fig. 1–2); raster assets extracted from the arXiv PDF.</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/ibp-fig1.png" alt="IBP schematic" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 1 — Manager imagines vs. acts; memory aggregates the trace.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/ibp-fig2.png" alt="IBP imagination strategies" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 2 — 1-step / <i>n</i>-step / tree imagination strategies over imagined states.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  <b>For us:</b> explicit <i>when-to-imagine</i>, but <b>end-to-end</b> training and an <b>RNN</b> trace → strong on <i>order</i>, weak on explicit <b>tree topology</b> in the representation.
</p>

---

# Thinker

<div class="text-xs opacity-80 mb-2">S. Chung, I. Anokhin &amp; D. Krueger — <i>Thinker: Learning to Plan and Act</i>, arXiv:2307.14993. Figures reproduced from the paper (Fig. 2–3); source SVGs from the arXiv package.</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/thinker-fig2-stage.svg" alt="Thinker stage" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 2 — One stage: <i>K − 1</i> imaginary model steps, then one real env step.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/thinker-fig3-tree.svg" alt="Thinker tree traversal" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 3 — Rollout / reset dynamics under a stage budget and max depth.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  <b>Progress:</b> planning and acting separated via augmented actions. <b>Limits here:</b> fixed stage geometry and a learned model-specific interface — not a separate halting policy over arbitrary search trees.
</p>

---

# Learned tree search (fixed budget)

<div class="text-sm leading-relaxed max-w-3xl space-y-3 mt-1">
  <p>
    <b>AlphaZero</b> (Silver et al., 2018, <i>Science</i>): policy + value coupled to MCTS — strong structure, but simulations per move are set externally, not a per-node "worth another expansion?" learner.
  </p>
  <p>
    <b>MCTSnets</b> (Guez et al., 2018, ICML; arXiv:1802.04697): backups and visit patterns through learned embeddings — topology in the net, still not an independent meta-controller for compute.
  </p>
</div>

---

# Gap → this deck

<div class="grid grid-cols-2 gap-6 mt-2 text-sm leading-relaxed">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider">Meta-control</b>
    <p class="mt-1">Good on <i>when</i> to stop / imagine; often sequential summaries of the trace.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary">
    <b class="text-secondary text-xs uppercase tracking-wider">Tree search</b>
    <p class="mt-1">Good on <i>how</i> to search; budget usually fixed, not a trained halt policy on the tree.</p>
  </div>
</div>

<p class="mt-5 text-sm max-w-3xl">
  <b>Our thread:</b> halting / allocation using a <b>structural</b> view of the search tree (later slides), trainable <b>beside</b> a fixed engine — with chess oracle + human timing as targets.
</p>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section I</div>
    <h1 class="text-4xl">Part 1 — The Problem</h1>
  </div>
</div>

---

# Why Meta-Control?

<v-clicks>

- **Fixed Budgets are Wasteful:** Standard engines spend the same time on a "forced" move as a complex tactical blunder.
- **The Trade-off:** Is the move-quality I'm about to discover worth the computational "electricity" (time/tokens) I'm about to spend?
- **Real-World Constraints:** Tokens cost money, but more importantly, **real-time decisions have a physical cost**. Most LLM tasks are sufficiently time-intensive that every token of "thought" must justify itself.
- **Human Intuition:** Skilled players know *when* to stop thinking—a stopping problem we can formalize.

</v-clicks>

---

# Part 2: High-Level Architecture

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <LeelaSearchLoop />
  
  <div class="mt-8 p-3 bg-neutral-soft border-l-2 border-accent italic text-[11px] opacity-80">
    Implementation details (GNN sweeps, DP Oracle, Halt Controller) are in the Appendix.
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl">Part 2 — Validation & Alignment</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      How do we know the meta-controller works as intended? Humans serve as an <b>existence proof</b> and a <b>target distribution</b> for efficient search.
    </div>
  </div>
</div>

---

# Behavioral data: how we chose these games

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-3">
  <p>
    Every timing slide uses the <b>same</b> human move pool in DuckDB. Code: <code>human_analytics/slurm/scripts/preprocess.py</code>.
  </p>
  <ul class="list-disc pl-5 space-y-2">
    <li><b>Which games</b> — Strong rapid: <b>10+0</b>, <b>both players 2000+ Elo</b>, <b>Oct–Dec 2023</b>.</li>
    <li><b>Scale</b> — 1.97M games · 145M moves · <b>135M non-zero-time moves</b> after bad-game filtering (berserk, negative move_time, grant-more-time).</li>
    <li><b>Engine analysis</b> — Stockfish depth=5/1 on a filtered subset (ply 15–75, opponent clock ≥ 60s); 88M positions pass; 1M sampled for analysis.</li>
  </ul>
</div>

---

# 1. Move time distribution

<div class="text-xs opacity-60 mb-2 -mt-2">Excludes premoves (<code>move_time = 0</code>). 50-bin histogram; n = 135M moves.</div>

<div class="grid grid-cols-2 gap-6 mt-4 items-start">
  <div>
    <img class="w-full object-contain" src="/figures/movetime_histogram.png" />
    <div class="text-[11px] opacity-70 mt-1 text-center">Move time (s)</div>
  </div>
  <div>
    <img class="w-full object-contain" src="/figures/log_movetime_histogram.png" />
    <div class="text-[11px] opacity-70 mt-1 text-center">log(move time)</div>
  </div>
</div>

<div class="takeaway border-secondary bg-neutral-soft text-sm py-3 mt-3">
  Heavy right skew in raw time; log-normal in log space — consistent with Weber's Law. Median ≈ 3s, mean pulled by long-tailed deliberation.
</div>

---

# 2. Game stage: ply vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">x = move ply; y = log(move time). 2×2 dashboard: global trend, quantile bins, ×2 by ply tertile.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/ply_vs_movetime.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">ply_vs_movetime</b><br><br>
    Non-monotone arc: fast openings (book), slower middlegame, faster endgame. Think time peaks around ply 30–60.
  </div>
</div>

---

# 3. Remaining clock vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">x = player clock remaining; y = log(move time). Excludes <code>move_time = 0</code> and clock ≥ 600s.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_vs_movetime.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">clock_vs_movetime</b><br><br>
    More clock → longer think time. Stratified by ply tertile: pattern holds in all game stages. Consistent with deliberation budgeting.
  </div>
</div>

---

# 4. Opponent clock vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">x = opponent clock remaining; y = log(move time).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_opp_vs_movetime.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">clock_opp_vs_movetime</b><br><br>
    Weaker effect than own clock. Players are partially sensitive to opponent time pressure — think less when opponent is also pressed.
  </div>
</div>

---

# 5. Branching factor vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">x = number of legal moves; y = log(move time).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/npossiblemoves_vs_movetime.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">npossiblemoves_vs_movetime</b><br><br>
    More legal moves → more thinking. Branching factor is a proxy for decision complexity. Effect is strongest in middlegame (tertile 2).
  </div>
</div>

---

# 6. Material vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">x = non-pawn pieces on board; y = log(move time).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/pieces_exc_pawns_vs_movetime.png" />
  
  <div class="takeaway border-sky-500 bg-sky-50/90 text-sm py-4">
    <b class="text-sky-700 uppercase tracking-wider text-xs">pieces_exc_pawns_vs_movetime</b><br><br>
    More pieces → more time. Richer material = more candidate interactions. Effect compresses in the endgame tertile.
  </div>
</div>

---

# 7. Own material vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">x = player's own non-pawn pieces; y = log(move time).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/self_pieces_exc_pawns_vs_movetime.png" />

  <div class="takeaway border-amber-500 bg-amber-50/90 text-sm py-4">
    <b class="text-amber-900 uppercase tracking-wider text-xs">self_pieces_exc_pawns_vs_movetime</b><br><br>
    Similar to §6 but predictor is <i>your</i> remaining officers. Controls for opponent material separately.
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section III</div>
    <h1 class="text-4xl">Engine Analysis</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      Value of Computation (VOC) and Move Quality (MQ) — connecting human timing to engine evaluation.
    </div>
  </div>
</div>

---

# Engine quantities: what we compute

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-2">
  <p>For each move in <code>pos_with_engine_eval</code> (ply 15–75, opponent clock ≥ 60s), Stockfish depth 5/1 gives:</p>
  
  <table class="text-xs w-full mt-3 border-collapse">
    <thead>
      <tr class="border-b border-gray-300">
        <th class="text-left py-1 pr-4">Column</th>
        <th class="text-left py-1 pr-4">Formula</th>
        <th class="text-left py-1">Interpretation</th>
      </tr>
    </thead>
    <tbody class="opacity-90">
      <tr><td class="py-1 pr-4 font-mono">e_win_best</td><td class="pr-4">V<sub>deep</sub>(a<sub>deep</sub>)</td><td>Win prob of objectively best move</td></tr>
      <tr><td class="py-1 pr-4 font-mono">e_win_second_best</td><td class="pr-4">V<sub>deep</sub>(rank-2)</td><td>Win prob of second-best</td></tr>
      <tr><td class="py-1 pr-4 font-mono">e_win_taken</td><td class="pr-4">V<sub>deep</sub>(move played)</td><td>Win prob of the actual move</td></tr>
      <tr class="border-t border-gray-200"><td class="py-1 pr-4 font-mono text-accent">voc</td><td class="pr-4">e_win_best − V<sub>deep</sub>(a<sub>shallow</sub>)</td><td>Gain from deep vs. shallow best (≥ 0)</td></tr>
      <tr><td class="py-1 pr-4 font-mono text-accent">mq</td><td class="pr-4">e_win_taken − e_win_best</td><td>How suboptimal was the move played (≤ 0)</td></tr>
      <tr><td class="py-1 pr-4 font-mono text-accent">toptwo</td><td class="pr-4">e_win_best − e_win_second_best</td><td>How decisively best beats 2nd-best (≥ 0)</td></tr>
    </tbody>
  </table>

  <p class="mt-3 opacity-70 text-xs">All probabilities are from the <b>side to move</b> before the move is made. VOC = ΔUC from Russek et al. (2022).</p>
</div>

---

# VOC distribution (n = 1M)

<div class="text-xs opacity-60 mb-2 -mt-2">Stockfish depth 5/1 · ply 15–75 · opponent clock ≥ 60s.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/voc_histogram.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">VOC</b><br><br>
    Heavy mass at 0 (depth-1 already optimal for ~67% of positions). Long right tail: tactical positions where deeper search finds substantially better moves. Mean ≈ 0.10.
  </div>
</div>

---

# MQ distribution (n = 1M)

<div class="text-xs opacity-60 mb-2 -mt-2">MQ = e_win_taken − e_win_best ≤ 0.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/mq_histogram.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">MQ</b><br><br>
    Spike at 0: 62% of moves are near-optimal (MQ ≥ −0.005). Long left tail from blunders. Mean ≈ −0.12. Distribution reflects that strong players play well but occasionally err badly.
  </div>
</div>

---

# VOC vs. log(RT): humans think longer on hard positions

<div class="text-xs opacity-60 mb-2 -mt-2">x = VOC; y = log(move time). 2×2 dashboard by ply tertile.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/voc_vs_movetime.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">voc_vs_movetime</b><br><br>
    r(log RT, VOC) = <b>+0.097</b> (n=100K). Positive: higher VOC → more think time. Consistent with Russek et al. (2022). Depth=15 would sharpen the signal (33% non-zero VOC at depth=5 vs. more at depth=15).
  </div>
</div>

---

# MQ vs. clock: a surprising finding

<div class="text-xs opacity-60 mb-2 -mt-2">x = player clock remaining; y = MQ. 2×2 dashboard by ply tertile.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/mq_vs_clock.png" />
  
  <div class="takeaway border-amber-500 bg-amber-50/90 text-sm py-4">
    <b class="text-amber-900 uppercase tracking-wider text-xs">mq_vs_clock</b><br><br>
    <b>r = −0.098</b> — more clock → <i>worse</i> MQ, even within ply tertiles. Players with more clock have played quickly through low-VOC (book) positions; depth-5 penalises those strategic choices. Effect is mediated by VOC, not ply.
  </div>
</div>

---

# Summary: human timing and VOC

<div class="mt-6 text-sm leading-relaxed max-w-3xl space-y-4">
  <div class="grid grid-cols-2 gap-6">
    <div class="p-3 bg-neutral-soft border-l-2 border-accent">
      <b class="text-accent text-xs uppercase tracking-wider">Confirmed (Russek et al.)</b>
      <ul class="mt-2 list-disc pl-4 space-y-1 text-xs">
        <li>log(RT) ∝ VOC: r = +0.097 ✓</li>
        <li>VOC distribution right-skewed with mass at 0 ✓</li>
        <li>Effect holds across game stages ✓</li>
      </ul>
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-amber-500">
      <b class="text-amber-700 text-xs uppercase tracking-wider">Novel / Open</b>
      <ul class="mt-2 list-disc pl-4 space-y-1 text-xs">
        <li>MQ vs. clock is <i>negative</i> (within all tertiles)</li>
        <li>67% zero-VOC at depth=5 → depth=15 needed</li>
        <li>E[ΔUC] (expected VOC) not yet computed</li>
      </ul>
    </div>
  </div>

  <p class="opacity-70 text-xs mt-4">
    Next: scale to 1M positions at depth=5; then depth=15 on a targeted subset. Eventually loop VOC/MQ into <code>build_selected_moves_with_engine</code> for the full 88M-move dataset.
  </p>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Appendix</div>
    <h1 class="text-4xl">Technical Implementation Details</h1>
  </div>
</div>

---

# 1. Control Flow: The Loop

The Meta-Controller sits on top of a standard MCTS planner (like Leela), deciding at each step whether to expand further or act.

<div class="mt-8">
  <LeelaSearchLoop />
</div>

---

# 2. Representation & Architecture

<div class="mt-8">
  <MetaControllerZoom />
</div>

---

# 3. Learning: Bidirectional Sweeps

The GNN performs representation learning over the tree using two sequential passes.

<div class="mt-4">
  <GnnTwoSweeps />
</div>

---

# 4. Decision: The Halt Controller

We frame search depth as a learnable policy $\pi_\theta$. The **Halt Controller** solves a stopping problem by maximizing the net value of computation.

**The Economy of Thought**
The agent seeks an optimal depth $k^*$ that maximizes the expected reward:

$$
R(k) = \mathbb{E} \left[ \text{Value}(T_k) \right] - C \cdot k
$$

**Mechanistic Readout**
- **Input:** The current root hidden state $h_{root}$.
- **Output:** A halting probability $P(\text{halt} | h_{root})$ predicted by an MLP.
- **Optimization:** Trained via Policy Gradients ($\nabla_\theta J$) against a DP Oracle to learn the inflection point of the "Thinking Curve."

---

# 5. Training Stage 1: Representation

<div>
  <GnnPretrainDiagram />
</div>


---

# ChildWDL: Calibrating the Thinking Curve

<div class="flex flex-col items-center justify-start mt-4">
  <div class="w-4/5">
    <ChildWdlDiagram />
  </div>
</div>

---

# 6. Training Stage 2: Policy (The DP Oracle)

Identifying the **"Economy of Thought"** inflection point. The model learns to halt when the expected **net reward $R$** begins to drop, rather than just maximizing raw value $V$.

<div class="mt-8">
  <PolicyPretrainDiagram />
</div>

---

# Intuition: The DP Oracle

<div class="h-full flex flex-col items-center justify-start mt-4">
  <div class="w-full max-w-4xl">
    <DpOracleDiagram />
  </div>
</div>

<div class="grid grid-cols-3 gap-6 mt-8 text-sm px-8">
  <div>
    <b class="text-accent block mb-2">Why DP?</b>
    Optimal search is recursive. To know if a move is right, we must reason <b>backwards</b> from terminal leaf outcomes.
  </div>
  <div>
    <b class="text-accent block mb-2">Why does it work?</b>
    We solve for $V^*(s) = \max(V, \mathbb{E}[V^*_{child}] - C)$. This defines the <b>mathematical optimum</b> for halting.
  </div>
  <div>
    <b class="text-accent block mb-2">How do we train?</b>
    We use the DP result as <b>Ground Truth</b>. The GNN's $h_i$ is mapped to this "perfect" binary decision.
  </div>
</div>
