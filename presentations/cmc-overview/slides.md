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

<div class="grid grid-cols-3 gap-8 mt-12">
  <div class="border-l-4 border-accent pl-4">
    <h2 class="text-accent text-xl mb-0">Part 1: Motivation</h2>
    <div class="text-accent text-[10px] font-bold mb-4 uppercase tracking-tighter italic opacity-80">(feedback: does this make sense?)</div>
    <ul class="text-xs space-y-2 opacity-90">
      <li>Meta-control is hard but crucial.</li>
      <li>Fixed budgets waste tokens, money, and <b>time</b>.</li>
      <li>Humans are the existence proof for "thinking about thinking."</li>
    </ul>
  </div>

  <div class="border-l-4 border-accent pl-4">
    <h2 class="text-accent text-xl mb-4">Part 2: Methods</h2>
    <ul class="text-xs space-y-2 opacity-90">
      <li>Architecture: GNN + Halt Controller.</li>
      <li>Training via DP Oracle.</li>
      <li>Performance benchmarks vs. Baselines.</li>
    </ul>
  </div>

  <div class="border-l-4 border-accent pl-4 relative">
    <h2 class="text-accent text-xl mb-0">Part 3: Validation</h2>
    <div class="text-accent text-[10px] font-bold mb-4 uppercase tracking-tighter italic opacity-80">(feedback: what else should we show?)</div>
    <ul class="text-xs space-y-2 opacity-90">
      <li>Is it working as intended?</li>
      <li>Human Alignment: Does the "Thinking Curve" match human data?</li>
    </ul>
    <div class="absolute -top-3 -right-4 bg-accent text-white text-[8px] px-2 py-1 rotate-12 font-bold rounded">FEEDBACK WANTED</div>
  </div>
</div>

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
- **The Trade-off:** Is the move-quality I’m about to discover worth the computational "electricity" (time/tokens) I’m about to spend?
- **Real-World Constraints:** Tokens cost money, but more importantly, **real-time decisions have a physical cost**. Most LLM tasks are sufficiently time-intensive that every token of "thought" must justify itself.
- **Human Intuition:** Skilled players know *when* to stop thinking—a stopping problem we can formalize.

</v-clicks>

---

# Part 2: High-Level Architecture

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <div class="scale-110 transform origin-center">
    <LeelaSearchLoop />
  </div>
  
  <div class="mt-28 p-3 bg-neutral-soft border-l-2 border-accent italic text-[11px] opacity-80">
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

# 1. Move times are heavy-tailed

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/fe_clocktime_movetime/move_time_distribution.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">Key Takeaway</b><br><br>
    Most moves are near-instant, but the "long tail" of deep thinks dominates variance. 
    <br><br>
    Humans demonstrate that <b>non-trivial meta-control</b> is the default in skilled behavior.
  </div>
</div>

---

# 2. Stage of Game: The "Mid-game Bulge"

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/ply_impact_comparison.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">The Ply Paradox</b><br><br>
    Move times peak around move 40 and decay as the board simplifies. 
    <br><br>
    <b>Alignment Goal:</b> A rational mover should play *less optimally* early (where errors can be corrected) and focus resources on the critical mid-game transitions.
  </div>
</div>

---

# 3a. Elasticity: The Naive Aggregate

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/fe_clocktime_movetime/quad_0_baseline.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">Clock Awareness</b><br><br>
    The raw data shows a shallow positive trend ($\beta \approx 0.09$). 
    <br><br>
    People save/budget time for the endgame "just in case," even when current clock reserves are high.
  </div>
</div>

---

# 3b. Elasticity: The Resolution

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/fe_clocktime_movetime/quad_3_2way_fe.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">Budgeting Logic</b><br><br>
    Accounting for player identity reveals the true budget logic:
    <br><br>
    <div class="text-success font-bold text-2xl">
      $\beta \approx 0.54$
    </div>
    Humans consider both their own and their <b>opponent's</b> clock time to calibrate the "worth" of further thought.
  </div>
</div>

---

# 4. Value of Computation (The Demand)

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/voc_movetime/standard_voc_single_sample.png" />
  
  <div class="takeaway bg-neutral-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">Key Takeaway</b><br><br>
    <b>VOC</b> measures the potential gain from deep engine search over a shallow read.
    <br><br>
    $\log(T) \propto \sqrt{\text{VOC}}$: Humans spend the most "thought-capital" where depth matters most. 
    <br><br>
    <b>This is the target alignment for our Meta-Controller.</b>
  </div>
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
