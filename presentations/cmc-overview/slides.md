---
theme: default
title: Resource Rational Learned Meta-Control of Tree Search
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
    Resource Rational <br>
    Meta-Control for Tree Search
  </h1>
  
  <div class="mt-2 text-lg opacity-80">
    Yotam Sagiv & Jordan Lei
  </div>

  <div class="mt-6 text-[10px] font-bold uppercase tracking-widest opacity-40">
    Mechanistic Interpretability · Resource Rationality · Chess Search
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
- **Human Intuition:** Skilled players know *when* to stop thinking—a stopping problem we can formalize.
- **The Trade-off:** Is the move-quality I’m about to discover worth the computational "electricity" (time/tokens) I’m about to spend?

</v-clicks>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl">Part 2 — The Methods</h1>
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

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section III</div>
    <h1 class="text-4xl">Part 3 — Empirical Results</h1>
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
    Log-transforming to $\log(T)$ is required to stabilize variance and isolate the behavioral signal.
  </div>
</div>

---

# 2a. Elasticity: The Naive Aggregate

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/fe_clocktime_movetime/quad_0_baseline.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">Attempt 1: Opening Theory</b><br><br>
    The raw data shows a shallow positive trend ($\beta \approx 0.09$). 
    <br><br>
    However, this is corrupted by opening moves where players have maximum clocks but move instantly due to preparation.
  </div>
</div>

---

# 2b. Elasticity: The Ply Paradox

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/fe_clocktime_movetime/quad_1_ply_fe.png" />
  
  <div class="takeaway border-danger bg-danger-soft text-sm py-4">
    <b class="text-danger uppercase tracking-wider text-xs">Attempt 2: Selection Bias</b><br><br>
    Controlling for ply reveals a paradox: while aggregate bins look positive, <b>within-ply slopes are negative</b>. 
    <br><br>
    Faster players dominate the high-clock buckets, masking the true relation.
  </div>
</div>

---

# 2c. Elasticity: The Resolution

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/fe_clocktime_movetime/quad_3_2way_fe.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">The Thinking Hypothesis</b><br><br>
    Accounting for both player identity and game stage resolves the paradox.
    <br><br>
    <div class="text-success font-bold text-2xl">
      $\beta \approx 0.54$
    </div>
    A 10% increase in clock time leads to a ~5.4% increase in the thinking budget.
  </div>
</div>

---

# 2d. Elasticity: The Ply Impact

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/ply_impact_comparison.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">Stage of Game</b><br><br>
    Move times exhibit a characteristic "mid-game bulge". 
    <br><br>
    Thinking time peaks around move 40 and then decays as the board simplifies into the endgame, independent of clock budget.
  </div>
</div>

---

# 3. Value of Computation (The Demand)

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/voc_movetime/standard_voc_single_sample.png" />
  
  <div class="takeaway bg-neutral-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">Key Takeaway</b><br><br>
    <b>VOC</b> measures the potential gain from deep engine search over a shallow read.
    <br><br>
    $\log(T) \propto \sqrt{\text{VOC}}$: Humans spend the most "thought-capital" on positions where depth matters most.
  </div>
</div>
