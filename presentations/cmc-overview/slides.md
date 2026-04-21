---
theme: default
title: Chess Meta-Control (CMC)
info: |
  A didactic breakdown of the CMC architecture: Control Flow · Representation · GNN Mechanics · Decision · Training.
css: ./style.css
class: text-left
mdc: true
font:
  sans: 'Inter'
  mono: 'Fira Code'
---

# Chess Meta-Control (CMC)

**Optimizing the Economy of Thought in Complex Search**

<div class="mt-40 text-sm opacity-60">
`lmcos/demos/understanding.md` · `project.md`
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
$$ R(k) = \mathbb{E} \left[ \text{Value}(T_k) \right] - C \cdot k $$

**Mechanistic Readout**
- **Input:** The current root hidden state $\mathbf{h}_{root}$.
- **Output:** A halting probability $P(\text{halt} | \mathbf{h}_{root})$ predicted by an MLP.
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

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/move_time_distribution.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-base py-6">
    <b>Key Takeaway:</b><br><br>
    Most moves are near-instant, but the "long tail" of deep thinks dominates variance. 
    <br><br>
    Log-transforming to <b>$\ln(T)$</b> is required to stabilize variance and isolate the behavioral signal.
  </div>
</div>

---

# 2a. Elasticity: The Naive Aggregate

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/attempt1_naive_trend.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-base py-6">
    <b>Attempt 1 (Primary Confound): Opening Theory</b><br><br>
    The raw data shows a shallow positive trend ($\beta \approx 0.09$). 
    <br><br>
    However, this is corrupted by opening moves where players have maximum clocks but move instantly due to preparation.
  </div>
</div>

---

# 2b. Elasticity: The Ply Paradox

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/attempt2_ply_wise_trend.png" />
  
  <div class="takeaway border-danger bg-danger-soft text-base py-6">
    <b>Attempt 2 (Secondary Confound): Selection Bias</b><br><br>
    Controlling for ply reveals a paradox: while aggregate bins look positive, <b>within-ply slopes are negative</b>. 
    <br><br>
    Faster players (who maintain higher clocks) dominate the high-clock buckets, masking the true relation.
  </div>
</div>

---

# 2c. Elasticity: The Resolution

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/attempt3_controlled_trend.png" />
  
  <div class="takeaway border-success bg-success-soft text-base py-6">
    <b>Attempt 3 (The Thinking Hypothesis): Fixed-Effect Control</b><br><br>
    Accounting for <b>both</b> player identity and game stage resolves the paradox.
    <br><br>
    <span class="text-success font-bold text-xl">$\beta \approx 0.54$</span><br>
    A 10% increase in clock time leads to a ~5.4% increase in the thinking budget.
  </div>
</div>

---

# 3. Value of Computation (The Demand)

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/voc_analysis/voc_quad_view.png" />
  
  <div class="takeaway text-base py-6">
    <b>Key Takeaway:</b><br><br>
    <b>VOC</b> measures the potential gain from deep engine search over a shallow read.
    <br><br>
    <b>$\ln(T) \propto \sqrt{VOC}$</b>: Humans spend the most "thought-capital" on positions where depth matters most.
  </div>
</div>
