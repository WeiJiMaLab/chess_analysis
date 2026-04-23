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

<div class="grid grid-cols-1 gap-4 mt-4 text-sm leading-relaxed max-w-5xl">
  <p>
    All timing plots use the same DuckDB sample: <b>move-level rows in <code>selected_moves</code></b> inside the project personal database, joined to analysis tables built by the Python scripts.
  </p>
  <ul class="list-disc pl-6 space-y-2">
    <li>
      <b>Game list:</b> We restrict to game IDs listed in the <code>selected_games</code> table (in that same database). The exact inclusion rule (Elo, time control, etc.) is whatever that table encodes.
    </li>
    <li>
      <b>Extracting moves (<code>src/utils/load_data.py</code>):</b> A Slurm array walks Lichess parquet shards under the archive root, keeps only those <code>gid</code>s, and by default <b>drops an entire game</b> if any row has a negative <code>move_time</code> (bad timestamps). Staging <code>.parquet</code> files are merged into <code>selected_moves</code>.
    </li>
    <li>
      <b>Launch script (<code>src/slurm/script_load_moves.sh</code>):</b> Reserves a clean temp directory, submits the load-shards array job, and submits a <b>merge</b> job that runs only after the array succeeds, repopulating <code>selected_moves</code> from the combined parquet.
    </li>
    <li>
      <b>Downstream analysis:</b> Preprocessing is now centralized in <code>load_data.py</code>, which precalculates <code>ln_move_time</code>, clock features, <b><code>n_possible_moves</code></b>, and quantile bins. Analysis scripts assume these tables (<code>_selected_moves</code> or <code>_selected_moves_nonzero_T</code>) already exist.
    </li>
  </ul>
</div>

---

# 1a. Move times: raw and log (default)

<div class="text-xs opacity-60 mb-2 -mt-2">Excludes premoves / <code>move_time = 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/move_time_summary/combined.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">move_time_summary</b><br><br>
    Histograms of T and ln T (SQL equal-width bins). Most mass is very fast; the tail is why we often use log time downstream.
  </div>
</div>

---

# 1b. Move times: raw and log (with premoves)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--include_zeroT</code> — including zero-time moves.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/move_time_summary/combined_include_zeroT.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">vs 1a</b><br><br>
    Strips the origin spike; “deliberation-only” when paired with the same filter in other plots.
  </div>
</div>

---

# 2a. Game stage: ply vs. mean log think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2">Excludes <code>move_time = 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/combined.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">ply_movetime</b><br><br>
    Left: mean ln T by integer ply. Right: by ply decile. Game stage, not clock.
  </div>
</div>

---

# 2b. Game stage: ply vs. mean log think time (with premoves)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--include_zeroT</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/combined_include_zeroT.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">vs 2a</b><br><br>
    How much the fast end is real zeros vs short thinks.
  </div>
</div>

---

# 3a. Remaining clock vs. think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2">Player clock; excludes <code>move_time = 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">clock_movetime</b><br><br>
    2×2: binned log clock vs ln T, scatter + OLS, per-ply β of ln T on your log remaining clock.
  </div>
</div>

---

# 3b. Remaining clock vs. think time (with premoves)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--include_zeroT</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_include_zeroT.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">vs 3a</b><br><br>
    Slopes and per-ply β are sensitive to dropping instants.
  </div>
</div>

---

# 4a. Opponent clock vs. your think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--opp</code>: x = their remaining clock; excludes 0s.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_opp.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">clock_movetime --opp</b><br><br>
    Same 2×2 as §3, but the predictor is opponent clock. Contrast to §3a.
  </div>
</div>

---

# 4b. Opponent clock vs. your think time (with premoves)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--include_zeroT --opp</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_include_zeroT_opp.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">vs 4a</b><br><br>
    Opponent-clock analogue of §3b: instants out.
  </div>
</div>

---

# 5a. Branching: legal moves vs. raw think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2">x = <code>n_possible_moves</code>; y = raw T (s), not log. Excludes 0s.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/npossiblemoves_movetime/combined.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">npossiblemoves_movetime</b><br><br>
    2×2: branching vs raw T, OLS, per-ply β. Complements the log-time / clock slides.
  </div>
</div>

---

# 5b. Branching: legal moves vs. raw think time (with premoves)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--include_zeroT</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/npossiblemoves_movetime/combined_include_zeroT.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">vs 5a</b><br><br>
    Instants out; slopes / β panel can move.
  </div>
</div>

---

# 6. Value of Computation (The Demand)

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/voc_movetime/standard_voc_single_sample.png" />
  
  <div class="takeaway bg-neutral-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">VOC</b><br><br>
    Shallow vs deep value gap. $\log T$ tracks $\sqrt{\text{VOC}}$: more think time where search depth actually pays. Controller alignment target.
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
