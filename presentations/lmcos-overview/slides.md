---
theme: default
title: Learned Meta-control of Tree Search
info: Project status — human behavioral analysis and normative DP agent.
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
    The Great Reunification — sprint plan · June 2026
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">The plan</div>
    <h1 class="text-5xl">The Great Reunification</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      Put the human dataset and the normative agent on <b>the same positions</b>,
      measure the normative ceiling on human RT, and open the model-comparison thread.
    </div>
  </div>
</div>

---

# The big idea

<div class="mt-6 max-w-3xl space-y-4 text-sm">
  <div class="p-4 bg-neutral-soft border-2 border-accent rounded-lg text-center">
    <div class="text-lg font-semibold">How should a meta-controller know <i>when to stop thinking</i>?</div>
    <div class="mt-2 text-sm opacity-80"><b>Input = search tree &nbsp;→&nbsp; Output = continue / halt</b></div>
  </div>

  <div class="p-3 bg-accent-soft border-l-2 border-accent rounded text-xs">
    <b>Background (Russek et al.):</b> the <i>value of computation</i>
    $\;\text{gain} = V_\text{deep}(a_\text{deep}) - V_\text{deep}(a_\text{shallow})\;$
    tracks human reaction time. We ask whether a <b>normatively optimal</b> stopping rule —
    and a <b>learned controller</b> that approximates it — also tracks human RT on the
    <b>same positions</b>.
  </div>
</div>

---

# The model — greatest possible world

<div class="mt-6 max-w-3xl space-y-4 text-sm">
  <p class="font-semibold">In the best of all possible worlds, the agent looks like this:</p>
  <div class="grid grid-cols-2 gap-3 text-xs">
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary">Architecture</b><br>GNN + readout (MC) head over the search tree.
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary">Objective</b><br>Self-play to <i>win</i>, minus a cost that grows with tree size.
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary">Inner loop</b><br>Lc0 MCTS+PUCT rollouts — the planning loop, abstracted.
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary">Learning</b><br>RL discovers the <b>optimal stopping step (OSS)</b>.
    </div>
  </div>
  <p class="text-xs opacity-60">...but five constraints stand in the way. Each has a solution.</p>
</div>

---

# The model — constraints &rarr; solutions

<div class="mt-3 max-w-4xl text-xs">

| Constraint | Solution |
| :--- | :--- |
| GNN root encoding is a **sparse bottleneck** — `z_root` may learn nothing | **Auxiliary head.** `ChildWDLEncoder(z_root, child_pos) → child WDL`. Forces `z_root` to be informative. Data: edge `(parent, child)` → search-encoded child WDL. |
| **Reward is a sparse signal** for meta-control | **Dense target.** Train MC: `(z_root, β) → advantage A`. Per-step, dense. |
| Most **leaves are unexpanded** → child-WDL target trivial | **Weight the loss** by subtree size; trivial children contribute ≈ 0. |
| Advantage **matters most near the boundary** | **BCE penalty on the sign** of $A$, weight $\lambda$. |
| Many **FENs need no thought** | **Filter them out** (PUCT stability filter). |

</div>

<div class="mt-3 text-xs opacity-60">The first two are the load-bearing moves: a representation that means something, and a dense target to learn from.</div>

---

# Advantage &mdash; a dense, computable target

<div class="mt-5 max-w-3xl space-y-4 text-sm">
  <div class="p-4 bg-neutral-soft border-2 border-accent rounded-lg text-center">
    $V_t = \max\big(V_\text{halt},\; V_\text{continue}\big)
       = \max\big(V_\text{deep}(a_t),\; V_{t+1} - \text{cost}(t{+}1,\beta)\big)$
    <div class="mt-2 text-xs opacity-70">Optimal stopping is recursive &rarr; solve backwards by <b>dynamic programming</b>.</div>
  </div>

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs">
    <b>Co-located data.</b> Snapshots $T_0 \dots T_\text{max}$ fall out for free while MCTS+PUCT
    grows the tree. <b>One artifact</b> trains both heads: the GNN uses the <i>nodes</i> of the
    full tree $T_\text{max}$; the MC uses the <i>snapshots</i> $T_0\dots T_\text{max}$ to encode
    advantages.
  </div>
</div>

---

# The model pipeline today

<div class="mt-4 grid grid-cols-2 gap-4 max-w-4xl text-xs">
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
    <b class="text-secondary uppercase tracking-wider">Data</b>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
      <li>~40K filtered FENs (39,668 trees)</li>
      <li>$T_0\dots T_\text{max}$ snapshots → GNN + MC targets</li>
    </ul>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
    <b class="text-secondary uppercase tracking-wider">Engine / Model</b>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
      <li>Lc0 inner loop</li>
      <li>GNN: GRU cell, up+down sweeps, child-attention</li>
      <li>MC: MLP readout of `z_root` → advantage</li>
    </ul>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
    <b class="text-secondary uppercase tracking-wider">Metrics</b>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
      <li>child-WDL CE curves</li>
      <li>MC advantage MSE + λ·BCE curves</li>
      <li>greedy stop (first $A<0$) vs DP-OSS</li>
    </ul>
  </div>
  <div class="p-3 bg-green-50 border-l-2 border-green-500 rounded">
    <b class="text-green-700 uppercase tracking-wider">Status</b>
    <p class="mt-1 opacity-80">Learns well across a wide variety of positions; competitive with what we hoped to see.</p>
  </div>
</div>

---

# The data — greatest possible world

<div class="mt-6 max-w-3xl space-y-4 text-sm">
  <div class="p-4 bg-neutral-soft border-2 border-accent rounded-lg text-center">
    <div class="text-lg font-semibold">$\text{OSS}(s) \;\leftrightarrow\; \log \text{RT}(s)$</div>
    <div class="mt-2 text-xs opacity-70">
      Take the GNN trained <b>only on self-play</b>; show its optimal stopping step is strongly
      correlated with human thinking time.
    </div>
  </div>
  <div class="p-3 bg-accent-soft border-l-2 border-accent rounded text-xs">
    If humans, the model, and the normative target all align → evidence for
    <b>resource rationality</b>, and a compelling application of the model.
  </div>
  <p class="text-xs opacity-60">...but again, constraints — and a deliberately model-free first step.</p>
</div>

---

# The data — constraints &rarr; solutions

<div class="mt-4 max-w-4xl text-xs">

| Constraint | Solution |
| :--- | :--- |
| Time-control & **skill confounds** | One control: **60+0 (10-min)**, no berserk, fixed Elo tranche (≥2000). |
| Don't want results to **depend on the model** | **Zero-parameter analyses first** — show humans behave sensibly with no model at all. |

</div>

<div class="mt-4 p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs max-w-4xl">
  <b>Zero-parameter findings:</b> Ply · Branching · Own pieces · Gain (Russek) · (neg) action gap
  all track RT — with <b>Branching</b> playing a large role that Russek et al. do <b>not</b>
  explain and that is largely <b>orthogonal</b> to gain.
</div>

---

# The human pipeline today

<div class="mt-4 grid grid-cols-2 gap-4 max-w-4xl text-xs">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
    <b class="text-accent uppercase tracking-wider">Data / Engine</b>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
      <li>~2M games → 60+0, ≥2000 Elo</li>
      <li>Stockfish (for convenience)</li>
      <li>0-parameter</li>
    </ul>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
    <b class="text-accent uppercase tracking-wider">Metrics</b>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
      <li>Ply / Branching / Own pieces</li>
      <li>Gain / Action gap</li>
    </ul>
  </div>
  <div class="col-span-2 p-3 bg-green-50 border-l-2 border-green-500 rounded">
    <b class="text-green-700 uppercase tracking-wider">Status</b>
    <span class="opacity-80"> Interesting trends on what people actually do — with surprising results we'd like validated.</span>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Where it stands</div>
    <h1 class="text-4xl">Three problems block the reunification</h1>
  </div>
</div>

---

# The three problems

<div class="mt-6 max-w-3xl space-y-3 text-sm">
  <div class="p-4 bg-red-50 border-l-4 border-red-500 rounded">
    <b class="text-red-700 uppercase tracking-wider text-xs">Urgent — data mismatch</b>
    <p class="mt-1 text-xs opacity-90">Human-filtered FENs ≠ model-training FENs. Run the data-gen pipeline on the <b>human</b> FENs, then compare the <b>target OSS (no model)</b> with human RT — the <b>upper bound</b> on what any normative model can explain. <span class="opacity-60">→ U1</span></p>
  </div>
  <div class="p-4 bg-amber-50 border-l-4 border-amber-400 rounded">
    <b class="text-amber-700 uppercase tracking-wider text-xs">Engine mismatch — gain</b>
    <p class="mt-1 text-xs opacity-90">Human gain uses Stockfish; Lc0 is the study standard. Replicate Russek's gain↔RT with <b>Lc0</b>. <span class="opacity-60">→ U2</span></p>
  </div>
  <div class="p-4 bg-neutral-soft border-l-4 border-secondary rounded">
    <b class="text-secondary uppercase tracking-wider text-xs">Important — model comparison</b>
    <p class="mt-1 text-xs opacity-90">One model, no baselines. Profile lesions & alternatives on the <b>existing shards</b> — runs concurrently. <span class="opacity-60">→ U3</span></p>
  </div>
</div>

---

# U1 — Reunify on the human FENs

<div class="mt-4 max-w-3xl space-y-2 text-sm">
  <p class="font-semibold">Migrate the model's data-gen pipeline onto the filtered human FENs, then read off the normative ceiling.</p>
  <div class="space-y-2 text-xs">
    <div class="flex gap-3 items-start"><div class="text-accent font-bold w-8 shrink-0">U1.0</div><div><b>Timing smoke</b> — 20 FENs, budget 96. <span class="text-green-700 font-semibold">Done.</span> Sets the cost everything depends on (next slide).</div></div>
    <div class="flex gap-3 items-start"><div class="text-accent font-bold w-8 shrink-0">U1.1</div><div><b>Generate trees</b> on 10K/50K/100K human FENs via `cts.data.build_tree` (Lc0, budget 96), snapshots + DP targets.</div></div>
    <div class="flex gap-3 items-start"><div class="text-accent font-bold w-8 shrink-0">U1.2</div><div><b>Upper bound</b> — `compute_budgeted_oracle()` → target OSS; correlate with `log(RT)` at matched positions. <b>The headline result.</b></div></div>
    <div class="flex gap-3 items-start"><div class="text-accent font-bold w-8 shrink-0">U1.3</div><div><b>Refit the model</b> (GNN + MC) on the unified data; greedy-stop vs DP-OSS; eventually predicted-stop ↔ RT.</div></div>
  </div>
  <div class="p-2 bg-amber-50 border-l-2 border-amber-400 rounded text-xs">Critical path: U1.0 → U1.1 → U1.2. U1.3 runs in parallel with U1.2 off the same trees.</div>
</div>

---

# U1 — Feasibility (the smoke already told us)

<div class="mt-3 max-w-4xl space-y-3 text-xs">
  <div class="grid grid-cols-2 gap-3">
    <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
      <b class="text-accent">Per-tree cost (A100)</b>
      <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
        <li>GPU: <b>~16–41 s/tree</b> (0.024–0.062 roots/s) — pin to one number with a clean steady-state smoke</li>
        <li>CPU/blas: <b>~14 min/tree</b> (~20–50× slower; lc0 needs `libcublas` even for blas)</li>
      </ul>
    </div>
    <div class="p-3 bg-red-50 border-l-2 border-red-400 rounded">
      <b class="text-red-700">Why `human_trees_10k/` is empty</b>
      <p class="mt-1 opacity-80">500 FENs × 1 h walls, but ≥16 s/tree needs <b>2–6 h/shard</b>. Shard-sizing bug, <b>not</b> a cost wall. Fix: <b>~20 fat shards on gpu-medium</b>, walls sized to FENs×s/tree.</p>
    </div>
  </div>

  <table>
  <thead><tr><th>Tier</th><th>Trees</th><th>GPU-h @25s</th><th>GPU-h @41s</th><th>Wall · 20 GPUs</th></tr></thead>
  <tbody>
  <tr><td>acceptable</td><td>10K</td><td>69</td><td>114</td><td>3.5 / 5.7 h</td></tr>
  <tr><td><b>good</b></td><td><b>50K</b></td><td>347</td><td>568</td><td><b>17 / 28 h</b></td></tr>
  <tr><td>ideal</td><td>100K</td><td>694</td><td>1,137</td><td>35 / 57 h</td></tr>
  </tbody>
  </table>

  <div class="p-2 bg-green-50 border-l-2 border-green-500 rounded">
    <b class="text-green-700">Verdict: feasible.</b> 50K < 1 day even pessimistically; 100K ≈ 1.5–2.4 days.
    Contingencies: CPU overflow (≈1,500 trees/h under `short`), depth 96→64 (~⅓ faster), Stockfish swap (needs sign-off).
  </div>
</div>

---

# U1.3 — Refit fast: profile, don't scale

<div class="mt-5 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Goal: an end-to-end loop that converges <i>good enough</i>, <i>quickly</i> — speed first.</p>
  <div class="grid grid-cols-2 gap-3 text-xs">
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary">Fitting smoke</b>
      <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
        <li>Tiny GNN (Config D): `d_embed=16`, 1 head, 1 layer</li>
        <li>Representation 16–32, not 128</li>
        <li>10–20 shards; plot loss curves live</li>
      </ul>
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary">Targets</b>
      <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
        <li>GNN: converge in ≤ ~2 h GPU</li>
        <li>MC: ≤ ~1 h, ≤ ~1000 steps (~1 epoch)</li>
        <li>Beat A0b anchor (54.6%) → toward 90.1%</li>
      </ul>
    </div>
  </div>
  <div class="p-2 bg-amber-50 border-l-2 border-amber-400 rounded text-xs">If tiny GNN underfits, widen D→C→B. Do not scale to production until the loop is fast.</div>
</div>

---

# U2 & U3 — the parallel threads

<div class="mt-5 grid grid-cols-2 gap-4 max-w-4xl text-xs">
  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded">
    <b class="text-amber-700 uppercase tracking-wider">U2 — Lc0 gain in human data</b>
    <ul class="mt-2 list-disc pl-4 space-y-1 opacity-90">
      <li>Replace Stockfish gain with <b>Lc0</b> gain on the same FENs</li>
      <li>Reuse the engine-eval harness → `lc0_evaluations`</li>
      <li><b>Test:</b> Lc0 reproduces the sign of SF gain↔RT; Lc0~SF rank-agree</li>
      <li><b>Fully independent</b> of U1/U3</li>
    </ul>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
    <b class="text-secondary uppercase tracking-wider">U3 — Baseline suite</b>
    <p class="mt-1 opacity-80">On the <b>existing</b> shards, vs OSS / Pr(halt) / regret:</p>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-90">
      <li>MLP over tree stats · gain-depth-only</li>
      <li>always-stop · never-stop</li>
      <li>geometric · tree-size-sensitive</li>
    </ul>
    <p class="mt-1 opacity-60">Fully independent — different data, different output dir.</p>
  </div>
</div>

---

# Sprint map

<div class="mt-4 max-w-3xl text-xs">

```
   U1.0 smoke ─► U1.1 generate ─► U1.2 OSS ↔ RT  (CRITICAL: the upper bound)
    (DONE)        human FENs   └► U1.3 refit (smoke → fit)

   U2 Lc0 gain   ──── independent, parallel ───►  (CPU/GPU)
   U3 baselines  ──── independent, parallel ───►  (existing shards)
```

</div>

<div class="mt-4 grid grid-cols-2 gap-4 max-w-3xl text-xs">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
    <b class="text-accent">Decisions needed</b>
    <ul class="mt-1 list-disc pl-4 space-y-1 opacity-80">
      <li>Tier: <b>50K recommended</b> (10K / 50K / 100K)</li>
      <li>Depth 96 vs 64 for the first run</li>
      <li>Schedule the Stockfish-swap profiling now, or on first wall?</li>
      <li>Pin the per-tree number (16 vs 41 s) with one clean smoke</li>
    </ul>
  </div>
  <div class="p-3 bg-green-50 border-l-2 border-green-500 rounded">
    <b class="text-green-700">Bottom line</b>
    <p class="mt-1 opacity-80">The reunification is feasible within the brief's "couple of days, max GPU QoS." The only true blocker was shard sizing — now understood. Full plan: <code>unify.md</code>.</p>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Appendix</div>
    <h1 class="text-4xl">Prior framing, results &amp; architecture</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      The earlier two-project framing, the human-behavior section, Analysis 0–4, and the
      architecture diagrams that motivate the plan above.
    </div>
  </div>
</div>

---

# Two projects, one open question

<div class="mt-8 flex items-center gap-0 max-w-4xl">
  <div class="flex flex-col items-center flex-1">
    <div class="p-4 bg-neutral-soft border-2 border-accent rounded-lg w-full text-center">
      <div class="text-accent text-xs font-bold uppercase tracking-wider mb-2">Human analysis</div>
      <div class="text-sm font-semibold">What do strong players <i>actually</i> do?</div>
      <div class="text-xs opacity-60 mt-1">1M positions · 10+0 · ≥2000 Elo</div>
    </div>
  </div>
  <div class="flex flex-col items-center px-5">
    <div class="text-xl opacity-40">→</div>
    <div class="mt-2 p-3 bg-amber-50 border-2 border-amber-400 rounded-lg text-center text-xs">
      <b class="text-amber-800 block">Do they agree?</b>
    </div>
    <div class="text-xl opacity-40">←</div>
  </div>
  <div class="flex flex-col items-center flex-1">
    <div class="p-4 bg-neutral-soft border-2 border-secondary rounded-lg w-full text-center">
      <div class="text-secondary text-xs font-bold uppercase tracking-wider mb-2">lmcos agent</div>
      <div class="text-sm font-semibold">What <i>should</i> an agent do?</div>
      <div class="text-xs opacity-60 mt-1">DP oracle · GNN + MC controller</div>
    </div>
  </div>
</div>

<div class="mt-8 grid grid-cols-2 gap-6 max-w-4xl text-sm">
  <div class="p-3 bg-green-50 border-l-3 border-green-500 rounded">
    <b class="text-green-700">If yes →</b> one paper: humans approximate optimal compute allocation; the agent learns to do the same
  </div>
  <div class="p-3 bg-neutral-soft border-l-3 border-gray-400 rounded">
    <b>If no →</b> two papers: descriptive account of human deliberation + normative model of efficient search
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

# Think time is log-normal — Weber's Law

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">Finding</span> Think time is approximately log-normal, not exponential or uniform</div>
    <div class="finding"><span class="label">Implication</span> Players scale thinking time multiplicatively with difficulty — consistent with Weber's Law. Justifies working in log(RT) throughout.</div>
    <div class="text-xs opacity-60 mt-4">n = 135M non-zero-time moves from 1.97M games</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/log_movetime_histogram.png" />
</div>

---

# What predicts when humans think longer?

<div class="mt-4 text-sm max-w-3xl">
  <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-3">Four board features — all computed from position, none from RT</div>
  <table class="text-xs w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold">Feature</th>
        <th class="text-left py-2 pr-4 font-bold">r with log(RT)</th>
        <th class="text-left py-2 font-bold">Intuition</th>
      </tr>
    </thead>
    <tbody class="text-[12px]">
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 font-mono">branching factor</td>
        <td class="text-blue-600 font-bold">+0.20</td>
        <td class="opacity-80">More candidates → more uncertainty to spread across</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 font-mono">gain_depth (ΔUC@5)</td>
        <td class="text-blue-500 font-bold">+0.10</td>
        <td class="opacity-80">Deeper search demonstrably finds a better move here</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 font-mono">own material</td>
        <td class="text-blue-400 font-bold">+0.04</td>
        <td class="opacity-80">More pieces → more interactions to resolve</td>
      </tr>
      <tr>
        <td class="py-2 pr-4 font-mono">toptwo</td>
        <td class="text-red-400 font-bold">−0.06</td>
        <td class="opacity-80">One move clearly better → less need to distinguish</td>
      </tr>
    </tbody>
  </table>
  <div class="mt-3 text-xs opacity-60">Replicates Russek et al. (2022): r(log RT, ΔUC) = +0.096 at depth=5 vs their depth=15. All effects hold within ply tertiles.</div>
</div>

---

# Correlation matrix — the full picture

<div class="grid grid-cols-[60fr_40fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <img class="w-full object-contain max-h-[72vh]" src="/figures/correlation_matrix.png" />
  <div>
    <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-3">n = 1,000,000 · Pearson r</div>
    <table class="text-xs w-full border-collapse">
      <thead>
        <tr class="border-b border-gray-200">
          <th class="text-left py-1 pr-3 font-bold">Pair</th>
          <th class="text-right py-1 font-bold">r</th>
        </tr>
      </thead>
      <tbody>
        <tr><td class="py-1 pr-3 opacity-80">Ply ↔ own material</td><td class="text-right font-mono text-red-500 font-semibold">−0.81</td></tr>
        <tr><td class="py-1 pr-3 opacity-80">gain_depth ↔ MQ</td><td class="text-right font-mono text-red-500 font-semibold">−0.42</td></tr>
        <tr><td class="py-1 pr-3 opacity-80">toptwo ↔ MQ</td><td class="text-right font-mono text-red-400 font-semibold">−0.30</td></tr>
        <tr class="border-t border-gray-100"><td class="py-1 pr-3 opacity-80">branching ↔ log(RT)</td><td class="text-right font-mono text-blue-600 font-semibold">+0.20</td></tr>
        <tr><td class="py-1 pr-3 opacity-80">gain_depth ↔ log(RT)</td><td class="text-right font-mono text-blue-400 font-semibold">+0.10</td></tr>
        <tr><td class="py-1 pr-3 opacity-80">MQ ↔ log(RT)</td><td class="text-right font-mono text-red-300 font-semibold">−0.12</td></tr>
        <tr><td class="py-1 pr-3 opacity-80">toptwo ↔ log(RT)</td><td class="text-right font-mono text-red-300 font-semibold">−0.06</td></tr>
      </tbody>
    </table>
  </div>
</div>

---

# Three counterintuitive findings

<div class="mt-6 max-w-3xl space-y-6">
  <div class="flex gap-4 items-start">
    <div class="text-3xl font-bold text-amber-400 shrink-0 w-8">1</div>
    <div>
      <div class="text-base font-semibold">More clock → worse moves</div>
      <div class="text-sm opacity-70 mt-1">r(MQ, clock) = −0.091, within every ply tertile. Players with more time have played quickly through low-VOC positions; depth-5 penalises their choices.</div>
    </div>
  </div>
  <div class="flex gap-4 items-start">
    <div class="text-3xl font-bold text-amber-400 shrink-0 w-8">2</div>
    <div>
      <div class="text-base font-semibold">67% of positions have VOC = 0 at depth=5</div>
      <div class="text-sm opacity-70 mt-1">Depth-1 already finds the correct move — yet humans deliberate. Suggests candidate-set uncertainty drives RT, not search-depth uncertainty.</div>
    </div>
  </div>
  <div class="flex gap-4 items-start">
    <div class="text-3xl font-bold text-amber-400 shrink-0 w-8">3</div>
    <div>
      <div class="text-base font-semibold">Branching beats VOC as an RT predictor</div>
      <div class="text-sm opacity-70 mt-1">r = +0.20 vs +0.10. Width of the decision problem drives deliberation more than realized value of deeper search.</div>
    </div>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl">The lmcos Agent</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      What should a rational agent do? And does it agree with humans?
    </div>
  </div>
</div>

---

# What the lmcos agent is trained to do

<div class="mt-6 max-w-3xl space-y-4 text-sm">
  <p>The lmcos agent faces a stopping problem: at each expansion step, halt and act on the current best move, or pay a compute cost and continue searching.</p>

  <div class="p-4 bg-neutral-soft rounded space-y-2">
    <p><b>The oracle target:</b> backward DP computes the optimal stopping step — the expansion count at which halting maximises expected value minus cost. This is <code>oracle_stop_step</code>. The GNN + MC controller learns to predict it.</p>
    <p class="text-xs opacity-70">Current accuracy: 80% exact stop-step, 90% sign. MC controller training: 52 seconds.</p>
  </div>
</div>

---

# Data sources — the comparison problem

<div class="mt-4 max-w-3xl text-sm">
  <table class="text-xs w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold w-1/4">Property</th>
        <th class="text-left py-2 pr-4 font-bold">lmcos training positions</th>
        <th class="text-left py-2 font-bold">Human behavioral data</th>
      </tr>
    </thead>
    <tbody class="text-[11px]">
      <tr class="border-b border-gray-100"><td class="py-1.5 pr-4 font-semibold">Source</td><td>100K FENs sampled from Lichess 2023</td><td>1.97M games from Lichess Oct–Dec 2023</td></tr>
      <tr class="border-b border-gray-100"><td class="py-1.5 pr-4 font-semibold">ELO range</td><td>1800–2600 (broad)</td><td>Both ≥ 2000 (strong players only)</td></tr>
      <tr class="border-b border-gray-100"><td class="py-1.5 pr-4 font-semibold">Time controls</td><td>All (bullet, blitz, rapid, classical)</td><td>10+0 only</td></tr>
      <tr class="border-b border-gray-100"><td class="py-1.5 pr-4 font-semibold">Ply range</td><td>8–120</td><td>15–75 (excl. opening/endgame)</td></tr>
      <tr class="border-b border-gray-100"><td class="py-1.5 pr-4 font-semibold">Engine</td><td>Lc0 t1-256x10 (~3000+ ELO)</td><td>Stockfish depth=5 (~2500+ ELO)</td></tr>
      <tr><td class="py-1.5 pr-4 font-semibold">Target variable</td><td>oracle_stop_step (normative)</td><td>log(RT) (observed behavior)</td></tr>
    </tbody>
  </table>

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded mt-4 text-xs">
    <b>The datasets are different positions.</b> A0a gives a directional signal using lmcos trees. A1 (next) will generate oracle trees on the <i>same</i> positions as the human data — removing this disanalogy.
  </div>
</div>

---

# Two stopping criteria — a key tension

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="grid grid-cols-2 gap-5">
    <div class="p-4 bg-neutral-soft border-l-3 border-accent rounded">
      <b class="text-accent text-xs uppercase tracking-wider">min_expansions</b>
      <p class="text-sm italic mt-1">"If I keep thinking, my action won't change — so my realized reward can't change."</p>
      <ul class="text-xs list-disc pl-4 space-y-1 opacity-80 mt-2">
        <li>Stops when <b>action identity</b> stabilises</li>
        <li>Matches Russek et al. (ΔUC = 0 when actions agree)</li>
        <li class="font-semibold text-accent">→ Theoretically correct for single decisions</li>
        <li class="text-red-600">r(gain_depth) = −0.557 — opposite of humans</li>
      </ul>
    </div>
    <div class="p-4 bg-neutral-soft border-l-3 border-secondary rounded">
      <b class="text-secondary text-xs uppercase tracking-wider">oracle_stop_step</b>
      <p class="text-sm italic mt-1">"My Q-estimate is still improving — that outweighs the cost."</p>
      <ul class="text-xs list-disc pl-4 space-y-1 opacity-80 mt-2">
        <li>Stops when <b>Q-refinement</b> no longer justifies cost</li>
        <li>What the lmcos model is trained on</li>
        <li class="font-semibold text-secondary">→ Matches human RT direction (3/4 features)</li>
        <li>r(gain_depth) = +0.797</li>
      </ul>
    </div>
  </div>

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs">
    <b>The paradox:</b> min_expansions is the theoretically correct stopping criterion — but both humans and the oracle follow oracle_stop_step in practice. Neither stops when the action has stabilised. Why?
  </div>
</div>

---

# Analysis 0a: oracle vs human RT on the same features

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-3 text-sm">
    <div><span class="label">What</span> Compute <code>oracle_stop_step</code> on 39,668 lmcos training trees. Extract board features from root FEN.</div>
    <div><span class="label">x</span> Board features: branching, material, gain_depth, toptwo</div>
    <div><span class="label">y</span> oracle_stop_step (DP-optimal expansions, budget=43)</div>
    <div><span class="label">Compare</span> r(feature, oracle_stop_step) vs r(feature, human log RT)</div>
    <div class="finding"><span class="label">Finding</span> All 4 of 4 features match direction. gain_depth r = +0.233 for oracle vs +0.096 for humans. toptwo r = -0.290 for oracle vs -0.064 for humans.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/oracle_stop_step_vs_human_rt.png" />
</div>

---

# oracle_stop_step ↔ VOC proxies vs human RT ↔ VOC proxies

<div class="mt-4 max-w-3xl text-sm">
  <table class="text-xs w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold">Feature</th>
        <th class="text-right py-2 pr-4 font-bold">oracle_stop_step r</th>
        <th class="text-right py-2 pr-4 font-bold">human log(RT) r</th>
        <th class="text-left py-2 font-bold"></th>
      </tr>
    </thead>
    <tbody class="text-[12px]">
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 font-mono">branching</td>
        <td class="text-right pr-4 text-blue-600 font-bold">+0.014</td>
        <td class="text-right pr-4 text-blue-600 font-bold">+0.195</td>
        <td class="text-green-700 font-semibold">✓ match (tiny)</td>
      </tr>
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 font-mono">material</td>
        <td class="text-right pr-4 text-blue-500 font-bold">+0.011</td>
        <td class="text-right pr-4 text-blue-400 font-bold">+0.039</td>
        <td class="text-green-700 font-semibold">✓ match (tiny)</td>
      </tr>
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 font-mono">gain_depth</td>
        <td class="text-right pr-4 font-bold text-blue-700">+0.233</td>
        <td class="text-right pr-4 text-blue-500 font-bold">+0.096</td>
        <td class="text-green-700 font-semibold">✓ match</td>
      </tr>
      <tr class="bg-green-50">
        <td class="py-2 pr-4 font-mono">toptwo</td>
        <td class="text-right pr-4 text-red-500 font-bold">−0.290</td>
        <td class="text-right pr-4 text-red-400 font-bold">−0.064</td>
        <td class="text-green-700 font-semibold">✓ match</td>
      </tr>
    </tbody>
  </table>
  <div class="mt-3 p-3 bg-neutral-soft rounded text-xs space-y-1">
    <p><b>Value Landscape Alignment:</b> Both humans and the oracle stop faster when one move is clearly better (negative toptwo correlation) and search longer when deeper search yields higher value (positive gain_depth/VOC correlation).</p>
    <p><b>Structural Complexity Divergence:</b> Branching factor and piece count strongly drive human deliberation due to explicit move/threat enumeration, whereas the oracle's PUCT search and value network handle this natively, showing near-zero correlation with optimal stopping.</p>
  </div>
</div>

---

# The epistemology of stopping

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="opacity-80">At any step t, the agent faces three locally indistinguishable situations:</p>

  <div class="space-y-2">
    <div class="flex gap-3 items-start p-2.5 bg-green-50 border-l-2 border-green-400 rounded">
      <span class="font-bold text-green-700 w-5 shrink-0">1</span>
      <div><b>Converging to the right answer.</b> Q rising; this IS the globally best move. <span class="text-green-700 font-semibold">Should stop.</span></div>
    </div>
    <div class="flex gap-3 items-start p-2.5 bg-red-50 border-l-2 border-red-400 rounded">
      <span class="font-bold text-red-700 w-5 shrink-0">2</span>
      <div><b>Converging to the wrong answer.</b> Q rising — but a better move is unexplored. Signal identical to case 1. <span class="text-red-700 font-semibold">Should keep going.</span></div>
    </div>
    <div class="flex gap-3 items-start p-2.5 bg-amber-50 border-l-2 border-amber-400 rounded">
      <span class="font-bold text-amber-700 w-5 shrink-0">3</span>
      <div><b>Not converging.</b> Q plateaued — either genuinely ambiguous, or the better move hasn't been reached. <span class="text-amber-700 font-semibold">Unknown.</span></div>
    </div>
  </div>

  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded text-xs space-y-1 mt-1">
    <p><b>Circular problem:</b> "would stopping now change my action?" requires knowing what further search would reveal — which requires completing the search.</p>
    <p><b>"Chasing tails":</b> in dominant positions, Q keeps improving even after the decision is made. Both humans and the oracle rationally continue — systematic over-computation in dominant positions, under-computation in ambiguous ones.</p>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section III</div>
    <h1 class="text-4xl">Next Steps</h1>
  </div>
</div>

---

# Analysis 1 — Unify the datasets

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Run Yotam's lmcos pipeline on positions from the human behavioral dataset. Compare oracle_stop_step with actual human RT at the <i>same positions</i>.</p>

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs mb-3">
    <b>Why not use Yotam's existing trees?</b> His dataset (ELO 1800–2600, all time controls) carries different human confounds than the filtered behavioral data (10+0, ≥2000 Elo). Unifying removes this disanalogy and enables a direct position-level test.
  </div>

  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">1.</div>
      <div>Extract FENs from <code>processed_moves_nonzero</code>. Verify 6-field FEN format. Start with 1K as a smoke test — must complete in minutes on A100.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">2.</div>
      <div>Run <code>build_tree.py</code> (Lc0, budget=96) → generate oracle trees. Run <code>compute_budgeted_oracle()</code> → <code>oracle_stop_step</code> per position.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">3.</div>
      <div>Join with <code>personal.db</code> on FEN → recover <code>log(RT)</code>. Plot oracle_stop_step vs log(RT). Does the oracle predict where humans think longer?</div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">Status: 1K smoke test (644 FENs) running on SLURM (Job ID 9217163). Re-submitted after fixing a PUCT infinite loop bug in tree generation.</div>
</div>

---

# Analysis 2 — Tree-Statistic Summary & Minimal Model

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Find the smallest architecture that trains end-to-end in hours. <b>Burning question: can we skip GNN pretraining entirely?</b></p>

  <div class="p-3 bg-accent-soft border-l-2 border-accent rounded text-xs mb-3">
    <b>A0b result (2026-06-05, corrected labels):</b> 4 raw tree-stat scalars → MLP → sign acc <b>54.6%</b> val (≈ chance), exact stop <b>5.2%</b>, r(pred,oracle) = +0.29. GNN+MC packed baseline: <b>90.1%</b>. GNN encoder confirmed essential — scalar features have no discriminative power for halt/continue.
  </div>

  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">1.</div>
      <div>Train Config D (d_embed=16, 1 layer) from <b>random initialisation</b> on 10 existing training shards. Track GNN loss — does it converge in &lt;2 hours?</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">2.</div>
      <div>If yes: skip pretraining entirely. If no: escalate to Config C (d_embed=32) until acceptable loss is reached.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">3.</div>
      <div>Success = acceptable GNN loss (&lt;pretrained baseline) + training time ≤2 hours on a single GPU.</div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">Status: Old Config D pretraining job (Job ID 9215254) cancelled to pivot first to A2.1 (Tree-Statistic Summaries) before attempting A2.2 (Tiny-GNN) as requested.</div>
</div>

---

# Analysis 3 — Weaker engine (conditional)

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Only pursue if Analysis 1 shows no correspondence between oracle_stop_step and human RT.</p>

  <div class="p-3 bg-neutral-soft border-l-2 border-gray-400 rounded text-xs mb-3">
    <b>Hypothesis:</b> Lc0 at ~3000 ELO trivially resolves positions that a 2000-ELO human finds hard. Matching engine strength to player ELO may improve alignment.
  </div>

  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-amber-600 font-bold w-5 shrink-0">1.</div>
      <div>Use Stockfish with <code>UCI_LimitStrength=true, UCI_Elo=2000</code> for tree building on the same 10K human FENs from A1.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-amber-600 font-bold w-5 shrink-0">2.</div>
      <div>Centipawn → WDL: <code>pwin = 1/(1 + exp(−cp/400))</code>. Compute oracle_stop_step. Compare r(oracle, human RT) with A1 result.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-amber-600 font-bold w-5 shrink-0">3.</div>
      <div>If alignment improves: retrain the full pipeline with the weaker engine — this becomes the human-aligned normative agent.</div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">CPU-only. No consistency requirement with Lc0 frozen weights — fresh pipeline.</div>
</div>

---

# Analysis 4 — Information-theoretic stopping (entropy VoI)

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Test policy entropy reduction as a normative stopping rule over Stockfish multi-depth search traces.</p>

  <div class="p-3 bg-green-50 border-l-2 border-green-400 rounded text-xs mb-3">
    <b>Motivation:</b> Human RT correlates with branching factor ($r \approx +0.20$). Expected VOC ($E[\Delta\text{UC}]$) incorporates candidate-set uncertainty, but requires tuning decision temperatures. Information gain (Shannon entropy reduction of the policy) offers a direct, parameter-free stopping signal.
  </div>

  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">1.</div>
      <div>Query Stockfish on 1K human positions for depths $d \in [1, 8]$. Extract Centipawns for all legal moves. Convert to win probabilities: $Q_{d, k} = 1/(1 + e^{-cp/400})$.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">2.</div>
      <div>Compare two policy formulations $P_k(d) = \text{softmax}(\beta Q_d)$: 
        <br>• <b>Approach A (Subset Softmax):</b> Evaluate over top-K moves returned by engine.
        <br>• <b>Approach B (Neutral Imputation):</b> Evaluate over all legal moves, imputing unevaluated ones to $0.5$.
      </div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-5 shrink-0">3.</div>
      <div>Determine stopping depth $d^*$ where information gain drops below threshold: $H(d-1) - H(d) < \theta$. Correlate $d^*$ with human $log(RT)$ and branching factor.
      </div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">Status: Complete (10K scale-up). Run locally on Della with 16 CPU workers. Swept over θ ∈ [0, 0.1] on 6,494 valid traces.</div>
</div>

---

# Analysis 4 — Results of 10K scale-up

<div class="mt-4 text-sm max-w-3xl">
  <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-3">Pearson correlation (r) across θ thresholds (n = 6,494 multi-depth traces, β = 1.0)</div>
  <table class="text-xs w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-right py-2 pr-4 font-bold">θ threshold</th>
        <th class="text-right py-2 pr-4 font-bold">r(d* with log RT)</th>
        <th class="text-right py-2 pr-4 font-bold">r(d* with branching)</th>
        <th class="text-right py-2 font-bold">Mean stopping depth (d*)</th>
      </tr>
    </thead>
    <tbody class="text-[12px] font-mono">
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 text-right">0.0</td>
        <td class="py-2 pr-4 text-right">-0.029</td>
        <td class="py-2 pr-4 text-right">-0.103</td>
        <td class="py-2 text-right">2.46</td>
      </tr>
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 text-right font-bold">0.001</td>
        <td class="py-2 pr-4 text-right text-blue-600 font-bold">+0.014</td>
        <td class="py-2 pr-4 text-right text-blue-600 font-bold">+0.101</td>
        <td class="py-2 text-right font-bold">1.64</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 text-right">0.005</td>
        <td class="py-2 pr-4 text-right">-0.005</td>
        <td class="py-2 pr-4 text-right">+0.039</td>
        <td class="py-2 text-right">1.15</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 text-right">0.01</td>
        <td class="py-2 pr-4 text-right">-0.020</td>
        <td class="py-2 pr-4 text-right">-0.017</td>
        <td class="py-2 text-right">1.05</td>
      </tr>
      <tr>
        <td class="py-2 pr-4 text-right">0.05</td>
        <td class="py-2 pr-4 text-right">-0.018</td>
        <td class="py-2 pr-4 text-right">-0.067</td>
        <td class="py-2 text-right">1.00</td>
      </tr>
    </tbody>
  </table>
  <div class="mt-4 p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs space-y-1">
    <p><b>Key Finding:</b> The strong initial correlation with log(RT) disappeared at the 10K scale. While the correlation with branching factor remains positive at θ = 0.001 ($r \approx 0.101$), the correlation with human log(RT) drops to near-zero ($r \approx 0.014$).</p>
    <p><b>Implication:</b> This specific normative stopping rule based on full-strength engine policy-entropy reduction does not explain human deliberation time. This suggests that standard engine evaluations do not align well with human cognitive effort (supporting Analysis 3's weaker engine hypothesis).</p>
  </div>
</div>

---

# The core open question

<div class="mt-8 max-w-3xl space-y-5">
  <div class="p-5 bg-neutral-soft border-2 border-accent rounded-lg text-center">
    <div class="text-lg font-semibold">
      oracle_stop_step(s) &nbsp;↔&nbsp; actual log RT(s)
    </div>
    <div class="text-sm opacity-70 mt-2">at the <b>same positions</b> s from the human behavioral dataset</div>
  </div>

  <div class="grid grid-cols-2 gap-4 text-sm">
    <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
      <b class="text-accent text-xs uppercase tracking-wider">What we know so far</b>
      <ul class="mt-2 text-xs list-disc pl-4 space-y-1 opacity-80">
        <li>4/4 board features match direction between oracle and humans (A0a, corrected)</li>
        <li>A0b minimal MLP vs GNN+MC — val sign acc <b>54.6%</b> (≈ chance) vs GNN+MC 90.1%; GNN essential ✅</li>
        <li>Both oracle and humans over-compute in dominant positions</li>
      </ul>
    </div>
    <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded">
      <b class="text-amber-700 text-xs uppercase tracking-wider">Analysis 1 will answer</b>
      <p class="mt-2 text-xs text-amber-900">Run Yotam's pipeline on the human dataset FENs. A positive correlation between oracle_stop_step and log(RT) at matched positions → one paper. No correlation → two papers.</p>
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

# Paper Sketch

<div class="h-full flex flex-col justify-start mt-2">
  <PaperSketch />
</div>

---

# Control flow

<div class="h-full flex flex-col items-center justify-center">
  <LeelaSearchLoop />
</div>

---

# Representation & GNN

<div class="mt-4">
  <MetaControllerZoom />
</div>

---

# Halt controller & DP Oracle

<div class="mt-2">
  <DpOracleDiagram />
</div>

<div class="grid grid-cols-3 gap-6 mt-4 text-sm px-4">
  <div><b class="text-accent block mb-1">Why DP?</b>Optimal stopping is recursive — reason backwards from leaves.</div>
  <div><b class="text-accent block mb-1">Solution</b>$V^*(s) = \max(V,\, \mathbb{E}[V^*_\text{child}] - C)$</div>
  <div><b class="text-accent block mb-1">Supervision</b>DP result as ground truth for the GNN halt head.</div>
</div>
