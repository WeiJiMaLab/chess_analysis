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
    Project status — June 2026
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section I</div>
    <h1 class="text-4xl">Where we stand</h1>
  </div>
</div>

---

# Two projects, one open question

<div class="mt-6 max-w-3xl space-y-4 text-sm">
  <div class="grid grid-cols-2 gap-6">
    <div class="p-4 bg-neutral-soft border-l-3 border-accent rounded">
      <b class="text-accent text-xs uppercase tracking-wider">Human analysis</b>
      <p class="mt-2">What do strong players <i>actually do</i>?<br>When and why do they think longer?</p>
      <p class="mt-2 text-xs opacity-60">Status: 1M-position dataset established, key predictors measured</p>
    </div>
    <div class="p-4 bg-neutral-soft border-l-3 border-secondary rounded">
      <b class="text-secondary text-xs uppercase tracking-wider">ML agent (lmcos)</b>
      <p class="mt-2">What <i>should</i> an agent do?<br>Train a controller to stop optimally under compute cost.</p>
      <p class="mt-2 text-xs opacity-60">Status: DP oracle + GNN + MC controller trained and running</p>
    </div>
  </div>

  <div class="p-4 bg-amber-50 border-l-3 border-amber-400 rounded mt-4">
    <b class="text-amber-700 text-xs uppercase tracking-wider">The open question</b>
    <p class="mt-2">Do they agree? Does the normative agent predict where humans spend time?</p>
  </div>

  <div class="grid grid-cols-2 gap-6 mt-2 text-xs">
    <div class="p-3 bg-neutral-soft rounded">
      <b>If yes →</b> one paper: humans approximate optimal compute allocation, and here is an agent that learns to do the same
    </div>
    <div class="p-3 bg-neutral-soft rounded">
      <b>If no →</b> two papers: a descriptive account of human deliberation, and separately, a normative agent for efficient search
    </div>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl">What humans actually do</h1>
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
    <div class="finding"><span class="label">Implication</span> Players scale thinking time multiplicatively with difficulty — consistent with Weber's Law on deliberation. Justifies working in log(RT) throughout.</div>
    <div class="text-xs opacity-60 mt-4">n = 135M non-zero-time moves from 1.97M games</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/log_movetime_histogram.png" />
</div>

---

# What predicts when humans think longer?

<div class="mt-4 text-sm max-w-3xl">
  <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-3">Four board features, all computed from position (no RT feedback)</div>
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
        <td class="opacity-80">More candidates → spread uncertainty over more moves</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 font-mono">gain_depth (ΔUC@5)</td>
        <td class="text-blue-500 font-bold">+0.10</td>
        <td class="opacity-80">Deeper search actually finds a better move here</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2 pr-4 font-mono">own material</td>
        <td class="text-blue-400 font-bold">+0.04</td>
        <td class="opacity-80">More pieces → more interactions to resolve</td>
      </tr>
      <tr>
        <td class="py-2 pr-4 font-mono">toptwo</td>
        <td class="text-red-400 font-bold">−0.06</td>
        <td class="opacity-80">Decisive best move → less need to distinguish candidates</td>
      </tr>
    </tbody>
  </table>
  <div class="mt-4 text-xs opacity-60">
    Also tested: gain_depth replicates Russek et al. (2022) r(log RT, ΔUC) = +0.096 at depth=5 vs their depth=15.
    All effects hold within ply tertiles (not a game-stage confound).
  </div>
</div>

---

# Correlation matrix — the full picture

<div class="grid grid-cols-[58fr_42fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <img class="w-full object-contain max-h-[72vh]" src="/figures/correlation_matrix.png" />
  <div class="text-xs space-y-2">
    <div class="font-semibold text-sm mb-3">n = 1,000,000 · Pearson r  <span class="opacity-50 font-normal">(blue = positive, red = negative)</span></div>
    <div class="p-2 bg-blue-50 rounded border-l-2 border-blue-400">
      <b>branching ↔ log(RT) = +0.20</b><br>
      <span class="opacity-80">Strongest predictor. Width of choice problem drives deliberation.</span>
    </div>
    <div class="p-2 bg-blue-50 rounded border-l-2 border-blue-300">
      <b>gain_depth ↔ log(RT) = +0.10</b><br>
      <span class="opacity-80">Humans linger where deeper computation demonstrably pays.</span>
    </div>
    <div class="p-2 bg-red-50 rounded border-l-2 border-red-400">
      <b>gain_depth ↔ MQ = −0.42</b><br>
      <span class="opacity-80">Positions where thinking helps are also where humans err most. <i>Errors drive deliberation.</i></span>
    </div>
    <div class="p-2 bg-red-50 rounded border-l-2 border-red-300">
      <b>MQ ↔ log(RT) = −0.12</b><br>
      <span class="opacity-80 text-[11px]">Better moves take longer — or: long thinkers play harder positions.</span>
    </div>
  </div>
</div>

---

# What's counterintuitive

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="p-3 bg-amber-50 border-l-3 border-amber-500 rounded">
    <b class="text-amber-800">1. More clock → worse move quality (r = −0.091, within every ply tertile)</b>
    <p class="mt-1 opacity-80 text-xs">Expected: more time available → more deliberation → better moves. Observed: players with more clock have been playing quickly through low-VOC positions where depth-5 disagrees with their "book" choices. The clock effect is confounded by VOC, not ply.</p>
  </div>
  <div class="p-3 bg-amber-50 border-l-3 border-amber-500 rounded">
    <b class="text-amber-800">2. 67% of positions have VOC = 0 (at depth=5)</b>
    <p class="mt-1 opacity-80 text-xs">Depth-1 search already finds the "correct" move for most positions. Yet humans still deliberate on them (clock, branching, material all positive). This suggests humans are computing something orthogonal to engine search depth — possibly resolving uncertainty across a <i>set</i> of candidates, not just the depth dimension.</p>
  </div>
  <div class="p-3 bg-amber-50 border-l-3 border-amber-500 rounded">
    <b class="text-amber-800">3. Branching is stronger than VOC (r = +0.20 vs +0.10)</b>
    <p class="mt-1 opacity-80 text-xs">Width of the decision problem predicts RT better than realized value of deeper search. This is inconsistent with a pure depth-based model of deliberation — candidate set size seems to be the primary driver.</p>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section III</div>
    <h1 class="text-4xl">What we trained the model to do</h1>
  </div>
</div>

---

# The DP loss target

<div class="mt-6 max-w-3xl space-y-5 text-sm">
  <p>The lmcos agent solves a stopping problem: at each expansion step, halt and act on the current best move, or pay a compute cost and continue searching.</p>

  <div class="p-4 bg-neutral-soft rounded">
    $$V^*(s) = \max\bigl(\underbrace{V_\text{halt}(s)}_{\text{play best move now}},\ \underbrace{V_\text{continue}(s) - C}_{\text{search one more step}}\bigr)$$
    <p class="mt-2 text-xs opacity-70">Solved by backward DP through the oracle Q-trace. Output: <code>oracle_stop_step</code> — the expansion count at which halting is DP-optimal. This is the training target for the meta-controller.</p>
  </div>

  <div class="grid grid-cols-2 gap-4 text-xs mt-2">
    <div class="p-3 bg-neutral-soft rounded">
      <b>Architecture:</b> GNN encodes search tree snapshot → MLP predicts P(halt). Trained by fitted-Q with oracle as supervisor.
    </div>
    <div class="p-3 bg-neutral-soft rounded">
      <b>Performance:</b> 80% exact stop-step accuracy, 90% sign accuracy on held-out trees. MC controller training: 52 seconds for 3 epochs.
    </div>
  </div>
</div>

---

# Data sources — the comparison problem

<div class="mt-4 max-w-3xl text-sm">
  <table class="text-xs w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold w-1/3">Property</th>
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
      <tr><td class="py-1.5 pr-4 font-semibold">Oracle target</td><td>oracle_stop_step (DP-optimal)</td><td>log(RT) (observed behavior)</td></tr>
    </tbody>
  </table>

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded mt-4 text-xs">
    <b>The datasets are different.</b> Before comparing oracle_stop_step to human RT, we need to generate oracle trees on the <i>same</i> positions that appear in the human dataset. This is Analysis 1.
  </div>
</div>

---

# Do humans approximate the DP target?

<div class="mt-4 max-w-3xl text-sm space-y-4">
  <p class="opacity-80">We don't yet know — Analysis 1 will answer this. But we can already enumerate the interpretive possibilities:</p>

  <table class="text-xs w-full border-collapse mt-2">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold">oracle_stop_step<br>↔ board features</th>
        <th class="text-left py-2 pr-4 font-bold">human RT<br>↔ board features</th>
        <th class="text-left py-2 font-bold">Interpretation</th>
      </tr>
    </thead>
    <tbody class="text-[11px]">
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 text-green-700 font-semibold">Same direction</td>
        <td class="py-2 pr-4 text-green-700 font-semibold">Same direction</td>
        <td class="py-2">Humans quasi-optimal; DP is a good model of VOC <span class="opacity-60">→ one paper</span></td>
      </tr>
      <tr class="border-b border-gray-100 bg-blue-50">
        <td class="py-2 pr-4 text-blue-700 font-semibold">Same direction</td>
        <td class="py-2 pr-4 text-red-700 font-semibold">Opposite direction</td>
        <td class="py-2">Humans not tracking VOC; DP model is correct but humans are irrational / use different cost structure</td>
      </tr>
      <tr class="border-b border-gray-100 bg-amber-50">
        <td class="py-2 pr-4 text-red-700 font-semibold">Opposite direction</td>
        <td class="py-2 pr-4 text-green-700 font-semibold">Same direction</td>
        <td class="py-2">Humans sensitive to something real; DP metric is wrong (VOC is not what we compute) <span class="opacity-60">→ revise the model</span></td>
      </tr>
      <tr class="bg-red-50">
        <td class="py-2 pr-4 text-red-700 font-semibold">Opposite direction</td>
        <td class="py-2 pr-4 text-red-700 font-semibold">Opposite direction</td>
        <td class="py-2">Neither is measuring the right thing <span class="opacity-60">→ two separate papers</span></td>
      </tr>
    </tbody>
  </table>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section IV</div>
    <h1 class="text-4xl">Next steps</h1>
  </div>
</div>

---

# Two stopping criteria — and which is closer to VOC

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="grid grid-cols-2 gap-5">
    <div class="p-4 bg-neutral-soft border-l-3 border-accent rounded space-y-2">
      <b class="text-accent text-xs uppercase tracking-wider">min_expansions</b>
      <p class="text-sm italic">"If I keep thinking, my action won't change — so my realized reward can't change. There is no decision-theoretic value in continuing."</p>
      <ul class="text-xs list-disc pl-4 space-y-1 opacity-80 mt-2">
        <li>Stops when <b>action identity</b> has stabilised</li>
        <li>Regret-free: additional compute cannot alter the chosen move</li>
        <li>Matches Russek et al.: ΔUC = 0 when a_shallow = a_deep</li>
        <li class="text-accent font-semibold">→ Closer to the spirit of VOC for human decisions</li>
      </ul>
    </div>
    <div class="p-4 bg-neutral-soft border-l-3 border-secondary rounded space-y-2">
      <b class="text-secondary text-xs uppercase tracking-wider">oracle_stop_step</b>
      <p class="text-sm italic">"My estimate of my action's value is still improving — that improvement outweighs the cost. Continue, even though the action won't change."</p>
      <ul class="text-xs list-disc pl-4 space-y-1 opacity-80 mt-2">
        <li>Stops when <b>Q-value refinement</b> no longer justifies cost</li>
        <li>Epistemic: MCTS Q(a) improves through deeper search even if action is fixed</li>
        <li>Appropriate for multi-episode training (generalising across budgets)</li>
        <li class="text-secondary font-semibold">→ What the lmcos model is trained on</li>
      </ul>
    </div>
  </div>

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs mt-2">
    <b class="text-amber-800">Implication:</b> The sign flip between our earlier (min_expansions) and current (oracle_stop_step) oracle analysis is not a mistake — they measure different things and their correlations with board features are near-inverses by construction. For the human comparison, <b>min_expansions may be the more natural target</b> since it directly asks "would thinking more change what I do?"
  </div>
</div>

---

# The epistemology of stopping

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="opacity-80">At any step t during search, the agent faces three indistinguishable situations:</p>

  <div class="space-y-2">
    <div class="flex gap-3 items-start p-2 bg-green-50 border-l-2 border-green-400 rounded">
      <span class="font-bold text-green-700 w-4 shrink-0">1</span>
      <div><b>Converging to the right answer.</b> Q of the current best is rising; this IS the globally optimal move. <span class="text-green-700 font-semibold">→ Should stop.</span></div>
    </div>
    <div class="flex gap-3 items-start p-2 bg-red-50 border-l-2 border-red-400 rounded">
      <span class="font-bold text-red-700 w-4 shrink-0">2</span>
      <div><b>Converging to the wrong answer.</b> Q keeps rising — but a superior move exists in the unexplored region. Signal identical to case 1. <span class="text-red-700 font-semibold">→ Should keep going.</span></div>
    </div>
    <div class="flex gap-3 items-start p-2 bg-amber-50 border-l-2 border-amber-400 rounded">
      <span class="font-bold text-amber-700 w-4 shrink-0">3</span>
      <div><b>Not converging.</b> Q has plateaued — either the position is genuinely ambiguous, or search isn't reaching the better move. <span class="text-amber-700 font-semibold">→ Unknown action.</span></div>
    </div>
  </div>

  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded mt-2 text-xs space-y-1">
    <p><b>The circular problem:</b> the correct stopping criterion — "would stopping now change my action?" — requires knowing what further search would reveal. That requires completing the search.</p>
    <p><b>The "chasing tails" effect:</b> Q-improvement is highest in dominant positions (cases 1 or 2). The agent rationally continues — but in dominant positions, the decision was already made. Each expansion confirms what was already known, not what was unknown.</p>
    <p><b>The dissatisfying asymmetry:</b> cases 1 and 2 are indistinguishable locally. So is "stuck for good" vs "stuck temporarily" in case 3. The agent has no corrective signal for the cases where more compute is genuinely needed but the search looks unproductive.</p>
  </div>
</div>

---

# Analysis 0 — Results

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="grid grid-cols-2 gap-4">
    <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
      <b class="text-accent text-xs uppercase tracking-wider">0a — Oracle vs human RT features (n=5K)</b>
      <table class="text-xs w-full mt-2 border-collapse">
        <thead><tr class="border-b border-gray-200"><th class="text-left py-1 pr-2">Feature</th><th class="text-right pr-2">Oracle r</th><th class="text-right pr-2">Human r</th><th class="text-right">Match</th></tr></thead>
        <tbody>
          <tr><td class="py-0.5 pr-2">branching</td><td class="text-right pr-2 text-blue-600">+0.165</td><td class="text-right pr-2 text-blue-600">+0.195</td><td class="text-right text-green-600">✓</td></tr>
          <tr><td class="py-0.5 pr-2">material</td><td class="text-right pr-2 text-blue-600">+0.154</td><td class="text-right pr-2 text-blue-500">+0.039</td><td class="text-right text-green-600">✓</td></tr>
          <tr><td class="py-0.5 pr-2">gain_depth</td><td class="text-right pr-2 font-bold text-blue-700">+0.797</td><td class="text-right pr-2 text-blue-500">+0.096</td><td class="text-right text-green-600">✓</td></tr>
          <tr><td class="py-0.5 pr-2">toptwo</td><td class="text-right pr-2 text-blue-500">+0.436</td><td class="text-right pr-2 text-red-400">−0.064</td><td class="text-right text-amber-500">✗</td></tr>
        </tbody>
      </table>
      <p class="mt-2 text-[11px] opacity-70">3/4 directions match → A1 is motivated. gain_depth r=+0.797: oracle is extremely sensitive to search value. toptwo mismatch: oracle verifies winner; humans satisfice on decisiveness.</p>
    </div>
    <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
      <b class="text-secondary text-xs uppercase tracking-wider">0b — Minimal MC baseline (4 scalars → MLP)</b>
      <div class="mt-2 space-y-1 text-xs">
        <div class="flex justify-between"><span>GNN+MC baseline</span><span class="font-mono font-bold">90.1%</span></div>
        <div class="flex justify-between"><span>Minimal MLC (val)</span><span class="font-mono font-bold text-accent">86.4%</span></div>
        <div class="flex justify-between text-[11px] opacity-60 mt-1"><span>Gap</span><span>3.7pp</span></div>
        <div class="flex justify-between text-[11px] opacity-60"><span>Inference</span><span>0.054 ms/snap</span></div>
      </div>
      <p class="mt-2 text-[11px] opacity-70">86% of the oracle's halt/continue signal captured by 4 raw scalars. GNN adds only ~4pp. → Skip GNN pretraining (A2) is strongly motivated.</p>
    </div>
  </div>
</div>

---

# Analysis 1 — The fair comparison

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Generate oracle trees for human board positions; compare DP-optimal stopping with actual RT.</p>
  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-6 shrink-0">1.</div>
      <div>Extract FENs from <code>processed_moves_nonzero</code> (human dataset). Run Lc0 tree building on those FENs — same pipeline as lmcos training, now on human positions. <span class="opacity-60 text-xs">Est. 2–4 hrs on A100 for 10K positions.</span></div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-6 shrink-0">2.</div>
      <div>Run <code>compute_budgeted_oracle()</code> on each tree → <code>oracle_stop_step</code>. Both this and <code>log(RT)</code> are now defined on the same positions.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-accent font-bold w-6 shrink-0">3.</div>
      <div>Plot (a) <code>oracle_stop_step</code> vs <code>log(RT)</code>, (b) <code>oracle_stop_step</code> vs board features, (c) model predicted stop step vs <code>oracle_stop_step</code>. Determine which of the 4 interpretive cases applies.</div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">Open question: should budget be fixed (96 expansions) or proportional to player clock time?</div>
</div>

---

# Analysis 2 — Minimal model

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">Find the smallest architecture that trains in hours, not days. Do this in parallel — unblocks iteration speed regardless of Analysis 1 outcome.</p>
  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-blue-600 font-bold w-8 shrink-0">1a.</div>
      <div><b>Logistic regression baseline</b> (today, no GPU): extract tree statistics (n_nodes, best_q, wdl_var, branching, budget) → train LR to predict sign(advantage). If this reaches 80%+ accuracy, the GNN is not adding useful representation.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-blue-600 font-bold w-8 shrink-0">1b.</div>
      <div><b>GNN architecture ablation</b>: shrink embeddings (128→32), layers (3→1), heads (4→1) on existing data. Target: ≥75% accuracy in ≤2 hrs total training. No new data needed.</div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">Current MC controller: 52s for 3 epochs, already fast. Bottleneck is GNN pretraining — ablation targets this.</div>
</div>

---

# Analysis 3 — Weaker engine (if A1 shows mismatch)

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="font-semibold">lmcos uses Lc0 at ~3000 ELO. Human players are ~2000 ELO. A position that's trivial for the engine may be hard for a human. Does matching engine strength to player ELO improve the human-oracle alignment?</p>
  <div class="space-y-2">
    <div class="flex gap-3 items-start">
      <div class="text-amber-600 font-bold w-6 shrink-0">1.</div>
      <div>Regenerate trees with a weaker engine (smaller Lc0 network, or Stockfish with <code>UCI_Elo=2000</code>). Run oracle on these trees → new oracle_stop_step estimates.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-amber-600 font-bold w-6 shrink-0">2.</div>
      <div>Check whether oracle_stop_step from the weaker engine correlates better with human RT. If yes: the agent's "difficulty" model was miscalibrated; matching ELO is necessary for the normative comparison.</div>
    </div>
    <div class="flex gap-3 items-start">
      <div class="text-amber-600 font-bold w-6 shrink-0">3.</div>
      <div>If weak-engine oracle aligns with human RT: retrain the full pipeline with the weaker engine. This becomes the version of the model that best predicts human behavior.</div>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-2">Do not pursue until Analysis 1 gives a directional answer.</div>
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

$$R(k) = \mathbb{E}\!\left[\text{Value}(T_k)\right] - C \cdot k \qquad V^*(s) = \max(V,\, \mathbb{E}[V^*_\text{child}] - C)$$

<div class="mt-4">
  <DpOracleDiagram />
</div>

<div class="grid grid-cols-3 gap-6 mt-4 text-sm px-4">
  <div><b class="text-accent block mb-1">Why DP?</b>Optimal stopping is recursive — reason backwards from leaves.</div>
  <div><b class="text-accent block mb-1">Solution</b>$V^*(s) = \max(V,\, \mathbb{E}[V^*_\text{child}] - C)$</div>
  <div><b class="text-accent block mb-1">Supervision</b>DP result as ground truth for the GNN halt head.</div>
</div>
