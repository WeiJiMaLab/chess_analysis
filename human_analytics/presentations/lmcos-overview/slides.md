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

# Dataset

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-2">
  <ul class="list-disc pl-5 space-y-2">
    <li><b>Source:</b> Lichess, Oct–Dec 2023 · 10+0 · both players ≥ 2000 Elo</li>
    <li><b>Scale:</b> 1.97M games · 145M moves · <b>135M non-zero-time moves</b></li>
    <li><b>Exclusions:</b> berserk, negative move_time, grant-more-time games dropped at shard time</li>
    <li><b>Engine analysis subset:</b> ply 15–75, opponent clock ≥ 60s → 88M positions eligible; <b>1M sampled</b> at Stockfish depth 5/1</li>
  </ul>
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
    <div><span class="label">Expect</span> Near-symmetric if Weber's Law holds for deliberation</div>
    <div class="finding"><span class="label">Finding</span> Approximately log-normal — consistent with Weber's Law. Deliberation time scales multiplicatively, not additively.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/log_movetime_histogram.png" />
</div>

---

# Do players think more in the middlegame?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Move ply (game stage)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Less thinking in opening (book), more in middlegame</div>
    <div class="finding"><span class="label">Finding</span> Non-monotone arc — peaks around ply 30–60. Opening and endgame both faster. Stratified by ply tertile to confirm the arc is not an artifact.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/ply_vs_movetime.png" />
</div>

---

# Do players with more time think longer?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Player clock remaining (s)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> More time available → more deliberation</div>
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
    <div class="finding"><span class="label">Finding</span> Real but weaker — players think less when the opponent is also pressed. Consistent across ply stages.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/clock_opp_vs_movetime.png" />
</div>

---

# Do more legal moves mean more thinking?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Number of legal moves</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> More candidate moves → more uncertainty → more deliberation</div>
    <div class="finding"><span class="label">Finding</span> <b>Strongest behavioral predictor</b> (r = +0.20). Effect is largest in the middlegame tertile where tactical complexity peaks.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/npossiblemoves_vs_movetime.png" />
</div>

---

# Does own material drive longer thinking?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Own non-pawn pieces</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> More pieces → more candidate interactions → more thinking</div>
    <div class="finding"><span class="label">Finding</span> Positive (r = +0.04). Related to branching but captures a different dimension: how many pieces must I reason about deploying? Effect compresses in endgame.</div>
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

<table class="text-xs max-w-2xl mt-6 border-collapse">
  <thead>
    <tr class="border-b border-gray-300">
      <th class="text-left py-2 pr-5 font-bold">Quantity</th>
      <th class="text-left py-2 pr-5 font-bold">Formula</th>
      <th class="text-left py-2 font-bold">Interpretation</th>
    </tr>
  </thead>
  <tbody class="text-[11px] opacity-90">
    <tr><td class="py-1.5 pr-5 font-mono">e_win_best</td><td class="pr-5">V<sub>deep</sub>(a<sub>deep</sub>)</td><td>Win prob of objectively best move</td></tr>
    <tr><td class="py-1.5 pr-5 font-mono">e_win_second_best</td><td class="pr-5">V<sub>deep</sub>(rank-2)</td><td>Win prob of second-best move</td></tr>
    <tr><td class="py-1.5 pr-5 font-mono">e_win_taken</td><td class="pr-5">V<sub>deep</sub>(move played)</td><td>Win prob of the actual move</td></tr>
    <tr class="border-t border-gray-200 text-accent font-semibold">
      <td class="py-1.5 pr-5 font-mono">gain_depth</td><td class="pr-5">e_win_best − V<sub>deep</sub>(a<sub>shallow</sub>)</td><td>Gain from depth-5 over depth-1 best &nbsp;(≥ 0)</td>
    </tr>
    <tr class="text-accent font-semibold">
      <td class="py-1.5 pr-5 font-mono">MQ</td><td class="pr-5">e_win_taken − e_win_best</td><td>How suboptimal was the move played &nbsp;(≤ 0)</td>
    </tr>
    <tr class="text-accent font-semibold">
      <td class="py-1.5 pr-5 font-mono">toptwo</td><td class="pr-5">e_win_best − e_win_second_best</td><td>How decisive is the best move &nbsp;(≥ 0)</td>
    </tr>
  </tbody>
</table>

<p class="mt-4 opacity-50 text-[10px]">gain_depth ≈ ΔUC from Russek et al. (2022) computed at depth=5 vs depth=15. Win probabilities from Stockfish WDL, side-to-move perspective before the move.</p>

---

# How often does deeper search find a better move?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> gain_depth (win-prob gain, depth-5 over depth-1)</div>
    <div><span class="label">y</span> Count</div>
    <div><span class="label">Expect</span> Many positions trivial (gain=0); right-skewed tail for tactically complex positions</div>
    <div class="finding"><span class="label">Finding</span> 67% zero — depth-1 already optimal for most positions. Long right tail where deeper search genuinely helps. Mean ≈ 0.10.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/voc_histogram.png" />
</div>

---

# How optimal are human moves?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> MQ = e_win_taken − e_win_best &nbsp;(≤ 0)</div>
    <div><span class="label">y</span> Count</div>
    <div><span class="label">Expect</span> Spike at 0 (optimal move played); left tail for errors</div>
    <div class="finding"><span class="label">Finding</span> 62% near-optimal (MQ ≥ −0.005). Long left tail from rare but large blunders. Strong players rarely err, but errors are costly when they do.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/mq_histogram.png" />
</div>

---

# Do humans think longer when it pays?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> gain_depth (value of deeper search)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Higher gain_depth → more thinking (Russek et al., 2022)</div>
    <div class="finding"><span class="label">Finding</span> <b>r = +0.096</b> (n = 1M). Positive and consistent across all ply tertiles — humans allocate compute where it demonstrably helps. Replicates Russek et al. at depth=5.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/voc_vs_movetime.png" />
</div>

---

# Does a decisive best move mean more or less thinking?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> toptwo = e_win_best − e_win_second_best &nbsp;(≥ 0)</div>
    <div><span class="label">y</span> log(Move Time)</div>
    <div><span class="label">Expect</span> Ambiguous — decisive positions may require finding or recognising the winning move</div>
    <div class="finding"><span class="label">Finding</span> <b>r = −0.064</b> (n = 1M) — negative. When one move is clearly superior at depth-5, players think less. Contrast with gain_depth: here the engine has already resolved the choice; there, it hasn't.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/toptwo_vs_movetime.png" />
</div>

---

# Does more time mean better moves?

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-4 text-sm">
    <div><span class="label">x</span> Player clock remaining (s)</div>
    <div><span class="label">y</span> MQ (move quality ≤ 0)</div>
    <div><span class="label">Expect</span> More clock → longer deliberation → better move quality</div>
    <div class="finding"><span class="label">Finding</span> <b>r = −0.091</b> — counterintuitive. More clock → worse MQ, within every ply tertile. Players with more time have played quickly through low–gain_depth positions; depth-5 penalises those strategic choices.</div>
  </div>
  <img class="w-full object-contain max-h-[72vh]" src="/figures/mq_vs_clock.png" />
</div>

---

# Correlation matrix

<div class="grid grid-cols-[60fr_40fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <img class="w-full object-contain max-h-[72vh]" src="/figures/correlation_matrix.png" />
  <div>
    <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-2">n = 1,000,000 · Pearson r</div>
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

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section III</div>
    <h1 class="text-4xl">Towards a Normative Baseline</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      What <i>should</i> a rational agent's think time look like, and how close can we get to measuring it?
    </div>
  </div>
</div>

---

# The core question

<div class="mt-6 max-w-3xl space-y-5">
  <div class="p-4 bg-neutral-soft border-l-4 border-accent rounded">
    <div class="text-accent font-bold uppercase tracking-widest text-[10px] mb-2">What we want to show</div>
    <div class="text-xl font-semibold font-mono">
      opt_think_steps(s) &nbsp;&nbsp;↔&nbsp;&nbsp; actual_think_steps(s)
    </div>
    <div class="mt-3 text-sm opacity-80 space-y-1">
      <div><b>Left:</b> normatively optimal compute — the depth where marginal gain from one more step equals its cost</div>
      <div><b>Right:</b> observed RT in seconds — a proxy for the agent's internal evaluation steps</div>
    </div>
  </div>

  <div class="text-sm space-y-2 opacity-80">
    <p>If the correlation is high, the agent (human or model) is allocating compute rationally — thinking longer when and only when it pays.</p>
    <p>This is the fundamental validation for both the human behavioral data and the lmcos learned controller.</p>
    <p class="opacity-60 text-xs">Everything in the previous section is a proxy for the left side. The challenge is building a good approximation of opt_think_steps from observable position features.</p>
  </div>
</div>

---

# The thought process: building opt_think_steps

<div class="mt-4 text-sm max-w-3xl">
  <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-3">Proxy hierarchy — increasingly faithful to the normative ideal</div>
  <table class="text-xs w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold">Proxy</th>
        <th class="text-left py-2 pr-4 font-bold">Formula</th>
        <th class="text-left py-2 font-bold">Key limitation</th>
      </tr>
    </thead>
    <tbody>
      <tr class="border-b border-gray-100">
        <td class="py-1.5 pr-4 font-mono text-amber-700">gain_depth</td>
        <td class="pr-4 opacity-80">V<sub>5</sub>(a<sub>5</sub>) − V<sub>5</sub>(a<sub>1</sub>)</td>
        <td class="opacity-80 text-[11px]">Two arbitrary depths; total gain, not marginal; no candidate-set noise</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-1.5 pr-4 font-mono text-amber-700">gain_budget</td>
        <td class="pr-4 opacity-80">V<sub>96</sub>(a<sub>96</sub>) − V<sub>96</sub>(a<sub>1</sub>)</td>
        <td class="opacity-80 text-[11px]">Interesting: node budget → shallower in complex positions → gain_budget is <i>negatively</i> correlated with branching</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-1.5 pr-4 font-mono text-blue-600">entropy_topk</td>
        <td class="pr-4 opacity-80">H(softmax(V<sub>1</sub>) over top-K)</td>
        <td class="opacity-80 text-[11px]">Captures candidate-set width but not depth dimension</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-1.5 pr-4 font-mono text-blue-600">marginal_gain(d)</td>
        <td class="pr-4 opacity-80">V(d+1) − V(d)</td>
        <td class="opacity-80 text-[11px]">Correct shape but requires evaluation at many depths per position</td>
      </tr>
      <tr class="border-b border-gray-100 font-semibold">
        <td class="py-1.5 pr-4 font-mono text-green-700">min_expansions</td>
        <td class="pr-4 opacity-90">min{d : a<sub>d</sub> = a<sub>∞</sub>}</td>
        <td class="opacity-80 text-[11px] font-normal">Best so far — decision consistency, no value comparison; not yet computed</td>
      </tr>
      <tr>
        <td class="py-1.5 pr-4 font-mono text-green-700">E[ΔUC]</td>
        <td class="pr-4 opacity-80">E<sub>shallow</sub>[V<sub>deep</sub>(a<sub>deep</sub>) − V<sub>deep</sub>(k)]</td>
        <td class="opacity-80 text-[11px]">Requires β — see next slide</td>
      </tr>
    </tbody>
  </table>
</div>

---

# Why we chose to pause here

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="p-3 bg-neutral-soft border-l-3 border-amber-400 rounded text-xs space-y-2">
    <div class="font-bold text-amber-700 uppercase tracking-wide text-[10px]">Rabbit hole 1: E[ΔUC] and the β problem</div>
    <p>E[ΔUC] requires estimating β (the "decision temperature" of the shallow policy) from observed moves. But observed moves were generated by <i>deep</i> thinking — not by a depth-1 process. Fitting β against deep-thinking outputs gives a biased, endogenous estimate: the chicken-and-the-egg problem.</p>
    <p class="opacity-70">Clean fix would require observing moves under extreme time pressure (near-zero RT) — a counterfactual not available in the data.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-3 border-amber-400 rounded text-xs space-y-2">
    <div class="font-bold text-amber-700 uppercase tracking-wide text-[10px]">Rabbit hole 2: Satisficing stopping rule</div>
    <p>Even "deep" search leaves residual uncertainty. Agents stop not when σ = 0 but when σ is small enough — a satisficing threshold θ ("80% sure"). This makes RT a <i>stopping time</i>, not a budget, and requires a full Drift-Diffusion or Bayesian sequential test model.</p>
  </div>
  <div class="p-3 bg-accent-soft border-l-3 border-accent rounded text-xs">
    <div class="font-bold text-accent uppercase tracking-wide text-[10px]">Why our proxies are sufficient for now</div>
    <p>We need opt_think_steps to be <i>correlated</i> with actual RT in the right directions — not to be perfectly measured. gain_depth, branching, and toptwo are interpretable, computable, and make qualitatively correct predictions. That is enough to compare against the lmcos model's feature sensitivity.</p>
  </div>
</div>

---

# Our chosen proxies: intuition

<div class="mt-4 max-w-3xl space-y-0">
  <div class="text-[10px] font-bold uppercase tracking-widest opacity-50 mb-3">Four exogenous predictors — all computable, all interpretable</div>
  <table class="text-sm w-full border-collapse">
    <thead>
      <tr class="border-b-2 border-gray-300">
        <th class="text-left py-2 pr-4 font-bold w-1/4">Predictor</th>
        <th class="text-left py-2 pr-4 font-bold w-1/2">Intuition: why should you think longer?</th>
        <th class="text-right py-2 font-bold">r with log(RT)</th>
      </tr>
    </thead>
    <tbody class="text-[12px]">
      <tr class="border-b border-gray-100">
        <td class="py-2.5 pr-4 font-mono font-semibold">branching</td>
        <td class="pr-4 opacity-90">More legal moves = more candidates to evaluate; uncertainty spreads across more options</td>
        <td class="text-right font-mono text-blue-600 font-bold">+0.20</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2.5 pr-4 font-mono font-semibold">own material</td>
        <td class="pr-4 opacity-90">More pieces = more interactions and threats to reason about; related to branching but captures compositional complexity</td>
        <td class="text-right font-mono text-blue-400 font-bold">+0.04</td>
      </tr>
      <tr class="border-b border-gray-100">
        <td class="py-2.5 pr-4 font-mono font-semibold">gain_depth</td>
        <td class="pr-4 opacity-90">Empirical evidence: depth-5 finds a better move than depth-1 here. Direct signal that shallow evaluation is unreliable for this position.</td>
        <td class="text-right font-mono text-blue-500 font-bold">+0.10</td>
      </tr>
      <tr>
        <td class="py-2.5 pr-4 font-mono font-semibold">toptwo</td>
        <td class="pr-4 opacity-90">Large gap between best and 2nd-best means the position is already "resolved" at depth-5 — less need to deliberate further</td>
        <td class="text-right font-mono text-red-400 font-bold">−0.06</td>
      </tr>
    </tbody>
  </table>
  <div class="mt-3 text-xs opacity-60">Note on toptwo: the negative sign is correct — when one move is clearly superior, deliberation adds little. Contrast with gain_depth (positive): there the engine's own deeper search is still changing its mind.</div>
</div>

---

# The minimum viable path forward

<div class="mt-6 max-w-3xl space-y-4 text-sm">
  <p class="opacity-80">To connect the human data to the lmcos learned controller, we need to run both on the same positions and ask: do they respond to the same features?</p>

  <div class="space-y-3">
    <div class="p-3 bg-neutral-soft rounded flex gap-4 items-start">
      <div class="text-green-600 font-bold text-lg mt-0.5">✓</div>
      <div>
        <b>Phase 1 (done) — Human behavioral analysis</b>
        <div class="text-xs opacity-70 mt-1">Four exogenous predictors established. Correlations with log(RT) measured on 1M positions.</div>
      </div>
    </div>
    <div class="p-3 bg-neutral-soft rounded flex gap-4 items-start">
      <div class="text-amber-500 font-bold text-lg mt-0.5">→</div>
      <div>
        <b>Phase 2 (next) — lmcos model predictions</b>
        <div class="text-xs opacity-70 mt-1">Run Lc0 on ~1000 positions from the human dataset to generate search trees. Feed trees to the trained lmcos halt controller → P(halt | position). Requires: (a) running Lc0 on FENs from the dataset, (b) calling the trained model checkpoint.</div>
      </div>
    </div>
    <div class="p-3 bg-neutral-soft rounded flex gap-4 items-start">
      <div class="text-gray-400 font-bold text-lg mt-0.5">◯</div>
      <div>
        <b>Phase 3 (synthesis) — Side-by-side comparison</b>
        <div class="text-xs opacity-70 mt-1">Correlate P(halt) with the same four features. Build a 2×4 comparison table: r(feature, log RT) vs r(feature, P(halt)). The claim: both humans and the model think longer in positions that normatively warrant more compute.</div>
      </div>
    </div>
  </div>
</div>

---

# Summary

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider">Key results (n = 1M)</b>
    <ul class="mt-2 list-none space-y-1 text-xs">
      <li>✓ <b>log(RT) ∝ gain_depth</b> — r = +0.096: humans think longer when computation demonstrably pays</li>
      <li>✓ <b>Branching is the strongest RT predictor</b> — r = +0.20; more candidates → more spreading of deliberation</li>
      <li>✓ <b>Decisive positions → faster play</b> — toptwo ↔ log(RT) = −0.06; when one move is clearly best, stop early</li>
      <li>✓ <b>gain_depth ↔ MQ = −0.42</b> — positions where computation matters are also where humans err the most</li>
      <li>⚠ <b>MQ–clock is negative</b> (within all ply tertiles) — likely a depth-5 artifact; more clock = more low–gain_depth positions</li>
    </ul>
  </div>
  <div class="p-3 bg-amber-50 border-l-2 border-amber-400">
    <b class="text-amber-700 text-xs uppercase tracking-wider">Explicit pause points</b>
    <ul class="mt-2 list-none space-y-1 text-xs text-amber-900">
      <li>∅ &nbsp;E[ΔUC] — β endogeneity problem; requires near-zero-RT observations to identify cleanly</li>
      <li>∅ &nbsp;min_expansions — best normative proxy but requires streaming iterative deepening; not yet computed</li>
      <li>∅ &nbsp;Full DDM / Bayesian stopping model — a complete research project on its own</li>
      <li>∅ &nbsp;depth=15 replication of Russek et al. — gain_depth at depth=5 is sufficient for the model comparison</li>
    </ul>
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
    <li>Mirrors <b>fast vs. slow</b> reasoning (Kahneman, 2011; MB/MF RL — Daw et al., 2005).</li>
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

<p class="mt-3 text-sm">Sigmoidal halting unit + ponder cost $\tau$. Depth-per-input learned end-to-end. Operates on sequential traces, not tree topology.</p>

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
