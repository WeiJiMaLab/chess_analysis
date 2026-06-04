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

# Correlation matrix

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
      <div class="text-sm opacity-70 mt-1">r = +0.20 vs +0.10. Width of the decision problem drives deliberation more than realized value of deeper search — inconsistent with a pure depth-based VOC model.</div>
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

  <div class="p-3 bg-amber-50 border-l-2 border-amber-400 rounded text-xs">
    <b>Important:</b> the lmcos training trees use FENs sampled from Lichess 2023 with ELO 1800–2600 and all time controls — a broader, different population than the human behavioral dataset (10+0, ≥2000 Elo). The datasets are not matched positions.
  </div>
</div>

---

# Analysis 0a: oracle vs human RT on the same features

<div class="grid grid-cols-[38fr_62fr] gap-8 items-center h-[calc(100%-3.5rem)]">
  <div class="space-y-3 text-sm">
    <div><span class="label">What</span> Compute <code>oracle_stop_step</code> on 5K lmcos training trees using the actual DP oracle. Extract board features from root FEN.</div>
    <div><span class="label">x</span> Board features: branching, material, gain_depth, toptwo</div>
    <div><span class="label">y</span> oracle_stop_step (DP-optimal number of expansions, budget=43)</div>
    <div><span class="label">Compare</span> r(feature, oracle_stop_step) vs r(feature, human log RT)</div>
    <div class="finding"><span class="label">Finding</span> 3 of 4 features match direction. gain_depth is the strongest oracle predictor by far (r = +0.797). toptwo is the one mismatch.</div>
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
        <td class="text-right pr-4 text-blue-600 font-bold">+0.165</td>
        <td class="text-right pr-4 text-blue-600 font-bold">+0.195</td>
        <td class="text-green-700 font-semibold">✓ match</td>
      </tr>
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 font-mono">material</td>
        <td class="text-right pr-4 text-blue-500 font-bold">+0.154</td>
        <td class="text-right pr-4 text-blue-400 font-bold">+0.039</td>
        <td class="text-green-700 font-semibold">✓ match</td>
      </tr>
      <tr class="border-b border-gray-100 bg-green-50">
        <td class="py-2 pr-4 font-mono">gain_depth</td>
        <td class="text-right pr-4 font-bold text-blue-700">+0.797</td>
        <td class="text-right pr-4 text-blue-500 font-bold">+0.096</td>
        <td class="text-green-700 font-semibold">✓ match — oracle much stronger</td>
      </tr>
      <tr class="bg-red-50">
        <td class="py-2 pr-4 font-mono">toptwo</td>
        <td class="text-right pr-4 text-blue-500 font-bold">+0.436</td>
        <td class="text-right pr-4 text-red-400 font-bold">−0.064</td>
        <td class="text-red-700 font-semibold">✗ mismatch — interpretable</td>
      </tr>
    </tbody>
  </table>
  <div class="mt-3 p-3 bg-neutral-soft rounded text-xs space-y-1">
    <p><b>toptwo mismatch:</b> oracle continues longer when toptwo is large (refining the dominant move's Q-estimate). Humans stop faster (decisiveness = satisficing signal).</p>
    <p><b>gain_depth gap (+0.797 vs +0.096):</b> oracle is defined to stop when Q-refinement no longer pays — so it's necessarily most sensitive to positions where Q is still changing. Humans track this signal, but weakly.</p>
  </div>
</div>

---

# Two stopping criteria — and which is closer to VOC

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="grid grid-cols-2 gap-5">
    <div class="p-4 bg-neutral-soft border-l-3 border-accent rounded">
      <b class="text-accent text-xs uppercase tracking-wider">min_expansions</b>
      <p class="text-sm italic mt-1">"If I keep thinking, my action won't change — so my realized reward can't change."</p>
      <ul class="text-xs list-disc pl-4 space-y-1 opacity-80 mt-2">
        <li>Stops when <b>action identity</b> stabilises</li>
        <li>Matches Russek et al. spirit (ΔUC = 0 when actions agree)</li>
        <li class="font-semibold text-accent">→ Theoretically correct for single decisions</li>
        <li class="text-red-600">But r(gain_depth) = −0.557 — opposite of humans</li>
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
    <b>The paradox:</b> min_expansions is the correct stopping criterion in theory — but both humans and the oracle follow oracle_stop_step in practice. Neither stops when the action has stabilised.
  </div>
</div>

---

# The epistemology of stopping

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <p class="opacity-80">At any step t, the agent faces three situations that are locally indistinguishable:</p>

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
      <div><b>Not converging.</b> Q plateaued — either genuinely ambiguous, or the better move hasn't been reached yet. <span class="text-amber-700 font-semibold">Unknown.</span></div>
    </div>
  </div>

  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded text-xs space-y-1">
    <p><b>Circular problem:</b> "would stopping now change my action?" requires knowing what further search would reveal — which requires completing the search.</p>
    <p><b>"Chasing tails":</b> in dominant positions, each expansion confirms the dominant move. Q keeps improving. Both humans and the oracle rationally continue — but the decision was already made. Systematic over-computation in dominant positions; under-computation in ambiguous ones.</p>
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section III</div>
    <h1 class="text-4xl">Next Steps</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">What each analysis would answer, and what from the current evidence motivates it</div>
  </div>
</div>

---

# What motivates the next analyses

<div class="mt-4 max-w-3xl space-y-3 text-sm">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent rounded">
    <b class="text-accent text-xs uppercase tracking-wider">Analysis 1 — Unify the datasets</b>
    <p class="mt-1 text-xs">Run Yotam's lmcos pipeline (Lc0 trees + DP oracle) on positions from the <b>human behavioral dataset</b> (processed_moves_nonzero: 10+0, ≥2000 Elo, ply 15–75). Compare oracle_stop_step with actual human RT at the <i>same positions</i>. <b>Why not use Yotam's existing trees?</b> His dataset (ELO 1800–2600, all time controls) carries different human confounds than our filtered behavioral data. Unifying removes this disanalogy and enables a direct position-level test of "does the oracle predict where humans think longer?"</p>
    <p class="mt-1 text-[11px] opacity-60 text-accent">Motivated by: A0a shows 3/4 features match — worth the SLURM cost to test at matched positions.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary rounded">
    <b class="text-secondary text-xs uppercase tracking-wider">Analysis 2 — Skip GNN pretraining</b>
    <p class="mt-1 text-xs">Train a small MLP from random initialisation on the existing packed episodes. 4 raw scalars already achieve 86.4% sign accuracy (vs 90.1% GNN+MC). If a tiny model from scratch trains in hours not days, iteration speed improves dramatically.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-amber-400 rounded">
    <b class="text-amber-700 text-xs uppercase tracking-wider">Analysis 3 — Weaker engine (conditional on A1)</b>
    <p class="mt-1 text-xs">If A1 shows no oracle-human correspondence, test Stockfish at ELO=2000 (matching the human player pool). Lc0 at 3000+ Elo trivially resolves positions a 2000-Elo human finds genuinely hard — engine strength may explain the mismatch.</p>
  </div>
</div>

---

# Summary

<div class="mt-4 max-w-3xl space-y-4 text-sm">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider">Key results</b>
    <ul class="mt-2 list-none space-y-1 text-xs">
      <li>✓ <b>log(RT) ∝ gain_depth</b> — r = +0.096 (n=1M): humans think longer when computation pays</li>
      <li>✓ <b>Branching strongest RT predictor</b> — r = +0.20: width drives deliberation more than depth</li>
      <li>✓ <b>oracle_stop_step matches human RT on 3/4 features</b> — directional signal from A0a</li>
      <li>✓ <b>4 scalars → 86.4% sign accuracy</b> (vs 90.1% GNN+MC, gap = 3.7pp) — from A0b</li>
      <li>⚠ <b>toptwo mismatch</b>: oracle and humans apply decisiveness differently</li>
      <li>⚠ <b>Chasing tails</b>: both over-compute in dominant positions, under-compute in ambiguous ones</li>
    </ul>
  </div>
  <div class="p-3 bg-amber-50 border-l-2 border-amber-400">
    <b class="text-amber-700 text-xs uppercase tracking-wider">Core open question</b>
    <p class="mt-1 text-xs text-amber-900">Does oracle_stop_step predict human RT at the <i>same positions</i>? Analysis 1 will answer this by unifying the datasets — the only remaining confound between the two halves of this project.</p>
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
