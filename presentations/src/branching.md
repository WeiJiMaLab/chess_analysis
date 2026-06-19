---
theme: default
title: Branching & resource-rational deliberation
info: Why decision width — not realized value-of-search — paces human think time.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search · proposal</span>

# Why does decision *width* drive deliberation?

<div class="mt-2 text-lg opacity-80">A resource-rational account of the branching effect on human think time.</div>

<div class="mt-4 text-sm opacity-50">Draft proposal (R-BRANCH) · branching ρ(log RT) ≈ +0.35, far above any value-search metric.</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">The phenomenon</span>Branching paces think time; value-of-search barely does</div>
      <div class="mdl-text">
        <ul>
          <li><strong>Branching</strong> — the number of legal candidate moves — is the strongest single predictor of log RT: <strong>ρ ≈ +0.35</strong>.</li>
          <li>Every engine value-search quantity is far weaker: Gain ≈ +0.07, GSS ≈ +0.12, MQ ≈ −0.15.</li>
          <li>A value-convergence oracle (stop once the value gap is decided) reproduces the value <em>directions</em> but <strong>never represents decision width</strong> — so it misses what actually paces humans.</li>
        </ul>
      </div>
    </div>
    <div class="mdl-figbox"><img src="../public/figures/legal_moves_vs_movetime.png" alt="branching vs move time" /></div>
  </div>
  <div class="mdl-found">Humans deliberate in proportion to the <strong>width of the decision</strong> (how many moves they must weigh), not to the <em>realized</em> value-of-computation. The paradox dissolves once "value of computation" is read as <em>expected</em> (ex-ante), not <em>realized</em> (ex-post).</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">The account</span>Deliberate to resolve <em>which move is best</em></div>

<div class="mdl-content">
  <div class="mdl-text">
    <p class="mdl-lead">Move choice as a <strong>metalevel decision problem</strong> (Russell &amp; Wefald; Hay; Lieder &amp; Griffiths): each expansion is a noisy observation that sharpens a belief over which move is best; keep computing while the <strong>expected</strong> value of computation exceeds its cost.</p>
    <ul>
      <li>The key quantity is <strong>prior uncertainty over the argmax</strong>. More plausible candidates (higher branching, flatter policy π) → larger uncertainty → <strong>expected VOC stays high</strong> → keep searching.</li>
      <li>As candidates are ruled out and the posterior over the argmax concentrates, expected VOC drops below cost → stop. So <strong>stopping time ∝ prior argmax-uncertainty ≈ policy entropy H(π) ≈ branching.</strong></li>
      <li><strong>Realized Gain is the outcome, not the driver</strong> — usually ≈ 0, because the prior-favored move survives. RT tracks <em>expected</em> VOC (uncertainty) while only weakly relating to <em>realized</em> VOC — exactly the dissociation in the data.</li>
      <li>A chess <strong>Hick's law</strong>: choice RT grows with the number of near-tied alternatives. (It also reframes "lc0 too strong": value collapses — |final Q| ≥ 0.9 in ~79% — yet the <em>argmax</em> among near-equal moves can stay wide.)</li>
    </ul>
  </div>
  <div class="mdl-found">◆ <strong>Proposal:</strong> replace "stop when the <em>value</em> is decided" with "stop when the <em>argmax</em> is decided." Optimal deliberation grows with the number / uncertainty of competing candidates.</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Mechanisms</span>How the tree could operate on branching</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide" style="margin-top:0.2rem">
    <thead>
      <tr><th>Mechanism</th><th>Where branching enters lc0 search</th></tr>
    </thead>
    <tbody>
      <tr class="hl"><td><strong>1. Width-gated stop</strong> (lc0-native)</td><td>Stop when the root <strong>visit distribution concentrates</strong> (top-move share &gt; θ, or visit entropy &lt; τ). Wide decisions start diffuse → take more expansions to concentrate. = "stop when the posterior over the argmax is peaked."</td></tr>
      <tr><td><strong>2. Candidate-coverage search</strong></td><td>Visit each plausible candidate ≥ k times before committing → effort ∝ effective branching. Branching sets the <em>width</em> of exploration; value-convergence sets the <em>depth</em>.</td></tr>
      <tr><td><strong>3. Expected-VOC / metalevel cost</strong></td><td>Make the continue-value depend on E[VOC] = P(an unevaluated move overturns the best), monotone in H(π) → the normative <em>target</em> rewards searching wide positions longer.</td></tr>
      <tr><td><strong>4. We already discard a width channel</strong></td><td>GSS (expansions-to-find-best) is partly branching-mediated — which is <em>why</em> GSS↔RT &gt; Gain↔RT. The value-convergence stop halts early and throws that width signal away.</td></tr>
    </tbody>
  </table>
  <div class="mdl-found">All four are the same principle — allocate computation to resolve <strong>argmax-uncertainty</strong> — differing only in where branching enters the search.</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Preliminary evidence</span>Branching is the scale; Gain modulates where lc0 is unsure</div>

<div class="mdl-content">
  <div class="grid grid-cols-2 gap-6 items-center" style="min-height:0">
    <div class="mdl-figbox"><img src="../public/figures/gss_vs_rt.png" alt="greedy stop step vs RT" /></div>
    <div class="mdl-figbox"><img src="../public/figures/gain_vs_rt.png" alt="Gain vs RT" /></div>
  </div>
  <div class="mdl-text" style="margin-top:0.4rem">
    <ul>
      <li><strong>Branching dominates:</strong> alone it explains <strong>R² ≈ 0.12</strong> of log RT; both tree metrics <em>together</em> add ΔR² ≤ 0.01 — the width channel is the story.</li>
      <li><strong>GSS = branching-mediated effort</strong> (left): most collinear with branching (ρ +0.19), wins the <em>linear</em> race (Pearson +0.114 vs +0.073), unique signal in <em>decisive</em> positions ("effort to confirm the obvious best").</li>
      <li><strong>Gain = the orthogonal value-of-search signal</strong> (right): zero in ~60% of positions, so Pearson understates it; on <strong>ranks</strong> it's stronger (Spearman +0.138 vs +0.123) and adds more unique rank variance over branching. Concentrated in <strong>non-decisive</strong> positions (~21%) — there Gain ≈ +0.13 while GSS ≈ 0.</li>
    </ul>
  </div>
  <div class="mdl-found">Branching sets the <strong>scale</strong> of deliberation; realized Gain <strong>modulates</strong> it only where the engine is genuinely uncertain — the resource-rational picture, on a first pass over 112.8K FENs.</div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Predictions</span>What would confirm the account?</div>

<div class="mdl-content">
  <table class="mdl-table mdl-table--wide">
    <thead>
      <tr><th>Prediction</th><th>Test</th></tr>
    </thead>
    <tbody>
      <tr class="hl"><td><strong>P1</strong> · entropy ≈ branching</td><td>Root policy entropy H(π) predicts log RT about as well as raw branching (ρ ≈ +0.3+), better than any value metric. <em>Cheap — the prior is already in <code>node_features</code>.</em></td></tr>
      <tr class="hl"><td><strong>P2</strong> · width-gated &gt; value-convergence</td><td>A halt-on-visit-concentration rule predicts human RT better than the value-convergence oracle. Slots into the budgeted-baselines harness.</td></tr>
      <tr><td><strong>P3</strong> · expected ≫ realized VOC</td><td>An ex-ante argmax-entropy signal dominates realized Gain and adds unique variance over branching.</td></tr>
      <tr><td><strong>P4</strong> · scale vs modulation</td><td>Within fixed branching, residual RT tracks Gain/value — branching sets the scale, value-of-computation modulates it.</td></tr>
      <tr><td><strong>P5</strong> · Hick form</td><td>Does RT scale with <code>branching</code> (linear, "enumerate moves") or <code>log(branching)</code> / H(π) (entropy, "evidence accumulation")?</td></tr>
    </tbody>
  </table>
  <div class="mdl-found">The hypothesis: the <em>expected</em>-VOC / entropy signal — not branching per se — is the true driver, with branching its observable proxy. <span class="opacity-50">Draft proposal; P1–P5 not yet run.</span></div>
</div>
