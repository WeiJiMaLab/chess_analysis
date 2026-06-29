# When ought one to think? A meta-rational account of chess deliberation time

**Ref:** `R-TREESEARCH` · [Index](reference.md)

> **Status:** 📝 active. The **model paper** — it **picks up where [(R-VOC)](voc.md) leaves off**. R-VOC showed
> that value-of-computation (in every form) is a legal-moves *proxy* and the real structure is *satisficed
> decision difficulty*; here we formalize that as one optimization — deliberation time as the *optimal amount of
> computation under a cost* — and lay out the fit-to-RT program. See R-VOC for the full empirical journey +
> figures (this report owns the **model + all forward proposals**). Powered on filtered elo2000 (n≈65k moves,
> bootstrap 95% CIs); the decisive *fit-cost-to-RT on a construal substrate* run is the open frontier.
> Calibration detail in [(R-HALT-CALIB)](halt_calibration.md); data in [(R-DATA)](reference.md).

## 1 · The claim, as one optimization

**Hypothesis:** human deliberation time is the *optimal amount of computation* a meta-rational agent would
spend, given a cost. Formally, a **meta-level MDP**:

- **state** = the consideration set / search built so far;
- **actions** = *include the next-most-promising move* (grow the construal) **or** *stop and play current best*;
- **reward** = on stop, the supervisor value `V_sup(chosen)`; each inclusion costs `c`;
- **objective** = `max  E[V_sup(chosen)] − c·(#inclusions)` — i.e. **reward − cost**, which is exactly **−regret**;
- **optimal policy** = grow while marginal VOC > `c`, stop otherwise (= **satisfice**); solvable by backward-DP
  **or** learnable by RL.

**Free parameters: ~1–2** — the cost `c` and a prior (which move to consider next / how sharply to prune); the
value supervisor is given by the engine. **Prediction:** RT ∝ the number of computations the meta-optimal
policy makes. **The test that makes this an *account* and not a *description*:** fit `c` (+prior) to human RT,
and ask **how much of the RT structure a 1–2-parameter meta-RL reproduces vs the descriptive ceiling.**

> **Current answer (63k):** the 1-parameter meta-RL (DP-optimal stop, best cost) tracks RT at **ρ≈+0.16**; the
> descriptive ceiling (legal-moves / satisfaction) is **±0.31**. So on the current substrate the rational model
> explains **about half** — and closing that gap (or proving it can't be closed) is the whole game. §2 is what
> the model must reproduce; §3 is its parameters; §4 is how we fit it.

## 1b · The architecture, and two questions about it

Three knobs are easy to conflate — keep them separate:

| knob | role | changes the values? |
|---|---|---|
| **`UCI_Elo`** | a *play* handicap | **no-op** for us — we read SF's *eval/WDL*, not its played move ([[sf-uci-elo-noop-for-eval]]) |
| **N = `sf_search_limit_nodes`** | the **leaf evaluator** (each node valued by an N-node SF search; N=1 ≈ static head, N=100 ≈ shallow search) — the **heuristic / "gut"** | **yes** (n1 vs n100 differ per-FEN, ρ≈0.93) |
| **M = `search_budget` = 96** | the **MCTS/PUCT expansions** built *on* the N-heuristic — the **planning** | the thing whose *trace* we analyze |

The "thinking" we model as human-like is the **M-trace** (1→96 expansions, `oracle_root_q_trace`), bootstrapped
on the N-heuristic. *(An earlier framing wrongly used n1→n100 as the prior→truth axis — that varies the
heuristic N, not the planning M.)*

### Q1 — Are we already doing best-first search? (yes — with a UCT selector)

`build_tree` does **no rollouts**: each node is valued by the SF heuristic, and expansions are chosen by
**PUCT**. With our **uniform priors**, PUCT reduces to **UCB** — exploit `Q` + a visit-count exploration bonus.
So we are *already* running a **heuristic best-first tree search whose selector is UCT rather than greedy-argmax**,
and the exploration bonus makes it **breadth-leaning** (it expands many root moves early before deepening). So
the construal need not be a *new* generator: the existing expansion trace **is** an incremental, heuristic
best-first inclusion. The real levers are the **selector** (UCT exploration vs optimism/argmax) and a **per-move
cost + satisficing stop** — not "MCTS vs BeFS" wholesale.

> **Decision:** first read the construal off the *existing* expansion trace (heuristic best-first inclusion
> under UCT) + impose H1's per-move cost and a satisficing stop, before building a bespoke BeFS.

### Q2 — Does the tree search actually add value? (the missing check)

Planning is worth modelling only if it **changes the prognosis**. We have been varying the *prior* (N, the
dumb-eval) assuming *it* is the fault while holding the tree-building fixed — but the search itself could be
adding little, in which case the M-trace is ~flat and there is nothing for any VOC/cost to capture.
**Provisional probe (n≈3k):** heuristic↔planned root-value rank-corr ≈ **0.33**, and the root argmax keeps
changing until ~step **78 of 95** — so the search **reorders the policy heavily** (planning seems to add a lot,
not little). But the probe is artifact-prone (the early trace is nearly *tied* — moves are undifferentiated
until search separates them), so it needs a **clean 1-ply-heuristic-vs-M-converged measurement**, properly
aligned, before the magnitudes are trustworthy.

> **Decision:** before more prior/cost tuning, **measure the tree's value-add directly** — the decision-flip
> rate and value-movement from the 1-ply heuristic to the M-converged policy. Large ⇒ the search is sound and
> the prior/cost framing holds; small ⇒ the **tree-building is the weak link**, and the lever is the *search*
> (selector/expansion), not the prior. *(There are two failure modes — a bad **prior** (N) and a weak
> **search** (M); we have only stress-tested the prior.)*

## 2 · The targets the model must reproduce

What predicts human RT (n≈65k, bootstrap CIs):

![What predicts human RT](../figures/lmcos_tiny/rt_headline.png)

- **size +0.31** (# legal moves) and **satisfaction −0.31** (fraction of near-best moves) — two dominant,
  equal-and-opposite drivers; **sharpness +0.24** (action gap).
- the **value-of-computation family** (regret/Gain, step\*, softmax-VOC) sits at **+0.12–0.16**; the **causal
  halter ≈ 0**.

And the VOC family is not even *independent* — partial out legal-moves and every one collapses to a small
residual (the proxy result):

![VOC signals are legal-moves proxies](../figures/lmcos_tiny/rt_partials.png)

The signed structure is a **sign flip** — more options slower, more *good* options faster:

![The sign flip](../figures/lmcos_tiny/good_moves_signflip.png)

And — the decisive one for the normative claim — **satisfaction reshapes the entire RT-vs-n curve**, exactly as
a satisficing stop predicts:

![Plateau shift by satisfaction](../figures/lmcos_tiny/rt_vs_n_concavity.png)

High-satisfaction positions **plateau low** (you stop early no matter how many legal moves); low-satisfaction
positions are **steep** (`n` drives you). The aggregate "RT ≈ linear in `n`" *hides* this family of
satisfaction-conditioned curves — some saturating, some not.

> **Result:** RT = **satisficed decision difficulty** (size − satisfaction + sharpness). The plateau-shift
> shows the satisficing structure is *real*, not bare problem-size (Hick's law) — which is what makes a
> meta-rational account viable rather than merely descriptive. These four signatures are what a correct
> 1–2-parameter meta-RL must *generate from its parameters*, not fit feature-by-feature.

## 3 · The model's parameters — and where each is mis-specified (H1–H6)

Every hypothesis is a **parameter or substrate of the meta-MDP**, in two groups.

**(A) The construal — actions, cost, process** (where the ±0.31 lives):

- **H1 — the cost `c`.** Currently there is *no* per-inclusion cost (the oracle pre-includes every move free),
  so no size effect can arise. `c` is the central parameter — it sets the *saturation* of consideration-set
  size (`S* ≈ min(n, S_unconstrained)`), not the sign.
- **H4 — the process.** Construct-then-satisfice with a prior ordering; the oracle only *refines* a given tree,
  it never *constructs* the set.
- **H3 — the implementation.** Optimistic Best-First-Search **is** the incremental construal (grow breadth, pay
  per inclusion, stop when marginal VOC < `c`); its **expansion-count is the RT prediction**. The current
  substrate spends budget on **depth**, not **breadth** — *the* reason the meta-RL underfits (+0.16).
- **H6 — the prior strength.** How hard the prior prunes sets the set size (and saturation). A weak prior (our
  ≥2000 Lichess pool — club-level, ratings high vs FIDE, time-pressured) ⇒ consider ~all moves ⇒ RT linear in
  `n` (the unsaturated regime we see). Predicts **more saturation at higher player strength** *(data-gapped:
  `personal.db` keeps only the ≥2000 filter, not per-player Elo)*.

The cost sweep (below) shows the current-substrate behaviour: regret is **flat** across all 12 cost regimes
(it's a monotone transform of Gain), while step\* *responds* to cost — but tops out at the same ceiling.

![Normative signals vs RT, by cost regime](../figures/lmcos_tiny/cost_sweep_rt.png)

The uncertainty knob (softmax temperature, the prior's sharpness) is similar — calibrated τ≈0.1 recovers the
argmax value but does not exceed it:

![Softmax-VOC vs RT by temperature](../figures/lmcos_tiny/voc_tau_sweep.png)

**(B) The values — the supervisor:**

- **H2 — engine strength.** `V_sup` must be *human-relevant*: the sign-flip is a **litmus**. **Update
  (2026-06-29):** the `UCI_Elo` ladder is a **no-op** — SF-1350 and SF-2000 trees are *bit-identical* per FEN
  (max|Δ|=0, ρ=1), because `StockfishDirectEvalProvider` returns the static eval/WDL and `UCI_Elo` only weakens
  *play*, not the eval [[sf-uci-elo-noop-for-eval]]. So strength must be varied via the **search**
  (`sf_search_limit_nodes`/depth), not `UCI_Elo` — which is exactly the dumb-eval run below (and folds H2 into
  H3). The elo1350 rung is a wasted duplicate (cleanup candidate).

**(H5 — hindsight asymmetry: rejected** — the causal halter ≈ the hindsight oracle on RT.)

> **Clarification:** "weaken the generator" (H3) is one *parameter* of the optimization, not a rival theory.
> And **H1 (low cost) and H6 (weak prior) are the same knob** — both make the set ≈ all-`n` (the linear regime);
> they separate only by varying cost directly (the breadth ladder) or player strength (data-gapped).

## 4 · Fitting it: the program

We already have the meta-RL machinery — the budgeted-oracle DP **is** the meta-optimal stop for a given `c`,
and the **PG controller is a learned meta-policy** maximizing `E[reward − cost]`:

![Fitted controllers — regret vs compute](../figures/lmcos_tiny/regret_vs_compute.png)

The one thing we have **not** done: fit `c` (+prior) to **human RT** — we fit it to the agent's own regret, then
*checked* against RT. The decisive runs:

| # | experiment | what it fits / tests | status |
|---|---|---|---|
| **P0** | 63k powered re-run | every §2 signature, tight CIs | ✅ done |
| **P-FIT** | **fit `c`(+prior) to RT on the construal (incremental-inclusion) substrate** | the headline: meta-RL variance explained vs the ±0.31 ceiling, and whether the §2 signatures fall out of 1–2 params | **the frontier** |
| **P5** | SF **MultiPV** (`n_pvs`) sweep — breadth-value vs RT | the construal/saturation claim | needs a **prior** (caveat ↓) |
| **P2** | **dumb-eval** gen: `sf_search_limit_nodes` 100→1 → `elo2000_n1` | does a dumber *search* recover width↔RT (H2 folds in + H3) | **running overnight** (`10399953`→`55`) |
| ~~P1~~ | ~~SF-1350 vs SF-2000 (UCI_Elo)~~ | **RETIRED** — `UCI_Elo` no-op; 1350≡2000 bit-identical [[sf-uci-elo-noop-for-eval]] | retired |
| **P3** | optimistic **BeFS** generator | RT ∝ expansion-count (the construal, H3) | gated on P2/P5 |
| **P4** | lc0 **policy-prior** hybrid | prior-focused consideration (H4) | gated on P3 |

**The breadth ladder** (how P5→P3 build up, cheap → faithful; unit cost = `n_nodes`):
1. **MultiPV (`n_pvs`)** — SF's top-`k` lines; marginal value of the k-th = breadth-VOC; `k*` where it drops
   below cost. Tests the core claim off-the-shelf. *(Caveat: each PV is deeply searched.)*
2. **Fixed-width beam** (beam=1…K) — adds *lookahead* to breadth; a discretized BeFS.
3. **Incremental optimistic BeFS** — adaptive width grown to satisficing; the full construal.

*(Granularity is a secondary axis: the consideration unit could be **moves** or **pieces** — the latter closer
to perceptual "looking at a piece," cheap to probe by aggregating moves by source piece.)*

> **Caveat — P5/P-FIT need a *prior*, not just `final_Q`.** Breadth-VOC over the *true* converged values is
> degenerate: the best move is always considered first, so nothing can flip. The construal model is non-trivial
> only under **uncertainty** — a noisy prior over move values (you *discover* values by including, paying `c`),
> so a later candidate can overtake your current best. So P-FIT/P5 require a small uncertainty/prior model
> (noise σ, cost `c`) — light to code, but **not** parameter-free, and they make **H4's policy-prior
> essential** rather than optional.

> **Decision:** the **headline number is the variance a 1–2-parameter meta-RL explains vs the ±0.31
> descriptive ceiling** — *if it reaches it with the §2 signatures emerging, we have a parsimonious
> meta-rational account; if it stalls at +0.16, the gap is the genuinely non-rational residual.* P5 (MultiPV)
> is the cheapest probe; **P-FIT is definitive.** P3/P4 re-open the strength-ladder/faithfulness contracts
> ([(R-DATA)](reference.md)) — explicit v2-generator work.

---

## Appendix

**Softmax-VOC model.** Policy = `softmax(v_t/τ)`; halt value = `E_softmax[V_sup]` (closed form, no sampling);
benefit of thinking = the sharpening = uncertainty reduction. Supervisor = deep-tree `final_Q` (free) or an
independent stronger Stockfish (→ folded into the strength litmus, P1). τ=1 degenerate, τ≈0.1 best.

**How the data were made (brief).** SF-2000 (`UCI_Elo`) search trees via `build_tree` — `search_budget=96`,
`max_depth=4`, leaf eval `sf_search_limit_nodes=100`, uniform priors; 250k roots sampled from the 110M human
pool; **PUCT-stable ∧ value-monotone** filter (≈24.6% kept → ~61k). step\* = `argmax_s[V(s) − cost(s)]`. OSS
distribution (note the power-law's late mode is truncated by the forced-choice cap):

![OSS distribution by cost shape](../figures/lmcos_tiny/oss_dist_elo2000.png)

**Fitted-RL detail** (regret by model — tree-stats PG ≻ GNN-z ≻ blind; [(R-LMCOS-TINY)](lmcos_tiny.md)):

![Regret by readout model](../figures/lmcos_tiny/regret_by_model.png)

**Figure index** (all `figures/lmcos_tiny/`, n≈65k unless noted): `rt_headline`, `rt_partials`,
`good_moves_signflip`, `rt_vs_n_concavity` (§2) · `cost_sweep_rt`, `voc_tau_sweep` (§3) · `regret_vs_compute`
(§4) · `oss_dist_elo2000`, `regret_by_model` (appendix). One visual language (horizontal bars, ρ on x, names on
y; blue=size, red=satisfaction, grey=value-of-computation, green dashed=legal-moves reference), produced by
`lmcos_tiny/analysis/make_rt_figures.py`. Method: Spearman with percentile-bootstrap 95% CIs
([[bootstrap-cis-always]]); partials via the rank formula; metric pre-committed (human log-RT).
