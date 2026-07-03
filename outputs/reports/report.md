# When do people think? A search-model account of human deliberation time

**The retrospective paper.** This is the single top-to-bottom story the project tells, assembled
from the per-inquiry reports in [`reports/`](reports/reference.md). The didactic, click-driven
version is the deck [`presentations/src/tree-search.md`](presentations/src/tree-search.md); the
per-inquiry detail and methods live in the reports this paper points to. Here we keep one
narrative, in paper order, and say plainly what is **settled** and what is still **open**.

> **The one-line story.** Humans deliberate longest when the decision is *wide* — many legal moves to weigh.
> The number of legal moves is the strongest predictor of response time, but that is the **fact to explain, not a
> model**: a count has no normative content. So the question is not "what beats legal-moves" (we chased that and
> everything *correctly* collapsed onto it); it is **"is it resource-rational to spend effort proportional to your
> options?"** The result we are after is a resource-rational planner — one that pays per operation and stops when
> the marginal value no longer justifies the cost — that **reproduces** the `size − satisfaction` curves, i.e.
> *explains* the legal-moves effect rather than rivaling it.

The paper has **three sections**, matching the figure categories in `figures/{board,engine,normative}`:

1. **Board** — model-free board regressors. *Legal moves dominate; response time is log-normal.*
2. **Engine** — Stockfish value-of-computation signals. *They track RT only weakly, and every one collapses toward the move count.*
3. **Normative (TODO)** — the resource-rational planner that should *reproduce* the move-count effect. *Future work; one positive sign so far.*

---

## Board — how do people allocate thinking time?

Humans do not spend equal time on every move. Working **model-free** — no engine, just features
*of the position itself* — we ask which features predict how long a person thinks, on Lichess
**10+0** games (both players Elo ≥ 2000). Full inquiry: [(R-MOVETIME-BOARD)](reports/board.md).

First, the shape of the target. Log(response time) is approximately **normal** — response time is
**log-normal**, not exponential — so people scale thinking *multiplicatively* with difficulty.

![log response time — histogram + normal QQ](figures/board/rt_distribution.png)

> **Decision:** Work in **log(RT)** throughout (response time is log-normal, Weber's law), and screen
> structural features with **Spearman** rank correlation — they are skewed/bounded, so rank
> correlation is the honest measure.

Then the regressors. Each board feature gets the same dashboard (global + by game-stage tertile,
K=10 tie-safe quantile bins with per-bin SEM):

![legal moves vs response time](figures/board/legal_moves.png)
![ply vs response time](figures/board/ply.png)
![player clock vs response time](figures/board/clock.png)

| Feature (from the position) | ρ with log(RT) | Reading |
|---|---|---|
| **legal moves** | **+0.343** | more candidate moves → more to weigh |
| ply | **−0.248** | later in the game → faster (fewer pieces, more forced) |
| clock (time left) | **+0.122** | more time on the clock → more willing to spend it |

![Spearman correlation — board features](figures/board/board_feature_corr.png)

The legal-move count (ρ **+0.343**) is the strongest single board-feature tie to RT, and it is
**not** reducible to the ply/material/clock complex (which all move together as games progress).

> **Result:** The **width of the decision — the number of legal moves — is the strongest single
> predictor of human response time** (ρ +0.343), stronger than ply (−0.248) or clock (+0.122). This is
> the fact the rest of the paper has to explain.

---

## Engine — can an engine's value tell us when to think?

The legal-moves effect is suggestive but *structural* — it says nothing about whether thinking is
*worth it*. The resource-rational hypothesis is sharper: people should think longer where an engine
would *gain* more from searching. To test it we read value quantities off a **Stockfish search tree**
on the same position — the canonical **SF1** set (`n1md36`: Stockfish, n=1 leaf eval, depth-36, no
pruning; **n = 105,141** played moves matched). The analysis lives inline in
`/human/engine.py`.

The signals, in the user's terms:

- **Gain** — *value of computation*: how much value the full search finds beyond its own first guess.
- **MQ (move quality)** — *satisfaction of the played move*: `final_Q(played) − final_Q(best)` ≤ 0,
  the value the human actually left on the table.
- **Greedy action-gap** — *decisiveness*: top-1 minus top-2 of the root children's value, computed
  *myopically* (pre-search).
- **GSS / OSS** — *stop step*: when a greedy / cost-optimal stop locks onto its best move.
- **Greedy frac-good** — fraction of moves within ≤ 0.1 of the best, computed myopically.

![Gain vs RT](figures/engine/gain.png)
![MQ vs RT](figures/engine/mq.png)
![greedy action gap vs RT](figures/engine/action_gap.png)
![GSS vs RT](figures/engine/gss.png)
![greedy frac-good vs RT](figures/engine/frac_good.png)
![OSS vs RT](figures/engine/oss.png)

| Metric (SF1 `n1md36` tree) | ρ with log RT | Direction |
|---|---|---|
| **Gain** (value of computation) | **+0.140** | as predicted — more to gain, longer think |
| **MQ** (played-move quality) | **−0.172** | longer thinks land on *worse* moves |
| greedy action-gap (decisiveness) | **+0.061** | small/near-null |
| GSS (greedy stop step) | **+0.096** | as predicted |
| greedy frac-good (≤ 0.1 of best, myopic) | **−0.166** | more good moves → faster (satisficing) |
| OSS (optimal stop step) | **+0.120** | as predicted |

**Why these are all myopic (pre-search) signals.** The *deep* (post-search) action gap is a
post-search **outcome** confounded with think-time — a tree that searched longer has, by
construction, a more resolved gap — so the engine plots use the **pre-search myopic** values
throughout, which are the genuine inputs available *before* deciding how long to think. This is also
why **sharpness was dropped**: the deep action-gap that once looked like a "sharpness" signal was a
post-search artifact, and the myopic gap is ~null (+0.061).

> **Clarification (H(π) is degenerate).** Stockfish has no policy head — its prior is **uniform** — so
> the policy entropy H(π) ≡ log(#legal moves): it is the move count relabeled, not a real signal. Its
> apparent correlation dies completely once legal-moves is partialled out (partial | legal ≈ +0.002).
> We do not present it as an engine result.

Every magnitude here is faint (|ρ| ≲ 0.17), and the values are **stable** across depth/eval budget.
The dominant driver of human RT is the **move count**, not value — and indeed each engine signal
collapses toward it: the engine confirms *that* width drives RT but offers no value-based reason for
why a *count* should set response time.

![Spearman matrix — engine metrics, board structure, RT](figures/engine/correlation_matrix.png)

> **Result:** The engine's value-of-computation quantities track human RT only **weakly** (|ρ| ≲ 0.17),
> and every one collapses toward the **legal-move count**. The engine captures the *direction* of the
> resource-rational effects but offers no value-based account of the dominant driver. That sets up the
> normative question: legal-moves is the thing to *explain*, not a rival to beat.

---

## Normative (TODO) — does a resource-rational planner explain the move-count effect?

This section is a **placeholder** — `figures/normative/` is intentionally empty for now — but it
carries the paper's thesis, so we state the reclaimed framing precisely.

**The reframing.** The legal-move count is the **explanandum, not a floor to beat.** A count has no
normative content: "RT correlates with legal moves" is a fact in search of a mechanism, not a model.
The goal is therefore **not** to find a value signal that *beats* legal-moves in magnitude (we
chased that across VOC, pruning, and construal proxies, and everything *correctly* collapsed onto the
count). The goal is a **resource-rational planner** — one that pays a cost per operation and stops
when the marginal value of more search no longer justifies the cost — that **reproduces** the
`size − satisfaction` curves. Such a planner would *explain* the legal-moves effect (more options ⇒
more to consider ⇒ longer, satisficing down when many moves are good) rather than rival it. A bare
count can never be an explanation; a planner that *produces* the count from a cost-benefit stop can.

**The one positive sign we have.** A **no-prune node-cost stop** — a planner that simply pays per
expanded node and stops at the cost-adjusted value peak — already reproduces the two key signs:
**+size** (RT grows with the number of options) and **−satisfaction** (RT shrinks when more moves
are good). This is in `/analysis/normative_curves.py`. It is the first evidence that the
move-count effect *falls out of* a resource-rational stop rather than needing a separate explanation.

**Sharpness was dropped.** The "+sharpness" term from earlier framings rested on the *deep* (post-
search) action-gap, which the Engine section shows is a post-search artifact confounded with
think-time. The honest **myopic** gap is ~null (+0.061), so sharpness is no longer part of the target
decomposition; the curves to reproduce are `size − satisfaction`.

**Future work (the actual fit + plots).** What remains — and what the empty `figures/normative/`
will hold — is the model fit itself: take the resource-rational planner with sensible cost
parameters and show it **regenerates the `RT-vs-{n_moves, fraction_good}` curves** with the right
shape (the satisficing concavity: high-satisfaction positions plateau low, low-satisfaction ones
stay steep). Only then does the normative story earn its keep, turning "RT re-describes decision
width" into "a resource-rational planner *predicts* when people think." That is the next step, not a
result yet.

> **Result (so far):** A no-prune node-cost stop reproduces **+size** and **−satisfaction**
> (`/analysis/normative_curves.py`) — the move-count effect *falls out of* a
> resource-rational stop. The full model fit and the normative plots are **future work**.

---

## What is settled, and what is open

| Claim | Status | Evidence |
|---|---|---|
| Human think-time is log-normal; decision **width** (legal moves) is its strongest single predictor (ρ +0.343) | **Settled** | board regressors; survives ply (−0.248) / clock (+0.122) |
| Engine value-of-computation tracks RT only weakly, and every signal collapses toward the move count | **Settled** | SF1 `n1md36`: Gain +0.140, MQ −0.172, greedy gap +0.061, GSS +0.096, frac-good −0.166, OSS +0.120 |
| H(π) is **not** a real engine signal (uniform SF prior ⇒ H(π) ≡ log #legal-moves) | **Settled** | partial \| legal ≈ +0.002 |
| Sharpness is dropped — the deep action-gap was a post-search artifact; the myopic gap is ~null | **Settled** | engine plots use pre-search myopic values; myopic gap +0.061 |
| Searching for a signal that **beats** legal-moves | **Retired (wrong target)** | a count is the *explanandum*, not a rival model; everything correctly collapses onto it |
| A resource-rational (per-op cost) stop **reproduces** +size and −satisfaction | **Settled (one sign)** | `/analysis/normative_curves.py` |
| A resource-rational model **reproduces** the `size − satisfaction` *curves* with sensible cost params | **Open (the fit)** | future work; check RT-vs-{n_moves, fraction_good} |

> **The working conclusion.** RT tracks **decision width** with a **satisficing signature**, and that is the
> **fingerprint of resource-rational option-consideration** — *not* "people irrationally count moves," and *not* a
> phenomenon waiting for a signal that beats it. Legal-moves is what a cost-based planner *should* produce; the
> open work is the **fit** — does such a planner regenerate the RT curves with reasonable parameters? That is an
> *explanation* of the legal-moves effect, which a bare count can never be.

---

## Pointers (per-inquiry reports & methods)

This paper is a synthesis; each section's full methods, data lineage, and caveats live in its report
(cross-references are collected in the [index](reports/reference.md), not duplicated here):

- **Board** — [(R-MOVETIME-BOARD)](reports/board.md): board regressors, the log-normal RT, the width axis.
- **Engine** — the engine value signals (Gain / MQ / greedy gap / GSS / frac-good / OSS) on the SF1
  `n1md36` trees, computed **inline** by `/human/engine.py` (figures in `figures/engine/`).
  There is no standalone engine report; the analysis is folded in here.
- **Normative** — [(R-TREESEARCH)](reports/treesearch.md): the resource-rational framing and the
  reclaimed result (a per-operation-cost stop *reproduces* the curves). The pruning refinement is
  [(R-PRUNING)](reports/pruning.md). The positive `+size / −satisfaction` sign lives in
  `/analysis/normative_curves.py`.
- **Data** — [(R-DATA)](reports/reference.md#data-reference-r-data): the human Lichess dataset and the
  search-tree dataset.

All correlations are Spearman ρ with **percentile-bootstrap 95% CIs**; RT is always log(response time);
the click-driven walkthrough is [`presentations/src/tree-search.md`](presentations/src/tree-search.md).
