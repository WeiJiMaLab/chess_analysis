# Construal: effort as the complexity of the game you choose to play

**Ref:** `R-CONSTRUAL` · [Index](reference.md) · parents [(R-TREESEARCH)](treesearch.md) / [(R-PRUNING)](pruning.md)

> **Verdict (2026-06-30): ❌ the subgame collapses to the null.** The construal-rank proxy on 259K moves: `|S*|`
> (gut-rank of the winning piece) ↔ RT = **+0.095, partial|legal +0.025**; the value-based reward−cost stop
> `|S*|(c)`, swept c∈{0.02..0.4}, tops out at **partial|legal +0.027**. *Why it was forced:* with both sides frozen,
> the restricted search can't generate refutations by frozen pieces, so `regret(S) → 0` exactly when the best
> move's piece enters S — i.e. `|S*| ≡ s_star`. The expensive symmetric build would only **reproduce the proxy**;
> do not build it. This is the third generative "effort" account to collapse to decision-width (after VOC and
> pruning), which triangulates the real result: **effort is the breadth of the root decision, not the look-ahead.**
> Next: consolidate the satisficed-decision-difficulty finding, or pivot to a *perceptual/representational* cost
> (chunking; de Groot / Chase-Simon / Gobet) — not another search variant. See [[construal-framing]].

> **Status:** ❌ ruled out (proxy null). The framing that survives the **tree-readout paradox**: effort is not searching a tree
> you have already built — it is choosing **how much of the board to represent** *before* you search. Human
> think-time = the **complexity of the simplified game (construal)** you build to find a good-enough move. This is
> **value-guided construal** (Ho, Griffiths, and colleagues: people plan over deliberately simplified task
> representations). It is the first cost on the table whose unit is *not* a tree node — which is the only way it
> can dissociate from the legal-moves floor instead of re-describing it.

## Why construal? The cost-of-search paradox

Every tree-derived cost we tried — `n_total`, `n_frontier`, `max_depth`, the pruned variants — either **collapses
to legal-moves** (`n_frontier ≡ n_total`, 0.86-collinear) or **falls short of it** (depth −0.18, washes on pruned
trees), and raw pruned node-count washes to ≈0 ([(R-PRUNING)](pruning.md)). The deeper reason is a paradox: **if
you have built the search tree, you already know the best move** — there is nothing left to *pay* to read it out.
So the unit of effort was wrong. It is not "how big is the tree I expanded," it is **"how much of the problem did
I have to represent to decide."**

> **Decision:** model effort as **construal complexity** — the number of pieces brought into a simplified game —
> not as tree expansions. The full tree is the thing you are *avoiding*, not paying for.

## The model

A construal `S ⊆ {your pieces}` is the set of pieces you actually calculate with, grown in **criticality order**.

- `a_sub(S)` = the root move a **restricted search** (only S-pieces may move) prefers, its leaves scored by the
  **full-board evaluator**.
- **construal regret** `R(S) = V_full(a_full) − V_full(a_sub(S)) ≥ 0`, monotone ↓ to 0 as `S →` full board.
- **meta-MDP:** state = `S`; action = add the next-most-critical piece / **stop**; reward on stop =
  `V_full(a_sub(S)) − c·|S|`; stop when the marginal value of one more piece `< c`.
- **RT ∝ |S\*|**, the satisficing construal size.

> This is the same reward − cost meta-rational object as the budgeted oracle, but the **atom of computation is a
> piece, not a node** — so `|S\*|` can be small in a wide-but-obvious position and large in a quiet-but-subtle one,
> which raw move-count cannot.

## Freeze, not remove — and the evaluation problem (the crux)

**Freeze beats remove.** Removing the non-attended pieces produces **illegal FENs** (king left in check, impossible
configs) and **degenerate evals** (two same-colored bishops can never mate a wrong-colored king, so a 1-piece
subgame is meaningless). Freezing keeps a legal board.

**But freezing does not simplify the *value*:** an FEN-only evaluator still *sees* the frozen pieces, so it cannot
return the "as if those pieces were absent" value. Trying to make it do so is the wall.

> **Decision (resolves the crux):** **split perception from calculation.** The construal restricts which pieces
> you **calculate moves for** (the search action set), *not* what you perceive. The **leaf value stays the full
> holistic gut-eval** (SF on the true FEN). A human's gut judges the whole board at a glance; only deliberate
> calculation is selective. So we never need "as-if-absent" value — and the same-colored-bishop degeneracy
> dissolves, because the eval is always the real full-board eval.

> **Clarification:** this is a *weaker* construal than a literal subgame — you still **perceive** the whole board,
> you just do not **calculate** the frozen pieces' moves. That is the honest price of an FEN-only evaluator, and
> arguably the more realistic model of attention.

**A clean consequence.** With holistic leaf eval, `R(S) → 0` the moment `S` contains the piece behind a good move,
so **`|S\*| ≈ the criticality-rank of the best move's piece`** (refined by depth: a move that is only good because
of a follow-up needs the search). RT ∝ how far down your gut-ordered piece list the good move hides.

> **Design dimension (open):** restrict only your **root** move to S ("which piece do I move"), or your moves at
> **all plies** ("which pieces do I calculate with"). Start root-restricted + a few ply; tighten later. A known
> residual approximation: opponent threats from unattended pieces are invisible to the restricted search but *are*
> seen at the leaf eval.

## The criticality prior

`criticality(piece) = |Δ V_full if that piece is frozen/removed|`, computed by the **same evaluator** that scores
leaves — so there is **one evaluator**, doing double duty (the relevance prior and the value), no second
supervisor. Where the drop is illegal, fall back to a proxy ordering (piece mobility, involvement in the engine
PV, attack/defense maps).

> **Decision:** one SF evaluator computes both criticality (drop-deltas) and leaf value. Criticality orders the
> construal growth.

## Computation & storage

- **SF-n1** (the fast gut), **M ≈ 16–24** expansions per subgame — smaller trees are fine here.
- Per position, store a **sequence of subgame BeFS runs**, one per construal level `k = 1..K` (criticality-ordered)
  — *not* incremental subsets of a single tree.
- Worst case ≈ `#movable pieces` runs/position (≤ ~16), each `M ≈ 24` → roughly **4× the old per-position cost**;
  SF-n1 keeps it tractable. **Profile** on a few hundred FENs first, then generate; store alongside the existing
  trees (`sf_trees/construal/...`).

> **Decision:** one stored BeFS subgame-tree per construal level; cap `M` at ~24; SF-n1; profile-then-generate.

## Predictions & the test

- **RT ∝ |S\*|** where `R(S\*) <` threshold.
- Does `|S\*|` **beat legal-moves (+0.34)** *and* **survive partialling it**? The discriminating case: a
  high-legal-move position with an obvious-piece tactic → small `|S\*|` → fast, where move-count predicts slow.
- Does **`size − satisfaction + sharpness` emerge**? A clear best move ⇒ small construal ⇒ fast; no dominant move ⇒
  large construal ⇒ slow.
- **Throttling (the novel, sharp prediction):** the legal-moves effect should **saturate** — once `|S\*|` caps (you
  can only attend so many pieces), extra legal moves stop adding RT. Raw move-count cannot predict a plateau.

> **Open test (not a result):** whether construal size `|S\*|` is a genuinely *new* RT predictor (beats + survives
> legal-moves) or re-describes mobility. This is engine-derived and 0–1 parameter (the criticality prior + the
> cost/threshold `c`) — `2d`-style analysis, **no training**.

## Open risks

- **`|S\*|` vs legal-moves collinearity** — mobility ≈ active pieces ≈ legal-moves; must beat *and* survive partial.
- **The criticality prior is load-bearing** — wrong ordering ⇒ construal grows in the wrong order.
- **Eval-faithfulness residual** — unattended-piece threats only enter at the leaf (the perception/calculation split).
- **Cost** — profile before committing a full generation.
