# Meta-controller cost structure — does convex cost force budget-dominance?

**Ref:** `R-MC-COST` · [Index](reference.md)

## The Question

Why does the neural meta-controller fail to outperform a simple budget-only floor, and why does budget information appear to be the only signal the controller utilizes?

We hypothesize that the **convex power-law time-cost structure** currently used in the budgeted oracle ($p=2.8$, $\lambda=18.537$) acts as a dominant mathematical ceiling. Because the cost of continuing search grows exponentially as the remaining budget shrinks:
$$c_{\mathrm{time}}(T) = \lambda\Big[ (T-1+\tau)^{-(p-1)} - (T+\tau)^{-(p-1)} \Big]$$
the agent is forced to stop early purely due to the steep cost curve. This makes the optimal stopping boundary heavily dominated by the budget countdown, drowning out any subtle value signals from the tree state $z$.

---

## Procedural Proposal: The Cost-Scale and Linear-Cost Sweep

We propose to isolate and test this hypothesis by running two parallel, low-cost offline experiments on the cached tree datasets.

### Phase 1: Software Restructure (The Unified `Readout` Harness)
We will implement the unified `Readout` harness (as detailed in `R-MC-READOUT`) so that the baseline and GNN-MC models are evaluated on the exact same dataset and split.

### Phase 2: Offline Cost-Model Experiments

#### Experiment A: The Constant/Linear Cost Oracle
We will test if GNN features $z$ become predictive when the cost of search is decoupled from the remaining budget.
* **Procedure:**
  1. Regenerate the DP oracle labels for the cached tree dataset using a **flat, linear continue cost** ($c_{\mathrm{time}}(T) = \bar{c}$) instead of the power-law cost, where the cost of each expansion is constant.
  2. Train the following `Readout` models:
     - `FractionStopReadout` (budget-only floor)
     - `GnnMetaControllerReadout` (full GNN-MC model)
  3. Compare their validation regret. If the full GNN-MC significantly outperforms the budget-only floor under a linear cost, it proves that the convex cost structure was the primary cause of budget-dominance.

#### Experiment B: The Cost-Scale ($\lambda$) Sweep
We will locate the transition point where value-based stopping decisions emerge as the cost penalty becomes less restrictive.
* **Procedure:**
  1. Sweep the scale parameter $\lambda$ of the power-law cost down towards zero (e.g., $\lambda \in [0.1, 1.0, 5.0, 18.537]$).
  2. For each $\lambda$, regenerate the DP oracle and train both the budget-only and full GNN-MC models.
  3. Plot the **value of the root state** (the regret gap: $\text{Regret}_{\text{budget-only}} - \text{Regret}_{\text{full}}$) as a function of $\lambda$. We expect a transition threshold where value-based stopping becomes prominent as $\lambda$ decreases.

---

## Expected Outcomes
* If **Theory 1 (Restrictive Cost)** is correct, the GNN-MC will achieve a substantial regret reduction over the budget-only floor under a linear cost or low-$\lambda$ regime, demonstrating that the representation $z$ does carry stopping utility when not suppressed by the environment's cost dynamics.
* If the GNN-MC still fails to beat the floor even under a zero or linear cost, it points directly to **Theory 2 (Lossy Signal)** or **Theory 3 (Low Tree Value)**.
