# Meta-controller training signal — is advantage regression too lossy?

**Ref:** `R-MC-SIGNAL` · [Index](reference.md)

## The Question

Is the counterfactual advantage target $A = Q_{\mathrm{continue}} - Q_{\mathrm{halt}}$ too flat, and does MSE minimization act as a lossy filter that wipes out the subtle value signals from the tree representation $z$?

We hypothesize that because the difference in move quality between stopping at step $t$ and step $t+k$ is extremely small (frequently $< 0.01$ in win probability), the advantage signal is tiny. During training, standard MSE minimization treats these tiny advantages as noise. The network minimizes loss by defaulting to predict $A \approx 0$ (which triggers an immediate stop) and keys entirely off the clean, large-magnitude budget features.

---

## Procedural Proposal: Value Filtering and Regret Weighting

We propose two parallel experiments to test whether changing the dataset's value sensitivity or modifying the loss function to be regret-aware restores the utility of the tree representation $z$.

### Experiment A: Nontrivial Tree Filtering
If the network is swamped by "flat" episodes where stopping time doesn't matter, we can filter them out.
* **Procedure:**
  1. Characterize the cached episodes using the metrics in `episode_difficulty.py`.
  2. Filter the dataset to keep only "nontrivial" episodes where:
     - The **range of halt rewards** is high ($\max h_t - \min h_t > 0.10$).
     - The **regret of immediate halting** (stopping at step 0) is high, meaning deliberation is mandatory.
  3. Train both the budget-only floor and the full GNN-MC model on this filtered dataset.
  4. Compare their validation regret. If the GNN-MC achieves a significant lift over the budget-only floor on this subset, it proves that flat episodes were washing out the value signal.

### Experiment B: Regret-Weighted Advantage Loss
Instead of treating all advantage errors equally, we can penalize errors more severely when they lead to costly decision mistakes.
* **Procedure:**
  1. For each step $t$ in an episode, compute the **regret of making the wrong stopping decision** at that step:
     - If the optimal action is Continue ($A^* > 0$): the penalty of halting is $A^*$.
     - If the optimal action is Halt ($A^* \le 0$): the penalty of continuing is $-A^*$.
  2. Define a **Regret-Weighted Loss**:
     $$\text{Loss} = \sum_t |A^*_t| \cdot \left( (A_t - A^*_t)^2 + \text{sign\_weight} \cdot \text{BCE}(\sigma(A_t), \mathbb{1}_{\{A^*_t > 0\}}) \right)$$
  3. Train the GNN-MC model using this regret-weighted loss. This forces the gradients to focus on the high-regret decision boundaries rather than the low-regret flat regions.
  4. Evaluate whether this regret-weighted training allows the full model to outperform the budget-only floor.

---

## Expected Outcomes
* If **Theory 2 (Lossy Signal)** is correct, the GNN-MC will achieve a substantial regret reduction over the budget-only floor when trained on the filtered nontrivial subset (Experiment A) or when using the regret-weighted loss (Experiment B).
* This would prove that the GNN embedding $z$ does contain stopping utility, but standard MSE on flat value landscapes fails to align the network's parameters with the decision boundary.
