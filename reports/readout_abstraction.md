# The Unified Readout Abstraction — software architecture for stopping policies

**Ref:** `R-MC-READOUT` · [Index](reference.md)

## The Question

Why do we treat baseline stopping rules (Always-Stop, Never-Stop, Fraction-of-Budget) as hand-written Python functions while training the GNN-MC as a PyTorch neural network on a surrogate loss? 

This architectural asymmetry introduces structural opacity and evaluation risk. If the GNN-MC underperforms its own nested baselines, we must ensure they exist in the **same mathematical space** and are optimized and evaluated under the **exact same harness**.

We propose to unify all stopping policies under a single base class: the **`Readout`** abstraction. Every stopping rule becomes a parameterized readout head mapping the joint tree-budget state $\langle z, B \rangle$ to a scalar continue-advantage $A$.

---

## The Readout Formalism

Under this unified framework, a stopping policy is a function:
$$A = f(z, B)$$
where:
* $z \in \mathbb{R}^d$ is the tree representation (GNN root embedding or hand-crafted features).
* $B \in \mathbb{R}^k$ is the budget context (remaining budget $T_t$, starting budget, and current step index $t$).
* $A \in \mathbb{R}$ is the predicted continue-advantage.

The decision rule is identical for all models: **continue search if $A > 0$, halt at the first step where $A \le 0$**.

### The Readout Family

We represent all baseline and neural models as concrete subclasses of the `Readout` abstraction:

| Model / Readout Subclass | Input Scope | Parameter Space | Advantage Equation |
|---|---|---|---|
| **`AlwaysStopReadout`** | Discards $z$ and $B$ | None (constant) | $A(z, B) = -C_{\infty}$ (e.g., $-10^5$) |
| **`NeverStopReadout`** | Discards $z$ and $B$ | None (constant) | $A(z, B) = +C_{\infty}$ (e.g., $+10^5$) |
| **`FractionStopReadout`** | Uses $B$ only, discards $z$ | 1 scalar ($\theta \in [0, 1]$) | $A(z, B) = \theta \cdot B_{\text{total}} - t$ |
| **`GnnMetaControllerReadout`**| Uses both $z$ and $B$ | MLP weights | $A(z, B) = \text{MLP}([z, B])$ |

---

## Procedural Proposal: The Unified Readout Harness

We propose to implement this unified architecture and evaluate a range of model complexities using the same optimization loop.

### Phase 1: Base Class Implementation
1. Create `lmcos/src/models/readout.py` and define the `Readout(nn.Module)` abstract class.
2. Implement the four subclasses (`AlwaysStopReadout`, `NeverStopReadout`, `FractionStopReadout`, and `GnnMetaControllerReadout`).
3. Enforce the surrogate loss directly in the base class:
   $$\text{Loss} = \text{MSE}(A, A^*) + \text{sign\_weight} \cdot \text{BCE}(\sigma(A), \mathbb{1}_{\{A^* > 0\}})$$

### Phase 2: Optimization Alignment
1. **Fit the 1-Parameter Fraction Head via GD:** Train the `FractionStopReadout` using standard gradient descent (Adam) on the surrogate loss over the training cache, allowing the PyTorch parameter $\theta$ to converge.
2. **Fit the 1-Parameter Fraction Head via Direct Regret:** In parallel, fit $\theta$ by sweeping it over a fine grid to directly minimize empirical regret on the training set (matching the legacy `f*` selection).
3. **Compare Mismatch Cost:** Evaluate both fitted Fraction heads on the held-out validation set. The difference in their regrets is the **direct surrogate-to-regret mismatch cost** for a 1-parameter model.

### Phase 3: Complexity Escalation
Once the training loops are aligned, we will evaluate the model family across a spectrum of inputs to locate the exact complexity boundary:
1. **Budget-only MLP:** Feed only $B$ into a small MLP readout to represent a non-linear budget stopping boundary.
2. **Hand-crafted + Budget MLP:** Feed a few interpretable root features (e.g., policy entropy, Q-margin) along with $B$ into the MLP.
3. **Frozen GNN + Budget MLP:** Feed the full 128-dimensional GNN root embedding $z$ along with $B$.

---

## Expected Outcomes
* By training the `FractionStopReadout` via gradient descent, we isolate whether the surrogate loss is capable of finding the optimal threshold even in the simplest 1-D case.
* By placing all models behind a single PyTorch `forward` pass, we guarantee structural fairness in evaluation, removing any possibility of subtle evaluation leaks (harness differences).
* This provides a clean, modular plug-and-play seam to swap in different tree encoders $z$ (from hand-crafted to GNN) without altering the training or evaluation loops.
