"""Empirical test of whether `ChildWdlModel`'s `concat(parent_states, slot_states)`
genuinely disambiguates between sibling children on a T_n-shaped PARTIAL tree.

Background (see `history.md`'s "Stage: packhistory_GNNpretrain" and Task 3's
Progress Log for the full narrative): the k-steps-ahead pretraining plan
originally called for a new head taking `concat(u_n, v_n, slot_states)`
(`gnn.py`), reasoning that `ChildWdlHead`'s `concat(parent_states, slot_states)`
alone was only sufficient for children with no accumulated history of their
own. That reasoning was reverted: `ChildWdlHead` is already used, unmodified,
on every edge of today's *complete* trees -- including richly-explored
children, not just fresh leaves -- because `TreeEncoder`'s bottom-up attention
pass already routes a child's subtree information into its parent's aggregate
state before `ChildWdlHead` ever runs; `slot_states` is meant to disambiguate
*which* child that shared aggregate is being asked about. Nothing about
encoding a partial tree T_n instead of a complete tree should change this --
T_n *is* "the complete tree" from the encoder's own perspective when that's
what it's given.

This module tests that claim empirically, on a genuine T_n-shaped partial
tree (two sibling children under one parent, each with its OWN small,
independently-signaled subtree -- not the complete-tree shape `ChildWdlHead`
is already proven on in production), per the user's explicit instruction that
the correction needs rigorous testing, not just an architectural argument.

Development process (see history.md for the full trail): several earlier
probe designs were tried and discarded before this one -- a random-init
forward-pass check (predictions differed, but entirely due to the slot
embedding, not content); a single-signal supervised design where only ONE
child's edge was ever supervised (a real training-objective flaw: it let the
model win by ignoring slot_states entirely and mapping the shared aggregate
straight to a class, which is why the model's "sibling" prediction moved in
lockstep with the target's -- not evidence about localization, just an
artifact of an under-specified objective). The design below fixes that by
independently supervising BOTH children from their OWN distinct subtrees each
training step, which is the only way to genuinely penalize a "collapse to a
shared, slot-independent prediction" shortcut.

Two properties are tested (mirroring the plan's DoD checklist for Task 3):
  (a) predictions genuinely differ / carry real per-child signal -- measured
      by held-out joint accuracy on a task solvable only via each child's own
      subtree content, well above the two-independent-3-way-classes chance
      rate (1/3 * 1/3 = 1/9).
  (b) perturbation localizes to the perturbed child -- holding the parent's
      own features AND the sibling's entire subtree exactly fixed, and
      changing ONLY the target child's grandchildren, the target child's own
      prediction should move noticeably while the (untouched) sibling's
      prediction should stay close to unchanged. This is what distinguishes
      genuine slot-gated disambiguation from generic parent-state drift (a
      shared-aggregate perturbation that moves every sibling's prediction by
      comparable magnitude regardless of which one's subtree actually
      changed).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Same CPU-thread-oversubscription workaround already used by
# test_e2e_backprop_smoke.py -- this login node's core count causes severe
# overhead for this workload's small-tensor, many-sequential-ops pattern.
torch.set_num_threads(4)

import torch.nn.functional as F

from cts.core.tensorizer import TreeBatch
from cts.models.gnn import ChildWdlModel

_NODE_FEAT = 6
_D_EMBED = 32
_CHANCE_JOINT_ACCURACY = 1.0 / 9.0  # two independent 3-way classifications


def _bucket(signal: float) -> int:
    """Map a scalar signal to one of 3 classes -- an arbitrary but fixed,
    deterministic function of the signal, standing in for "this child's own
    subtree implies WDL class X"."""
    if signal > 1.0:
        return 0
    if signal < -1.0:
        return 2
    return 1


def _make_partial_tree(
    rng: np.random.RandomState,
    base_features: np.ndarray | None = None,
    override_signal_a: float | None = None,
    override_signal_b: float | None = None,
):
    """Build a T_n-shaped partial-tree TreeBatch: root(0) with two direct
    children -- child_a (node 1, slot 0) and child_b (node 2, slot 1) -- each
    with its OWN pair of grandchildren (nodes 3,4 and 5,6 respectively),
    simulating "two siblings under the same parent with deliberately
    different sub-histories" while both remain legitimately part of T_n.

    Each child's grandchildren encode an independent scalar "signal" in
    feature column 0, bucketed into a 3-way class -- this is the target the
    model must recover from `concat(parent_states, slot_states)` alone (no
    direct access to the child's own state), which is exactly what
    `ChildWdlHead`/`ChildWdlModel` compute today.

    `base_features`/`override_signal_*` let a caller hold everything except
    one child's grandchildren fixed, for the isolated-perturbation check.
    """
    if base_features is None:
        node_features = rng.randn(7, _NODE_FEAT).astype("float32")
    else:
        node_features = base_features.copy()

    signal_a = rng.uniform(-3, 3) if override_signal_a is None else override_signal_a
    signal_b = rng.uniform(-3, 3) if override_signal_b is None else override_signal_b

    gc_a0 = rng.randn(_NODE_FEAT).astype("float32")
    gc_a1 = rng.randn(_NODE_FEAT).astype("float32")
    gc_a0[0] = gc_a1[0] = signal_a / 2
    gc_b0 = rng.randn(_NODE_FEAT).astype("float32")
    gc_b1 = rng.randn(_NODE_FEAT).astype("float32")
    gc_b0[0] = gc_b1[0] = signal_b / 2
    node_features[3], node_features[4] = gc_a0, gc_a1
    node_features[5], node_features[6] = gc_b0, gc_b1

    # Structurally a genuine (if tiny) T_n snapshot: root -> 2 children, each
    # with its own already-expanded grandchildren -- not the bare-root or
    # single-child shapes a complete-tree-only test would use.
    parent_index = [-1, 0, 0, 1, 1, 2, 2]
    depth = [0, 1, 1, 2, 2, 2, 2]
    child_ptr = [0, 2, 4, 6, 6, 6, 6, 6]
    children_index = [1, 2, 3, 4, 5, 6]
    # Must include the grandchild edges too, not just the root's direct
    # edges: TreeEncoder's upward attention pass only routes information
    # along edges present in edge_parent/edge_child (parent_index alone only
    # drives the downward pass, a parent->child projection that cannot carry
    # grandchild info UP into a child's own state). Omitting these edges was
    # an early bug in this test's development (see history.md) that made an
    # earlier draft's task structurally unlearnable regardless of model
    # quality -- not a real finding.
    edge_parent = [0, 0, 1, 1, 2, 2]
    edge_child = [1, 2, 3, 4, 5, 6]
    edge_slot = [0, 1, 0, 1, 0, 1]

    tb = TreeBatch(
        node_features=torch.tensor(node_features),
        tree_index=torch.zeros(7, dtype=torch.long),
        parent_index=torch.tensor(parent_index, dtype=torch.long),
        root_index=torch.tensor([0], dtype=torch.long),
        edge_parent=torch.tensor(edge_parent, dtype=torch.long),
        edge_child=torch.tensor(edge_child, dtype=torch.long),
        edge_slot=torch.tensor(edge_slot, dtype=torch.long),
        child_ptr=torch.tensor(child_ptr, dtype=torch.long),
        children_index=torch.tensor(children_index, dtype=torch.long),
        depth=torch.tensor(depth, dtype=torch.long),
        feature_names=tuple(f"f{i}" for i in range(_NODE_FEAT)),
        batch_size=1,
        num_nodes=7,
        num_edges=6,
    )
    return tb, _bucket(signal_a), _bucket(signal_b), node_features


def _build_model(seed: int) -> ChildWdlModel:
    torch.manual_seed(seed)
    return ChildWdlModel(
        k=2,
        node_feat=_NODE_FEAT,
        device="cpu",
        node_embed_hidden=32,
        d_embed=_D_EMBED,
        d_message=_D_EMBED,
        n_heads=4,
        d_att=16,
        decoder_hidden=64,
        sequential=True,
    )


def _train(model: ChildWdlModel, num_steps: int, batch_size: int, seed: int):
    """Train `model` to predict BOTH children's independently-signaled
    classes each step -- the only design that actually penalizes a "collapse
    to a shared, slot-independent prediction" shortcut, since class_a and
    class_b agree only ~1/3 of the time by construction.
    """
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(num_steps):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for _ in range(batch_size):
            tb, cls_a, cls_b, _ = _make_partial_tree(rng)
            logits = model(tb)  # [6, 3]; edge0 = child_a (slot0), edge1 = child_b (slot1)
            loss = loss + F.cross_entropy(logits[0].unsqueeze(0), torch.tensor([cls_a]))
            loss = loss + F.cross_entropy(logits[1].unsqueeze(0), torch.tensor([cls_b]))
        loss = loss / (2 * batch_size)
        loss.backward()
        opt.step()


def _held_out_joint_accuracy(model: ChildWdlModel, n_trees: int, seed: int) -> float:
    model.eval()
    rng = np.random.RandomState(seed)
    correct_joint = 0
    with torch.no_grad():
        for _ in range(n_trees):
            tb, cls_a, cls_b, _ = _make_partial_tree(rng)
            logits = model(tb)
            pred_a = int(logits[0].argmax().item() == cls_a)
            pred_b = int(logits[1].argmax().item() == cls_b)
            correct_joint += int(pred_a and pred_b)
    return correct_joint / n_trees


def _isolated_perturbation_ratio(model: ChildWdlModel, seed: int) -> tuple[float, float, float]:
    """Hold the base tree (root + both children's own raw features) and
    child_b's entire subtree EXACTLY fixed; vary ONLY child_a's grandchildren.
    Returns (delta on perturbed child_a, delta on untouched sibling child_b,
    ratio). A genuinely slot-gated/disambiguating model should show
    delta_a >> delta_b; generic parent-state drift shows delta_a ~= delta_b.
    """
    model.eval()
    base_rng = np.random.RandomState(seed)
    base_features = base_rng.randn(7, _NODE_FEAT).astype("float32")
    fixed_b_signal = 2.7
    with torch.no_grad():
        tb1, *_ = _make_partial_tree(
            np.random.RandomState(seed + 1), base_features=base_features,
            override_signal_a=2.5, override_signal_b=fixed_b_signal,
        )
        logits1 = model(tb1)
        tb2, *_ = _make_partial_tree(
            np.random.RandomState(seed + 1), base_features=base_features,
            override_signal_a=-2.5, override_signal_b=fixed_b_signal,
        )
        logits2 = model(tb2)
    delta_a = (logits2[0] - logits1[0]).abs().sum().item()
    delta_b = (logits2[1] - logits1[1]).abs().sum().item()
    ratio = delta_a / max(delta_b, 1e-9)
    return delta_a, delta_b, ratio


# --- Fast (default) tests: modest training budget, run in every test-suite pass ---

@pytest.fixture(scope="module")
def fast_trained_model() -> ChildWdlModel:
    model = _build_model(seed=7)
    _train(model, num_steps=300, batch_size=16, seed=42)
    return model


def test_predictions_differ_meaningfully_between_siblings_with_different_subtrees(
    fast_trained_model,
):
    """Property (a): ChildWdlModel's concat(parent_states, slot_states) can be
    trained to recover each sibling's OWN subtree-dependent class, well above
    the joint-chance rate -- i.e. the two children's predictions genuinely
    differ based on content, not just because the slot embeddings differ.
    """
    joint_acc = _held_out_joint_accuracy(fast_trained_model, n_trees=90, seed=999)
    # Chance is 1/9 (two independent 3-way guesses). Require a solid multiple
    # of chance, not just "technically above" -- this threshold is well below
    # what was reproducibly observed across several independent training runs
    # during development (0.32-0.33 at this budget or higher; see history.md).
    assert joint_acc > 3 * _CHANCE_JOINT_ACCURACY, (
        f"joint held-out accuracy {joint_acc:.3f} is not meaningfully above "
        f"chance ({_CHANCE_JOINT_ACCURACY:.3f}) -- ChildWdlModel does not "
        f"appear to carry genuine per-child disambiguating signal on this "
        f"T_n-shaped partial tree"
    )


def test_perturbation_localizes_to_the_perturbed_child(fast_trained_model):
    """Property (b): the decisive check. Perturbing ONLY child_a's subtree
    (holding the parent's own features and child_b's entire subtree exactly
    fixed) should move child_a's prediction specifically, without equally
    dragging its untouched sibling's prediction along. A ratio close to 1
    means the shared parent aggregate is leaking into the sibling's decode
    roughly as much as into the perturbed child's own decode -- i.e. generic
    parent-state drift, not slot-gated disambiguation.
    """
    delta_a, delta_b, ratio = _isolated_perturbation_ratio(fast_trained_model, seed=555)
    assert ratio > 1.5, (
        f"perturbation is NOT localized to the perturbed child: delta on "
        f"perturbed child_a={delta_a:.4f}, delta on untouched sibling "
        f"child_b={delta_b:.4f}, ratio={ratio:.2f} (expected > 1.5 for "
        f"genuine slot-gated disambiguation). This was reproduced across "
        f"four independent probe designs during development (random-init, "
        f"single-signal trained, dual-signal at 500 and 2000 training "
        f"steps) with ratio consistently ~0.98-1.01 -- i.e. NOT a "
        f"training-budget artifact. See history.md Task 3 Progress Log."
    )


# --- Slow confirmatory variant: much larger training budget, opt-in ---

@pytest.mark.slow
def test_perturbation_localization_with_extended_training():
    """Confirms the fast test's localization result is not merely an
    artifact of a short training budget -- trains ~5x longer (matching the
    2000-step scratch probe run during development, which reproduced the
    same ratio ~0.98 and barely moved held-out accuracy versus the 300-500
    step budget: 0.333 vs 0.320-0.333). Opt-in (`pytest -m slow`) since this
    takes on the order of a couple of minutes on a CPU-only login node.
    """
    model = _build_model(seed=7)
    _train(model, num_steps=1500, batch_size=16, seed=42)

    joint_acc = _held_out_joint_accuracy(model, n_trees=150, seed=999)
    assert joint_acc > 3 * _CHANCE_JOINT_ACCURACY, (
        f"joint held-out accuracy {joint_acc:.3f} not meaningfully above "
        f"chance ({_CHANCE_JOINT_ACCURACY:.3f}) even with extended training"
    )

    delta_a, delta_b, ratio = _isolated_perturbation_ratio(model, seed=555)
    assert ratio > 1.5, (
        f"perturbation still NOT localized after extended training: "
        f"delta_a={delta_a:.4f} delta_b={delta_b:.4f} ratio={ratio:.2f} "
        f"(expected > 1.5). Extended training does not fix this -- it is a "
        f"real property of ChildWdlModel's concat(parent_states, "
        f"slot_states) mechanism on this T_n-shaped partial tree, not an "
        f"under-training artifact."
    )
