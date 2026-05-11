"""Meta-control target derivation via dynamic programming over search history.

This module implements the "derive once from the full tree" strategy described
in phase_1_plan.md (Stage 3).  Given a completed SearchTree with expansion
history and accumulated edge stats, it:

1. Computes the *halt reward* at each snapshot (the Q-value of the best root
   child at that point in the search).
2. Runs a backward DP pass to compute the *optimal continue value* at each
   snapshot under a given continue_cost.
3. Packages the results as SearchSnapshot objects with a pre-computed
   advantage (continue_value − halt_reward).
"""

from __future__ import annotations


from typing import Dict, List, Optional, Tuple

from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import SearchNode, EdgeStats, SearchSnapshot



# ---------------------------------------------------------------------------
# Halt-reward computation
# ---------------------------------------------------------------------------

def _best_root_q(
    tree: SearchTree,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
) -> float:
    """Return the Q-value of the best root child, or the root value if no children."""
    root = tree.root
    if not root.children:
        return root.features.get("value", 0.0)

    best_q = float("-inf")
    for child_node in root.children.values():
        key = (root.node_id, child_node.node_id)
        stats = edge_stats.get(key)
        if stats is not None and stats.visit_count > 0:
            if stats.q_value > best_q:
                best_q = stats.q_value
        else:
            # Unvisited child — use its static value (negated, since it's
            # from the child's perspective)
            child_val = -child_node.features.get("value", 0.0)
            if child_val > best_q:
                best_q = child_val

    return best_q if best_q > float("-inf") else root.features.get("value", 0.0)


def _replay_edge_stats_at_prefix(
    tree: SearchTree,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
    expansion_count: int,
) -> Dict[Tuple[int, int], EdgeStats]:
    """Return a copy of edge_stats limited to edges that exist in the
    first *expansion_count* expansions of the tree.

    This avoids re-running PUCT from scratch for each prefix.  We re-use
    the full-tree edge stats as an upper bound; the halt reward is computed
    from whatever edges are present in the prefix, using the accumulated
    stats (which is a deterministic function of the expansion history).
    """
    # Nodes present after `expansion_count` expansions:
    #   - root (always present, id=0)
    #   - children added in the first `expansion_count` expansion steps
    history = tree.expansion_history[:expansion_count]
    present_ids = {tree.root.node_id}
    for node in history:
        present_ids.add(node.node_id)
        # Also include all siblings (children of the same parent)
        if node.parent is not None:
            for sibling in node.parent.children.values():
                present_ids.add(sibling.node_id)

    prefix_stats: Dict[Tuple[int, int], EdgeStats] = {}
    for (p, c), stats in edge_stats.items():
        if p in present_ids and c in present_ids:
            prefix_stats[(p, c)] = stats
    return prefix_stats


def compute_halt_rewards(
    tree: SearchTree,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
) -> List[float]:
    """Compute the halt reward at each expansion step [0 .. N].

    Index i corresponds to the state of the tree after i expansions.
    The list has length ``len(tree.expansion_history) + 1``.
    """
    n = len(tree.expansion_history)
    halt_rewards: List[float] = []

    for i in range(n + 1):
        prefix_stats = _replay_edge_stats_at_prefix(tree, edge_stats, i)
        # Build a set of root children present at this prefix
        root = tree.root
        root_children_present: Dict[str, SearchNode] = {}
        if i > 0:
            present_ids = {root.node_id}
            for node in tree.expansion_history[:i]:
                present_ids.add(node.node_id)
                if node.parent is not None:
                    for sibling in node.parent.children.values():
                        present_ids.add(sibling.node_id)
            for move, child in root.children.items():
                if child.node_id in present_ids:
                    root_children_present[move] = child
        # else: no expansions yet → root only

        if not root_children_present:
            halt_rewards.append(root.features.get("value", 0.0))
        else:
            best_q = float("-inf")
            for child_node in root_children_present.values():
                key = (root.node_id, child_node.node_id)
                stats = prefix_stats.get(key)
                if stats is not None and stats.visit_count > 0:
                    q = stats.q_value
                else:
                    q = -child_node.features.get("value", 0.0)
                if q > best_q:
                    best_q = q
            halt_rewards.append(best_q)

    return halt_rewards


# ---------------------------------------------------------------------------
# DP pass
# ---------------------------------------------------------------------------

def calculate_dp_values(
    halt_rewards: List[float],
    continue_cost: float,
) -> List[float]:
    """Backward DP to compute optimal value at each snapshot.

    V[i] = max(halt_rewards[i],  V[i+1] - continue_cost)
    V[N] = halt_rewards[N]  (forced halt at end)

    Returns a list of length len(halt_rewards).
    """
    n = len(halt_rewards)
    if n == 0:
        return []

    dp = [0.0] * n
    dp[-1] = halt_rewards[-1]
    for i in range(n - 2, -1, -1):
        continue_val = dp[i + 1] - continue_cost
        dp[i] = max(halt_rewards[i], continue_val)
    return dp


# ---------------------------------------------------------------------------
# Snapshot derivation
# ---------------------------------------------------------------------------

def derive_snapshots(
    tree: SearchTree,
    edge_stats: Dict[Tuple[int, int], EdgeStats],
    continue_cost: float,
) -> List[SearchSnapshot]:
    """Derive meta-control training snapshots from a completed search tree.

    For each prefix of the expansion history, computes:
    - halt_reward: value of halting (best root Q at that prefix)
    - continue_value: optimal future value under the stop/continue cost
    - advantage: continue_value − halt_reward (positive → should continue)

    Returns a list of SearchSnapshot of length ``len(expansion_history) + 1``.
    The last snapshot always has advantage ≤ 0 (forced halt).
    """
    halt_rewards = compute_halt_rewards(tree, edge_stats)
    dp_values = calculate_dp_values(halt_rewards, continue_cost)

    snapshots: List[SearchSnapshot] = []
    n = len(halt_rewards)
    for i in range(n):
        if i < n - 1:
            continue_value = dp_values[i + 1] - continue_cost
        else:
            continue_value = halt_rewards[i]  # no future → halt

        advantage = continue_value - halt_rewards[i]
        snapshots.append(SearchSnapshot(
            expansion_index=i,
            halt_reward=halt_rewards[i],
            continue_value=continue_value,
            advantage=advantage,
        ))

    return snapshots
