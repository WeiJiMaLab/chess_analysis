"""Utility for measuring child policy drift during incremental tree search expansions.

Policy is represented as a softmax probability distribution over the visit counts
of a parent's children at a specific search time t. Policy drift is computed as
the KL divergence between the "early" policy (at time k) and the "late" policy
(at time k+n).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class PolicyDriftMeasurement:
    """Represent the policy drift measured for a specific node in the search tree."""

    node_id: int
    early_visits: List[int]
    late_visits: List[int]
    policy_early: List[float]
    policy_late: List[float]
    kl_divergence: float


def softmax_policy(visits: Sequence[int | float], temperature: float = 1.0) -> List[float]:
    """Compute the policy probability distribution as a softmax over child visit counts.

    Using softmax provides a smooth, strictly positive probability distribution,
    which ensures the KL divergence is always stable and well-defined without
    requiring arbitrary clipping or epsilon injection.
    """
    if not visits:
        raise ValueError("Cannot compute policy over empty visit counts.")

    x = np.array(visits, dtype=np.float64) / temperature
    e_x = np.exp(x - np.max(x))
    probs = e_x / e_x.sum()
    return [float(p) for p in probs]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute the Kullback-Leibler divergence D_KL(P || Q) between two distributions.

    D_KL(P || Q) = sum(P(i) * log(P(i) / Q(i))).
    Because the input policies are generated via softmax, all probabilities are
    strictly positive (>0), making the divergence mathematically stable.
    """
    if len(p) != len(q):
        raise ValueError(f"Probability distributions must have same length: p={len(p)}, q={len(q)}.")
    if len(p) == 0:
        raise ValueError("Cannot compute KL divergence on empty distributions.")

    arr_p = np.array(p, dtype=np.float64)
    arr_q = np.array(q, dtype=np.float64)

    # Re-normalize to guard against minor float precision drift
    arr_p /= arr_p.sum()
    arr_q /= arr_q.sum()

    # Compute KL divergence
    return float(np.sum(arr_p * np.log(arr_p / arr_q)))


def compute_policy_drift(
    node_id: int,
    early_visits: Sequence[int],
    late_visits: Sequence[int],
    *,
    temperature: float = 1.0,
    n_min: int = 1,
) -> PolicyDriftMeasurement | None:
    """Compute policy drift for a given node between early and late visits.

    Returns None if the early or late state does not satisfy minimum visit constraints,
    ensuring we only evaluate nodes where the search policy is well-defined instead
    of trivial.

    Args:
        node_id: ID of the node being tracked.
        early_visits: Sequence of child visit counts at the early search time t_early.
        late_visits: Sequence of child visit counts at the late search time t_late.
        temperature: Temperature used in the softmax calculation.
        n_min: Minimum total visits required at the early stage to be considered well-defined.
    """
    if len(early_visits) != len(late_visits):
        raise ValueError("Early and late visits must have the same number of children.")
    if len(early_visits) < 2:
        # Policy is trivial/undefined for single-child or leaf nodes
        return None

    total_early_visits = sum(early_visits)
    if total_early_visits < n_min:
        # Not well-defined yet
        return None

    policy_early = softmax_policy(early_visits, temperature=temperature)
    policy_late = softmax_policy(late_visits, temperature=temperature)
    kl_div = kl_divergence(policy_early, policy_late)

    return PolicyDriftMeasurement(
        node_id=node_id,
        early_visits=list(early_visits),
        late_visits=list(late_visits),
        policy_early=policy_early,
        policy_late=policy_late,
        kl_divergence=kl_div,
    )
