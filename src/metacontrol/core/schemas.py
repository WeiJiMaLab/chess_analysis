"""Centralised schemas and data classes for the metacontrol project."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from metacontrol.core.tree import SearchTree


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

class ChessFeatureSchema:
    def __init__(self):
        # Match legacy shard width (``ysagiv`` controller packs use ``wdl_var``).
        self.feature_names: Tuple[str, ...] = (
            "value",
            "wdl_win",
            "wdl_draw",
            "wdl_loss",
            "wdl_var",
        )
        self.num_features: int = len(self.feature_names)

    def vectorize(self, features: dict) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]


@dataclass
class SearchNode:
    node_id: int
    fen: str
    parent: Optional['SearchNode'] = None
    children: Dict[str, 'SearchNode'] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    q_value: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    features: Dict[str, float] = field(default_factory=lambda: {
        "value": 0.0,
        "wdl_win": 0.0,
        "wdl_draw": 0.0,
        "wdl_loss": 0.0
    })

    def __repr__(self) -> str:
        return f"SearchNode(id={self.node_id}, depth={self.depth}, visits={self.visit_count}, q={self.q_value:.3f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "fen": self.fen,
            "visit_count": self.visit_count,
            "total_value": self.total_value,
            "q_value": self.q_value,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "features": self.features,
            "children": {move: child.node_id for move, child in self.children.items()}
        }


@dataclass
class TreeBatch:
    node_features: torch.Tensor
    parent_index: torch.Tensor
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_slot: torch.Tensor
    depth: torch.Tensor
    root_index: torch.Tensor
    tree_index: torch.Tensor
    num_nodes: int
    num_edges: int
    batch_size: int


# ---------------------------------------------------------------------------
# Generator Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ChildInfo:
    """Child expansion returned by a provider."""
    move_uci: str
    fen: str
    value: float
    prior: float
    wdl: Optional[Tuple[float, float, float]] = None
    is_terminal: bool = False


@dataclass(frozen=True)
class GeneratorConfig:
    max_nodes: int
    max_depth: int
    c_puct: float = 1.0

    def __post_init__(self) -> None:
        assert self.max_nodes > 0, "max_nodes must be positive"
        assert self.max_depth >= 0, "max_depth must be non-negative"
        assert self.c_puct >= 0.0, "c_puct must be non-negative"


@dataclass
class EdgeStats:
    """Running statistics for a parent→child edge during search."""
    visit_count: int = 0
    total_value: float = 0.0
    q_value: float = 0.0
    total_wdl: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_wdl: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GeneratorResult:
    tree: SearchTree
    edge_stats: Dict[Tuple[int, int], EdgeStats]
    num_expansions: int


# ---------------------------------------------------------------------------
# Target Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SearchSnapshot:
    """One observation in the meta-controller's decision trajectory.

    Each snapshot corresponds to a prefix of the full search tree after
    a certain number of expansions.
    """
    expansion_index: int
    halt_reward: float
    continue_value: float
    advantage: float

    # Reference to the tree prefix
    tree_prefix: Optional[SearchTree] = None


@dataclass
class GNNTargets:
    """Supervised targets for training the GNN on a generated search tree."""
    node_values: Dict[int, float]
    edge_wdls: Dict[Tuple[int, int], Tuple[float, float, float]]
