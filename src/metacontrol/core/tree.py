from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

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

class SearchTree:
    def __init__(self, root: SearchNode):
        self.root = root
        self.nodes_by_id: Dict[int, SearchNode] = {root.node_id: root}
        self.expansion_history: List[SearchNode] = []

    def add_node(self, parent_id: int, move: str, child_node: SearchNode):
        parent = self.nodes_by_id[parent_id]
        parent.children[move] = child_node
        child_node.parent = parent
        child_node.depth = parent.depth + 1
        self.nodes_by_id[child_node.node_id] = child_node
        self.expansion_history.append(child_node)

    def get_node(self, node_id: int) -> Optional[SearchNode]:
        return self.nodes_by_id.get(node_id)

    def render(self, node: Optional[SearchNode] = None, indent: int = 0) -> str:
        if node is None:
            node = self.root
        
        lines = []
        prefix = "  " * indent
        line = f"{prefix}Node {node.node_id}: {node.fen[:20]}... (v={node.visit_count}, q={node.q_value:.2f})"
        if node.is_terminal:
            line += " [TERMINAL]"
        lines.append(line)
        
        for move, child in sorted(node.children.items()):
            lines.append(f"{prefix}  --{move}-->")
            lines.append(self.render(child, indent + 2))
            
        return "\n".join(lines)
