from typing import Dict, List, Optional, Any


from metacontrol.core.schemas import SearchNode

class SearchTree:
    def __init__(self, root: SearchNode):
        self.root = root
        self.nodes_by_id: Dict[int, SearchNode] = {root.node_id: root}
        self.expansion_history: List[SearchNode] = []
        #: One entry per successful PUCT expand_leaf (the expanded leaf). Used
        #: for meta-control snapshots; falls back to :attr:`expansion_history`
        #: for hand-built trees in unit tests.
        self.search_expansion_history: List[SearchNode] = []

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
