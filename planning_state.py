from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tree import ExpansionChild, SearchTree


@dataclass
class PlanningState:
    tree: SearchTree
    steps_taken: int = 0
    halted: bool = False
    chosen_move_uci: Optional[str] = None
    history: List[str] = field(default_factory=list)

    def halt(self, value_feature: str = "value") -> str:
        if self.halted:
            raise ValueError("Planning has already halted.")

        best_child_id = self.tree.best_root_child(value_feature=value_feature)
        best_child = self.tree.get_node(best_child_id)
        if best_child.incoming_move_uci is None:
            raise ValueError("Best root child must have an incoming move.")

        self.halted = True
        self.chosen_move_uci = best_child.incoming_move_uci
        self.history.append(f"halt:{self.chosen_move_uci}")
        return self.chosen_move_uci

    def continue_planning(self) -> int:
        if self.halted:
            raise ValueError("Cannot continue planning after halt.")

        self.steps_taken += 1
        self.history.append("continue")
        return self.steps_taken

    def apply_expansion(self, parent_id: int, children: List[ExpansionChild]) -> List[int]:
        if self.halted:
            raise ValueError("Cannot expand the tree after halt.")

        child_ids = self.tree.add_children(parent_id, children)
        self.history.append(f"expand:{parent_id}:{len(child_ids)}")
        return child_ids

