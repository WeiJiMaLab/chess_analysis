import pytest
from metacontrol.core.tree import SearchNode, SearchTree

def test_single_leaf():
    root = SearchNode(node_id=0, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    tree = SearchTree(root)
    assert tree.root.node_id == 0
    assert len(tree.nodes_by_id) == 1
    assert tree.root.depth == 0

def test_child_of_parent_is_self():
    root = SearchNode(node_id=0, fen="start")
    tree = SearchTree(root)
    child = SearchNode(node_id=1, fen="start+e4")
    tree.add_node(0, "e2e4", child)
    assert tree.root.children["e2e4"] == child

def test_parent_of_child_is_self():
    root = SearchNode(node_id=0, fen="start")
    tree = SearchTree(root)
    child = SearchNode(node_id=1, fen="start+e4")
    tree.add_node(0, "e2e4", child)
    assert child.parent == root

def test_root_is_depth_zero():
    root = SearchNode(node_id=0, fen="start")
    tree = SearchTree(root)
    assert tree.root.depth == 0

def test_tree_depth_after_adding_child():
    root = SearchNode(node_id=0, fen="start")
    tree = SearchTree(root)
    child1 = SearchNode(node_id=1, fen="depth1")
    tree.add_node(0, "move1", child1)
    child2 = SearchNode(node_id=2, fen="depth2")
    tree.add_node(1, "move2", child2)
    
    assert child1.depth == 1
    assert child2.depth == 2

def test_no_cycles():
    # Tree structure is enforced by add_node logic (transposition unfolding)
    # Even if same FEN is added, it gets a new node_id
    root = SearchNode(node_id=0, fen="pos1")
    tree = SearchTree(root)
    child1 = SearchNode(node_id=1, fen="pos2")
    tree.add_node(0, "m1", child1)
    child2 = SearchNode(node_id=2, fen="pos1") # Transposition
    tree.add_node(1, "m2", child2)
    
    assert child2.depth == 2
    assert child2.node_id != root.node_id

def test_checkmate_is_leaf():
    root = SearchNode(node_id=0, fen="mate", is_terminal=True)
    assert root.is_terminal

def test_render_with_depth():
    root = SearchNode(node_id=0, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    tree = SearchTree(root)
    child = SearchNode(node_id=1, fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    tree.add_node(0, "e2e4", child)
    
    output = tree.render()
    assert "Node 0" in output
    assert "Node 1" in output
    assert "--e2e4-->" in output
