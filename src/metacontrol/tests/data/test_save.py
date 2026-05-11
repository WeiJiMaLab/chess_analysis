import torch

from metacontrol.core.schemas import EdgeStats, GeneratorConfig, GeneratorResult, SearchNode
from metacontrol.core.tree import SearchTree
from metacontrol.data.generator import TreeSearch


def _tiny_tree() -> GeneratorResult:
    root = SearchNode(node_id=0, fen="r0", features={"value": 0.1})
    tree = SearchTree(root)
    c1 = SearchNode(node_id=1, fen="r1", features={"value": -0.2, "wdl_win": 0.2, "wdl_draw": 0.6, "wdl_loss": 0.2})
    tree.add_node(0, "e2e4", c1)
    edge_stats = {(0, 1): EdgeStats(visit_count=1, total_value=0.15, q_value=0.15)}
    return GeneratorResult(tree=tree, edge_stats=edge_stats, num_expansions=0)


def test_tree_search_save_writes_payload(tmp_path):
    search = TreeSearch(provider=None, config=GeneratorConfig(max_nodes=1, max_depth=1))
    out = tmp_path / "tree.pt"

    payload = search.save(_tiny_tree(), out, continue_cost=0.05)

    loaded = torch.load(out, weights_only=False)
    assert loaded["format"] == "metacontrol_single_tree_v1"
    assert loaded["num_nodes"] == payload["num_nodes"]
    assert loaded["node_features"].shape[1] == 5
    assert loaded["edge_child"].dtype == torch.int32
    assert loaded["target_advantages"].numel() == loaded["num_snapshots"]
    assert loaded["mc_halt_rewards"].numel() == loaded["num_snapshots"]
