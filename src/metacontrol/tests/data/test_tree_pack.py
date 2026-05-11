import torch

from metacontrol.core.schemas import EdgeStats, GeneratorResult, SearchNode
from metacontrol.core.tree import SearchTree
from metacontrol.data.tree_pack import pack_single_tree_like_legacy_shard


def _tiny_tree() -> GeneratorResult:
    root = SearchNode(node_id=0, fen="r0", features={"value": 0.1})
    tree = SearchTree(root)
    c1 = SearchNode(node_id=1, fen="r1", features={"value": -0.2, "wdl_win": 0.2, "wdl_draw": 0.6, "wdl_loss": 0.2})
    tree.add_node(0, "e2e4", c1)
    edge_stats = {(0, 1): EdgeStats(visit_count=1, total_value=0.15, q_value=0.15)}
    return GeneratorResult(tree=tree, edge_stats=edge_stats, num_expansions=0)


def test_pack_single_tree_keys_and_dtypes():
    packed = pack_single_tree_like_legacy_shard(_tiny_tree(), continue_cost=0.05)
    assert packed["format"] == "metacontrol_single_tree_v1"
    assert packed["node_features"].shape[1] == 5
    assert packed["edge_child"].dtype == torch.int32
    assert packed["target_advantages"].numel() == packed["num_snapshots"]
    assert packed["mc_halt_rewards"].numel() == packed["num_snapshots"]
