from dataclasses import dataclass

import torch
import torch.nn as nn

from TreeMHA import TreeAttMsgLayer
from tensorizer import DEFAULT_CHILD_SLOT_COUNT


@dataclass
class TreeEncoderOutput:
    node_states: torch.Tensor
    root_states: torch.Tensor


@dataclass
class TreeSearchOutput:
    node_states: torch.Tensor
    root_states: torch.Tensor
    halt_logits: torch.Tensor
    halt_prob: torch.Tensor


@dataclass
class NodeValueOutput:
    node_states: torch.Tensor
    root_states: torch.Tensor
    node_values: torch.Tensor


@dataclass
class ChildWdlOutput:
    node_states: torch.Tensor
    root_states: torch.Tensor
    edge_logits: torch.Tensor


@dataclass
class PolicyValueOutput:
    node_states: torch.Tensor
    root_states: torch.Tensor
    halt_logits: torch.Tensor
    halt_prob: torch.Tensor
    state_value: torch.Tensor


class TreeNN(nn.Module):
    """
    Tree encoder over the flattened TreeBatch representation.

    The encoder embeds per-node scalar features, runs alternating upward and
    downward message passing, and returns both all node states and the root state
    for each tree in the batch.
    """

    def __init__(
        self,
        k,
        node_feat,
        device,
        node_embed_hidden=128,
        d_embed=512,
        d_message=512,
        n_heads=4,
        d_att=128,
        child_slot_count=DEFAULT_CHILD_SLOT_COUNT,
    ):
        super().__init__()
        if child_slot_count < 2:
            raise ValueError("child_slot_count must be at least 2 so the final slot can act as overflow.")
        self.k = k
        self.node_feat = node_feat
        self.device = torch.device(device)
        self.d_embed = d_embed
        self.d_message = d_message
        self.child_slot_count = int(child_slot_count)

        self.node_embed = nn.Sequential(
            nn.Linear(node_feat, node_embed_hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(node_embed_hidden, d_embed, device=self.device),
        )
        self.child_slot_embedding = nn.Embedding(self.child_slot_count, d_embed, device=self.device)
        self.node_gru = nn.GRUCell(d_message, d_embed, device=self.device)
        self.upward_msg = TreeAttMsgLayer(
            n_heads=n_heads,
            d_att=d_att,
            d_embed=d_embed,
            d_message=d_message,
            device=self.device,
        )
        self.downward_msg = nn.Linear(d_embed, d_message, bias=False, device=self.device)

    def _downward_messages(self, node_states, parent_index):
        messages = node_states.new_zeros(node_states.size(0), self.d_message)
        has_parent = parent_index >= 0
        if has_parent.any():
            parent_states = node_states[parent_index[has_parent]]
            messages[has_parent] = self.downward_msg(parent_states)
        return messages

    def canonicalize_edge_slots(self, edge_slot: torch.Tensor) -> torch.Tensor:
        return edge_slot.clamp(min=0, max=self.child_slot_count - 1)

    def forward(self, tree_batch):
        node_features = tree_batch.node_features.to(self.device)
        child_ptr = tree_batch.child_ptr.to(self.device)
        children_index = tree_batch.children_index.to(self.device)
        edge_parent = tree_batch.edge_parent.to(self.device)
        edge_child = tree_batch.edge_child.to(self.device)
        edge_slot = self.canonicalize_edge_slots(tree_batch.edge_slot.to(self.device))
        parent_index = tree_batch.parent_index.to(self.device)
        root_index = tree_batch.root_index.to(self.device)

        node_states = self.node_embed(node_features)
        for _ in range(self.k):
            edge_slot_embed = self.child_slot_embedding(edge_slot)
            upward = self.upward_msg(
                node_states,
                edge_parent,
                edge_child,
                edge_slot_embed=edge_slot_embed,
                child_ptr=child_ptr,
                children_index=children_index,
            )
            node_states = self.node_gru(upward, node_states)

            downward = self._downward_messages(node_states, parent_index)
            node_states = self.node_gru(downward, node_states)

        return TreeEncoderOutput(
            node_states=node_states,
            root_states=node_states[root_index],
        )


class HaltController(nn.Module):
    def __init__(self, d_embed, hidden_dim=128, device="cpu"):
        super().__init__()
        device = torch.device(device)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, device=device),
        )

    def forward(self, root_states):
        halt_logits = self.mlp(root_states).squeeze(-1)
        halt_prob = torch.sigmoid(halt_logits)
        return halt_logits, halt_prob


class NodeValueHead(nn.Module):
    def __init__(self, d_embed, hidden_dim=128, device="cpu"):
        super().__init__()
        device = torch.device(device)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, device=device),
        )

    def forward(self, node_states):
        return self.mlp(node_states).squeeze(-1)


class ChildWdlHead(nn.Module):
    def __init__(self, d_embed, hidden_dim=128, device="cpu"):
        super().__init__()
        device = torch.device(device)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_embed, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3, device=device),
        )

    def forward(self, parent_states, slot_states):
        return self.mlp(torch.cat([parent_states, slot_states], dim=-1))


class RootValueHead(nn.Module):
    def __init__(self, d_embed, hidden_dim=128, device="cpu"):
        super().__init__()
        device = torch.device(device)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, device=device),
        )

    def forward(self, root_states):
        return self.mlp(root_states).squeeze(-1)


class TreeSearchModel(nn.Module):
    """
    End-to-end stage-1 model: tree encoder plus halt controller on root states.
    """

    def __init__(
        self,
        k,
        node_feat,
        device,
        node_embed_hidden=128,
        d_embed=512,
        d_message=512,
        n_heads=4,
        d_att=128,
        controller_hidden=128,
        child_slot_count=DEFAULT_CHILD_SLOT_COUNT,
    ):
        super().__init__()
        self.encoder = TreeNN(
            k=k,
            node_feat=node_feat,
            device=device,
            node_embed_hidden=node_embed_hidden,
            d_embed=d_embed,
            d_message=d_message,
            n_heads=n_heads,
            d_att=d_att,
            child_slot_count=child_slot_count,
        )
        self.halt_controller = HaltController(
            d_embed=d_embed,
            hidden_dim=controller_hidden,
            device=device,
        )

    def forward(self, tree_batch):
        encoded = self.encoder(tree_batch)
        halt_logits, halt_prob = self.halt_controller(encoded.root_states)
        return TreeSearchOutput(
            node_states=encoded.node_states,
            root_states=encoded.root_states,
            halt_logits=halt_logits,
            halt_prob=halt_prob,
        )


class NodeValueModel(nn.Module):
    """
    Encoder plus temporary node-value readout used for supervised pretraining.
    """

    def __init__(
        self,
        k,
        node_feat,
        device,
        node_embed_hidden=128,
        d_embed=512,
        d_message=512,
        n_heads=4,
        d_att=128,
        value_hidden=128,
        encoder=None,
        child_slot_count=DEFAULT_CHILD_SLOT_COUNT,
    ):
        super().__init__()
        if encoder is None:
            encoder = TreeNN(
                k=k,
                node_feat=node_feat,
                device=device,
                node_embed_hidden=node_embed_hidden,
                d_embed=d_embed,
                d_message=d_message,
                n_heads=n_heads,
                d_att=d_att,
                child_slot_count=child_slot_count,
            )
        self.encoder = encoder
        self.node_value_head = NodeValueHead(
            d_embed=encoder.d_embed,
            hidden_dim=value_hidden,
            device=encoder.device,
        )

    def forward(self, tree_batch):
        encoded = self.encoder(tree_batch)
        node_values = self.node_value_head(encoded.node_states)
        return NodeValueOutput(
            node_states=encoded.node_states,
            root_states=encoded.root_states,
            node_values=node_values,
        )


class PolicyValueTreeSearchModel(nn.Module):
    """
    Tree encoder with halt policy and root value heads for PPO.
    """

    def __init__(
        self,
        k,
        node_feat,
        device,
        node_embed_hidden=128,
        d_embed=512,
        d_message=512,
        n_heads=4,
        d_att=128,
        controller_hidden=128,
        value_hidden=128,
        encoder=None,
        child_slot_count=DEFAULT_CHILD_SLOT_COUNT,
    ):
        super().__init__()
        if encoder is None:
            encoder = TreeNN(
                k=k,
                node_feat=node_feat,
                device=device,
                node_embed_hidden=node_embed_hidden,
                d_embed=d_embed,
                d_message=d_message,
                n_heads=n_heads,
                d_att=d_att,
                child_slot_count=child_slot_count,
            )
        self.encoder = encoder
        self.halt_controller = HaltController(
            d_embed=encoder.d_embed,
            hidden_dim=controller_hidden,
            device=encoder.device,
        )
        self.value_head = RootValueHead(
            d_embed=encoder.d_embed,
            hidden_dim=value_hidden,
            device=encoder.device,
        )

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, tree_batch):
        encoded = self.encoder(tree_batch)
        halt_logits, halt_prob = self.halt_controller(encoded.root_states)
        state_value = self.value_head(encoded.root_states)
        return PolicyValueOutput(
            node_states=encoded.node_states,
            root_states=encoded.root_states,
            halt_logits=halt_logits,
            halt_prob=halt_prob,
            state_value=state_value,
        )


class ChildWdlModel(nn.Module):
    """
    Encoder plus slot-conditioned parent-to-child WDL decoder used for edge-level pretraining.
    """

    def __init__(
        self,
        k,
        node_feat,
        device,
        node_embed_hidden=128,
        d_embed=512,
        d_message=512,
        n_heads=4,
        d_att=128,
        decoder_hidden=128,
        encoder=None,
        child_slot_count=DEFAULT_CHILD_SLOT_COUNT,
    ):
        super().__init__()
        if encoder is None:
            encoder = TreeNN(
                k=k,
                node_feat=node_feat,
                device=device,
                node_embed_hidden=node_embed_hidden,
                d_embed=d_embed,
                d_message=d_message,
                n_heads=n_heads,
                d_att=d_att,
                child_slot_count=child_slot_count,
            )
        self.encoder = encoder
        self.child_wdl_head = ChildWdlHead(
            d_embed=encoder.d_embed,
            hidden_dim=decoder_hidden,
            device=encoder.device,
        )

    def forward(self, tree_batch):
        encoded = self.encoder(tree_batch)
        edge_parent = tree_batch.edge_parent.to(self.encoder.device)
        edge_slot = self.encoder.canonicalize_edge_slots(tree_batch.edge_slot.to(self.encoder.device))
        parent_states = encoded.node_states[edge_parent]
        slot_states = self.encoder.child_slot_embedding(edge_slot)
        edge_logits = self.child_wdl_head(parent_states, slot_states)
        return ChildWdlOutput(
            node_states=encoded.node_states,
            root_states=encoded.root_states,
            edge_logits=edge_logits,
        )
