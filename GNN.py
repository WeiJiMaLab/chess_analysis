from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from TreeMHA import TreeAttMsgLayer


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


class SinusoidalSlotEncoding(nn.Module):
    def __init__(self, d_embed: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        if d_embed <= 0:
            raise ValueError("d_embed must be positive.")
        self.d_embed = int(d_embed)
        self.device = torch.device(device)
        half_dim = max(1, math.ceil(self.d_embed / 2))
        exponent = torch.arange(half_dim, dtype=torch.float32, device=self.device)
        scale = torch.exp(-math.log(10000.0) * exponent / half_dim)
        self.register_buffer("inverse_frequencies", scale, persistent=False)

    def forward(self, slot_index: torch.Tensor) -> torch.Tensor:
        if slot_index.dtype != torch.long:
            slot_index = slot_index.long()
        positions = slot_index.to(self.inverse_frequencies.device, dtype=torch.float32).unsqueeze(-1)
        angles = positions * self.inverse_frequencies.unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoding.shape[-1] < self.d_embed:
            padding = encoding.new_zeros(encoding.shape[0], self.d_embed - encoding.shape[-1])
            encoding = torch.cat([encoding, padding], dim=-1)
        return encoding[:, : self.d_embed]


class TreeNN(nn.Module):
    """
    Tree encoder over the flattened TreeBatch representation.

    The encoder embeds per-node scalar features, runs alternating upward and
    downward message passing, and returns both all node states and the root state
    for each tree in the batch.

    When ``sequential=True`` (default), messages propagate in topological order
    within each pass — leaves-to-root for the upward sweep, root-to-leaves for the
    downward sweep — so that each node sees its children's (or parent's) *already-
    updated* states.  A single pass (``k=1``) therefore gives every node a receptive
    field covering the entire tree, regardless of depth.

    When ``sequential=False``, the original synchronous (Jacobi-style) update is
    used: all nodes update simultaneously from the previous round's states, giving
    a receptive-field radius of *k* hops.
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
        sequential=True,
    ):
        super().__init__()
        self.k = k
        self.node_feat = node_feat
        self.device = torch.device(device)
        self.d_embed = d_embed
        self.d_message = d_message
        self.sequential = sequential

        self.node_embed = nn.Sequential(
            nn.Linear(node_feat, node_embed_hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(node_embed_hidden, d_embed, device=self.device),
        )
        self.child_slot_encoding = SinusoidalSlotEncoding(d_embed=d_embed, device=self.device)
        self.child_slot_projection = nn.Linear(d_embed, d_embed, device=self.device)
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

    def slot_embeddings(self, edge_slot: torch.Tensor) -> torch.Tensor:
        slot_encoding = self.child_slot_encoding(edge_slot.to(self.device))
        return self.child_slot_projection(slot_encoding)

    def _forward_synchronous(
        self, node_states, edge_parent, edge_child, edge_slot_embed,
        parent_index, child_ptr, children_index,
    ):
        for _ in range(self.k):
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
        return node_states

    def _forward_sequential(
        self, node_states, edge_parent, edge_child, edge_slot_embed,
        parent_index, depth,
    ):
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        # Precompute per-level node indices and edge masks (static across k).
        level_nodes = []
        if edge_parent.numel() > 0:
            edge_parent_depth = depth[edge_parent]
        else:
            edge_parent_depth = depth.new_empty(0)

        for d in range(max_depth + 1):
            level_nodes.append((depth == d).nonzero(as_tuple=False).view(-1))

        for _ in range(self.k):
            # --- Upward pass: leaves → root ---
            for d in range(max_depth, -1, -1):
                nodes_d = level_nodes[d]
                if nodes_d.numel() == 0:
                    continue

                edge_mask = (edge_parent_depth == d) if edge_parent_depth.numel() > 0 else None
                if edge_mask is not None and edge_mask.any():
                    upward = self.upward_msg(
                        node_states,
                        edge_parent[edge_mask],
                        edge_child[edge_mask],
                        edge_slot_embed=edge_slot_embed[edge_mask],
                    )
                    level_msg = upward[nodes_d]
                else:
                    level_msg = node_states.new_zeros(nodes_d.size(0), self.d_message)

                new_level = self.node_gru(level_msg, node_states[nodes_d])
                node_states = node_states.clone()
                node_states[nodes_d] = new_level

            # --- Downward pass: root → leaves ---
            for d in range(max_depth + 1):
                nodes_d = level_nodes[d]
                if nodes_d.numel() == 0:
                    continue

                level_parents = parent_index[nodes_d]
                msg = node_states.new_zeros(nodes_d.size(0), self.d_message)
                has_parent = level_parents >= 0
                if has_parent.any():
                    msg[has_parent] = self.downward_msg(
                        node_states[level_parents[has_parent]]
                    )

                new_level = self.node_gru(msg, node_states[nodes_d])
                node_states = node_states.clone()
                node_states[nodes_d] = new_level

        return node_states

    def forward(self, tree_batch):
        node_features = tree_batch.node_features.to(self.device)
        child_ptr = tree_batch.child_ptr.to(self.device)
        children_index = tree_batch.children_index.to(self.device)
        edge_parent = tree_batch.edge_parent.to(self.device)
        edge_child = tree_batch.edge_child.to(self.device)
        edge_slot = tree_batch.edge_slot.to(self.device)
        parent_index = tree_batch.parent_index.to(self.device)
        root_index = tree_batch.root_index.to(self.device)
        depth = tree_batch.depth.to(self.device)

        node_states = self.node_embed(node_features)
        edge_slot_embed = self.slot_embeddings(edge_slot)

        if self.sequential:
            node_states = self._forward_sequential(
                node_states, edge_parent, edge_child, edge_slot_embed,
                parent_index, depth,
            )
        else:
            node_states = self._forward_synchronous(
                node_states, edge_parent, edge_child, edge_slot_embed,
                parent_index, child_ptr, children_index,
            )

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
        sequential=True,
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
            sequential=sequential,
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
        sequential=True,
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
                sequential=sequential,
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


class ChildWdlModel(nn.Module):
    """
    Encoder plus slot-conditioned child-WDL decoder used for supervised pretraining.
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
        sequential=True,
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
                sequential=sequential,
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
        edge_slot = tree_batch.edge_slot.to(self.encoder.device)
        slot_states = self.encoder.slot_embeddings(edge_slot)
        edge_logits = self.child_wdl_head(encoded.node_states[edge_parent], slot_states)
        return ChildWdlOutput(
            node_states=encoded.node_states,
            root_states=encoded.root_states,
            edge_logits=edge_logits,
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
        sequential=True,
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
                sequential=sequential,
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
