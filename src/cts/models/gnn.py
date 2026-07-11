import math
from typing import NamedTuple

import torch
import torch.nn as nn

from .tree_mha import TreeAttMsgLayer


class TreeEncoderOutput(NamedTuple):
    """Result of one ``TreeEncoder`` forward pass over a batched tree."""

    node_states: torch.Tensor  # [N, d_embed] per-node embeddings after k iterations of message passing
    root_states: torch.Tensor  # [B, d_embed] root embedding per tree in the batch (the "z_root" downstream conditions on)


class SinusoidalSlotEncoding(nn.Module):
    """Fixed sinusoidal positional encoding indexed by child slot.

    Used to give each child a slot-specific signal so the parent's attention
    over children is order-aware. Consumed by ``TreeAttMsgLayer`` as the
    additive ``edge_slot_embed`` and by ``ChildWdlHead`` as the slot query.
    """

    def __init__(self, d_embed: int, device: torch.device | str = "cpu") -> None:
        """
        Args:
            d_embed: dimensionality of the encoding (same width as node embeddings).
            device: device on which the cached inverse-frequency buffer lives.
        """
        super().__init__()
        self.d_embed = int(d_embed)
        self.device = torch.device(device)
        # Standard sinusoidal frequency schedule: geometric in 1/10000^(2i/d).
        # Cached as a non-persistent buffer so it moves with .to(device) but
        # doesn't bloat checkpoints.
        half_dim = max(1, math.ceil(self.d_embed / 2))
        exponent = torch.arange(half_dim, dtype=torch.float32, device=self.device)
        scale = torch.exp(-math.log(10000.0) * exponent / half_dim)
        self.register_buffer("inverse_frequencies", scale, persistent=False)

    def forward(self, slot_index: torch.Tensor) -> torch.Tensor:
        """Map an integer slot index tensor to its sinusoidal encoding.

        Args:
            slot_index: [E] integer tensor of per-edge child slot indices.

        Returns:
            [E, d_embed] tensor; concatenated sin/cos of the slot positions,
            zero-padded if ``d_embed`` is odd so half_dim*2 doesn't fit exactly.
        """
        if slot_index.dtype != torch.long:
            slot_index = slot_index.long()
        positions = slot_index.to(self.inverse_frequencies.device, dtype=torch.float32).unsqueeze(-1)
        angles = positions * self.inverse_frequencies.unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # Trim or zero-pad to exactly d_embed (sin/cos concat is 2*half_dim
        # which can be one wider than d_embed when d_embed is odd).
        if encoding.shape[-1] < self.d_embed:
            padding = encoding.new_zeros(encoding.shape[0], self.d_embed - encoding.shape[-1])
            encoding = torch.cat([encoding, padding], dim=-1)
        return encoding[:, : self.d_embed]


class TreeEncoder(nn.Module):
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
        """
        Args:
            k: number of message-passing iterations (each iteration is one
                upward + one downward sweep).
            node_feat: number of input scalar features per node (must match
                the schema in ``schema.py``).
            device: device for all parameters and buffers.
            node_embed_hidden: hidden width of the per-node feature-embedding MLP.
            d_embed: dimensionality of node embeddings throughout the network.
            d_message: dimensionality of per-edge messages.
            n_heads: number of attention heads in the upward message layer.
            d_att: per-head key/query/value dimension.
            sequential: if True, use level-sequential message passing; otherwise
                use synchronous Jacobi-style updates.
        """
        super().__init__()
        self.k = k
        self.node_feat = node_feat
        self.device = torch.device(device)
        self.d_embed = d_embed
        self.d_message = d_message
        self.sequential = sequential

        # Per-node feature → embedding MLP (applied once at the start of forward).
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat, node_embed_hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(node_embed_hidden, d_embed, device=self.device),
        )
        # Slot positional encoding + a learnable projection that lets the
        # network rescale the fixed sinusoid before mixing it into child keys/values.
        self.child_slot_encoding = SinusoidalSlotEncoding(d_embed=d_embed, device=self.device)
        self.child_slot_projection = nn.Linear(d_embed, d_embed, device=self.device)
        # Shared GRU cell applied for both upward and downward updates; the
        # message tensor is the input and the prior node state is the hidden state.
        self.node_gru = nn.GRUCell(d_message, d_embed, device=self.device)
        self.upward_msg = TreeAttMsgLayer(
            n_heads=n_heads,
            d_att=d_att,
            d_embed=d_embed,
            d_message=d_message,
            device=self.device,
        )
        # Downward message is a simple linear projection of the parent's state;
        # no attention needed since each node has at most one parent.
        self.downward_msg = nn.Linear(d_embed, d_message, bias=False, device=self.device)

    def _downward_messages(self, node_states, parent_index):
        """Compute one downward message per node from its parent's current state.

        Args:
            node_states: [N, d_embed] current activations.
            parent_index: [N] long tensor; entry i is the parent's index, or -1 for roots.
        """
        messages = node_states.new_zeros(node_states.size(0), self.d_message)
        # Roots get the zero message; everyone else gets a linear projection
        # of their parent's current state.
        has_parent = parent_index >= 0
        if has_parent.any():
            parent_states = node_states[parent_index[has_parent]]
            messages[has_parent] = self.downward_msg(parent_states)
        return messages

    def slot_embeddings(self, edge_slot: torch.Tensor) -> torch.Tensor:
        """Encode per-edge child slot indices and apply the learnable projection."""
        slot_encoding = self.child_slot_encoding(edge_slot.to(self.device))
        return self.child_slot_projection(slot_encoding)

    def _forward_synchronous(
        self, node_states, edge_parent, edge_child, edge_slot_embed,
        parent_index,
    ):
        """Run k Jacobi-style iterations: every node updates simultaneously each round.

        Receptive field after k rounds is k hops, so depth-D trees need k >= D
        for the root to see every leaf.
        """
        for _ in range(self.k):
            upward = self.upward_msg(
                node_states,
                edge_parent,
                edge_child,
                edge_slot_embed=edge_slot_embed,
            )
            node_states = self.node_gru(upward, node_states)

            downward = self._downward_messages(node_states, parent_index)
            node_states = self.node_gru(downward, node_states)
        return node_states

    def _forward_sequential(
        self, node_states, edge_parent, edge_child, edge_slot_embed,
        parent_index, depth,
    ):
        """Run k level-sequential iterations (leaves→root, then root→leaves).

        Within each iteration, nodes at level d update *after* their descendants
        (upward) or *after* their ancestors (downward), so one iteration suffices
        for tree-wide receptive field regardless of depth.

        Args:
            depth: [N] long tensor of per-node depths from the root.
        """
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
            # Walk depths from deepest to root; at each level, every node sees
            # children whose state has already been refreshed this iteration.
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
                    # Pure-leaf level: no outgoing edges, so message is zero.
                    level_msg = node_states.new_zeros(nodes_d.size(0), self.d_message)

                new_level = self.node_gru(level_msg, node_states[nodes_d])
                # Clone before writing back so autograd sees a fresh tensor
                # at each level (in-place write on a leaf of the graph breaks gradients).
                node_states = node_states.clone()
                node_states[nodes_d] = new_level

            # --- Downward pass: root → leaves ---
            # Walk depths from root to deepest; each level sees its parent's
            # already-updated state.
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
        """Encode a batched tree and return per-node + per-root embeddings.

        Args:
            tree_batch: a flattened ``TreeBatch`` with ``node_features``,
                ``edge_parent``, ``edge_child``, ``edge_slot``, ``parent_index``,
                ``root_index``, and ``depth`` tensors.
        """
        node_features = tree_batch.node_features.to(self.device)
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
                parent_index,
            )

        return TreeEncoderOutput(
            node_states=node_states,
            root_states=node_states[root_index],
        )


class ChildWdlHead(nn.Module):
    """Slot-conditioned MLP that predicts a child's WDL logits from its parent.

    Concatenates the parent's encoded state with the child's slot embedding
    and passes both through a small MLP. Used as the pretraining head on top
    of ``TreeEncoder``.
    """

    def __init__(self, d_embed, hidden_dim=128, device="cpu"):
        """
        Args:
            d_embed: encoder embedding dimension (sets the input width to ``2 * d_embed``).
            hidden_dim: hidden width of the MLP.
            device: device for the MLP parameters.
        """
        super().__init__()
        device = torch.device(device)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_embed, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3, device=device),
        )

    def forward(self, parent_states, slot_states):
        """Return [E, 3] WDL logits given per-edge parent states and slot embeddings."""
        return self.mlp(torch.cat([parent_states, slot_states], dim=-1))


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
        """
        Args:
            k, node_feat, device, node_embed_hidden, d_embed, d_message,
            n_heads, d_att, sequential: forwarded to ``TreeEncoder`` when constructing
                a fresh encoder.
            decoder_hidden: hidden width of the ``ChildWdlHead`` MLP.
            encoder: optional pre-existing ``TreeEncoder`` instance; if provided, the
                constructor arguments above are ignored and the head is wired to
                this encoder's ``d_embed`` and device.
        """
        super().__init__()
        # Build a fresh encoder unless the caller hands one in (used when
        # fine-tuning or sharing an encoder across heads).
        if encoder is None:
            encoder = TreeEncoder(
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

    def forward(self, tree_batch) -> torch.Tensor:
        """Encode the batch, then return per-edge WDL logits.

        Returns just the ``[E, 3]`` logits tensor; the wrapped encoder's
        intermediate node/root states are not exposed because no consumer
        reads them through this head.
        """
        encoded = self.encoder(tree_batch)
        edge_parent = tree_batch.edge_parent.to(self.encoder.device)
        edge_slot = tree_batch.edge_slot.to(self.encoder.device)
        # Re-encode slot indices through the encoder's projection so the head
        # sees the same slot representation the encoder used internally.
        slot_states = self.encoder.slot_embeddings(edge_slot)
        return self.child_wdl_head(encoded.node_states[edge_parent], slot_states)
