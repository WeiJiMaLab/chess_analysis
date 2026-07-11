"""Operates on a flattened batch (edge_parent/edge_child indexed), not a padded
dense tensor, since child counts vary widely -- hence segmented scatter instead
of F.softmax + matmul."""

import math

import torch.nn as nn


class TreeAttMsgLayer(nn.Module):
    """One step of slot-conditioned attention message passing.

    For each parent node ``i`` in the batched tree, compute a query from its
    current activation and attend over the keys/values of its children
    (optionally with an additive slot positional encoding on the child side).
    The aggregated per-parent vector becomes the upward "message" that the
    encoder's outer loop combines with the parent's own state.

    Shapes use:
        N = total nodes in the batch
        E = total parent→child edges in the batch
    Both are flattened across trees in the batch.
    """

    def __init__(self, n_heads, d_att, d_embed, d_message=None, device="cpu"):
        """
        Args:
            n_heads: number of attention heads.
            d_att: per-head key/query/value dimension.
            d_embed: input node activation dimension.
            d_message: output message dimension; defaults to ``d_embed``.
            device: device on which the linear-projection parameters live.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_att = d_att
        self.d_embed = d_embed
        self.d_message = d_embed if d_message is None else d_message
        self.model_width = n_heads * d_att  # total width across all heads, used for fused proj dims

        # Standard Q/K/V/O projections. No biases — matches the convention
        # used elsewhere in the encoder and saves a handful of parameters.
        self.W_q = nn.Linear(d_embed, self.model_width, bias=False, device=device)
        self.W_k = nn.Linear(d_embed, self.model_width, bias=False, device=device)
        self.W_v = nn.Linear(d_embed, self.model_width, bias=False, device=device)
        self.W_o = nn.Linear(self.model_width, self.d_message, bias=False, device=device)

    def forward(self, tree_acts, edge_parent, edge_child, edge_slot_embed=None):
        """Compute one upward message per node in the flattened batch.

        Args:
            tree_acts: [N, d_embed] current node activations.
            edge_parent: [E] long tensor; for each edge, the parent's node id.
            edge_child: [E] long tensor; for each edge, the child's node id.
            edge_slot_embed: optional [E, d_embed] additive encoding mixed
                into the child's key/value (the slot positional encoding).

        Returns:
            [N, d_message] tensor; entry ``i`` is the aggregated message for
            parent ``i``. Nodes that have no children (leaves) get zero.
        """
        n_nodes, _ = tree_acts.shape

        # Edgeless case (e.g. all-root mini-batch): every node returns the
        # zero message. Short-circuit so the scatter ops below don't see
        # empty index tensors with ambiguous dtype/device.
        if edge_child.numel() == 0:
            head_messages = tree_acts.new_zeros(n_nodes, self.n_heads, self.d_att)
            return self.W_o(head_messages.reshape(n_nodes, self.model_width))

        # Project every node into Q-space once; per-edge K and V are
        # projected from the child's activation (plus optional slot embed).
        query = self.W_q(tree_acts).view(n_nodes, self.n_heads, self.d_att)
        edge_input = tree_acts[edge_child]
        if edge_slot_embed is not None:
            edge_input = edge_input + edge_slot_embed
        edge_key = self.W_k(edge_input).view(-1, self.n_heads, self.d_att)
        edge_value = self.W_v(edge_input).view(-1, self.n_heads, self.d_att)

        # Scaled dot-product score per (parent, child) edge, per head.
        edge_query = query[edge_parent]
        logits = (edge_key * edge_query).sum(dim=-1) / math.sqrt(self.d_att)

        # Segmented softmax (per parent), implemented as scatter ops because
        # child counts vary. Standard log-sum-exp stabilization: subtract
        # each parent's max logit before exponentiating to avoid overflow.
        expanded_parent = edge_parent.unsqueeze(-1).expand(-1, self.n_heads)
        parent_max = logits.new_full((n_nodes, self.n_heads), float("-inf"))
        parent_max.scatter_reduce_(0, expanded_parent, logits, reduce="amax", include_self=True)

        stabilized = logits - parent_max[edge_parent]
        exp_logits = stabilized.exp()

        parent_sums = logits.new_zeros(n_nodes, self.n_heads)
        parent_sums.scatter_add_(0, expanded_parent, exp_logits)
        attn = exp_logits / parent_sums[edge_parent].clamp_min(1e-12)

        # Weighted-sum aggregation: again scatter-add to handle variable
        # child counts without padding. Output per parent, per head.
        weighted_values = attn.unsqueeze(-1) * edge_value
        head_messages = tree_acts.new_zeros(n_nodes, self.n_heads, self.d_att)
        head_messages.scatter_add_(
            0,
            edge_parent.view(-1, 1, 1).expand(-1, self.n_heads, self.d_att),
            weighted_values,
        )
        return self.W_o(head_messages.reshape(n_nodes, self.model_width))
