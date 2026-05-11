import math

import torch
import torch.nn as nn


class TreeAttMsgLayer(nn.Module):
    """
    Multi-head attention over a node's children in the flattened tree batch.

    Input node states are shaped [N, d_embed]. For each parent node i, the layer
    attends over the child node states listed in child_ptr / children_index and
    returns one aggregated upward message per node shaped [N, d_message].
    """

    def __init__(self, n_heads, d_att, d_embed, d_message=None, device="cpu"):
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_att <= 0:
            raise ValueError("d_att must be positive.")

        self.n_heads = n_heads
        self.d_att = d_att
        self.d_embed = d_embed
        self.d_message = d_embed if d_message is None else d_message
        self.model_width = n_heads * d_att

        self.W_q = nn.Linear(d_embed, self.model_width, bias=False, device=device)
        self.W_k = nn.Linear(d_embed, self.model_width, bias=False, device=device)
        self.W_v = nn.Linear(d_embed, self.model_width, bias=False, device=device)
        self.W_o = nn.Linear(self.model_width, self.d_message, bias=False, device=device)

    def forward(self, tree_acts, edge_parent, edge_child, edge_slot_embed=None):
        n_nodes, _ = tree_acts.shape
        if edge_child.numel() == 0:
            head_messages = tree_acts.new_zeros(n_nodes, self.n_heads, self.d_att)
            return self.W_o(head_messages.reshape(n_nodes, self.model_width))

        query = self.W_q(tree_acts).view(n_nodes, self.n_heads, self.d_att)
        edge_input = tree_acts[edge_child]
        if edge_slot_embed is not None:
            edge_input = edge_input + edge_slot_embed
        edge_key = self.W_k(edge_input).view(-1, self.n_heads, self.d_att)
        edge_value = self.W_v(edge_input).view(-1, self.n_heads, self.d_att)

        edge_query = query[edge_parent]
        logits = (edge_key * edge_query).sum(dim=-1) / math.sqrt(self.d_att)

        expanded_parent = edge_parent.unsqueeze(-1).expand(-1, self.n_heads)
        parent_max = logits.new_full((n_nodes, self.n_heads), float("-inf"))
        parent_max.scatter_reduce_(0, expanded_parent, logits, reduce="amax", include_self=True)

        stabilized = logits - parent_max[edge_parent]
        exp_logits = stabilized.exp()

        parent_sums = logits.new_zeros(n_nodes, self.n_heads)
        parent_sums.scatter_add_(0, expanded_parent, exp_logits)
        attn = exp_logits / parent_sums[edge_parent].clamp_min(1e-12)

        weighted_values = attn.unsqueeze(-1) * edge_value
        head_messages = tree_acts.new_zeros(n_nodes, self.n_heads, self.d_att)
        head_messages.scatter_add_(
            0,
            edge_parent.view(-1, 1, 1).expand(-1, self.n_heads, self.d_att),
            weighted_values,
        )
        return self.W_o(head_messages.reshape(n_nodes, self.model_width))
