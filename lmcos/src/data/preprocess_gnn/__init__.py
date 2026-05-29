"""Encoder pretraining preprocessing chain.

Three stages, each independently runnable:

1. ``cts.data.preprocess_gnn.split``           — train/validation split of generated trees
2. ``cts.data.preprocess_gnn.derive_prefixes`` — emit prefix snapshots per tree
3. ``cts.data.preprocess_gnn.pack``            — pack prefix snapshots into compact shards
"""
