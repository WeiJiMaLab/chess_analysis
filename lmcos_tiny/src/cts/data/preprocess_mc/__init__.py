"""Meta-controller training data preprocessing chain.

- ``cts.data.preprocess_mc.oracle``      — budgeted oracle that produces halt/expand targets per snapshot
- ``cts.data.preprocess_mc.pack``        — pack controller episodes into on-disk shards
- ``cts.data.preprocess_mc.materialize`` — precompute frozen-encoder ``z_root`` cache for fast training
"""
