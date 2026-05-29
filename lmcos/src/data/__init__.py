"""Data generation, target preprocessing, and on-disk packing for CTS.

- ``cts.data.build_tree``         — engine-backed tree generation (PUCT + lc0)
- ``cts.data.sample_fens``        — root-position sampling pipeline
- ``cts.data.validate_fens``      — bestmove-instability filtering on root FENs
- ``cts.data.preprocess_gnn.*``   — encoder pretraining target chain (split → derive prefixes → pack)
- ``cts.data.preprocess_mc.*``    — controller target chain (budgeted oracle → pack episodes → materialize encoder cache)
- ``cts.data.filter_packed_episodes`` — post-filter on packed controller episodes
"""
