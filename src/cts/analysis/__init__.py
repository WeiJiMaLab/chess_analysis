"""Diagnostics tools for trained CTS models.

- ``zt_probe`` — full-split topology decodability probe for the materialized
  ``z_t`` encoder cache (linear + small-MLP R^2 against tree-stats targets).
  Invoked directly (``python -m cts.analysis.zt_probe``) and by
  ``slurm/pipeline/zt_probe.slurm``; its helpers (``_linear_r2``, ``_mlp_r2``,
  ``_episode_split_mask``) are also imported by ``analysis.evaluate``.
"""
