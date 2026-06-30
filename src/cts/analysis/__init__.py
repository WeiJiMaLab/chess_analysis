"""Diagnostics, evaluation, and plotting tools for trained CTS models.

Main tools:

- ``analyze_budgeted_controller_run`` — the end-of-run dashboard for one
  trained controller: 25+ plots, ``summary.json``, ``report.md``. Body
  lives in themed sub-modules under ``analysis/``.
- ``analyze_compute_advantage_training_log`` — parse a training ``.out``
  log into per-epoch metrics.
- ``analyze_oversearch_preference_evolution`` — track the controller's
  preferred move as a function of how long it has been searching.
- ``analyze_tree_stratification`` — bucket source trees by stratification
  axes (stop-depth excess, budget-action variance).
- ``compare_controller_diagnostics`` — diff two trained controllers'
  per-episode diagnostics.
- ``evaluate_controller`` — closed-loop replay of a trained controller
  against the budgeted oracle on held-out data.
- ``plot_advantage_loss_from_log`` — quick visualization helper.

Shared regex/parse/decomposition helpers live in ``cts.analysis._common``.
"""
