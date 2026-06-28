"""Themed sub-modules for the budgeted-controller diagnostics analyzer.

The parent script ``cts.analysis.analyze_budgeted_controller_run`` wires
these together; each sub-module owns one bucket of plots/summaries
(training curves, regret, calibration, oracle-stop drivers, root-action
churn, future-worse move switching, halt-reward trajectories, baselines,
and the Markdown report).
"""
