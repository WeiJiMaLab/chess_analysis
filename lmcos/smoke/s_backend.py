"""S-backend: cross-backend numeric drift + determinism on a shared FEN set.

Report §5 ID: **S-backend** — "ONNX-CPU vs ONNX-GPU vs lc0; same backend twice".
What it proves: the L2 (evaluator) bar of report §2 — that our in-process net
returns value/WDL/policy close enough to lc0 (and to itself across devices) that
trajectories won't fork on numeric noise (report §9, risk: "cross-backend drift
forks trajectories"). It compares raw ``PositionEval`` outputs, NOT trees, so it
isolates evaluator numerics from search logic.

Backends compared (each a name -> ``Evaluator``):
  - ``net-cpu``   : ``NetEvaluator(backend=ONNX, device="cpu")``
  - ``net-gpu``   : ``NetEvaluator(backend=ONNX, device="cuda")``
  - ``lc0``       : ``Lc0UciEvaluator`` wrapping the real engines (the reference)
Any backend that isn't available (no GPU, scaffolded net) is skipped with a
note rather than failing the whole run.

Metrics per backend pair: max/mean abs diff on value and on the WDL vector,
and policy-argmax agreement (fraction of FENs whose top-prior move matches).
Determinism: run the lead backend twice and assert bit-identical PositionEvals.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack

from _lc0_baseline import lc0_direct_provider
from _smoke_common import (
    ScaffoldNotReady,
    SmokeResult,
    add_common_arguments,
    import_symbol,
    run_smoke,
    sample_fens,
)

VALUE_TOL = 1e-4  # fp32 cross-device tolerance; loosen for fp16 backends
WDL_TOL = 1e-4


def _build_backends(args, stack: ExitStack):
    """Return ``({name: evaluator}, skipped)``; keep the lc0 engines alive on ``stack``.

    A backend that raises (no GPU, net still a stub) is recorded in ``skipped``
    with the reason and omitted; the test still runs on whatever is available.
    The lc0 provider is entered on the caller's ExitStack so its engines stay
    alive through ``.evaluate()`` and are torn down when the caller exits.
    """
    backends = {}
    skipped = {}

    def _attempt(name, factory):
        try:
            backends[name] = factory()
        except (ScaffoldNotReady, NotImplementedError, Exception) as exc:  # noqa: BLE001
            skipped[name] = f"{type(exc).__name__}: {exc}"

    NetEvaluator = import_symbol("net_evaluator", "NetEvaluator")
    NetBackend = import_symbol("net_evaluator", "NetBackend")
    Lc0UciEvaluator = import_symbol("evaluator", "Lc0UciEvaluator")

    # ONNX-RT runs the same graph on CPU and GPU, so it is the backend that
    # makes the net-cpu vs net-gpu comparison apples-to-apples (report §6).
    _attempt("net-cpu", lambda: NetEvaluator(args.lc0_weights, backend=NetBackend.ONNX, device="cpu"))
    _attempt("net-gpu", lambda: NetEvaluator(args.lc0_weights, backend=NetBackend.ONNX, device="cuda"))

    def _build_lc0():
        provider = stack.enter_context(lc0_direct_provider(args.lc0_binary, args.lc0_weights))
        return Lc0UciEvaluator(provider)

    _attempt("lc0", _build_lc0)
    return backends, skipped


def _argmax_move(position_eval):
    """Return the highest-prior move for a PositionEval (priors: move -> float)."""
    priors = position_eval.priors
    if not priors:
        return None
    return max(priors.items(), key=lambda item: item[1])[0]


def _compare(evals_a, evals_b):
    """Max/mean abs diff on value + WDL, plus policy-argmax agreement."""
    value_diffs = [abs(float(a.value) - float(b.value)) for a, b in zip(evals_a, evals_b)]
    wdl_diffs = [
        max(abs(float(x) - float(y)) for x, y in zip(a.wdl, b.wdl))
        for a, b in zip(evals_a, evals_b)
    ]
    argmax_agree = [
        _argmax_move(a) == _argmax_move(b) for a, b in zip(evals_a, evals_b)
    ]
    n = max(len(value_diffs), 1)
    return {
        "value_max_abs_diff": max(value_diffs) if value_diffs else 0.0,
        "value_mean_abs_diff": sum(value_diffs) / n,
        "wdl_max_abs_diff": max(wdl_diffs) if wdl_diffs else 0.0,
        "wdl_mean_abs_diff": sum(wdl_diffs) / n,
        "policy_argmax_agreement": sum(argmax_agree) / n,
    }


def _main(args: argparse.Namespace) -> SmokeResult:
    fens = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=args.n,
        seed=args.seed,
    )
    # ExitStack keeps the lc0 engines alive across .evaluate() and the
    # determinism re-run, then tears them down on scope exit.
    with ExitStack() as stack:
        backends, skipped = _build_backends(args, stack)
        if not backends:
            raise ScaffoldNotReady(
                f"No evaluator backend could be constructed. Skipped: {skipped}"
            )

        evals = {name: evaluator.evaluate(fens) for name, evaluator in backends.items()}

        # Determinism: re-evaluate with the lead backend (lc0 if present, else
        # the first available) and require identical outputs.
        lead = "lc0" if "lc0" in backends else next(iter(backends))
        backends[lead].clear_caches()
        evals_repeat = backends[lead].evaluate(fens)

    determinism = _compare(evals[lead], evals_repeat)
    is_deterministic = (
        determinism["value_max_abs_diff"] == 0.0
        and determinism["wdl_max_abs_diff"] == 0.0
        and determinism["policy_argmax_agreement"] == 1.0
    )

    # Pairwise comparisons against lc0 when present, else against the lead.
    reference = "lc0" if "lc0" in backends else lead
    pair_metrics = {}
    for name in backends:
        if name == reference:
            continue
        pair_metrics[f"{name}_vs_{reference}"] = _compare(evals[name], evals[reference])

    within_tol = all(
        metrics["value_max_abs_diff"] <= VALUE_TOL and metrics["wdl_max_abs_diff"] <= WDL_TOL
        for metrics in pair_metrics.values()
    )
    return SmokeResult(
        smoke_id="S-backend",
        passed=bool(is_deterministic and within_tol),
        summary={
            "n_fens": len(fens),
            "backends_available": list(backends),
            "backends_skipped": skipped,
            "reference_backend": reference,
            "determinism_backend": lead,
            "deterministic": is_deterministic,
            "value_tol": VALUE_TOL,
            "wdl_tol": WDL_TOL,
            "pairwise": pair_metrics,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=256)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_smoke(_main, _parse_args()))
