"""
Smoke-test `GeneratedTreeHaltEnv` without training a controller.

Requirements
------------
Each path in ``example_paths`` must be a ``torch.save``d `PretrainExample` (has ``.tree``).
Typical CTS **raw** checkpoints (``cts_raw_pretrain_example_v1`` dicts in
``generated_trees_oracle*_...`` shards) **do not** qualify — ``GeneratedTreeHaltEnv``
will raise when loading.

You **must supply** paths (manifest, directory of ``*.pt``, or rely on ``--minimal`` fallback).
On each ``reset()``, if ``shuffle=True`` the env **picks uniformly** among those paths;
if ``shuffle=False`` it cycles in sorted order via an internal RNG/seed convention.

Paths can be anchored with ``repo_root / ...`` when run from checkout.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_LMCOS = _REPO / "lmcos"
if str(_LMCOS) not in sys.path:
    sys.path.insert(0, str(_LMCOS))

import torch  # noqa: E402
from cts_episode_envs import GeneratedTreeHaltEnv  # noqa: E402
from cts_pretrain import PretrainExample, TeacherSearchConfig  # noqa: E402
from tree import ExpansionChild, SearchTree  # noqa: E402


def _minimal_pretrain_example() -> PretrainExample:
    """Same shape as supervised_branch tests needing non-trivial expansion order."""
    tree = SearchTree()
    root_id = tree.create_root("root", {"value": 0.0, "prior": 1.0})
    root_children = tree.add_children(
        root_id,
        [
            ExpansionChild("a", "a", {"value": 0.1, "prior": 0.5}),
            ExpansionChild("b", "b", {"value": 0.2, "prior": 0.5}),
        ],
    )
    b_children = tree.add_children(
        root_children[1],
        [ExpansionChild("b1", "b1", {"value": 0.3, "prior": 1.0})],
    )
    tree.add_children(
        b_children[0],
        [ExpansionChild("b1a", "b1a", {"value": -0.4, "prior": 1.0}, is_terminal=True)],
    )
    tree.add_children(
        root_children[0],
        [ExpansionChild("a1", "a1", {"value": 0.4, "prior": 1.0}, is_terminal=True)],
    )
    return PretrainExample(tree=tree, node_target_values=[0.0] * tree.num_nodes())


def _load_paths(train_data: str | None) -> list[str]:
    if train_data is None:
        return []
    p = Path(train_data)
    if p.is_dir():
        return sorted(str(x) for x in p.rglob("*.pt") if x.is_file())
    if p.is_file() and train_data.endswith(".pt"):
        return [train_data]
    with open(train_data, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Directory of *.pt PretrainExamples, single .pt file, or text manifest (one path per line). "
        "If omitted, use --minimal or the script exits with a message.",
    )
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="Ignore --paths and use a tiny synthetic PretrainExample in a tempfile (matches lmcos test pattern).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle which example is chosen each reset()")
    ap.add_argument("--continue-cost", type=float, default=0.05)
    args = ap.parse_args()

    quality_config = TeacherSearchConfig(
        max_depth=10,
        search_budget=64,
        c_puct=1.0,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="smoke_generated_tree_halt_env",
    )

    paths: list[str]
    cleanup: tempfile.TemporaryDirectory | None = None

    if args.minimal:
        cleanup = tempfile.TemporaryDirectory()
        tmp = Path(cleanup.name) / "minimal.pt"
        torch.save(_minimal_pretrain_example(), tmp)
        paths = [str(tmp)]
    else:
        if not args.paths:
            raise SystemExit(
                "Provide --paths to PretrainExample .pt files/dir/manifest, or use --minimal. "
                "Raw cts_raw_pretrain_example_v1 shards (e.g. many ysagiv oracle*_trace .pt files) will not load here."
            )
        paths = _load_paths(args.paths)
        if not paths:
            raise SystemExit(f"No .pt paths found from {args.paths!r}")

    try:
        env = GeneratedTreeHaltEnv(
            example_paths=paths,
            quality_config=quality_config,
            continue_cost=args.continue_cost,
            seed=args.seed,
            shuffle=args.shuffle,
            max_cache_size=8,
        )

        tree = env.reset()
        print(f"n_paths={len(paths)} first_path={paths[0]}")
        print(f"reset(): root_children={bool(tree.root_children())} num_nodes={tree.num_nodes()}")

        done = False
        steps = 0
        step_result = None
        while not done:
            step_result = env.step(0)
            steps += 1
            done = step_result.done
            if steps > 10_000:
                raise RuntimeError("infinite loop guard")

        assert step_result is not None and step_result.info
        info = step_result.info
        print(f"stopped after_step={steps} episode_return={info.get('episode_return')} expansions={info.get('expansions')}")
        print(f"info keys sample: episode_return terminal_quality halted example_path...")
        print(f"  terminal_quality={info.get('terminal_quality')} halted={info.get('halted')}")
        print(f"  example_path={info.get('example_path')}")
        print("ok")
    finally:
        if cleanup is not None:
            cleanup.cleanup()


if __name__ == "__main__":
    main()
