"""Orchestrator that submits one sbatch job per shard of a FEN dataset.

Reads a base ``cts.data.build_tree`` YAML config, counts the FENs at
``config.fens``, slices the index range into contiguous shards of size
``--shard-size``, writes one derived YAML per shard (the base config with
``start_index``/``end_index`` overridden), and emits one sbatch
invocation per shard against ``slurm/generate_dataset_shard.slurm`` with
``CONFIG=<shard.yaml>``. Commands are printed by default; ``--submit``
actually schedules them.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import yaml


def count_fens(path: Path) -> int:
    """Return the number of non-blank lines in ``path`` (one FEN per line)."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    """Parse CLI args, slice the FEN file into shards, write per-shard YAMLs, submit."""
    parser = argparse.ArgumentParser(description="Submit sharded dataset-generation jobs.")
    parser.add_argument("--config", required=True, help="base YAML config for cts.data.build_tree")
    parser.add_argument("--shard-size", type=int, default=6250)
    parser.add_argument(
        "--shard-config-dir",
        default=None,
        help="where to write per-shard YAMLs (default: <config_dir>/<config_stem>_shards/)",
    )
    parser.add_argument("--project-dir", default="/home/ysagiv/chess/cts/async_soph")
    parser.add_argument("--job-name-prefix", default="cts-gen")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--time", default="01:00:00")
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--mem")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_config_path}")
    base_config = yaml.safe_load(base_config_path.read_text())

    fens_path = Path(base_config["fens"]).resolve()
    if not fens_path.exists():
        raise FileNotFoundError(f"Missing FEN file: {fens_path}")

    project_dir = Path(args.project_dir).resolve()
    slurm_script = project_dir / "slurm" / "generate_dataset_shard.slurm"
    if not slurm_script.exists():
        raise FileNotFoundError(f"Missing Slurm script: {slurm_script}")

    total_fens = count_fens(fens_path)
    if total_fens == 0:
        raise ValueError("FEN file is empty.")
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive.")

    # Ceiling division: the final shard may be partial (covers the tail).
    n_shards = math.ceil(total_fens / args.shard_size)

    shard_config_dir = (
        Path(args.shard_config_dir).resolve()
        if args.shard_config_dir
        else base_config_path.parent / f"{base_config_path.stem}_shards"
    )
    shard_config_dir.mkdir(parents=True, exist_ok=True)

    print(f"total_fens={total_fens}")
    print(f"shard_size={args.shard_size}")
    print(f"n_shards={n_shards}")
    print(f"shard_config_dir={shard_config_dir}")

    commands: list[list[str]] = []
    for shard_id in range(n_shards):
        start_index = shard_id * args.shard_size
        # Clamp end_index for the trailing partial shard.
        end_index = min(total_fens, start_index + args.shard_size)

        shard_config = dict(base_config)
        shard_config["start_index"] = start_index
        shard_config["end_index"] = end_index
        shard_config_path = shard_config_dir / f"shard_{shard_id:05d}.yaml"
        shard_config_path.write_text(yaml.safe_dump(shard_config, sort_keys=False))

        # "ALL," preserves the submitter's environment in addition to the keys
        # listed; otherwise sbatch would drop everything else (PATH, modules, etc).
        export_arg = f"ALL,PROJECT_DIR={project_dir},CONFIG={shard_config_path}"
        command = ["sbatch"]
        if args.job_name_prefix:
            command.extend(["--job-name", f"{args.job_name_prefix}-{shard_id:05d}"])
        if args.account:
            command.extend(["--account", args.account])
        if args.partition:
            command.extend(["--partition", args.partition])
        if args.time:
            command.extend(["--time", args.time])
        if args.cpus_per_task is not None:
            command.extend(["--cpus-per-task", str(args.cpus_per_task)])
        if args.mem:
            command.extend(["--mem", args.mem])
        command.extend(["--export", export_arg, str(slurm_script)])
        commands.append(command)
        print(" ".join(command))

    # Two-phase design: always print so the user can dry-run and audit the
    # commands first; only mutate the queue when --submit is passed.
    if args.submit:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
