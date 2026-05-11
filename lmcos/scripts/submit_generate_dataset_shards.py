from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path


def count_fens(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_sbatch_command(
    slurm_script: Path,
    project_dir: Path,
    fens_path: Path,
    output_root: Path,
    shard_id: int,
    start_index: int,
    end_index: int,
    args: argparse.Namespace,
) -> list[str]:
    exports = {
        "PROJECT_DIR": str(project_dir),
        "FENS_PATH": str(fens_path),
        "OUTPUT_ROOT": str(output_root),
        "SHARD_ID": str(shard_id),
        "START_INDEX": str(start_index),
        "END_INDEX": str(end_index),
        "PYTHON_BIN": args.python_bin,
        "ENGINE_PATH": args.engine_path,
        "WEIGHTS_PATH": args.weights_path,
        "BACKEND": args.backend,
        "MULTIPV": str(args.multipv),
        "MAX_DEPTH": str(args.max_depth),
        "SEARCH_BUDGET": str(args.search_budget),
        "MIN_NODES": str(args.min_nodes),
        "MAX_NODES": str(args.max_nodes),
        "SEED": str(args.seed),
        "CACHE_CLEAR_INTERVAL": str(args.cache_clear_interval),
        "LOG_INTERVAL": str(args.log_interval),
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
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
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit sharded dataset-generation jobs.")
    default_project_dir = Path("/home/ysagiv/chess/CTS")
    default_data_dir = Path("/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data")
    default_fens = default_data_dir / "roots" / "sampled_root_fens_2023.txt"
    default_output_root = default_data_dir / "generated_trees"

    parser.add_argument("--fens", default=str(default_fens))
    parser.add_argument("--output-root", default=str(default_output_root))
    parser.add_argument("--shard-size", type=int, default=6250)
    parser.add_argument("--project-dir", default=str(default_project_dir))
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--engine-path", default="/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0")
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-clear-interval", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--job-name-prefix", default="cts-gen")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    fens_path = Path(args.fens).resolve()
    output_root = Path(args.output_root).resolve()
    slurm_script = project_dir / "slurm" / "generate_dataset_shard.slurm"

    if not fens_path.exists():
        raise FileNotFoundError(f"Missing FEN file: {fens_path}")
    if not slurm_script.exists():
        raise FileNotFoundError(f"Missing Slurm script: {slurm_script}")

    default_weights_path = Path("/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz")
    weights_path = Path(args.weights_path) if args.weights_path else default_weights_path
    args.weights_path = str(weights_path.resolve())

    total_fens = count_fens(fens_path)
    if total_fens == 0:
        raise ValueError("FEN file is empty.")
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive.")

    n_shards = math.ceil(total_fens / args.shard_size)
    print(f"total_fens={total_fens}")
    print(f"shard_size={args.shard_size}")
    print(f"n_shards={n_shards}")
    print(f"output_root={output_root}")

    commands: list[list[str]] = []
    for shard_id in range(n_shards):
        start_index = shard_id * args.shard_size
        end_index = min(total_fens, start_index + args.shard_size)
        command = build_sbatch_command(
            slurm_script=slurm_script,
            project_dir=project_dir,
            fens_path=fens_path,
            output_root=output_root,
            shard_id=shard_id,
            start_index=start_index,
            end_index=end_index,
            args=args,
        )
        commands.append(command)
        print(" ".join(command))

    if args.submit:
        output_root.mkdir(parents=True, exist_ok=True)
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
