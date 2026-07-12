"""FEN/tree survival funnel per top-level pipeline config -- how many FEN board
positions (or ysagiv trees) survive each filtering stage, from the raw pool down
to the final train/validation split. Writes one fen_funnel.csv per config into
the config's own outputs/tables/<mirror of figures_dir> subfolder.

Reuses render_stage.py's config resolution (extends/variants/interpolation) via
subprocess so paths/table names never drift from what the pipeline actually reads.

    python -m analysis.fen_funnel [CONFIG.yaml ...]   # defaults to all 5 root configs
"""
import csv
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER_STAGE = os.path.join(REPO_ROOT, "render_stage.py")

DEFAULT_CONFIGS = [
    "config_minply15_maxply75.yaml",
    "config_ysagiv_xaba100k_minply15_maxply75.yaml",
    "config_ysagiv_xaba100k_minply15_maxply75_history.yaml",
    "config_ysagiv_xaba20k.yaml",
    "config_ysagiv_xaba20k_history.yaml",
]

# ysagiv's read-only source corpus (see slurm/pipeline/ysagiv_sample.slurm) -- external
# to every config, so it has no config key to resolve it from.
YSAGIV_SOURCE_CORPUS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
# The xaba-exclusion filter's own intermediate output (see ysagiv_xaba_filter.slurm) --
# some configs (e.g. xaba100k_minply15_maxply75) stack a ply-window filter on top of it
# (ysagiv_ply_window_filter.slurm's hardcoded --restrict-to), in which case split.include_list
# points past this intermediate file rather than at it.
YSAGIV_XABA_ONLY_INCLUDE_LIST = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/xaba/include_list.txt"


def get(config: str, key: str) -> str:
    result = subprocess.run(
        ["python3", RENDER_STAGE, config, "--get", key],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise KeyError(key)
    return result.stdout.strip()


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def is_own_corpus(config: str) -> bool:
    try:
        get(config, "treegen.command")
        return True
    except KeyError:
        return False


def figures_dir(config: str) -> str:
    try:
        return get(config, "human_analysis.figures_dir")
    except KeyError:
        return get(config, "globals.figures_dir")


def own_corpus_funnel(config: str):
    db = get(config, "human_analysis.selected_db_default")
    t_raw = get(config, "human_analysis.table_processed_moves_nonzero")
    t_filtered = get(config, "human_analysis.table_filtered")
    min_ply = get(config, "globals.min_ply")
    max_ply = get(config, "globals.max_ply")
    fens_sample = get(config, "globals.fens_sample")
    include_list = get(config, "split.include_list")
    split_root = get(config, "split.split_root")

    import duckdb
    conn = duckdb.connect(db, read_only=True)
    raw_n = conn.execute(f"SELECT count(DISTINCT fen) FROM {t_raw}").fetchone()[0]
    filtered_n = conn.execute(f"SELECT count(DISTINCT fen) FROM {t_filtered}").fetchone()[0]
    conn.close()

    return [
        ("raw_fen_pool",
         "distinct FEN board positions in personal.db (all plies, min_elo/date-filtered games)",
         raw_n),
        (f"ply_window_filter_{min_ply}_{max_ply}",
         f"distinct FENs surviving the ply {min_ply}-{max_ply} window (filtered_moves table)",
         filtered_n),
        ("tree_gen_subsample",
         "reservoir-sampled distinct FENs used as tree-generation roots (fens_sample.txt)",
         count_lines(fens_sample)),
        ("exclusion_filter",
         "trees surviving the argmax>2 'thinking helps' filter (split.include_list)",
         count_lines(include_list)),
        ("train_split",
         "training-set episodes (train_manifest.txt)",
         count_lines(os.path.join(split_root, "train_manifest.txt"))),
        ("validation_split",
         "validation-set episodes (validation_manifest.txt)",
         count_lines(os.path.join(split_root, "validation_manifest.txt"))),
    ]


def ysagiv_corpus_funnel(config: str):
    source_root = get(config, "split.source_root")
    include_list = get(config, "split.include_list")
    split_root = get(config, "split.split_root")

    rows = [
        ("source_corpus",
         "ysagiv human_trees read-only source corpus (all root FENs)",
         len(os.listdir(YSAGIV_SOURCE_CORPUS))),
        ("tree_subsample",
         "seeded random subsample of the source corpus (split.source_root)",
         len(os.listdir(source_root))),
    ]

    if os.path.abspath(include_list) == os.path.abspath(YSAGIV_XABA_ONLY_INCLUDE_LIST):
        rows.append((
            "xaba_exclusion_filter",
            "subsample survivors of the xaba-exclusion filter (split.include_list)",
            count_lines(include_list),
        ))
    else:
        rows.append((
            "xaba_exclusion_filter",
            "subsample survivors of the xaba-exclusion filter (intermediate; shared across configs)",
            count_lines(YSAGIV_XABA_ONLY_INCLUDE_LIST),
        ))
        rows.append((
            "ply_window_filter_15_75",
            "xaba survivors whose root FEN also falls at ply 15-75 in our own move DB (split.include_list)",
            count_lines(include_list),
        ))

    rows.append((
        "train_split",
        "training-set episodes (train_manifest.txt)",
        count_lines(os.path.join(split_root, "train_manifest.txt")),
    ))
    rows.append((
        "validation_split",
        "validation-set episodes (validation_manifest.txt)",
        count_lines(os.path.join(split_root, "validation_manifest.txt")),
    ))
    return rows


def write_funnel(config: str) -> str:
    rows = own_corpus_funnel(config) if is_own_corpus(config) else ysagiv_corpus_funnel(config)
    fdir = figures_dir(config)
    out_dir = fdir.replace("outputs/figures", "outputs/tables", 1)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fen_funnel.csv")

    raw_n = rows[0][2]
    prev_n = None
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "description", "n_fens", "pct_of_previous_stage", "pct_of_raw_pool"])
        for name, desc, n in rows:
            pct_prev = f"{100 * n / prev_n:.2f}" if prev_n else "100.00"
            pct_raw = f"{100 * n / raw_n:.2f}" if raw_n else ""
            w.writerow([name, desc, n, pct_prev, pct_raw])
            prev_n = n
    return out_path


def main(argv=None):
    configs = argv or DEFAULT_CONFIGS
    for config in configs:
        out_path = write_funnel(config)
        print(f"{config} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
