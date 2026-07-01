#!/bin/bash
# Submit the whole pipeline with chained dependencies, driven by pipeline.yaml.
# Reads the manifest (step order + `after` deps), submits each step in order, and
# wires Slurm afterok on the parents' job ids. `preprocess` is NOT submitted here
# (run `bash slurm/pipeline/preprocess.slurm` separately); steps depending on it
# simply start immediately.
#
#   bash slurm/pipeline/submit_all.sh            # submit for real
#   DRY_RUN=1 bash slurm/pipeline/submit_all.sh  # print the sbatch plan only
set -euo pipefail

TINY=/home/hl4291/chess_analysis
cd "$TINY"
MANIFEST="$TINY/slurm/pipeline/pipeline.yaml"
DRY_RUN="${DRY_RUN:-0}"

# Parse manifest -> one "name|script|array|after_csv|merge" line per step, in order.
# '|' (not TAB) is the delimiter so empty fields (e.g. no array) are preserved.
mapfile -t STEPS < <(python - "$MANIFEST" <<'PY'
import sys, yaml
for s in yaml.safe_load(open(sys.argv[1]))["steps"]:
    print("|".join([s["name"], s["script"], str(s.get("array", "")),
                    ",".join(s.get("after", [])), str(s.get("merge", ""))]))
PY
)

declare -A JOBID   # step name -> job id downstream should depend on

# submit <depstr> <array> <script...>  -> echoes the parsable job id
submit() {
  local depstr="$1" array="$2"; shift 2
  local a=(sbatch --parsable)
  [[ -n "$array" ]] && a+=(--array="$array")
  [[ -n "$depstr" ]] && a+=("--dependency=afterok:${depstr}")
  a+=("slurm/pipeline/$1")
  if [[ "$DRY_RUN" == "1" ]]; then echo "<$1>"; else "${a[@]}"; fi
}

for line in "${STEPS[@]}"; do
  IFS='|' read -r name script array after merge <<< "$line"

  # Parent job ids, for parents we actually submitted (skips preprocess / externals).
  deps=()
  if [[ -n "$after" ]]; then
    IFS=',' read -ra parents <<< "$after"
    for p in "${parents[@]}"; do
      [[ -n "${JOBID[$p]:-}" ]] && deps+=("${JOBID[$p]}")
    done
  fi
  depstr=$(IFS=:; echo "${deps[*]:-}")

  # Submit the (array) script gated on parents.
  jid=$(submit "$depstr" "$array" "$script")
  if [[ -n "$merge" ]]; then
    # Chain the merge afterok the array; downstream depends on the MERGE job.
    mid=$(submit "$jid" "" "$merge")
    JOBID[$name]="$mid"
    echo "submitted ${name}: array ${jid} [${array}] -> merge ${mid}${depstr:+ (afterok:${depstr})}"
  else
    JOBID[$name]="$jid"
    echo "submitted ${name}: ${jid}${array:+ [array ${array}]}${depstr:+ (afterok:${depstr})}"
  fi
done

echo "done."
