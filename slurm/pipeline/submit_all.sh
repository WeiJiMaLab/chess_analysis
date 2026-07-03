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

declare -A JOBID    # (name | name@variant) -> job id downstream should depend on
declare -A S_SCRIPT S_ARRAY S_AFTER S_MERGE   # per-step fields, for variant expansion
ORDER=()            # step names in manifest (topological) order

# submit <depstr> <array> <script> [variant]  -> echoes the parsable job id
submit() {
  local depstr="$1" array="$2" script="$3" variant="${4:-}"
  local a=(sbatch --parsable)
  [[ -n "$array" ]] && a+=(--array="$array")
  [[ -n "$depstr" ]] && a+=("--dependency=afterok:${depstr}")
  [[ -n "$variant" ]] && a+=("--export=ALL,VARIANT=${variant}")
  a+=("slurm/pipeline/$script")
  if [[ "$DRY_RUN" == "1" ]]; then echo "<${script}${variant:+ VARIANT=$variant}>"; else "${a[@]}"; fi
}

# submit_chain <name> <depstr> [variant]: submit the (array) script + optional merge;
# records JOBID for <key> (= name, or name@variant) as what downstream depends on.
submit_chain() {
  local name="$1" depstr="$2" variant="${3:-}"
  local key="$name${variant:+@$variant}"
  local jid; jid=$(submit "$depstr" "${S_ARRAY[$name]}" "${S_SCRIPT[$name]}" "$variant")
  if [[ -n "${S_MERGE[$name]}" ]]; then
    local mid; mid=$(submit "$jid" "" "${S_MERGE[$name]}" "$variant")
    JOBID[$key]="$mid"
    echo "submitted ${key}: array ${jid} [${S_ARRAY[$name]}] -> merge ${mid}${depstr:+ (afterok:${depstr})}"
  else
    JOBID[$key]="$jid"
    echo "submitted ${key}: ${jid}${S_ARRAY[$name]:+ [array ${S_ARRAY[$name]}]}${depstr:+ (afterok:${depstr})}"
  fi
}

# depstr for a step, remapping each parent to a variant job when it is itself in the
# variant tail, else the base job. INTAIL (assoc) marks the current variant's tail.
dep_string() {
  local after="$1" variant="${2:-}"; local -n tail_ref="$3"
  local deps=() p
  IFS=',' read -ra parents <<< "$after"
  for p in "${parents[@]}"; do
    [[ -z "$p" ]] && continue
    if [[ -n "$variant" && -n "${tail_ref[$p]:-}" ]]; then deps+=("${JOBID[${p}@${variant}]}")
    elif [[ -n "${JOBID[$p]:-}" ]]; then deps+=("${JOBID[$p]}"); fi
  done
  (IFS=:; echo "${deps[*]:-}")
}

# ---- base DAG ----
declare -A _NOTAIL
for line in "${STEPS[@]}"; do
  IFS='|' read -r name script array after merge <<< "$line"
  S_SCRIPT[$name]="$script"; S_ARRAY[$name]="$array"; S_AFTER[$name]="$after"; S_MERGE[$name]="$merge"
  ORDER+=("$name")
  submit_chain "$name" "$(dep_string "$after" "" _NOTAIL)"
done

# ---- variant tails (opt-in): RUN_VARIANTS=pruned,other bash submit_all.sh ----
# A variant re-runs the steps on the dependency path [from .. to] with VARIANT set:
# `from` forks after its BASE parent(s); the rest chain among the variant's own jobs.
# The path scope keeps side branches (board / engine) out (config `variants:` block).
IFS=',' read -ra WANT <<< "${RUN_VARIANTS:-}"
for vname in "${WANT[@]}"; do
  [[ -z "$vname" ]] && continue
  read -r vfrom vto < <(python3 -c "import yaml,os; v=(yaml.safe_load(open(os.environ['CONFIG'])) or {}).get('variants',{}).get('${vname}',{}); print(v.get('from',''), v.get('to',''))" 2>/dev/null || echo "")
  [[ -z "${vfrom:-}" || -z "${vto:-}" ]] && { echo "variant ${vname}: needs from+to in config (and CONFIG set) — skipping"; continue; }
  declare -A DESC=() ANC=() INTAIL=()
  DESC[$vfrom]=1                                            # descendants of `from` (forward, topo order)
  for name in "${ORDER[@]}"; do
    IFS=',' read -ra ps <<< "${S_AFTER[$name]}"
    for p in "${ps[@]}"; do [[ -n "${DESC[$p]:-}" ]] && DESC[$name]=1; done
  done
  ANC[$vto]=1                                               # ancestors of `to` (backward, reverse topo)
  for ((i=${#ORDER[@]}-1; i>=0; i--)); do
    name="${ORDER[$i]}"; [[ -n "${ANC[$name]:-}" ]] || continue
    IFS=',' read -ra ps <<< "${S_AFTER[$name]}"
    for p in "${ps[@]}"; do [[ -n "$p" ]] && ANC[$p]=1; done
  done
  for name in "${ORDER[@]}"; do                             # tail = path = DESC ∩ ANC
    [[ -n "${DESC[$name]:-}" && -n "${ANC[$name]:-}" ]] && INTAIL[$name]=1
  done
  for name in "${ORDER[@]}"; do
    [[ -n "${INTAIL[$name]:-}" ]] || continue
    submit_chain "$name" "$(dep_string "${S_AFTER[$name]}" "$vname" INTAIL)" "$vname"
  done
  unset DESC ANC INTAIL
done

echo "done."
