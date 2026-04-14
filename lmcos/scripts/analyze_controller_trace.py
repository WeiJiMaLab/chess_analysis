from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _state_key(features: Dict[str, float]) -> str:
    label = next((name for name, value in features.items() if name.startswith("label_") and value > 0.5), "label_?")
    phase = next((name for name, value in features.items() if name.startswith("phase_") and value > 0.5), "phase_?")
    return f"{label}|{phase}"


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _summarize_trace(trace_path: Path) -> Dict[str, Any]:
    rows = _load_jsonl(trace_path)

    pre_states: Dict[Tuple[int, int], Dict[str, Any]] = {}
    post_states: Dict[Tuple[int, int], Dict[str, Any]] = {}
    pre_by_state = defaultdict(lambda: {"halt": 0, "continue": 0, "rewards": [], "values": [], "updates": set()})
    paired_by_state_action = defaultdict(list)
    eval_by_state = defaultdict(lambda: {"halt": 0, "continue": 0, "halt_logits": []})

    for row in rows:
        event = row.get("event")
        if event == "rollout_transition" and row.get("phase") == "pre_gae":
            features = row["observation_root_features"]
            key = _state_key(features)
            pre_states[(row["update_index"], row["step_index"])] = row
            bucket = pre_by_state[key]
            bucket["updates"].add(row["update_index"])
            bucket["rewards"].append(float(row["reward"]))
            bucket["values"].append(float(row["value"]))
            if int(row["action"]) == 1:
                bucket["halt"] += 1
            else:
                bucket["continue"] += 1
        elif event == "rollout_transition" and row.get("phase") == "post_gae":
            post_states[(row["update_index"], row["rollout_index"])] = row
        elif event == "eval_step":
            features = row["observation_root_features"]
            key = _state_key(features)
            bucket = eval_by_state[key]
            bucket["halt_logits"].append(float(row["halt_logit"]))
            if int(row["action"]) == 1:
                bucket["halt"] += 1
            else:
                bucket["continue"] += 1

    for key, pre_row in pre_states.items():
        post_row = post_states.get(key)
        if post_row is None:
            continue
        state = _state_key(pre_row["observation_root_features"])
        action_key = "halt" if int(pre_row["action"]) == 1 else "continue"
        paired_by_state_action[(state, action_key)].append(
            {
                "advantage": float(post_row["advantage"]),
                "return_value": float(post_row["return_value"]),
                "value": float(pre_row["value"]),
                "reward": float(pre_row["reward"]),
                "update_index": int(pre_row["update_index"]),
            }
        )

    return {
        "pre_by_state": pre_by_state,
        "paired_by_state_action": paired_by_state_action,
        "eval_by_state": eval_by_state,
    }


def _state_report(summary: Dict[str, Any], state: str) -> Dict[str, Any]:
    pre = summary["pre_by_state"].get(state, {})
    eval_stats = summary["eval_by_state"].get(state, {})
    paired = summary["paired_by_state_action"]

    report = {
        "train": {
            "halt_count": pre.get("halt", 0),
            "continue_count": pre.get("continue", 0),
            "mean_reward": _mean(pre.get("rewards", [])),
            "mean_value": _mean(pre.get("values", [])),
            "num_updates_seen": len(pre.get("updates", set())),
        },
        "eval": {
            "halt_count": eval_stats.get("halt", 0),
            "continue_count": eval_stats.get("continue", 0),
            "mean_halt_logit": _mean(eval_stats.get("halt_logits", [])),
        },
        "action_conditioned": {},
    }
    for action_key in ("halt", "continue"):
        items = paired.get((state, action_key), [])
        report["action_conditioned"][action_key] = {
            "count": len(items),
            "mean_advantage": _mean(item["advantage"] for item in items),
            "mean_return": _mean(item["return_value"] for item in items),
            "mean_value": _mean(item["value"] for item in items),
            "mean_reward": _mean(item["reward"] for item in items),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two controller debug traces and summarize where they diverge.")
    parser.add_argument("--good-dir", required=True)
    parser.add_argument("--bad-dir", required=True)
    parser.add_argument(
        "--states",
        default="label_0|phase_0,label_1|phase_0,label_1|phase_1,label_2|phase_0,label_2|phase_1",
        help="Comma-separated state keys to compare.",
    )
    args = parser.parse_args()

    good_dir = Path(args.good_dir)
    bad_dir = Path(args.bad_dir)
    good = _summarize_trace(good_dir / "trace.jsonl")
    bad = _summarize_trace(bad_dir / "trace.jsonl")

    print(f"good_dir={good_dir}")
    print(f"bad_dir={bad_dir}")
    for state in args.states.split(","):
        print(f"state={state}")
        print("good=" + json.dumps(_state_report(good, state), sort_keys=True))
        print("bad=" + json.dumps(_state_report(bad, state), sort_keys=True))


if __name__ == "__main__":
    main()
