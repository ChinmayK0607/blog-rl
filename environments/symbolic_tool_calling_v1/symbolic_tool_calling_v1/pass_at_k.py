import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from symbolic_tool_calling_v1.artifacts import canonical_json


def analyze_pass_at_k(
    records: list[dict[str, Any]], expected_k: int, max_selected: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = record["task"]["spec"]["task_id"]
        grouped[task_id].append(record)

    groups = []
    for task_id in sorted(grouped):
        rollouts = grouped[task_id]
        if len(rollouts) != expected_k:
            raise ValueError(f"task {task_id} has {len(rollouts)} rollouts, expected {expected_k}")
        errors = sum(bool(record.get("errors")) for record in rollouts)
        if errors:
            raise ValueError(f"task {task_id} has {errors} errored rollouts")
        successes = sum(float(record["rewards"]["success"]) == 1.0 for record in rollouts)
        spec = rollouts[0]["task"]["spec"]
        bucket = "all_fail" if successes == 0 else "all_pass" if successes == expected_k else "mixed"
        groups.append(
            {
                "task_id": task_id,
                "seed": spec["seed"],
                "successes": successes,
                "rollouts": expected_k,
                "pass_rate": successes / expected_k,
                "bucket": bucket,
                "horizon_bucket": spec["horizon_bucket"],
                "imbalance_setting": spec["imbalance_setting"],
                "optimal_plan_length": spec["optimal_plan_length"],
            }
        )
    counts = {bucket: sum(group["bucket"] == bucket for group in groups) for bucket in ("all_fail", "mixed", "all_pass")}
    condition_counts = {}
    for horizon in sorted({group["horizon_bucket"] for group in groups}):
        subset = [group for group in groups if group["horizon_bucket"] == horizon]
        condition_counts[horizon] = {
            bucket: sum(group["bucket"] == bucket for group in subset)
            for bucket in ("all_fail", "mixed", "all_pass")
        }
    mixed = [group for group in groups if group["bucket"] == "mixed"]
    selected = mixed[:max_selected] if max_selected is not None else mixed
    summary = {
        "expected_k": expected_k,
        "num_groups": len(groups),
        "num_rollouts": len(records),
        "bucket_counts": counts,
        "mixed_fraction": counts["mixed"] / len(groups) if groups else 0.0,
        "mean_pass_rate": sum(group["pass_rate"] for group in groups) / len(groups) if groups else 0.0,
        "condition_bucket_counts": condition_counts,
        "mixed_task_ids": [group["task_id"] for group in mixed],
        "mixed_seeds": [group["seed"] for group in mixed],
        "selected_task_ids": [group["task_id"] for group in selected],
        "selected_seeds": [group["seed"] for group in selected],
        "num_selected": len(selected),
    }
    return groups, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify pass@k prompt groups for RL filtering.")
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-k", type=int, required=True)
    parser.add_argument("--max-selected", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.results_jsonl.read_text().splitlines() if line]
    groups, summary = analyze_pass_at_k(records, args.expected_k, args.max_selected)
    (args.output_dir / "groups.jsonl").write_text("".join(f"{canonical_json(group)}\n" for group in groups))
    (args.output_dir / "summary.json").write_text(f"{canonical_json(summary)}\n")
    print(json.dumps(summary, indent=2))
