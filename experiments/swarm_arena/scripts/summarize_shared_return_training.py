from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

VERSION = "arena-shared-return-training-summary-v1"


def _step_summary(row: dict[str, Any], previous: dict[str, str] | None) -> dict[str, Any]:
    groups = list(row["groups"])
    replicas = [replica for group in groups for replica in group["replicas"]]
    returns = [float(replica["return"]) for replica in replicas]
    advantages = [float(replica["advantage"]) for replica in replicas]
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        scenario = group["scenario"]
        if scenario.get("kind") not in {"critical", "decoy"}:
            continue
        by_pair[int(scenario["pair_index"])][str(scenario["kind"])] = group
    paired_differences = []
    namespace_match = True
    for kinds in by_pair.values():
        if set(kinds) != {"critical", "decoy"}:
            continue
        critical = kinds["critical"]
        decoy = kinds["decoy"]
        namespace_match &= (
            critical["scenario"].get("sampling_namespace")
            == decoy["scenario"].get("sampling_namespace")
        )
        paired_differences.extend(
            float(left["return"]) - float(right["return"])
            for left, right in zip(critical["replicas"], decoy["replicas"], strict=True)
        )
    digests = {str(key): str(value) for key, value in row["policy_adapter_sha256"].items()}
    kinds = [group["scenario"].get("kind", "ordinary") for group in groups]
    return {
        "step": int(row["step"]),
        "groups": len(groups),
        "replicas": len(replicas),
        "ordinary_groups": kinds.count("ordinary"),
        "critical_groups": kinds.count("critical"),
        "decoy_groups": kinds.count("decoy"),
        "mean_return": statistics.mean(returns),
        "return_stdev": statistics.pstdev(returns),
        "mean_absolute_advantage": statistics.mean(abs(value) for value in advantages),
        "nonzero_advantage_rate": statistics.mean(abs(value) > 1e-12 for value in advantages),
        "positive_advantages": sum(value > 1e-12 for value in advantages),
        "negative_advantages": sum(value < -1e-12 for value in advantages),
        "paired_sampling_namespaces": namespace_match,
        "critical_minus_decoy_mean_return": (
            statistics.mean(paired_differences) if paired_differences else None
        ),
        "policy_adapter_sha256": digests,
        "distinct_policy_adapters": len(set(digests.values())),
        "policies_changed_since_previous_step": (
            None
            if previous is None
            else sum(digests[policy] != previous[policy] for policy in sorted(digests))
        ),
    }


def summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        raise ValueError("training progress is empty")
    steps = []
    previous = None
    for expected, row in enumerate(payload):
        if int(row["step"]) != expected:
            raise ValueError(f"training steps are not contiguous at {expected}")
        summary = _step_summary(row, previous)
        steps.append(summary)
        previous = summary["policy_adapter_sha256"]
    return {
        "version": VERSION,
        "input": str(path),
        "completed_steps": len(steps),
        "steps": steps,
        "aggregate": {
            "mean_return": statistics.mean(step["mean_return"] for step in steps),
            "mean_absolute_advantage": statistics.mean(
                step["mean_absolute_advantage"] for step in steps
            ),
            "mean_nonzero_advantage_rate": statistics.mean(
                step["nonzero_advantage_rate"] for step in steps
            ),
        },
        "mechanical_checks": {
            "four_groups_per_step": all(step["groups"] == 4 for step in steps),
            "sixteen_replicas_per_step": all(step["replicas"] == 16 for step in steps),
            "balanced_critical_decoy": all(
                step["critical_groups"] == step["decoy_groups"] > 0 for step in steps
            ),
            "recognized_four_group_mixture": all(
                (
                    step["ordinary_groups"],
                    step["critical_groups"],
                    step["decoy_groups"],
                )
                in {(0, 2, 2), (2, 1, 1)}
                for step in steps
            ),
            "paired_sampling_namespaces": all(
                step["paired_sampling_namespaces"] for step in steps
            ),
            "nonzero_learning_signal_every_step": all(
                step["nonzero_advantage_rate"] > 0 for step in steps
            ),
            "four_distinct_policy_adapters": all(
                step["distinct_policy_adapters"] == 4 for step in steps
            ),
            "all_policies_change_between_updates": all(
                step["policies_changed_since_previous_step"] in {None, 4} for step in steps
            ),
        },
        "scope": "rollout and optimizer diagnostics only; not an evaluation of learned communication",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a four-policy shared-return RL run.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.input)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
