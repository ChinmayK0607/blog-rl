#!/usr/bin/env python3
"""Assess V13 ordinary rollout-only diagnostics against frozen signal thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EPSILON = 1e-12


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_signature(replica: dict[str, Any]) -> str:
    action = replica.get("focused_action")
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def assess(manifest: dict[str, Any], diagnostics: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected = {row["case_id"]: row for row in manifest["cases"]}
    if set(diagnostics) != set(expected):
        missing = sorted(set(expected) - set(diagnostics))
        extra = sorted(set(diagnostics) - set(expected))
        raise ValueError(f"screen diagnostics mismatch; missing={missing}, extra={extra}")
    cases = []
    for case_id, case in sorted(expected.items()):
        payload = diagnostics[case_id]
        if len(payload) != 1 or len(payload[0]["groups"]) != 1:
            raise ValueError(f"{case_id} must contain exactly one rollout-only group")
        group = payload[0]["groups"][0]
        scenario = group["scenario"]
        focused_agent = str(case["focused_agent"])
        if (
            scenario.get("source") != "ordinary"
            or int(scenario["seed"]) != int(case["seed"])
            or int(scenario["size"]) != int(case["size"])
            or int(scenario["scheduled_horizon"]) != int(case["horizon"])
            or scenario.get("focused_agent") != focused_agent
        ):
            raise ValueError(f"{case_id} diagnostic does not match its frozen case")
        replicas = group["replicas"]
        if len(replicas) != int(case["replicas"]):
            raise ValueError(f"{case_id} has the wrong replica count")
        returns = [float(replica["return"]) for replica in replicas]
        advantages = [
            float(replica["advantages"].get(focused_agent, 0.0)) for replica in replicas
        ]
        cases.append(
            {
                **case,
                "return_min": min(returns),
                "return_max": max(returns),
                "return_range": max(returns) - min(returns),
                "nonzero_advantage_rate": statistics.fmean(
                    abs(value) > EPSILON for value in advantages
                ),
                "positive_advantages": sum(value > EPSILON for value in advantages),
                "negative_advantages": sum(value < -EPSILON for value in advantages),
                "mean_absolute_advantage": statistics.fmean(abs(value) for value in advantages),
                "focused_action_diversity": len(
                    {action_signature(replica) for replica in replicas}
                ),
            }
        )

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_policy[str(row["focused_agent"])].append(row)
    thresholds = manifest["thresholds"]
    policy_metrics = {}
    failed = []
    for policy, rows in sorted(by_policy.items()):
        advantages = [
            value
            for row in rows
            for value in (
                [1.0] * row["positive_advantages"]
                + [-1.0] * row["negative_advantages"]
                + [0.0]
                * (
                    int(row["replicas"])
                    - row["positive_advantages"]
                    - row["negative_advantages"]
                )
            )
        ]
        variable_rate = statistics.fmean(row["return_range"] > EPSILON for row in rows)
        nonzero_rate = statistics.fmean(abs(value) > EPSILON for value in advantages)
        action_diversity = statistics.fmean(row["focused_action_diversity"] for row in rows)
        positive = sum(row["positive_advantages"] for row in rows)
        negative = sum(row["negative_advantages"] for row in rows)
        family_variable = Counter(
            row["opponent_family"] for row in rows if row["return_range"] > EPSILON
        )
        gates = {
            "variable_return_group_rate": (
                variable_rate
                >= thresholds["minimum_variable_return_group_rate_per_policy"]
            ),
            "nonzero_advantage_rate": (
                nonzero_rate >= thresholds["minimum_nonzero_advantage_rate_per_policy"]
            ),
            "focused_action_diversity": (
                action_diversity
                >= thresholds["minimum_mean_focused_action_diversity_per_policy"]
            ),
            "positive_advantages": positive >= thresholds["minimum_positive_advantages_per_policy"],
            "negative_advantages": negative >= thresholds["minimum_negative_advantages_per_policy"],
            "opponent_family_coverage": all(
                family_variable[family]
                >= thresholds["minimum_variable_groups_per_policy_opponent_family"]
                for family in ("base", "sft", "historical", "current")
            ),
        }
        policy_metrics[policy] = {
            "cases": len(rows),
            "replicas": len(advantages),
            "variable_return_group_rate": variable_rate,
            "nonzero_advantage_rate": nonzero_rate,
            "mean_focused_action_diversity": action_diversity,
            "positive_advantages": positive,
            "negative_advantages": negative,
            "variable_groups_by_opponent_family": dict(sorted(family_variable.items())),
            "gates": gates,
        }
        failed.extend(f"{policy}/{name}" for name, passed in gates.items() if not passed)
    ranked = sorted(
        cases,
        key=lambda row: (
            row["focused_agent"],
            -row["nonzero_advantage_rate"],
            -row["mean_absolute_advantage"],
            -row["focused_action_diversity"],
            row["case_id"],
        ),
    )
    return {
        "version": "arena-rl-v13-ordinary-signal-assessment-v1",
        "screen_manifest_sha256": manifest["sha256"],
        "status": "passed" if not failed else "failed",
        "failed_gates": failed,
        "protocol_admission_rate": 1.0,
        "policy_metrics": policy_metrics,
        "cases": ranked,
        "scope": "training-only rollout signal admission; never a capability evaluation",
        "protocol_note": (
            "run_live_rl writes a completed rollout-only diagnostic only after the "
            "shared-return group passes its fail-closed protocol admission"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    diagnostics = {}
    for case in manifest["cases"]:
        path = args.input_dir / case["case_id"] / "live_rl_diagnostic.json"
        diagnostics[case["case_id"]] = json.loads(path.read_text(encoding="utf-8"))
    result = assess(manifest, diagnostics)
    result["screen_manifest_file_sha256"] = file_sha256(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
