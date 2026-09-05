#!/usr/bin/env python3
"""Assess the V14.2 stable-frontier pilot without gating on probe-only cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.2-stable-frontier-assessment-v1"
EPSILON = 1e-12


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_signature(replica: dict[str, Any]) -> str:
    return json.dumps(replica.get("focused_action"), sort_keys=True, separators=(",", ":"))


def _case_result(case: dict[str, Any], diagnostic: list[dict[str, Any]]) -> dict[str, Any]:
    if len(diagnostic) != 1 or len(diagnostic[0].get("groups", [])) != 1:
        raise ValueError(f"{case['case_id']} must contain one rollout-only group")
    group = diagnostic[0]["groups"][0]
    scenario = group["scenario"]
    focused = str(case["focused_agent"])
    if (
        scenario.get("source") != "ordinary"
        or scenario.get("focused_agent") != focused
        or scenario.get("opponent", {}).get("family") != case["opponent_family"]
        or int(scenario["seed"]) != int(case["seed"])
        or int(scenario["size"]) != int(case["size"])
        or int(scenario["scheduled_horizon"]) != int(case["horizon"])
    ):
        raise ValueError(f"{case['case_id']} changed its frozen pilot binding")
    replicas = list(group["replicas"])
    if len(replicas) != int(case["replicas"]):
        raise ValueError(f"{case['case_id']} has the wrong replica count")
    returns = [float(replica["return"]) for replica in replicas]
    advantages = [
        float(replica.get("advantages", {}).get(focused, 0.0))
        for replica in replicas
    ]
    return {
        **case,
        "return_min": min(returns),
        "return_max": max(returns),
        "return_range": max(returns) - min(returns),
        "nonzero_advantages": sum(abs(value) > EPSILON for value in advantages),
        "positive_advantages": sum(value > EPSILON for value in advantages),
        "negative_advantages": sum(value < -EPSILON for value in advantages),
        "focused_action_diversity": len(
            {action_signature(replica) for replica in replicas}
        ),
    }


def assess(
    manifest: dict[str, Any],
    diagnostics: dict[str, list[dict[str, Any]]],
    *,
    manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != canonical_sha256(manifest_body):
        raise ValueError("V14.2 pilot manifest body hash mismatch")
    expected = {row["case_id"]: row for row in manifest["cases"]}
    if set(diagnostics) != set(expected):
        raise ValueError(
            "pilot diagnostics mismatch; "
            f"missing={sorted(set(expected) - set(diagnostics))}, "
            f"extra={sorted(set(diagnostics) - set(expected))}"
        )
    rows = [
        _case_result(expected[case_id], diagnostics[case_id])
        for case_id in sorted(expected)
    ]
    thresholds = manifest["thresholds"]
    blocking = defaultdict(list)
    probes = defaultdict(list)
    for row in rows:
        if row["admission_role"] == "frontier_candidate" and row["blocking"] is True:
            blocking[row["focused_agent"]].append(row)
        elif row["admission_role"] == "current_probe" and row["blocking"] is False:
            probes[row["focused_agent"]].append(row)
        else:
            raise ValueError(f"unknown pilot role/binding: {row['case_id']}")

    policy_metrics = {}
    probe_metrics = {}
    failed = []
    for policy in sorted(blocking):
        candidates = blocking[policy]
        current = probes[policy]
        if len(candidates) != 4 or len(current) != 2:
            raise ValueError(f"{policy} pilot allocation changed")
        replicas = sum(int(row["replicas"]) for row in candidates)
        variable = [row for row in candidates if row["return_range"] > EPSILON]
        nonzero = sum(int(row["nonzero_advantages"]) for row in candidates)
        positive = sum(int(row["positive_advantages"]) for row in candidates)
        negative = sum(int(row["negative_advantages"]) for row in candidates)
        diversity = statistics.fmean(
            float(row["focused_action_diversity"]) for row in candidates
        )
        variable_families = Counter(row["opponent_family"] for row in variable)
        gates = {
            "variable_groups": (
                len(variable) >= thresholds["minimum_variable_groups_per_policy"]
            ),
            "variable_families": (
                len(variable_families)
                >= thresholds["minimum_variable_families_per_policy"]
            ),
            "nonzero_advantage_rate": (
                nonzero / replicas
                >= thresholds["minimum_nonzero_advantage_rate_per_policy"]
            ),
            "focused_action_diversity": (
                diversity
                >= thresholds["minimum_mean_focused_action_diversity_per_policy"]
            ),
            "positive_advantages": (
                positive >= thresholds["minimum_positive_advantages_per_policy"]
            ),
            "negative_advantages": (
                negative >= thresholds["minimum_negative_advantages_per_policy"]
            ),
        }
        policy_metrics[policy] = {
            "blocking_cases": len(candidates),
            "replicas": replicas,
            "variable_groups": len(variable),
            "variable_families": dict(sorted(variable_families.items())),
            "nonzero_advantage_rate": nonzero / replicas,
            "mean_focused_action_diversity": diversity,
            "positive_advantages": positive,
            "negative_advantages": negative,
            "gates": gates,
        }
        failed.extend(f"{policy}/{name}" for name, passed in gates.items() if not passed)
        probe_variable = sum(row["return_range"] > EPSILON for row in current)
        probe_metrics[policy] = {
            "cases": len(current),
            "variable_groups": probe_variable,
            "nonzero_advantages": sum(
                int(row["nonzero_advantages"]) for row in current
            ),
            "blocking": False,
        }
    if set(blocking) != {"blue-0", "blue-1", "blue-2", "blue-3"}:
        raise ValueError("pilot does not cover all four policy slots")

    body = {
        "version": VERSION,
        "status": "passed" if not failed else "failed",
        "failed_gates": failed,
        "protocol_admission_rate": 1.0,
        "policy_metrics": policy_metrics,
        "current_probe_metrics": probe_metrics,
        "current_probe_is_blocking": False,
        "pilot_trajectories_used_for_optimization": False,
        "scope": "training-only scenario-selection pilot; never a capability evaluation",
        "cases": sorted(rows, key=lambda row: int(row["schedule_ordinal"])),
    }
    if manifest_file_sha256 is not None:
        body["screen_manifest_file_sha256"] = manifest_file_sha256
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    diagnostics = {
        case["case_id"]: json.loads(
            (args.input_dir / case["case_id"] / "live_rl_diagnostic.json").read_text(
                encoding="utf-8"
            )
        )
        for case in manifest["cases"]
    }
    result = assess(
        manifest,
        diagnostics,
        manifest_file_sha256=file_sha256(args.manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
