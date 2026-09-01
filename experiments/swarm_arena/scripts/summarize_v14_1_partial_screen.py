#!/usr/bin/env python3
"""Create a compact, hash-bound summary of the interrupted V14.1 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.1-partial-screen-summary-v1"
EPSILON = 1e-12


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_signature(replica: dict[str, Any]) -> str:
    return json.dumps(replica.get("focused_action"), sort_keys=True, separators=(",", ":"))


def summarize_case(
    case: dict[str, Any], diagnostic: list[dict[str, Any]]
) -> dict[str, Any]:
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
        raise ValueError(f"{case['case_id']} diagnostic changed its frozen binding")
    replicas = list(group["replicas"])
    if len(replicas) != int(case["replicas"]):
        raise ValueError(f"{case['case_id']} has the wrong replica count")
    returns = [float(replica["return"]) for replica in replicas]
    advantages = [
        float(replica.get("advantages", {}).get(focused, 0.0))
        for replica in replicas
    ]
    return_range = max(returns) - min(returns)
    classification = (
        "frontier"
        if return_range > EPSILON
        else "mastered" if max(returns) > EPSILON else "stalled"
    )
    return {
        "case_id": case["case_id"],
        "pool_case_id": case["pool_case_id"],
        "focused_agent": focused,
        "opponent_family": case["opponent_family"],
        "seed": int(case["seed"]),
        "size": int(case["size"]),
        "horizon": int(case["horizon"]),
        "replicas": len(replicas),
        "return_min": min(returns),
        "return_max": max(returns),
        "return_range": return_range,
        "nonzero_advantage_rate": statistics.fmean(
            abs(value) > EPSILON for value in advantages
        ),
        "positive_advantages": sum(value > EPSILON for value in advantages),
        "negative_advantages": sum(value < -EPSILON for value in advantages),
        "focused_action_diversity": len(
            {action_signature(replica) for replica in replicas}
        ),
        "classification": classification,
    }


def summarize(manifest: dict[str, Any], input_dir: Path) -> dict[str, Any]:
    manifest_body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != canonical_sha256(manifest_body):
        raise ValueError("V14.1 screen manifest body hash mismatch")
    rows = []
    diagnostic_hashes = {}
    for case in manifest["cases"]:
        path = input_dir / case["case_id"] / "live_rl_diagnostic.json"
        if not path.is_file():
            path = input_dir / f"{case['case_id']}.json"
        if not path.is_file():
            continue
        rows.append(summarize_case(case, json.loads(path.read_text(encoding="utf-8"))))
        diagnostic_hashes[case["case_id"]] = file_sha256(path)
    if not rows:
        raise ValueError("partial screen contains no durable diagnostics")
    body = {
        "version": VERSION,
        "scope": "training_only_partial_screen_no_development_or_frozen_data",
        "screen_manifest_sha256": manifest["sha256"],
        "expected_cases": len(manifest["cases"]),
        "completed_cases": len(rows),
        "complete": len(rows) == len(manifest["cases"]),
        "optimizer_updates": 0,
        "cases": rows,
        "case_diagnostic_sha256": dict(sorted(diagnostic_hashes.items())),
    }
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.manifest.read_text(encoding="utf-8")), args.input_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
