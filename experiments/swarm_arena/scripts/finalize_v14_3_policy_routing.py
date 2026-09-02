#!/usr/bin/env python3
"""Bind the complete V14.2 pilot to policy-specific online curriculum lanes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.3-policy-routing-finalizer-v1"
POOL_VERSION = "arena-rl-ordinary-case-pool-v1"
EPSILON = 1e-12


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hashed(value: dict[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != canonical_sha256(body):
        raise ValueError(f"{label} body hash mismatch")


def _classification(row: dict[str, Any]) -> str:
    if float(row["return_range"]) > EPSILON:
        return "frontier"
    return "mastered" if float(row["return_min"]) > EPSILON else "stalled"


def _policy_mode(metrics: dict[str, Any]) -> str:
    if all(bool(value) for value in metrics["gates"].values()):
        return "consolidate"
    if float(metrics["nonzero_advantage_rate"]) > 0:
        return "expand"
    return "discover"


def finalize(
    manifest: dict[str, Any],
    assessment: dict[str, Any],
    source_pool: dict[str, Any],
    curriculum_template: dict[str, Any],
    *,
    source_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    for label, value in (
        ("pilot manifest", manifest),
        ("pilot assessment", assessment),
        ("source ordinary pool", source_pool),
        ("curriculum template", curriculum_template),
    ):
        _verify_hashed(value, label)
    if int(manifest.get("case_count", -1)) != 24 or int(manifest.get("games", -1)) != 96:
        raise ValueError("V14.3 requires the complete frozen V14.2 pilot")
    if assessment.get("pilot_trajectories_used_for_optimization") is not False:
        raise ValueError("pilot trajectories must remain excluded from optimization")
    if float(assessment.get("protocol_admission_rate", 0.0)) != 1.0:
        raise ValueError("policy routing cannot bypass protocol admission")

    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    result_cases = {str(row["case_id"]): row for row in assessment["cases"]}
    if set(manifest_cases) != set(result_cases):
        raise ValueError("policy routing requires every frozen pilot case exactly once")
    source_cases = {str(row["case_id"]): row for row in source_pool["cases"]}
    pilot_metrics: dict[str, dict[str, Any]] = {}
    pilot_classifications: dict[str, str] = {}
    for pilot_id, binding in manifest_cases.items():
        result = result_cases[pilot_id]
        if any(
            result.get(key) != binding.get(key)
            for key in (
                "pool_case_id",
                "focused_agent",
                "opponent_family",
                "seed",
                "size",
                "horizon",
                "admission_role",
                "blocking",
            )
        ):
            raise ValueError(f"pilot result changed its case binding: {pilot_id}")
        source_id = str(binding["pool_case_id"])
        if source_id not in source_cases or source_id in pilot_metrics:
            raise ValueError(f"pilot source case binding is missing or duplicated: {source_id}")
        pilot_classifications[source_id] = _classification(result)
        pilot_metrics[source_id] = {
            "pilot_case_id": pilot_id,
            "admission_role": binding["admission_role"],
            "blocking": bool(binding["blocking"]),
            "return_range": float(result["return_range"]),
            "nonzero_advantages": int(result["nonzero_advantages"]),
            "positive_advantages": int(result["positive_advantages"]),
            "negative_advantages": int(result["negative_advantages"]),
            "focused_action_diversity": int(result["focused_action_diversity"]),
        }

    routed_cases = []
    for source in source_pool["cases"]:
        case = dict(source)
        case["initial_classification"] = pilot_classifications.get(
            str(case["case_id"]), "unseen"
        )
        routed_cases.append(case)
    modes = {
        policy: _policy_mode(metrics)
        for policy, metrics in sorted(assessment["policy_metrics"].items())
    }
    if set(modes) != {"blue-0", "blue-1", "blue-2", "blue-3"}:
        raise ValueError("policy routing must bind all four policy slots")
    if all(mode == "discover" for mode in modes.values()):
        raise ValueError("all-discovery pilot has no useful optimizer lane")

    cell_counts = Counter(
        (row["focused_agent"], row["opponent_family"]) for row in routed_cases
    )
    required_cells = {
        (f"blue-{policy}", family)
        for policy in range(4)
        for family in ("base", "sft", "historical", "current")
    }
    if set(cell_counts) != required_cells:
        raise ValueError("policy-routed pool must retain every policy/opponent cell")
    classification_counts = Counter(
        row["initial_classification"] for row in routed_cases
    )
    pool_body = {
        "version": POOL_VERSION,
        "scope": "training_only_policy_routed_no_development_or_frozen_data",
        "source_hashes": source_hashes,
        "pilot_manifest_sha256": manifest["sha256"],
        "pilot_assessment_sha256": assessment["sha256"],
        "selection_policy": {
            "pilot_observed_variable": "frontier",
            "pilot_observed_positive_flat": "mastered",
            "pilot_observed_nonpositive_flat": "stalled",
            "not_observed_in_frozen_pilot": "unseen",
            "pilot_trajectories_used_for_optimization": False,
        },
        "policy_modes": modes,
        "case_count": len(routed_cases),
        "cell_counts": {
            f"{policy}/{family}": count
            for (policy, family), count in sorted(cell_counts.items())
        },
        "classification_counts": dict(sorted(classification_counts.items())),
        "pilot_case_metrics": dict(sorted(pilot_metrics.items())),
        "cases": routed_cases,
    }
    pool = {**pool_body, "sha256": canonical_sha256(pool_body)}

    curriculum = json.loads(json.dumps(curriculum_template))
    curriculum["version"] = "arena-rl-v14.3-policy-routed-grounded-follow-through-v1"
    curriculum["ordinary_case_pool"] = pool
    curriculum["adaptive_curriculum"].update(
        {
            "policy_modes": [f"{policy}:{mode}" for policy, mode in sorted(modes.items())],
            "expand_frontier_fraction": 0.5,
            "discovery_frontier_fraction": 0.25,
        }
    )
    curriculum["ordinary_policy_routing"] = {
        "version": VERSION,
        "pilot_assessment_sha256": assessment["sha256"],
        "policy_modes": modes,
        "consolidate": "prefer observed frontier; retain fixed mastered/stalled anchors",
        "expand": "split frontier slots between observed frontier and unseen cases",
        "discover": "prefer unseen cases; exploit a bounded fraction after frontier appears",
        "selection_observation": "complete preceding training stage only",
        "fresh_rollouts_only": True,
        "zero_advantage_batch_action": "record telemetry and continue immutable schedule",
        "development_or_frozen_feedback": False,
    }
    curriculum_body = {
        key: value for key, value in curriculum.items() if key != "sha256"
    }
    curriculum = {**curriculum_body, "sha256": canonical_sha256(curriculum_body)}

    audit_body = {
        "version": VERSION,
        "status": "policy_routed_cpu_bound_runtime_freeze_pending",
        "source_hashes": source_hashes,
        "pilot_manifest_sha256": manifest["sha256"],
        "pilot_assessment_sha256": assessment["sha256"],
        "ordinary_case_pool_sha256": pool["sha256"],
        "curriculum_sha256": curriculum["sha256"],
        "policy_modes": modes,
        "classification_counts": dict(sorted(classification_counts.items())),
        "ordinary_cases": len(routed_cases),
        "policy_opponent_cells": len(cell_counts),
        "pilot_trajectories_used_for_optimization": False,
        "fresh_training_sampling_required": True,
        "credit_estimator_changed": False,
        "stage_gates_changed": False,
        "frozen_or_development_data_used": False,
        "optimizer_updates_authorized_after_runtime_preflight": 40,
    }
    audit = {**audit_body, "sha256": canonical_sha256(audit_body)}
    return {
        "ordinary_case_pool.json": pool,
        "curriculum.json": curriculum,
        "finalization_audit.json": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--pilot-assessment", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--curriculum-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "pilot_manifest": args.pilot_manifest,
        "pilot_assessment": args.pilot_assessment,
        "source_pool": args.source_pool,
        "curriculum_template": args.curriculum_template,
    }
    artifacts = finalize(
        json.loads(args.pilot_manifest.read_text(encoding="utf-8")),
        json.loads(args.pilot_assessment.read_text(encoding="utf-8")),
        json.loads(args.source_pool.read_text(encoding="utf-8")),
        json.loads(args.curriculum_template.read_text(encoding="utf-8")),
        source_hashes={name: file_sha256(path) for name, path in inputs.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(artifacts["finalization_audit.json"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
