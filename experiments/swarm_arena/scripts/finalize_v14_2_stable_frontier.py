#!/usr/bin/env python3
"""Bind a passed V14.2 pilot to the exact ordinary pool used by training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.2-stable-frontier-finalizer-v1"
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
    if assessment.get("status") != "passed" or assessment.get("failed_gates"):
        raise ValueError("V14.2 curriculum finalization requires a passed pilot")
    if assessment.get("pilot_trajectories_used_for_optimization") is not False:
        raise ValueError("pilot trajectories must be discarded before optimization")
    if int(manifest.get("case_count", -1)) != 24 or int(manifest.get("games", -1)) != 96:
        raise ValueError("V14.2 pilot shape changed")
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    result_cases = {str(row["case_id"]): row for row in assessment["cases"]}
    if set(manifest_cases) != set(result_cases):
        raise ValueError("pilot assessment does not cover the exact manifest")
    source_cases = {str(row["case_id"]): row for row in source_pool["cases"]}

    pool_cases = []
    pilot_case_metrics = {}
    for pilot_id in sorted(manifest_cases, key=lambda key: manifest_cases[key]["schedule_ordinal"]):
        binding = manifest_cases[pilot_id]
        result = result_cases[pilot_id]
        pool_case_id = str(binding["pool_case_id"])
        if pool_case_id not in source_cases:
            raise ValueError(f"pilot case is absent from source pool: {pool_case_id}")
        if any(result.get(key) != binding.get(key) for key in (
            "pool_case_id",
            "focused_agent",
            "opponent_family",
            "seed",
            "size",
            "horizon",
            "admission_role",
            "blocking",
        )):
            raise ValueError(f"pilot result changed its case binding: {pilot_id}")
        source = dict(source_cases[pool_case_id])
        source["initial_classification"] = _classification(result)
        pilot_case_metrics[pool_case_id] = {
            "pilot_case_id": pilot_id,
            "admission_role": binding["admission_role"],
            "blocking": bool(binding["blocking"]),
            "return_range": float(result["return_range"]),
            "nonzero_advantages": int(result["nonzero_advantages"]),
            "focused_action_diversity": int(result["focused_action_diversity"]),
        }
        pool_cases.append(source)

    if len({row["case_id"] for row in pool_cases}) != 24:
        raise ValueError("pilot-bound pool contains duplicate source cases")
    cell_counts = Counter(
        (row["focused_agent"], row["opponent_family"]) for row in pool_cases
    )
    required_cells = {
        (f"blue-{policy}", family)
        for policy in range(4)
        for family in ("base", "sft", "historical", "current")
    }
    if set(cell_counts) != required_cells:
        raise ValueError("pilot-bound pool must retain every policy/opponent cell")
    classification_counts = Counter(row["initial_classification"] for row in pool_cases)
    pool_body = {
        "version": POOL_VERSION,
        "scope": "training_only_pilot_bound_no_development_or_frozen_data",
        "source_hashes": source_hashes,
        "pilot_manifest_sha256": manifest["sha256"],
        "pilot_assessment_sha256": assessment["sha256"],
        "selection_policy": {
            "frontier": "complete pilot pass@4 return range is nonzero",
            "mastered": "complete pilot pass@4 returns are positive and flat",
            "stalled": "complete pilot pass@4 returns are nonpositive and flat",
            "frontier_fraction": 0.8,
            "mastered_anchor_fraction": 0.1,
            "stalled_anchor_fraction": 0.1,
            "current_probes": "nonblocking but retained as ordinary anchors/probes",
        },
        "case_count": len(pool_cases),
        "cell_counts": {
            f"{policy}/{family}": count
            for (policy, family), count in sorted(cell_counts.items())
        },
        "classification_counts": dict(sorted(classification_counts.items())),
        "pilot_case_metrics": dict(sorted(pilot_case_metrics.items())),
        "cases": pool_cases,
    }
    pool = {**pool_body, "sha256": canonical_sha256(pool_body)}

    curriculum = json.loads(json.dumps(curriculum_template))
    curriculum["version"] = "arena-rl-v14.2-pilot-bound-grounded-follow-through-v1"
    curriculum["ordinary_case_pool"] = pool
    curriculum["ordinary_frontier_stability_repair"].update(
        {
            "pilot_assessment_sha256": assessment["sha256"],
            "pilot_bound_ordinary_case_pool_sha256": pool["sha256"],
            "optimizer_updates_authorized_after_pilot": 40,
        }
    )
    curriculum_body = {
        key: value for key, value in curriculum.items() if key != "sha256"
    }
    curriculum = {**curriculum_body, "sha256": canonical_sha256(curriculum_body)}

    audit_body = {
        "version": VERSION,
        "status": "pilot_passed_curriculum_bound_runtime_freeze_pending",
        "source_hashes": source_hashes,
        "pilot_manifest_sha256": manifest["sha256"],
        "pilot_assessment_sha256": assessment["sha256"],
        "ordinary_case_pool_sha256": pool["sha256"],
        "curriculum_sha256": curriculum["sha256"],
        "ordinary_cases": len(pool_cases),
        "classification_counts": dict(sorted(classification_counts.items())),
        "policy_opponent_cells": len(cell_counts),
        "pilot_trajectories_used_for_optimization": False,
        "credit_estimator_changed": False,
        "stage_gates_changed": False,
        "frozen_or_development_data_used": False,
        "optimizer_updates_authorized": 40,
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
