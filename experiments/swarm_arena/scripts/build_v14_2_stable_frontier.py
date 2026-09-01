#!/usr/bin/env python3
"""Build V14.2 around stable frontier selection rather than pass@4 luck."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POLICIES = ("blue-0", "blue-1", "blue-2", "blue-3")
NON_CURRENT_FAMILIES = ("base", "sft", "historical")
VERSION = "arena-rl-v14.2-stable-frontier-v1"
SCREEN_VERSION = "arena-rl-v14.2-stable-frontier-screen-v1"
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


def _retest_by_pool_case(partial: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["pool_case_id"]): row for row in partial["cases"]}


def _source_rows(assessment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in assessment["cases"]}


def _candidate_rank(
    case: dict[str, Any],
    *,
    retest: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> tuple[int, float, str]:
    observed = source.get(str(case["source_case_id"]))
    result = retest.get(str(case["case_id"]))
    if result is not None and result["classification"] == "frontier":
        tier = 0
    elif (
        case["provenance"] == "complete_v14_screen"
        and observed is not None
        and float(observed["return_range"]) > EPSILON
        and result is None
    ):
        tier = 1
    elif case["provenance"] == "cross_opponent_transfer_from_observed_frontier":
        tier = 2
    elif case["provenance"] == "deterministic_unseen_neighbor_of_observed_frontier":
        tier = 3
    else:
        tier = 4
    source_range = 0.0 if observed is None else float(observed["return_range"])
    return tier, -source_range, str(case["case_id"])


def _blocking_candidates(
    policy: str,
    *,
    pool: list[dict[str, Any]],
    retest: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in pool
        if row["focused_agent"] == policy
        and row["opponent_family"] in NON_CURRENT_FAMILIES
        and row["initial_classification"] in {"frontier", "unseen"}
    ]
    selected = []
    for family in NON_CURRENT_FAMILIES:
        family_rows = [row for row in eligible if row["opponent_family"] == family]
        if not family_rows:
            raise ValueError(f"stable frontier pool lacks {policy}/{family}")
        selected.append(
            min(
                family_rows,
                key=lambda row: _candidate_rank(row, retest=retest, source=source),
            )
        )
    remaining = [row for row in eligible if row["case_id"] not in {x["case_id"] for x in selected}]
    if not remaining:
        raise ValueError(f"stable frontier pool lacks a fourth candidate for {policy}")
    selected.append(
        min(remaining, key=lambda row: _candidate_rank(row, retest=retest, source=source))
    )
    return selected


def _current_probes(policy: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probes = sorted(
        (
            row
            for row in pool
            if row["focused_agent"] == policy
            and row["opponent_family"] == "current"
            and row["provenance"] == "cross_opponent_transfer_from_observed_frontier"
        ),
        key=lambda row: str(row["case_id"]),
    )
    if len(probes) != 2:
        raise ValueError(f"stable frontier pool must have two transfer probes for {policy}")
    return probes


def build(
    assessment: dict[str, Any],
    partial: dict[str, Any],
    pool: dict[str, Any],
    curriculum: dict[str, Any],
    *,
    source_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if assessment.get("status") != "failed" or assessment.get("protocol_admission_rate") != 1.0:
        raise ValueError("V14.2 requires the complete protocol-valid V14 rejection")
    _verify_hashed(partial, "V14.1 partial summary")
    _verify_hashed(pool, "V14.1 ordinary pool")
    _verify_hashed(curriculum, "V14.1 curriculum")
    if partial.get("complete") or int(partial.get("optimizer_updates", -1)) != 0:
        raise ValueError("V14.2 requires the interrupted zero-update V14.1 screen")
    if partial.get("screen_manifest_sha256") != curriculum["ordinary_frontier_repair"][
        "screen_manifest_sha256"
    ]:
        raise ValueError("partial V14.1 summary is not bound to the supplied curriculum")

    source = _source_rows(assessment)
    retest = _retest_by_pool_case(partial)
    pool_rows = [dict(row) for row in pool["cases"]]
    cases = []
    ordinal = 0
    selected_pool_ids = set()
    for policy in POLICIES:
        blocking = _blocking_candidates(
            policy, pool=pool_rows, retest=retest, source=source
        )
        probes = _current_probes(policy, pool_rows)
        for role, candidates in (("frontier_candidate", blocking), ("current_probe", probes)):
            for index, candidate in enumerate(candidates):
                if candidate["case_id"] in selected_pool_ids:
                    raise ValueError("V14.2 pilot selected a duplicate pool case")
                selected_pool_ids.add(candidate["case_id"])
                cases.append(
                    {
                        "case_id": f"stable-frontier-{policy}-{role}-{index}",
                        "schedule_ordinal": ordinal,
                        "admission_role": role,
                        "blocking": role == "frontier_candidate",
                        "pool_case_id": candidate["case_id"],
                        "focused_agent": policy,
                        "opponent_family": candidate["opponent_family"],
                        "seed": int(candidate["seed"]),
                        "size": int(candidate["size"]),
                        "horizon": int(candidate["horizon"]),
                        "replicas": 4,
                        "provenance": candidate["provenance"],
                    }
                )
                ordinal += 1

    case_counts = Counter(row["focused_agent"] for row in cases)
    role_counts = Counter(row["admission_role"] for row in cases)
    blocking_family_counts = {
        policy: len(
            {
                row["opponent_family"]
                for row in cases
                if row["focused_agent"] == policy and row["blocking"]
            }
        )
        for policy in POLICIES
    }
    if len(cases) != 24 or set(case_counts.values()) != {6}:
        raise ValueError("V14.2 pilot must contain six cases per policy")
    if role_counts != {"frontier_candidate": 16, "current_probe": 8}:
        raise ValueError("V14.2 pilot role allocation changed")
    if set(blocking_family_counts.values()) != {3}:
        raise ValueError("V14.2 blocking cases must span all non-current families")

    thresholds = {
        "protocol_admission_rate": 1.0,
        "minimum_variable_groups_per_policy": 2,
        "minimum_variable_families_per_policy": 2,
        "minimum_nonzero_advantage_rate_per_policy": 0.25,
        "minimum_mean_focused_action_diversity_per_policy": 1.25,
        "minimum_positive_advantages_per_policy": 4,
        "minimum_negative_advantages_per_policy": 4,
        "current_probe_is_blocking": False,
    }
    screen_body = {
        "version": SCREEN_VERSION,
        "scope": "training_only_frontier_selection_pilot_no_development_or_frozen_data",
        "source_hashes": source_hashes,
        "ordinary_case_pool_sha256": pool["sha256"],
        "cases": cases,
        "case_count": len(cases),
        "games": sum(int(row["replicas"]) for row in cases),
        "policy_case_counts": dict(sorted(case_counts.items())),
        "role_case_counts": dict(sorted(role_counts.items())),
        "blocking_family_counts": blocking_family_counts,
        "thresholds": thresholds,
        "pilot_data_policy": (
            "pilot trajectories select scenario identities only and are discarded before "
            "optimizer update 1; training uses fresh sampling namespaces"
        ),
        "runner_contract": {
            "script": "experiments/swarm_arena/scripts/run_v13_ordinary_signal_screen.py",
            "optimizer_updates": 0,
            "one_case_per_invocation": True,
            "shared_return_replicas": 4,
            "focused_phase": "ACT",
        },
        "no_seed_search": True,
    }
    screen = {**screen_body, "sha256": canonical_sha256(screen_body)}

    observed_retests = []
    for pool_case_id, row in sorted(retest.items()):
        if not pool_case_id.startswith("observed:"):
            continue
        source_id = pool_case_id.removeprefix("observed:")
        old = source.get(source_id)
        if old is None or float(old["return_range"]) <= EPSILON:
            continue
        observed_retests.append(
            {
                "pool_case_id": pool_case_id,
                "focused_agent": row["focused_agent"],
                "opponent_family": row["opponent_family"],
                "v14_classification": "frontier",
                "v14_1_classification": row["classification"],
                "retained_frontier": row["classification"] == "frontier",
            }
        )
    retained = sum(row["retained_frontier"] for row in observed_retests)
    diagnosis_body = {
        "version": VERSION,
        "source_hashes": source_hashes,
        "observed_frontier_retests": observed_retests,
        "observed_frontier_retest_count": len(observed_retests),
        "observed_frontier_retained_count": retained,
        "observed_frontier_retention_rate": retained / len(observed_retests),
        "failure_mechanism": (
            "four-replica case classification is noisy: previously variable scenarios can "
            "produce four identical sampled actions and flat returns on a later block"
        ),
        "credit_estimator_decision": {
            "change": False,
            "reason": (
                "when focused actions varied, leave-one-out shared-return credit produced "
                "signed advantages; the failed groups had identical actions and returns, "
                "so the evidence points to scenario/sampling coverage rather than a broken critic"
            ),
        },
        "curriculum_decision": (
            "use a disjoint pilot to select a stable frontier, discard pilot trajectories, "
            "permit zero-signal current-opponent probes without blocking otherwise learnable "
            "policy slots, and retain the existing ten-update adaptive selector"
        ),
        "frozen_or_development_data_used": False,
        "optimizer_updates_used": 0,
    }
    diagnosis = {**diagnosis_body, "sha256": canonical_sha256(diagnosis_body)}

    repaired_curriculum = json.loads(json.dumps(curriculum))
    repaired_curriculum["version"] = "arena-rl-v14.2-grounded-follow-through-v1"
    repaired_curriculum["ordinary_frontier_stability_repair"] = {
        "diagnosis_sha256": diagnosis["sha256"],
        "pilot_screen_manifest_sha256": screen["sha256"],
        "blocking_lane": "four non-current frontier candidates per policy",
        "probe_lane": "two cross-opponent current-policy transfers per policy",
        "current_probe_is_blocking": False,
        "pilot_trajectories_used_for_optimization": False,
        "credit_estimator_changed": False,
    }
    repaired_curriculum["adaptive_scope"]["evidence"] = (
        "fresh training rollouts from the immediately preceding completed stage only; "
        "the zero-update pilot selects identities but contributes no optimizer samples"
    )
    curriculum_body = {
        key: value for key, value in repaired_curriculum.items() if key != "sha256"
    }
    repaired_curriculum = {
        **curriculum_body,
        "sha256": canonical_sha256(curriculum_body),
    }

    audit_body = {
        "version": VERSION,
        "status": "cpu_design_passed_gpu_pilot_pending",
        "source_hashes": source_hashes,
        "diagnosis_sha256": diagnosis["sha256"],
        "pilot_screen_manifest_sha256": screen["sha256"],
        "repaired_curriculum_sha256": repaired_curriculum["sha256"],
        "pilot_cases": len(cases),
        "pilot_games": screen["games"],
        "cases_per_policy": dict(sorted(case_counts.items())),
        "role_case_counts": dict(sorted(role_counts.items())),
        "current_probes_are_cross_opponent_transfers": all(
            row["provenance"] == "cross_opponent_transfer_from_observed_frontier"
            for row in cases
            if row["admission_role"] == "current_probe"
        ),
        "pilot_trajectories_used_for_optimization": False,
        "stage_gates_changed": False,
        "credit_estimator_changed": False,
        "frozen_or_development_data_used": False,
        "optimizer_updates_used": 0,
    }
    audit = {**audit_body, "sha256": canonical_sha256(audit_body)}
    return {
        "diagnosis.json": diagnosis,
        "pilot_screen_manifest.json": screen,
        "curriculum.json": repaired_curriculum,
        "audit.json": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-assessment", type=Path, required=True)
    parser.add_argument("--v14-1-partial", type=Path, required=True)
    parser.add_argument("--v14-1-pool", type=Path, required=True)
    parser.add_argument("--v14-1-curriculum", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "v14_assessment": args.v14_assessment,
        "v14_1_partial": args.v14_1_partial,
        "v14_1_pool": args.v14_1_pool,
        "v14_1_curriculum": args.v14_1_curriculum,
    }
    artifacts = build(
        json.loads(args.v14_assessment.read_text(encoding="utf-8")),
        json.loads(args.v14_1_partial.read_text(encoding="utf-8")),
        json.loads(args.v14_1_pool.read_text(encoding="utf-8")),
        json.loads(args.v14_1_curriculum.read_text(encoding="utf-8")),
        source_hashes={name: file_sha256(path) for name, path in inputs.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(artifacts["audit.json"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
