#!/usr/bin/env python3
"""Build a bounded V14.1 ordinary-frontier repair from training-only evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POLICIES = ("blue-0", "blue-1", "blue-2", "blue-3")
OPPONENTS = ("base", "sft", "historical", "current")
POOL_VERSION = "arena-rl-ordinary-case-pool-v1"
SCREEN_VERSION = "arena-rl-v14-ordinary-frontier-screen-v1"
REPAIR_VERSION = "arena-rl-v14-ordinary-frontier-repair-v1"
EPSILON = 1e-12


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row: dict[str, Any]) -> str:
    if float(row["return_range"]) > EPSILON:
        return "frontier"
    return "mastered" if float(row["return_max"]) > 0 else "stalled"


def unseen_seed(
    *, policy: str, family: str, source_case_id: str, index: int, used: set[int]
) -> int:
    nonce = index
    while True:
        value = int.from_bytes(
            hashlib.sha256(
                f"v14.1:{policy}:{family}:{source_case_id}:{nonce}".encode()
            ).digest()[:8],
            "big",
        )
        seed = 30_000_000 + value % 60_000_000
        if seed not in used:
            used.add(seed)
            return seed
        nonce += 1


def pool_case(
    *,
    case_id: str,
    policy: str,
    family: str,
    seed: int,
    size: int,
    horizon: int,
    initial_classification: str,
    provenance: str,
    source_case_id: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "focused_agent": policy,
        "opponent_family": family,
        "seed": seed,
        "size": size,
        "horizon": horizon,
        "initial_classification": initial_classification,
        "provenance": provenance,
        "source_case_id": source_case_id,
    }


def build(
    assessment: dict[str, Any],
    original_screen: dict[str, Any],
    curriculum: dict[str, Any],
    *,
    source_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if assessment.get("status") != "failed":
        raise ValueError("repair requires the completed failed V14 screen")
    if float(assessment.get("protocol_admission_rate", 0.0)) != 1.0:
        raise ValueError("ordinary frontier repair cannot mask protocol failure")
    if assessment.get("screen_manifest_sha256") != original_screen.get("sha256"):
        raise ValueError("assessment does not bind the supplied screen manifest")
    if set(assessment["policy_metrics"]) != set(POLICIES):
        raise ValueError("assessment must contain all four policy slots")

    rows = [dict(row) for row in assessment["cases"]]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    frontier_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        policy = str(row["focused_agent"])
        family = str(row["opponent_family"])
        if policy not in POLICIES or family not in OPPONENTS:
            raise ValueError("screen contains an unknown policy/opponent family")
        row["classification"] = classify(row)
        by_key[(policy, family)].append(row)
        if row["classification"] == "frontier":
            frontier_by_policy[policy].append(row)
    if set(by_key) != {(policy, family) for policy in POLICIES for family in OPPONENTS}:
        raise ValueError("screen must cover every policy/opponent family")
    if any(not frontier_by_policy[policy] for policy in POLICIES):
        raise ValueError("each policy must retain at least one observed frontier case")
    for values in by_key.values():
        values.sort(key=lambda row: (-float(row["return_range"]), str(row["case_id"])))
    for values in frontier_by_policy.values():
        values.sort(key=lambda row: (-float(row["return_range"]), str(row["case_id"])))

    used_seeds = {int(row["seed"]) for row in rows}
    pool_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    ordinal = 0
    for policy in POLICIES:
        for family in OPPONENTS:
            observed = by_key[(policy, family)]
            observed_frontier = [row for row in observed if row["classification"] == "frontier"]
            for row in observed:
                pool_rows.append(
                    pool_case(
                        case_id=f"observed:{row['case_id']}",
                        policy=policy,
                        family=family,
                        seed=int(row["seed"]),
                        size=int(row["size"]),
                        horizon=int(row["horizon"]),
                        initial_classification=str(row["classification"]),
                        provenance="complete_v14_screen",
                        source_case_id=str(row["case_id"]),
                    )
                )

            transfer_sources = sorted(
                frontier_by_policy[policy],
                key=lambda row: (
                    str(row["opponent_family"]) == family,
                    -float(row["return_range"]),
                    str(row["case_id"]),
                ),
            )[:2]
            transfers = []
            for index, source in enumerate(transfer_sources):
                candidate = pool_case(
                    case_id=f"transfer:{policy}:{family}:{index}:{source['case_id']}",
                    policy=policy,
                    family=family,
                    seed=int(source["seed"]),
                    size=int(source["size"]),
                    horizon=int(source["horizon"]),
                    initial_classification="unseen",
                    provenance="cross_opponent_transfer_from_observed_frontier",
                    source_case_id=str(source["case_id"]),
                )
                pool_rows.append(candidate)
                transfers.append(candidate)

            unseen = []
            for index, source in enumerate(transfer_sources):
                candidate = pool_case(
                    case_id=f"unseen:{policy}:{family}:{index}",
                    policy=policy,
                    family=family,
                    seed=unseen_seed(
                        policy=policy,
                        family=family,
                        source_case_id=str(source["case_id"]),
                        index=index,
                        used=used_seeds,
                    ),
                    size=int(source["size"]),
                    horizon=int(source["horizon"]),
                    initial_classification="unseen",
                    provenance="deterministic_unseen_neighbor_of_observed_frontier",
                    source_case_id=str(source["case_id"]),
                )
                pool_rows.append(candidate)
                unseen.append(candidate)

            first = (
                next(
                    row
                    for row in pool_rows
                    if row["case_id"] == f"observed:{observed_frontier[0]['case_id']}"
                )
                if observed_frontier
                else transfers[0]
            )
            selected = (first, unseen[0])
            for within_key, candidate in enumerate(selected):
                screen_rows.append(
                    {
                        "case_id": f"ordinary-frontier-{policy}-{family}-{within_key}",
                        "schedule_ordinal": ordinal,
                        "seed": candidate["seed"],
                        "size": candidate["size"],
                        "horizon": candidate["horizon"],
                        "stage": "v14_ordinary_frontier_repair",
                        "focused_agent": policy,
                        "opponent_family": family,
                        "replicas": 4,
                        "pool_case_id": candidate["case_id"],
                        "provenance": candidate["provenance"],
                    }
                )
                ordinal += 1

    pool_by_id = {row["case_id"]: row for row in pool_rows}
    if len(pool_by_id) != len(pool_rows) or len(pool_rows) != 128:
        raise ValueError("ordinary repair must produce 128 unique pool cases")
    if len(screen_rows) != 32:
        raise ValueError("ordinary repair screen must contain exactly 32 cases")
    for row in screen_rows:
        source = pool_by_id[row["pool_case_id"]]
        for key in (
            "focused_agent",
            "opponent_family",
            "seed",
            "size",
            "horizon",
            "provenance",
        ):
            if row[key] != source[key]:
                raise ValueError(f"ordinary repair screen changed pool binding: {key}")

    pool_body = {
        "version": POOL_VERSION,
        "scope": "training_only_complete_v14_screen_no_development_or_frozen_data",
        "selection_policy": {
            "frontier_fraction": 0.8,
            "mastered_anchor_fraction": 0.1,
            "stalled_anchor_fraction": 0.1,
            "classification": (
                "frontier iff complete pass@4 returns vary; otherwise mastered for "
                "positive flat return and stalled for nonpositive flat return"
            ),
            "unseen_cases": "frontier until one complete logical group is observed",
        },
        "source_hashes": source_hashes,
        "cases": pool_rows,
        "case_count": len(pool_rows),
    }
    pool = {**pool_body, "sha256": canonical_sha256(pool_body)}

    thresholds = dict(original_screen["thresholds"])
    screen_body = {
        "version": SCREEN_VERSION,
        "scope": "V14.1 training-only preflight; development and frozen data unopened",
        "source_assessment_sha256": source_hashes["assessment"],
        "source_screen_manifest_sha256": original_screen["sha256"],
        "ordinary_case_pool_sha256": pool["sha256"],
        "cases": screen_rows,
        "case_count": len(screen_rows),
        "games": len(screen_rows) * 4,
        "policy_case_counts": dict(
            sorted(Counter(row["focused_agent"] for row in screen_rows).items())
        ),
        "opponent_case_counts": dict(
            sorted(Counter(row["opponent_family"] for row in screen_rows).items())
        ),
        "thresholds": thresholds,
        "runner_contract": original_screen["runner_contract"],
        "no_seed_search": True,
    }
    screen = {**screen_body, "sha256": canonical_sha256(screen_body)}

    repaired_curriculum = json.loads(json.dumps(curriculum))
    repaired_curriculum["version"] = "arena-rl-v14.1-grounded-follow-through-v1"
    repaired_curriculum["ordinary_case_pool"] = pool
    repaired_curriculum["ordinary_frontier_repair"] = {
        "source_assessment_sha256": source_hashes["assessment"],
        "screen_manifest_sha256": screen["sha256"],
        "policy": pool_body["selection_policy"],
    }
    repaired_curriculum["adaptive_scope"] = {
        "changes": (
            "training handoff and ordinary case identities at ten-update boundaries"
        ),
        "does_not_change": [
            "stage group mix",
            "focused-policy balance",
            "opponent rotation",
            "reward or counterfactual",
            "development or frozen evaluation",
        ],
        "ordinary_policy": (
            "80% frontier with 10% mastered and 10% stalled anchors, selected only "
            "within the frozen policy/opponent-family slot"
        ),
        "evidence": "immediately preceding completed training stage only",
    }
    repaired_body = {
        key: value for key, value in repaired_curriculum.items() if key != "sha256"
    }
    repaired_curriculum = {
        **repaired_body,
        "sha256": canonical_sha256(repaired_body),
    }

    classification_counts = Counter(
        row["initial_classification"] for row in pool_rows
    )
    audit_body = {
        "version": REPAIR_VERSION,
        "status": "cpu_repair_passed_gpu_screen_pending",
        "source_hashes": source_hashes,
        "ordinary_case_pool_sha256": pool["sha256"],
        "screen_manifest_sha256": screen["sha256"],
        "repaired_curriculum_sha256": repaired_curriculum["sha256"],
        "pool_case_count": len(pool_rows),
        "pool_classification_counts": dict(sorted(classification_counts.items())),
        "screen_case_count": len(screen_rows),
        "screen_games": len(screen_rows) * 4,
        "screen_cases_per_policy": dict(
            sorted(Counter(row["focused_agent"] for row in screen_rows).items())
        ),
        "screen_cases_per_policy_family": {
            f"{policy}:{family}": sum(
                row["focused_agent"] == policy and row["opponent_family"] == family
                for row in screen_rows
            )
            for policy in POLICIES
            for family in OPPONENTS
        },
        "thresholds_unchanged": thresholds == original_screen["thresholds"],
        "blue_1_blue_2_current_cases_are_unseen": all(
            row["provenance"] != "complete_v14_screen"
            for row in screen_rows
            if row["focused_agent"] in {"blue-1", "blue-2"}
            and row["opponent_family"] == "current"
        ),
        "frozen_or_development_data_used": False,
        "optimizer_updates_used": 0,
    }
    audit = {**audit_body, "sha256": canonical_sha256(audit_body)}
    return {
        "ordinary_case_pool.json": pool,
        "ordinary_frontier_screen_manifest.json": screen,
        "curriculum.json": repaired_curriculum,
        "audit.json": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--original-screen", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
    original_screen = json.loads(args.original_screen.read_text(encoding="utf-8"))
    curriculum = json.loads(args.curriculum.read_text(encoding="utf-8"))
    artifacts = build(
        assessment,
        original_screen,
        curriculum,
        source_hashes={
            "assessment": file_sha256(args.assessment),
            "original_screen": file_sha256(args.original_screen),
            "curriculum": file_sha256(args.curriculum),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(artifacts["audit.json"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
