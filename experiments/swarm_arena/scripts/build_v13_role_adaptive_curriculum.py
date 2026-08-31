#!/usr/bin/env python3
"""Build the V13 role-adaptive consolidation curriculum from a V12 gap screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    CurriculumStage,
    exact_staged_curriculum_schedule,
)

VERSION = "arena-rl-v13-role-adaptive-consolidation-v1"
POLICIES = ("blue-0", "blue-1", "blue-2", "blue-3")
CHALLENGE_GROUPS = 80
MINIMUM_CHALLENGE_GROUPS_PER_POLICY = 12
CRITICAL_REHEARSAL_ROLE_QUOTAS = {policy: 5 for policy in POLICIES}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_order(quotas: dict[str, int]) -> list[str]:
    """Interleave an exact role quota without long same-role runs."""
    remaining = dict(quotas)
    result = []
    previous = None
    while sum(remaining.values()):
        available = [policy for policy, count in remaining.items() if count]
        nonrepeating = [policy for policy in available if policy != previous]
        candidates = sorted(
            nonrepeating or available,
            key=lambda policy: (-remaining[policy], policy),
        )
        selected = candidates[0]
        result.append(selected)
        remaining[selected] -= 1
        previous = selected
    return result


def case_stream(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
) -> Iterator[tuple[int, str]]:
    by_receiver: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        by_receiver[str(row["receiver"])].append((int(row["pair_index"]), str(row["world"])))
    cursors = Counter()
    for receiver in role_order(quotas):
        cases = by_receiver[receiver]
        if not cases:
            raise ValueError(f"no selected repair cases for {receiver}")
        yield cases[cursors[receiver] % len(cases)]
        cursors[receiver] += 1


def challenge_role_quotas(selection: dict[str, Any]) -> dict[str, int]:
    """Allocate the repair surplus by final training-only challenge severity."""
    by_receiver: dict[str, list[float]] = defaultdict(list)
    for row in selection["selected"]["challenge_repair"]:
        by_receiver[str(row["receiver"])].append(float(row["priority"]))
    if set(by_receiver) != set(POLICIES):
        raise ValueError("challenge selection must cover every policy")
    severity = {
        policy: sum(by_receiver[policy]) / len(by_receiver[policy]) for policy in POLICIES
    }
    floor_total = MINIMUM_CHALLENGE_GROUPS_PER_POLICY * len(POLICIES)
    surplus = CHALLENGE_GROUPS - floor_total
    total_severity = sum(severity.values())
    weights = (
        {policy: severity[policy] / total_severity for policy in POLICIES}
        if total_severity > 0
        else {policy: 1 / len(POLICIES) for policy in POLICIES}
    )
    raw = {policy: weights[policy] * surplus for policy in POLICIES}
    quotas = {
        policy: MINIMUM_CHALLENGE_GROUPS_PER_POLICY + int(raw[policy])
        for policy in POLICIES
    }
    unassigned = CHALLENGE_GROUPS - sum(quotas.values())
    remainder_order = sorted(
        POLICIES,
        key=lambda policy: (-(raw[policy] - int(raw[policy])), policy),
    )
    for policy in remainder_order[:unassigned]:
        quotas[policy] += 1
    if sum(quotas.values()) != CHALLENGE_GROUPS:
        raise AssertionError("challenge quota allocation must be exact")
    return quotas


def handoff_case_order(selection: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    challenge_quotas = challenge_role_quotas(selection)
    challenge = iter(
        case_stream(selection["selected"]["challenge_repair"], challenge_quotas)
    )
    rehearsal = iter(
        case_stream(
            selection["selected"]["critical_rehearsal"],
            CRITICAL_REHEARSAL_ROLE_QUOTAS,
        )
    )
    patterns = (
        [(2, 1, 1)] * 20
        + [(1, 2, 1), (2, 1, 1)] * 20
        + [(2, 1, 1)] * 20
    )
    ordered = []
    for _ordinary, critical, decoy in patterns:
        if decoy != 1:
            raise AssertionError("V13 requires exactly one matched challenge per update")
        ordered.append(next(challenge))
        ordered.extend(next(rehearsal) for _ in range(critical - decoy))
    try:
        next(challenge)
        raise AssertionError("unused challenge role quota")
    except StopIteration:
        pass
    try:
        next(rehearsal)
        raise AssertionError("unused critical rehearsal role quota")
    except StopIteration:
        pass
    return tuple(ordered)


def curriculum(selection: dict[str, Any]) -> dict[str, Any]:
    cases = handoff_case_order(selection)
    challenge_quotas = challenge_role_quotas(selection)
    retention = {"ordinary": 2, "critical": 1, "decoy": 1}
    transfer = {"ordinary": 1, "critical": 2, "decoy": 1}
    return {
        "version": VERSION,
        "total_updates": 80,
        "groups_per_update": 4,
        "reward": "verified_terminal_control_delta_only",
        "message_reward": None,
        "initializer": "four_distinct_public_v12_update160_nonadmitted_warmstart_adapters",
        "initializer_claim_boundary": (
            "V12 update160 is a continuation warm start, not a formally selected result"
        ),
        "launch_block": "no V13 launch until the ordinary pass@4 signal screen passes",
        "credit_assignment": {
            "critical": "receiver_ACT_absolute_factual_minus_receiver_only_target_swap",
            "decoy": "receiver_ACT_absolute_target_swap_challenge_minus_factual",
            "ordinary": "shared_terminal_return_leave_one_out",
        },
        "repair_hypotheses": [
            "blue-0 still over-obeys misleading broadcasts and needs the largest challenge quota",
            "easy factual following is saturated, so retain it as matched control rather than scale it",
            "fresh ordinary maps plus replay anchors are needed to prevent legacy/hard regression",
            "current and historical opponents remain the hardest communication counterparties",
        ],
        "schedule": {
            "pair_offset": 0,
            "ordinary_seed_base": 22_000_117,
            "shuffle_seed": 20_261_013,
        },
        "challenge_role_quotas": challenge_quotas,
        "critical_rehearsal_role_quotas": CRITICAL_REHEARSAL_ROLE_QUOTAS,
        "stages": [
            {
                "name": "role_adaptive_conflict_repair",
                "updates": 20,
                "update_pattern": [retention],
                "ordinary_sizes": [14, 16, 18],
                "ordinary_horizons": [6, 8, 10],
                "handoff_remaining_turns": 2,
                "handoff_cases": cases,
            },
            {
                "name": "hard_negative_transfer",
                "updates": 40,
                "update_pattern": [transfer, retention],
                "ordinary_sizes": [16, 18, 20, 22],
                "ordinary_horizons": [8, 10, 12],
                "handoff_remaining_turns": 4,
                "handoff_cases": cases,
            },
            {
                "name": "capability_consolidation",
                "updates": 20,
                "update_pattern": [retention],
                "ordinary_sizes": [18, 20, 22],
                "ordinary_horizons": [10, 12, 14],
                "handoff_remaining_turns": 4,
                "handoff_cases": cases,
            },
        ],
        "runtime": {
            "shared_return_replicas": 4,
            "paired_contrast_centering": "none",
            "target_swap_sender_retries": 8,
            "opponent_pool": ["base", "sft", "historical", "current"],
            "four_distinct_policy_adapters": True,
        },
        "ordinary_case_policy": {
            "fresh_groups": 140,
            "screen": "training-only pass@4 before launch; reject all-zero-credit batches",
            "replay_anchor_source": "V12 training-only gap screen; never held-out data",
            "minimum_slot_coverage": "every policy receives nonzero ordinary credit in preflight",
        },
        "online_eval_updates": [0, 20, 40, 60, 80],
        "development_candidates": [20, 40, 60, 80],
        "selection_rule": (
            "earliest candidate with positive semantic return and specificity, both ordinary "
            "clustered lower bounds at least -0.02, and no receiver challenge regression"
        ),
        "stop_rule": (
            "stop after two consecutive checkpoints with no improvement in blue-0 challenge "
            "robustness and any ordinary regression"
        ),
    }


def stage(row: dict[str, Any]) -> CurriculumStage:
    return CurriculumStage(
        name=row["name"],
        updates=row["updates"],
        update_pattern=tuple(CurriculumMix(**mix) for mix in row["update_pattern"]),
        ordinary_sizes=tuple(row["ordinary_sizes"]),
        ordinary_horizons=tuple(row["ordinary_horizons"]),
        handoff_focus_roles=("receiver",),
        handoff_cases=tuple(
            (pair_index, world) for pair_index, world in row["handoff_cases"]
        ),
        handoff_remaining_turns=row["handoff_remaining_turns"],
    )


def build(selection: dict[str, Any], train: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_status = selection["admission"]["status"]
    if selection_status not in {"interim_only", "training_only_complete"}:
        raise ValueError("V13 builder expects a training-only gap selection")
    plan = curriculum(selection)
    schedule = exact_staged_curriculum_schedule(
        tuple(stage(row) for row in plan["stages"]),
        groups_per_update=plan["groups_per_update"],
        pair_offset=plan["schedule"]["pair_offset"],
        ordinary_seed_base=plan["schedule"]["ordinary_seed_base"],
        shuffle_seed=plan["schedule"]["shuffle_seed"],
    )
    counts = Counter(row.kind for row in schedule)
    decoy_cases = {
        (row.pair_index, row.handoff_world) for row in schedule if row.kind == "decoy"
    }
    critical_cases = {
        (row.pair_index, row.handoff_world) for row in schedule if row.kind == "critical"
    }
    receivers = {
        kind: Counter(
            train["pairs"][row.pair_index][kind]["receiver"]
            for row in schedule
            if row.kind == kind and row.pair_index is not None
        )
        for kind in ("critical", "decoy")
    }
    audit = {
        "version": "arena-rl-v13-curriculum-audit-v1",
        "status": (
            "cpu_schedule_passed_gpu_gates_pending"
            if selection_status == "training_only_complete"
            else "interim_passed_refresh_required_at_v12_completion"
        ),
        "schedule_sha256": digest(
            [
                {
                    key: value
                    for key, value in row.__dict__.items()
                    if key != "handoff_trainable_turn_offsets" or value is not None
                }
                for row in schedule
            ]
        ),
        "selection_sha256": digest(selection),
        "group_counts": dict(sorted(counts.items())),
        "receiver_counts": {
            key: dict(sorted(value.items())) for key, value in receivers.items()
        },
        "decoys_are_matched_critical_subset": decoy_cases <= critical_cases,
        "ordinary_seeds_unique": len(
            {row.ordinary_seed for row in schedule if row.kind == "ordinary"}
        )
        == counts["ordinary"],
        "frozen_data_used": False,
        "launch_ready": False,
        "remaining_blockers": (
            [
                "run training-only ordinary pass@4 signal screen",
            ]
            if selection_status == "training_only_complete"
            else [
                "refresh gap selection from complete V12 progress",
                "run training-only ordinary pass@4 signal screen",
            ]
        ),
    }
    return plan, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    train = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    plan, audit = build(selection, train)
    audit["selection_file_sha256"] = file_sha256(args.selection)
    audit["train_manifest_file_sha256"] = file_sha256(args.train_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "curriculum.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "curriculum_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
