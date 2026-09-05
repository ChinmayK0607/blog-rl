#!/usr/bin/env python3
"""Build V14's gated multi-turn follow-through curriculum from public V13 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from swarm_ctf_eval.adaptive_curriculum import (
    select_handoff_cases,
    summarize_training_progress,
)
from swarm_ctf_eval.rl_production import (
    AdaptiveCurriculumConfig,
    CurriculumMix,
    CurriculumStage,
    exact_staged_curriculum_schedule,
)

VERSION = "arena-rl-v14-grounded-follow-through-v1"
POLICIES = ("blue-0", "blue-1", "blue-2", "blue-3")
TOTAL_UPDATES = 40
GROUPS_PER_UPDATE = 4


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _balanced_order(quotas: dict[str, int]) -> list[str]:
    remaining = dict(quotas)
    order: list[str] = []
    previous = None
    while sum(remaining.values()):
        available = [policy for policy, count in remaining.items() if count]
        candidates = [policy for policy in available if policy != previous] or available
        selected = sorted(candidates, key=lambda policy: (-remaining[policy], policy))[0]
        order.append(selected)
        remaining[selected] -= 1
        previous = selected
    return order


def _case_stream(
    cases_by_receiver: dict[str, list[tuple[int, str]]],
    quotas: dict[str, int],
) -> Iterator[tuple[int, str]]:
    cursors = Counter()
    for receiver in _balanced_order(quotas):
        cases = cases_by_receiver[receiver]
        if not cases:
            raise ValueError(f"V14 has no training-only handoff cases for {receiver}")
        yield cases[cursors[receiver] % len(cases)]
        cursors[receiver] += 1


def _balanced_quotas(total: int, *, policy_offset: int) -> dict[str, int]:
    quotient, remainder = divmod(total, len(POLICIES))
    extra = {
        POLICIES[(policy_offset + index) % len(POLICIES)] for index in range(remainder)
    }
    return {
        policy: quotient + (policy in extra)
        for policy in POLICIES
    }


def _source_cases(
    v13_curriculum: dict[str, Any],
    train_manifest: dict[str, Any],
) -> dict[str, list[tuple[int, str]]]:
    cases: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for stage in v13_curriculum["stages"]:
        for pair_index, world in stage["handoff_cases"]:
            case = (int(pair_index), str(world))
            if case in seen:
                continue
            seen.add(case)
            receiver = str(train_manifest["pairs"][case[0]]["critical"]["receiver"])
            cases[receiver].append(case)
    if set(cases) != set(POLICIES):
        raise ValueError("V14 source cases must cover all four receiver policies")
    return dict(cases)


def _handoff_case_order(
    patterns: list[dict[str, int]],
    cases_by_receiver: dict[str, list[tuple[int, str]]],
    *,
    policy_offset: int,
) -> tuple[tuple[int, str], ...]:
    decoy_groups = sum(row["decoy"] for row in patterns)
    extra_critical = sum(row["critical"] - row["decoy"] for row in patterns)
    decoy = iter(
        _case_stream(
            cases_by_receiver,
            _balanced_quotas(decoy_groups, policy_offset=policy_offset),
        )
    )
    rehearsal = iter(
        _case_stream(
            cases_by_receiver,
            _balanced_quotas(extra_critical, policy_offset=policy_offset),
        )
    )
    ordered = []
    for mix in patterns:
        ordered.append(next(decoy))
        ordered.extend(next(rehearsal) for _ in range(mix["critical"] - mix["decoy"]))
    return tuple(ordered)


def _adaptive_handoff_case_order(
    patterns: list[dict[str, int]],
    cases_by_receiver: dict[str, list[tuple[int, str]]],
    analysis: dict[str, Any],
    *,
    stage_name: str,
    policy_offset: int,
    adaptive_config: AdaptiveCurriculumConfig,
) -> tuple[tuple[int, str], ...]:
    base = _handoff_case_order(
        patterns,
        cases_by_receiver,
        policy_offset=policy_offset,
    )
    receiver_by_case = {
        case: receiver for receiver, cases in cases_by_receiver.items() for case in cases
    }
    decoy_positions: list[int] = []
    critical_positions: list[int] = []
    cursor = 0
    for mix in patterns:
        decoy_positions.extend(range(cursor, cursor + mix["decoy"]))
        critical_positions.extend(range(cursor + mix["decoy"], cursor + mix["critical"]))
        cursor += mix["critical"]
    result = list(base)
    if decoy_positions:
        selected = select_handoff_cases(
            kind="decoy",
            receiver_sequence=[receiver_by_case[base[index]] for index in decoy_positions],
            pool_by_receiver=cases_by_receiver,
            analysis=analysis,
            config=adaptive_config,
            selection_namespace=f"bootstrap:{stage_name}:decoy",
        )
        for index, case in zip(decoy_positions, selected, strict=True):
            result[index] = case
    if critical_positions:
        selected = select_handoff_cases(
            kind="critical",
            receiver_sequence=[receiver_by_case[base[index]] for index in critical_positions],
            pool_by_receiver=cases_by_receiver,
            analysis=analysis,
            config=adaptive_config,
            selection_namespace=f"bootstrap:{stage_name}:critical",
        )
        for index, case in zip(critical_positions, selected, strict=True):
            result[index] = case
    return tuple(result)


def _extract_evidence(summary: dict[str, Any]) -> dict[str, float | list[float]]:
    return {
        "normal_minus_dropped": summary["communication_effects"]["normal_minus_dropped"][
            "mean_difference"
        ],
        "normal_minus_dropped_95": summary["communication_effects"]["normal_minus_dropped"][
            "mean_difference_95"
        ],
        "specificity": summary["critical_minus_decoy_specificity"]["mean_difference"],
        "specificity_95": summary["critical_minus_decoy_specificity"]["mean_difference_95"],
        "rl_specific_lift": summary["rl_specific_communication_lift"]["mean_difference"],
        "rl_specific_lift_95": summary["rl_specific_communication_lift"][
            "mean_difference_95"
        ],
        "capture_normal_minus_dropped": summary["communication_mechanism"][
            "candidate_capture_normal_minus_dropped"
        ]["mean_difference"],
        "ordinary_hard": summary["capability_rl_minus_sft"]["ordinary_hard"][
            "mean_difference"
        ],
        "ordinary_legacy": summary["capability_rl_minus_sft"]["ordinary_legacy"][
            "mean_difference"
        ],
        "overall_gameplay": summary["overall_gameplay_rl_minus_sft"]["mean_difference"],
    }


def _gate_requirements(
    *,
    normal: float,
    specificity: float,
    lift: float,
    capture: float,
    ordinary: float,
    overall: float,
) -> list[dict[str, Any]]:
    requirements = [
        {
            "name": "terminal_message_dependence",
            "path": ["communication_effects", "normal_minus_dropped", "mean_difference"],
            "minimum": normal,
        },
        {
            "name": "critical_decoy_specificity",
            "path": ["critical_minus_decoy_specificity", "mean_difference"],
            "minimum": specificity,
        },
        {
            "name": "rl_specific_communication_lift",
            "path": ["rl_specific_communication_lift", "mean_difference"],
            "minimum": lift,
        },
        {
            "name": "capture_mechanism",
            "path": [
                "communication_mechanism",
                "candidate_capture_normal_minus_dropped",
                "mean_difference",
            ],
            "minimum": capture,
        },
        {
            "name": "ordinary_hard_retention",
            "path": ["capability_rl_minus_sft", "ordinary_hard", "mean_difference"],
            "minimum": ordinary,
        },
        {
            "name": "ordinary_legacy_retention",
            "path": ["capability_rl_minus_sft", "ordinary_legacy", "mean_difference"],
            "minimum": ordinary,
        },
        {
            "name": "overall_gameplay_retention",
            "path": ["overall_gameplay_rl_minus_sft", "mean_difference"],
            "minimum": overall,
        },
    ]
    for name in ("action_protocol_rate", "broadcast_protocol_rate", "broadcast_grounded_rate"):
        requirements.append(
            {
                "name": name,
                "path": ["candidate_protocol", name],
                "equals": 1.0,
            }
        )
    return requirements


def _stage_gates() -> dict[str, Any]:
    thresholds = {
        10: (0.0, -0.01, -0.01, 0.10, -0.02, 0.0),
        20: (0.015, 0.0, 0.0, 0.125, -0.02, 0.01),
        30: (0.025, 0.01, 0.01, 0.125, -0.01, 0.02),
        40: (0.030, 0.02, 0.02, 0.125, 0.0, 0.03),
    }
    names = {
        10: "two_turn_mechanism",
        20: "three_turn_transfer",
        30: "long_horizon_conversion",
        40: "specificity_consolidation",
    }
    body = {
        "version": "arena-rl-stage-gates-v1",
        "scope": "development_point_estimate_training_control_not_confirmatory_claim",
        "checkpoints": {
            str(step): {
                "stage": names[step],
                "on_fail": "stop_before_next_optimizer_update",
                "requirements": _gate_requirements(
                    normal=values[0],
                    specificity=values[1],
                    lift=values[2],
                    capture=values[3],
                    ordinary=values[4],
                    overall=values[5],
                ),
            }
            for step, values in thresholds.items()
        },
    }
    return {**body, "sha256": digest(body)}


def build(
    *,
    v13_curriculum: dict[str, Any],
    train_manifest: dict[str, Any],
    update60: dict[str, Any],
    update80: dict[str, Any],
    initializer_manifest: dict[str, Any],
    v13_progress: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    for step, summary in ((60, update60), (80, update80)):
        if summary.get("version") != "arena-rl-progress-eval-v5-rl-specific-communication":
            raise ValueError(f"V13 update{step} summary has the wrong version")
        if summary.get("tier") != "pulse" or summary.get("rows") != 192:
            raise ValueError(f"V13 update{step} summary is incomplete")
    ready = initializer_manifest["ready"]
    if int(ready["step"]) != 80:
        raise ValueError("V14 initializer must be public V13 update80")
    for policy, value in ready["policy_adapter_sha256"].items():
        adapter = f"checkpoints/step-80/policy-{policy}/adapter_model.safetensors"
        if initializer_manifest["files_sha256"].get(adapter) != value:
            raise ValueError(f"V13 initializer manifest hash mismatch for {policy}")

    stage_specs = [
        {
            "name": "two_turn_mechanism",
            "updates": 10,
            "update_pattern": [{"ordinary": 1, "critical": 2, "decoy": 1}],
            "ordinary_sizes": [16, 18, 20],
            "ordinary_horizons": [8, 10, 12],
            "handoff_remaining_turns": 4,
            "handoff_trainable_turn_offsets": [0, 1],
        },
        {
            "name": "three_turn_transfer",
            "updates": 10,
            "update_pattern": [{"ordinary": 1, "critical": 2, "decoy": 1}],
            "ordinary_sizes": [16, 18, 20],
            "ordinary_horizons": [8, 10, 12],
            "handoff_remaining_turns": 4,
            "handoff_trainable_turn_offsets": [0, 1, 2],
        },
        {
            "name": "long_horizon_conversion",
            "updates": 10,
            "update_pattern": [{"ordinary": 1, "critical": 2, "decoy": 1}],
            "ordinary_sizes": [18, 20, 22],
            "ordinary_horizons": [10, 12, 14],
            "handoff_remaining_turns": 6,
            "handoff_trainable_turn_offsets": [0, 1, 2, 3],
        },
        {
            "name": "specificity_consolidation",
            "updates": 10,
            "update_pattern": [{"ordinary": 2, "critical": 1, "decoy": 1}],
            "ordinary_sizes": [18, 20, 22],
            "ordinary_horizons": [10, 12, 14],
            "handoff_remaining_turns": 6,
            "handoff_trainable_turn_offsets": [1, 2, 3, 4],
        },
    ]
    cases_by_receiver = _source_cases(v13_curriculum, train_manifest)
    adaptive_config = AdaptiveCurriculumConfig(
        candidate_cases=tuple(
            f"{pair_index}:{world}:{receiver}"
            for receiver in POLICIES
            for pair_index, world in cases_by_receiver[receiver]
        )
    )
    frontier_analysis = summarize_training_progress(
        v13_progress,
        config=adaptive_config,
    )
    stages = []
    for index, stage in enumerate(stage_specs):
        patterns = stage["update_pattern"] * stage["updates"]
        cases = (
            _adaptive_handoff_case_order(
                patterns,
                cases_by_receiver,
                frontier_analysis,
                stage_name=stage["name"],
                policy_offset=index * 2,
                adaptive_config=adaptive_config,
            )
            if index == 0
            else _handoff_case_order(
                patterns,
                cases_by_receiver,
                policy_offset=index * 2,
            )
        )
        stages.append(
            {**stage, "handoff_focus_roles": ["receiver"], "handoff_cases": cases}
        )
    curriculum = {
        "version": VERSION,
        "total_updates": TOTAL_UPDATES,
        "groups_per_update": GROUPS_PER_UPDATE,
        "initializer": "four_distinct_public_v13_update80_adapters",
        "initializer_claim_boundary": "continuation warm start; V13 did not pass communication claim",
        "objective": "convert verified receiver capture dependence into specific terminal team return",
        "reward": "verified_terminal_control_delta_only",
        "message_reward": None,
        "credit_assignment": {
            "critical": "receiver_ACT_multiturn_factual_minus_receiver_only_target_swap",
            "decoy": "receiver_ACT_multiturn_target_swap_challenge_minus_factual",
            "ordinary": "shared_terminal_return_leave_one_out",
        },
        "schedule": {
            "pair_offset": 0,
            "ordinary_seed_base": 27_000_131,
            "shuffle_seed": 20_261_031,
        },
        "stages": stages,
        "runtime": {
            "shared_return_replicas": 4,
            "action_prompt_profile": "focused_handoff_compact",
            "shared_return_baseline": "paired_receiver_target_swap",
            "decoy_shared_return_baseline": "paired_receiver_target_swap_challenge",
            "paired_contrast_centering": "none",
            "target_swap_sender_retries": 8,
            "four_distinct_policy_adapters": True,
            "monitor_logical_update_signal": True,
        },
        "adaptive_curriculum": adaptive_config.__dict__,
        "adaptive_scope": {
            "changes": "training handoff case identities at ten-update boundaries",
            "does_not_change": [
                "ordinary retention schedule",
                "stage group mix",
                "receiver balance",
                "opponent rotation",
                "reward or counterfactual",
                "development or frozen evaluation",
            ],
            "evidence": "immediately preceding completed training stage only",
        },
        "online_eval_updates": [0, 10, 20, 30, 40],
        "stage_gate_policy": "withhold continuation on any failed predeclared gate",
        "final_selection_rule": (
            "earliest checkpoint with positive communication, specificity, and RL-specific lift; "
            "ordinary hard/legacy point estimates nonnegative; confirmatory intervals reserved for "
            "the larger development/frozen evaluation"
        ),
        "frozen_data_opened": False,
    }
    stage_objects = tuple(
        CurriculumStage(
            name=row["name"],
            updates=row["updates"],
            update_pattern=tuple(CurriculumMix(**mix) for mix in row["update_pattern"]),
            ordinary_sizes=tuple(row["ordinary_sizes"]),
            ordinary_horizons=tuple(row["ordinary_horizons"]),
            handoff_focus_roles=("receiver",),
            handoff_cases=tuple((int(pair), str(world)) for pair, world in row["handoff_cases"]),
            handoff_remaining_turns=row["handoff_remaining_turns"],
            handoff_trainable_turn_offsets=tuple(row["handoff_trainable_turn_offsets"]),
        )
        for row in stages
    )
    schedule = exact_staged_curriculum_schedule(
        stage_objects,
        groups_per_update=GROUPS_PER_UPDATE,
        pair_offset=curriculum["schedule"]["pair_offset"],
        ordinary_seed_base=curriculum["schedule"]["ordinary_seed_base"],
        shuffle_seed=curriculum["schedule"]["shuffle_seed"],
    )
    receiver_counts = {
        kind: Counter(
            train_manifest["pairs"][row.pair_index][kind]["receiver"]
            for row in schedule
            if row.kind == kind and row.pair_index is not None
        )
        for kind in ("critical", "decoy")
    }
    bootstrap_cases = stages[0]["handoff_cases"]
    bootstrap_classifications = {"critical": Counter(), "decoy": Counter()}
    cursor = 0
    for mix in stage_specs[0]["update_pattern"] * stage_specs[0]["updates"]:
        for index in range(mix["critical"]):
            pair_index, world = bootstrap_cases[cursor + index]
            kind = "decoy" if index < mix["decoy"] else "critical"
            key = f"{kind}:{pair_index}:{world}"
            bootstrap_classifications[kind][
                frontier_analysis["handoff_cases"].get(key, {}).get("classification", "unseen")
            ] += 1
        cursor += mix["critical"]
    audit = {
        "version": "arena-rl-v14-curriculum-audit-v1",
        "status": "cpu_schedule_passed_gpu_preflight_pending",
        "schedule_sha256": digest([row.__dict__ for row in schedule]),
        "group_counts": dict(sorted(Counter(row.kind for row in schedule).items())),
        "receiver_counts": {
            kind: dict(sorted(counts.items())) for kind, counts in receiver_counts.items()
        },
        "stage_offsets": {
            stage.name: list(stage.handoff_trainable_turn_offsets or ()) for stage in stage_objects
        },
        "adaptive_curriculum": {
            "config_sha256": digest(adaptive_config.__dict__),
            "v13_frontier_analysis_sha256": frontier_analysis["sha256"],
            "bootstrap_classification_counts": {
                kind: dict(sorted(counts.items()))
                for kind, counts in bootstrap_classifications.items()
            },
            "later_stage_source": "immediately preceding completed training stage",
            "ordinary_schedule_changed": False,
            "development_or_frozen_data_used": False,
        },
        "decoys_are_update_local_matched_critical_subset": all(
            {
                (row.pair_index, row.handoff_world)
                for row in schedule[index : index + 4]
                if row.kind == "decoy"
            }
            <= {
                (row.pair_index, row.handoff_world)
                for row in schedule[index : index + 4]
                if row.kind == "critical"
            }
            for index in range(0, len(schedule), 4)
        ),
        "ordinary_seeds_unique": len(
            {row.ordinary_seed for row in schedule if row.kind == "ordinary"}
        )
        == sum(row.kind == "ordinary" for row in schedule),
        "old_plan_hash_compatibility_required": True,
        "frozen_data_opened": False,
        "launch_ready": False,
        "remaining_blockers": [
            "render immutable V14 production/trainer configs from a fresh runtime certificate",
            "run zero-update ordinary signal and multiturn exact-replay preflight",
            "verify public V13-u80 adapter hashes on the eventual GPU host",
            "verify HF mirror, W&B, watcher, recovery supervisor, and hard budget cutoff",
        ],
    }
    diagnosis_body = {
        "version": "arena-rl-v14-v13-gap-diagnosis-v1",
        "public_source": {
            "repository": "CK0607/swarm-arena-live-runs",
            "run": "runs/rl-v13-role-adaptive4b80-21d756d1",
            "required_files": [
                "evaluations/update-60/summary.json",
                "evaluations/update-80/summary.json",
                "checkpoints/step-80/MANIFEST.json",
            ],
        },
        "source_hashes": source_hashes,
        "v13_training_frontier_analysis_sha256": frontier_analysis["sha256"],
        "v13_update60": _extract_evidence(update60),
        "v13_update80": _extract_evidence(update80),
        "conclusion": (
            "V13 learned factual reporting and receiver capture dependence while later training "
            "improved gameplay but attenuated terminal communication specificity; V14 trains "
            "the receiver's audited follow-through actions across later turns."
        ),
        "not_addressed_by_reward_shaping": True,
        "frozen_data_opened": False,
    }
    diagnosis = {**diagnosis_body, "sha256": digest(diagnosis_body)}
    gates = _stage_gates()
    bundle_body = {
        "version": "arena-rl-v14-cpu-bundle-v1",
        "status": "cpu_design_and_tests_complete_gpu_preflight_pending",
        "initializer": {
            "run": "rl-v13-role-adaptive4b80-21d756d1",
            "step": 80,
            "policy_revision": ready["policy_revision"],
            "policy_adapter_sha256": ready["policy_adapter_sha256"],
            "claim_boundary": "continuation warm start, not a communication-claim pass",
        },
        "curriculum": {
            "updates_maximum": TOTAL_UPDATES,
            "groups_per_update": GROUPS_PER_UPDATE,
            "schedule_sha256": audit["schedule_sha256"],
            "stage_gate_sha256": gates["sha256"],
            "adaptive_config_sha256": audit["adaptive_curriculum"]["config_sha256"],
        },
        "artifact_body_sha256": {
            "diagnosis": diagnosis["sha256"],
            "curriculum": digest(curriculum),
            "curriculum_audit": digest(audit),
            "stage_gates": gates["sha256"],
            "v13_frontier_analysis": frontier_analysis["sha256"],
        },
        "gpu_budget": {
            "maximum_usd": 15.0,
            "target_hardware": "4xL40S",
            "assumed_hourly_usd": 1.52,
            "maximum_wall_hours": 9.0,
            "auto_stop": "at failed stage gate, hard budget, or completed final artifact sync",
            "never_rent_before_cpu_and_zero_update_preflight_pass": True,
        },
        "required_preflight": [
            "all CPU tests and schedule audits pass",
            "multi-turn supervisor exact replay and parity pass",
            "ordinary pass@4 has nonzero positive and negative signal in every policy slot",
            "all services and compact mirrors are watched before update 1",
        ],
        "frozen_data_opened": False,
    }
    bundle = {**bundle_body, "sha256": digest(bundle_body)}
    return {
        "diagnosis.json": diagnosis,
        "curriculum.json": curriculum,
        "curriculum_audit.json": audit,
        "stage_gates.json": gates,
        "cpu_bundle.json": bundle,
        "v13_frontier_analysis.json": frontier_analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-curriculum", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--v13-update60-summary", type=Path, required=True)
    parser.add_argument("--v13-update80-summary", type=Path, required=True)
    parser.add_argument("--v13-update80-manifest", type=Path, required=True)
    parser.add_argument("--v13-progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "v13_curriculum": args.v13_curriculum,
        "train_manifest": args.train_manifest,
        "v13_update60_summary": args.v13_update60_summary,
        "v13_update80_summary": args.v13_update80_summary,
        "v13_update80_manifest": args.v13_update80_manifest,
        "v13_progress": args.v13_progress,
    }
    artifacts = build(
        v13_curriculum=json.loads(args.v13_curriculum.read_text(encoding="utf-8")),
        train_manifest=json.loads(args.train_manifest.read_text(encoding="utf-8")),
        update60=json.loads(args.v13_update60_summary.read_text(encoding="utf-8")),
        update80=json.loads(args.v13_update80_summary.read_text(encoding="utf-8")),
        initializer_manifest=json.loads(args.v13_update80_manifest.read_text(encoding="utf-8")),
        v13_progress=json.loads(args.v13_progress.read_text(encoding="utf-8")),
        source_hashes={name: file_sha256(path) for name, path in paths.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": artifacts["cpu_bundle.json"]["status"],
                "bundle_sha256": artifacts["cpu_bundle.json"]["sha256"],
                "schedule_sha256": artifacts["curriculum_audit.json"]["schedule_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
