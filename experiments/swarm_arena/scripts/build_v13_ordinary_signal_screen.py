#!/usr/bin/env python3
"""Build the hash-bound ordinary pass@4 preflight screen for V13."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    CurriculumStage,
    exact_staged_curriculum_schedule,
)

VERSION = "arena-rl-v13-ordinary-signal-screen-v1"
POLICIES = ("blue-0", "blue-1", "blue-2", "blue-3")
OPPONENTS = ("base", "sft", "historical", "current")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_screen_manifest(
    curriculum: dict[str, Any],
    curriculum_audit: dict[str, Any],
    initializer_manifest: dict[str, Any],
    *,
    curriculum_file_sha256: str,
    curriculum_audit_file_sha256: str,
    initializer_manifest_file_sha256: str,
) -> dict[str, Any]:
    if curriculum_audit["status"] != "cpu_schedule_passed_gpu_gates_pending":
        raise ValueError("ordinary screen requires a completed-run V13 CPU schedule")
    schedule = exact_staged_curriculum_schedule(
        tuple(stage(row) for row in curriculum["stages"]),
        groups_per_update=int(curriculum["groups_per_update"]),
        pair_offset=int(curriculum["schedule"]["pair_offset"]),
        ordinary_seed_base=int(curriculum["schedule"]["ordinary_seed_base"]),
        shuffle_seed=int(curriculum["schedule"]["shuffle_seed"]),
    )
    selected = []
    policy_counts = Counter()
    for assignment in schedule:
        if assignment.kind != "ordinary":
            continue
        focused_agent = f"blue-{assignment.ordinal % curriculum['groups_per_update']}"
        if policy_counts[focused_agent] >= 16:
            continue
        within_policy = policy_counts[focused_agent]
        selected.append(
            {
                "case_id": f"ordinary-screen-{focused_agent}-{within_policy:02d}",
                "schedule_ordinal": assignment.ordinal,
                "seed": assignment.ordinary_seed,
                "size": assignment.ordinary_size,
                "horizon": assignment.ordinary_horizon,
                "stage": assignment.stage,
                "focused_agent": focused_agent,
                "opponent_family": OPPONENTS[within_policy % len(OPPONENTS)],
                "replicas": 4,
            }
        )
        policy_counts[focused_agent] += 1
        if all(policy_counts[policy] == 16 for policy in POLICIES):
            break
    if policy_counts != Counter({policy: 16 for policy in POLICIES}):
        raise ValueError(f"V13 schedule lacks balanced ordinary screen cases: {policy_counts}")
    if len({row["seed"] for row in selected}) != len(selected):
        raise ValueError("ordinary screen seeds must be unique")
    ready = initializer_manifest["ready"]
    body = {
        "version": VERSION,
        "scope": "V13 training-only preflight; development and frozen data unopened",
        "curriculum_file_sha256": curriculum_file_sha256,
        "curriculum_audit_file_sha256": curriculum_audit_file_sha256,
        "initializer_manifest_file_sha256": initializer_manifest_file_sha256,
        "initializer": {
            "source_run": "rl-v12-counterfactual4b160-b9f6e7f3",
            "step": int(ready["step"]),
            "policy_revision": ready["policy_revision"],
            "policy_adapter_sha256": ready["policy_adapter_sha256"],
            "admission": (
                "hash-verified non-admitted V12 continuation warm start; "
                "does not retroactively select V12"
            ),
        },
        "cases": selected,
        "case_count": len(selected),
        "games": len(selected) * 4,
        "policy_case_counts": dict(sorted(policy_counts.items())),
        "opponent_case_counts": dict(
            sorted(Counter(row["opponent_family"] for row in selected).items())
        ),
        "thresholds": {
            "protocol_admission_rate": 1.0,
            "minimum_variable_return_group_rate_per_policy": 0.25,
            "minimum_nonzero_advantage_rate_per_policy": 0.25,
            "minimum_mean_focused_action_diversity_per_policy": 1.25,
            "minimum_positive_advantages_per_policy": 4,
            "minimum_negative_advantages_per_policy": 4,
            "minimum_variable_groups_per_policy_opponent_family": 1,
        },
        "runner_contract": {
            "script": "experiments/swarm_arena/scripts/run_live_rl.py",
            "one_case_per_invocation": True,
            "required_flags": [
                "--rollout-only",
                "--scenario-source ordinary",
                "--groups-per-step 1",
                "--steps 1",
                "--shared-return-replicas 4",
                "--shared-return-credit-assignment focused_agent",
                "--shared-return-trainable-phase ACT",
                "--ordinary-focused-agent <case.focused_agent>",
            ],
            "output": "<screen-root>/<case-id>/live_rl_diagnostic.json",
            "optimizer_updates": 0,
        },
    }
    return {**body, "sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--curriculum-audit", type=Path, required=True)
    parser.add_argument("--initializer-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curriculum = json.loads(args.curriculum.read_text(encoding="utf-8"))
    audit = json.loads(args.curriculum_audit.read_text(encoding="utf-8"))
    initializer = json.loads(args.initializer_manifest.read_text(encoding="utf-8"))
    manifest = build_screen_manifest(
        curriculum,
        audit,
        initializer,
        curriculum_file_sha256=file_sha256(args.curriculum),
        curriculum_audit_file_sha256=file_sha256(args.curriculum_audit),
        initializer_manifest_file_sha256=file_sha256(args.initializer_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
