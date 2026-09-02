#!/usr/bin/env python3
"""Freeze the CPU-complete V14.3 policy-routed continuation contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.3-cpu-bundle-v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: row for key, row in value.items() if key != "sha256"}
    if value.get("sha256") != canonical_sha256(body):
        raise ValueError(f"artifact body hash mismatch: {path}")
    return value


def build_bundle(
    *,
    base_bundle: dict[str, Any],
    assessment: dict[str, Any],
    audit: dict[str, Any],
    curriculum: dict[str, Any],
    ordinary_pool: dict[str, Any],
    stage_gates: dict[str, Any],
    artifact_paths: dict[str, Path],
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if audit.get("status") != "policy_routed_cpu_bound_runtime_freeze_pending":
        raise ValueError("V14.3 policy-routing audit is not CPU-ready")
    if assessment.get("protocol_admission_rate") != 1.0:
        raise ValueError("V14.3 cannot bypass protocol admission")
    if len(assessment.get("cases", [])) != 24:
        raise ValueError("V14.3 requires the complete 24-case V14.2 assessment")
    if assessment.get("pilot_trajectories_used_for_optimization") is not False:
        raise ValueError("V14.3 cannot optimize V14.2 pilot trajectories")
    modes = audit.get("policy_modes")
    if modes != {
        "blue-0": "expand",
        "blue-1": "consolidate",
        "blue-2": "consolidate",
        "blue-3": "discover",
    }:
        raise ValueError("V14.3 policy modes differ from the frozen pilot evidence")
    if curriculum.get("ordinary_case_pool", {}).get("sha256") != ordinary_pool["sha256"]:
        raise ValueError("V14.3 curriculum does not bind the ordinary pool")
    if curriculum.get("ordinary_policy_routing", {}).get("policy_modes") != modes:
        raise ValueError("V14.3 curriculum does not bind the policy modes")
    if curriculum.get("ordinary_policy_routing", {}).get("fresh_rollouts_only") is not True:
        raise ValueError("V14.3 requires fresh training rollouts")
    if curriculum.get("ordinary_policy_routing", {}).get(
        "zero_advantage_batch_action"
    ) != "record telemetry and continue immutable schedule":
        raise ValueError("V14.3 zero-advantage behavior changed")
    if audit.get("credit_estimator_changed") or audit.get("stage_gates_changed"):
        raise ValueError("V14.3 cannot change credit or stage gates")
    if stage_gates["sha256"] != base_bundle["pilot"]["stage_gate_sha256"]:
        raise ValueError("V14.3 stage-gate identity changed")
    if audit.get("frozen_or_development_data_used"):
        raise ValueError("V14.3 routing cannot use evaluation feedback")
    if int(curriculum.get("total_updates", -1)) != 40:
        raise ValueError("V14.3 must retain the bounded 40-update maximum")

    body = {
        "version": VERSION,
        "status": "cpu_frozen_runtime_preflight_pending",
        "initializer": base_bundle["initializer"],
        "gpu_budget": base_bundle["gpu_budget"],
        "frozen_data_opened": False,
        "routing": {
            "pilot_assessment_sha256": assessment["sha256"],
            "policy_modes": modes,
            "ordinary_case_pool_sha256": ordinary_pool["sha256"],
            "curriculum_sha256": curriculum["sha256"],
            "stage_gate_sha256": stage_gates["sha256"],
            "fresh_rollouts_only": True,
            "pilot_trajectories_used_for_optimization": False,
            "development_or_frozen_feedback": False,
        },
        "artifact_file_sha256": {
            name: file_sha256(path) for name, path in sorted(artifact_paths.items())
        },
        "code_file_sha256": {
            str(path): file_sha256(path) for path in sorted(code_paths)
        },
        "required_preflight": [
            "exact source commit is published and anonymously verified before renting",
            "HF write credential and W&B-only credential are verified before renting",
            "runtime certificate and parity pass on the exact serving and trainer stack",
            "public compact-mirror preflight passes before optimizer update 1",
            "update-0 development baseline and continuation record are valid",
            "watcher, recovery supervisor, spend cap, TTL, and exact-pod teardown are armed",
        ],
        "execution": {
            "maximum_optimizer_updates": 40,
            "stage_updates": 10,
            "stage_gate_steps": [10, 20, 30, 40],
            "selection_input": "complete preceding training stage only",
            "zero_advantage_batch_action": "telemetry_only_continue",
            "all_policy_updates_atomic": True,
            "no_retry_until_favorable": True,
            "no_seed_search": True,
        },
        "stop_contract": (
            "stage-gate rejection, hard budget, TTL, operationally unrecoverable fault, "
            "or verified completion requires compact sync and immediate exact-instance teardown"
        ),
        "optimizer_updates_authorized_after_runtime_preflight": 40,
    }
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--ordinary-pool", type=Path, required=True)
    parser.add_argument("--stage-gates", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_paths = {
        "assessment": args.assessment,
        "audit": args.audit,
        "curriculum": args.curriculum,
        "ordinary_pool": args.ordinary_pool,
        "stage_gates": args.stage_gates,
    }
    bundle = build_bundle(
        base_bundle=load_hashed(args.base_bundle),
        assessment=load_hashed(args.assessment),
        audit=load_hashed(args.audit),
        curriculum=load_hashed(args.curriculum),
        ordinary_pool=load_hashed(args.ordinary_pool),
        stage_gates=load_hashed(args.stage_gates),
        artifact_paths=artifact_paths,
        code_paths=tuple(args.code_file),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
