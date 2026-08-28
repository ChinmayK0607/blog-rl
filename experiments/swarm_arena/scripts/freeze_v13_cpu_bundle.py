#!/usr/bin/env python3
"""Verify and freeze the complete CPU-side V13 handoff bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze(
    *,
    progress_path: Path,
    initializer_path: Path,
    selection_path: Path,
    curriculum_path: Path,
    audit_path: Path,
    screen_path: Path,
) -> dict[str, Any]:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    initializer = json.loads(initializer_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    steps = [int(update["step"]) for update in progress]
    if steps != list(range(160)):
        raise ValueError("V13 bundle requires complete contiguous V12 progress through step 159")
    if selection["admission"]["status"] != "training_only_complete":
        raise ValueError("V13 gap selection is not bound to completed training evidence")
    if selection["source"]["progress_sha256"] != file_sha256(progress_path):
        raise ValueError("V13 gap selection does not match the completed progress file")
    ready = initializer["ready"]
    if (
        int(ready["step"]) != 160
        or ready["policy_revision"] != progress[-1]["policy_revision"]
        or ready["policy_adapter_sha256"]
        != progress[-1]["policy_adapter_sha256"]
    ):
        raise ValueError("public V12-u160 initializer disagrees with final progress")
    if audit["status"] != "cpu_schedule_passed_gpu_gates_pending":
        raise ValueError("V13 curriculum audit is not CPU-passed")
    if audit["remaining_blockers"] != ["run training-only ordinary pass@4 signal screen"]:
        raise ValueError("V13 CPU bundle has an unexpected remaining blocker")
    if screen["curriculum_file_sha256"] != file_sha256(curriculum_path):
        raise ValueError("ordinary screen does not bind the V13 curriculum")
    if screen["curriculum_audit_file_sha256"] != file_sha256(audit_path):
        raise ValueError("ordinary screen does not bind the V13 audit")
    if screen["initializer_manifest_file_sha256"] != file_sha256(initializer_path):
        raise ValueError("ordinary screen does not bind the V12 initializer")
    files = {
        "v12_final_progress": progress_path,
        "v12_update160_manifest": initializer_path,
        "v13_repair_case_selection": selection_path,
        "v13_curriculum": curriculum_path,
        "v13_curriculum_audit": audit_path,
        "v13_ordinary_signal_screen": screen_path,
    }
    file_hashes = {name: file_sha256(path) for name, path in files.items()}
    body = {
        "version": "arena-rl-v13-cpu-bundle-v1",
        "status": "cpu_complete_gpu_signal_screen_pending",
        "files_sha256": file_hashes,
        "initializer": {
            "run": "rl-v12-counterfactual4b160-b9f6e7f3",
            "step": 160,
            "policy_revision": ready["policy_revision"],
            "policy_adapter_sha256": ready["policy_adapter_sha256"],
            "claim_boundary": "non-admitted continuation warm start",
        },
        "curriculum": {
            "updates": curriculum["total_updates"],
            "groups_per_update": curriculum["groups_per_update"],
            "challenge_role_quotas": curriculum["challenge_role_quotas"],
            "schedule_sha256": audit["schedule_sha256"],
        },
        "next_gpu_action": {
            "name": "ordinary_pass_at_4_signal_screen",
            "games": screen["games"],
            "optimizer_updates": 0,
            "recommended_hardware": "4xL40S",
            "estimated_wall_time": "30-60 minutes after model/server setup",
            "on_pass": "render final trainer configs and launch V13 80-update continuation",
            "on_fail": "replace low-signal ordinary seed band before any optimizer update",
        },
        "frozen_data_opened": False,
    }
    return {**body, "sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-progress", type=Path, required=True)
    parser.add_argument("--v12-initializer-manifest", type=Path, required=True)
    parser.add_argument("--v13-selection", type=Path, required=True)
    parser.add_argument("--v13-curriculum", type=Path, required=True)
    parser.add_argument("--v13-audit", type=Path, required=True)
    parser.add_argument("--v13-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        progress_path=args.v12_progress,
        initializer_path=args.v12_initializer_manifest,
        selection_path=args.v13_selection,
        curriculum_path=args.v13_curriculum,
        audit_path=args.v13_audit,
        screen_path=args.v13_screen,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
