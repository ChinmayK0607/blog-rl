#!/usr/bin/env python3
"""Freeze the credential-free V14.1 CPU launch contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    audit: dict[str, Any],
    curriculum: dict[str, Any],
    pool: dict[str, Any],
    repair_screen: dict[str, Any],
    original_screen: dict[str, Any],
    stage_gates: dict[str, Any],
    artifact_paths: dict[str, Path],
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if audit["status"] != "cpu_repair_passed_gpu_screen_pending":
        raise ValueError("V14.1 repair audit is not launch-pending")
    if not audit["thresholds_unchanged"]:
        raise ValueError("V14.1 cannot freeze with changed admission thresholds")
    if repair_screen["thresholds"] != original_screen["thresholds"]:
        raise ValueError("repair screen changed the original admission thresholds")
    if repair_screen["case_count"] != 32 or repair_screen["games"] != 128:
        raise ValueError("repair screen must remain exactly 32 cases / 128 games")
    if curriculum["ordinary_case_pool"]["sha256"] != pool["sha256"]:
        raise ValueError("curriculum does not embed the exact ordinary case pool")
    if (
        curriculum["ordinary_frontier_repair"]["screen_manifest_sha256"]
        != repair_screen["sha256"]
    ):
        raise ValueError("curriculum does not bind the repair screen")
    if audit["ordinary_case_pool_sha256"] != pool["sha256"]:
        raise ValueError("repair audit does not bind the ordinary case pool")
    if audit["screen_manifest_sha256"] != repair_screen["sha256"]:
        raise ValueError("repair audit does not bind the repair screen")
    if audit["repaired_curriculum_sha256"] != curriculum["sha256"]:
        raise ValueError("repair audit does not bind the repaired curriculum")
    if base_bundle["frozen_data_opened"] or curriculum["frozen_data_opened"]:
        raise ValueError("V14.1 CPU freeze cannot use frozen evaluation data")

    body = {
        "version": "arena-rl-v14.1-cpu-bundle-v1",
        "status": "cpu_frozen_source_publication_and_gpu_screen_pending",
        "initializer": base_bundle["initializer"],
        "gpu_budget": base_bundle["gpu_budget"],
        "frozen_data_opened": False,
        "repair": {
            "failed_assessment_sha256": audit["source_hashes"]["assessment"],
            "ordinary_case_pool_sha256": pool["sha256"],
            "ordinary_case_pool_cases": pool["case_count"],
            "screen_manifest_sha256": repair_screen["sha256"],
            "screen_cases": repair_screen["case_count"],
            "screen_games": repair_screen["games"],
            "curriculum_sha256": curriculum["sha256"],
            "stage_gate_sha256": stage_gates["sha256"],
            "thresholds_unchanged": True,
        },
        "artifact_file_sha256": {
            name: file_sha256(path) for name, path in sorted(artifact_paths.items())
        },
        "code_file_sha256": {
            str(path): file_sha256(path) for path in sorted(code_paths)
        },
        "required_preflight": [
            "exact source commit is published and anonymously verified",
            "HF write credential availability is verified before renting",
            "runtime certificate and parity pass on the exact serving stack",
            "32-case / 128-game ordinary frontier screen passes every unchanged gate",
            "HF compact mirror, W&B, watcher, recovery, spend, and exact-pod teardown are armed before update 1",
        ],
        "stop_contract": (
            "screen rejection, stage-gate rejection, hard budget, or verified completion "
            "requires evidence sync and immediate exact-pod teardown"
        ),
        "optimizer_updates_authorized": 0,
    }
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--repair-screen", type=Path, required=True)
    parser.add_argument("--original-screen", type=Path, required=True)
    parser.add_argument("--stage-gates", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--admission-limits", type=Path, required=True)
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_paths = {
        "audit": args.audit,
        "curriculum": args.curriculum,
        "ordinary_case_pool": args.pool,
        "ordinary_frontier_screen": args.repair_screen,
        "stage_gates": args.stage_gates,
        "trainer_config": args.trainer_config,
        "base_plan": args.base_plan,
        "admission_limits": args.admission_limits,
        "handoff_manifest": args.handoff_manifest,
    }
    bundle = build_bundle(
        base_bundle=load_hashed(args.base_bundle),
        audit=load_hashed(args.audit),
        curriculum=load_hashed(args.curriculum),
        pool=load_hashed(args.pool),
        repair_screen=load_hashed(args.repair_screen),
        original_screen=load_hashed(args.original_screen),
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
