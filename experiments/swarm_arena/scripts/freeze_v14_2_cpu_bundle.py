#!/usr/bin/env python3
"""Freeze the credential-free, zero-update V14.2 pilot contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.2-cpu-bundle-v1"


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
    diagnosis: dict[str, Any],
    audit: dict[str, Any],
    curriculum: dict[str, Any],
    source_pool: dict[str, Any],
    pilot_manifest: dict[str, Any],
    stage_gates: dict[str, Any],
    artifact_paths: dict[str, Path],
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if audit.get("status") != "cpu_design_passed_gpu_pilot_pending":
        raise ValueError("V14.2 design audit is not launch-pending")
    if diagnosis.get("credit_estimator_decision", {}).get("change") is not False:
        raise ValueError("V14.2 CPU freeze cannot silently change credit assignment")
    if diagnosis.get("optimizer_updates_used") != 0:
        raise ValueError("V14.2 diagnosis must precede optimization")
    if pilot_manifest.get("case_count") != 24 or pilot_manifest.get("games") != 96:
        raise ValueError("V14.2 pilot must remain 24 cases / 96 games")
    if pilot_manifest.get("thresholds", {}).get("current_probe_is_blocking") is not False:
        raise ValueError("V14.2 current-policy probes must remain diagnostic")
    if pilot_manifest.get("pilot_data_policy") != (
        "pilot trajectories select scenario identities only and are discarded before "
        "optimizer update 1; training uses fresh sampling namespaces"
    ):
        raise ValueError("V14.2 pilot data-discard contract changed")
    repair = curriculum.get("ordinary_frontier_stability_repair", {})
    if repair.get("diagnosis_sha256") != diagnosis["sha256"]:
        raise ValueError("V14.2 curriculum does not bind the diagnosis")
    if repair.get("pilot_screen_manifest_sha256") != pilot_manifest["sha256"]:
        raise ValueError("V14.2 curriculum does not bind the pilot manifest")
    if audit.get("pilot_screen_manifest_sha256") != pilot_manifest["sha256"]:
        raise ValueError("V14.2 audit does not bind the pilot manifest")
    if audit.get("repaired_curriculum_sha256") != curriculum["sha256"]:
        raise ValueError("V14.2 audit does not bind the curriculum template")
    if audit.get("credit_estimator_changed") or audit.get("stage_gates_changed"):
        raise ValueError("V14.2 CPU freeze cannot change credit or stage gates")
    if base_bundle.get("frozen_data_opened") or curriculum.get("frozen_data_opened"):
        raise ValueError("V14.2 CPU freeze cannot use frozen evaluation data")

    body = {
        "version": VERSION,
        "status": "cpu_frozen_source_publication_and_gpu_pilot_pending",
        "initializer": base_bundle["initializer"],
        "gpu_budget": base_bundle["gpu_budget"],
        "frozen_data_opened": False,
        "pilot": {
            "diagnosis_sha256": diagnosis["sha256"],
            "source_ordinary_pool_sha256": source_pool["sha256"],
            "screen_manifest_sha256": pilot_manifest["sha256"],
            "cases": pilot_manifest["case_count"],
            "games": pilot_manifest["games"],
            "curriculum_template_sha256": curriculum["sha256"],
            "stage_gate_sha256": stage_gates["sha256"],
            "trajectories_used_for_optimization": False,
            "current_probe_is_blocking": False,
        },
        "artifact_file_sha256": {
            name: file_sha256(path) for name, path in sorted(artifact_paths.items())
        },
        "code_file_sha256": {
            str(path): file_sha256(path) for path in sorted(code_paths)
        },
        "required_preflight": [
            "exact source commit is published and anonymously verified",
            "HF write credential and W&B-only credential availability are verified before renting",
            "runtime certificate and parity pass on the exact serving stack",
            "24-case / 96-game stable-frontier pilot completes without retries or optimizer work",
            (
                "committed assessor passes the aggregate non-current gates while "
                "treating current probes as diagnostic"
            ),
            (
                "committed finalizer binds the exact pilot assessment to a 24-case "
                "ordinary pool and fresh sampling namespaces"
            ),
            (
                "HF compact mirror, watcher, recovery, spend, and exact-pod teardown "
                "are armed before GPU setup"
            ),
        ],
        "fail_closed_transition": (
            "optimizer update 1 is prohibited until the pilot passes, the exact assessment "
            "and pilot-bound curriculum are durably mirrored, and the production plan is "
            "rebuilt from that curriculum"
        ),
        "stop_contract": (
            "pilot rejection, stage-gate rejection, hard budget, or verified completion "
            "requires evidence sync and immediate exact-instance teardown"
        ),
        "optimizer_updates_authorized": 0,
    }
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--stage-gates", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_paths = {
        "audit": args.audit,
        "curriculum": args.curriculum,
        "diagnosis": args.diagnosis,
        "pilot_manifest": args.pilot_manifest,
        "source_pool": args.source_pool,
        "stage_gates": args.stage_gates,
    }
    bundle = build_bundle(
        base_bundle=load_hashed(args.base_bundle),
        diagnosis=load_hashed(args.diagnosis),
        audit=load_hashed(args.audit),
        curriculum=load_hashed(args.curriculum),
        source_pool=load_hashed(args.source_pool),
        pilot_manifest=load_hashed(args.pilot_manifest),
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
