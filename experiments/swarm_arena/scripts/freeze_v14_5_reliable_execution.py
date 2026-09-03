#!/usr/bin/env python3
"""Freeze V14.5's CPU-complete, bounded execution-recovery contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.5-reliable-execution-v2-topology-profile"
EXPECTED_PARENT_SHA256 = (
    "ef4c9c614856edbf23b525724e3cc9524a8fe749e6e7e5fc2e6f4e6dd887aef3"
)


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


def _trainer_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _stable_path(path: Path) -> str:
    parts = path.parts
    for root in ("experiments", "packages", "src", "skills"):
        if root in parts:
            return Path(*parts[parts.index(root) :]).as_posix()
    raise ValueError(f"bundle path is outside the tracked execution tree: {path}")


def _scientific_trainer_body(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"rollout_parity_recovery", "wandb"}
    }


def build_bundle(
    *,
    parent: dict[str, Any],
    prior_trainer_path: Path,
    trainer_path: Path,
    plan_path: Path,
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if parent.get("sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("V14.5 parent must be the frozen V14.4 CPU bundle")
    if parent.get("frozen_data_opened") is not False:
        raise ValueError("V14.5 cannot inherit a bundle that opened frozen data")
    prior_trainer = _trainer_config(prior_trainer_path)
    trainer = _trainer_config(trainer_path)
    if _scientific_trainer_body(trainer) != _scientific_trainer_body(
        prior_trainer
    ):
        raise ValueError("V14.5 changed trainer science outside execution recovery")
    parity_gate = trainer.get("rollout_parity_gate")
    if parity_gate != parent["trainer"]["parity_thresholds"]:
        raise ValueError("V14.5 changed the frozen V14.4 numerical thresholds")
    recovery = trainer.get("rollout_parity_recovery")
    if recovery != {
        "action": "quarantine_logical_update",
        "window_size": 10,
        "max_quarantined_updates_per_window": 1,
    }:
        raise ValueError("V14.5 parity recovery differs from the bounded contract")

    body = {
        "version": VERSION,
        "status": "cpu_validated",
        "parent_cpu_bundle_sha256": parent["sha256"],
        "initializer": parent["initializer"],
        "routing": parent["routing"],
        "artifact_file_sha256": parent["artifact_file_sha256"],
        "frozen_data_opened": False,
        "trainer": {
            "config": _stable_path(trainer_path),
            "config_file_sha256": file_sha256(trainer_path),
            "prior_config_file_sha256": file_sha256(prior_trainer_path),
            "unchanged_scientific_body_sha256": canonical_sha256(
                _scientific_trainer_body(trainer)
            ),
            "parity_thresholds": parity_gate,
            "parity_recovery": recovery,
            "optimization_dtype_changed": False,
            "reduce_dtype_changed": False,
        },
        "execution": {
            **parent["execution"],
            "logical_update_reporting_separate_from_optimizer_steps": True,
            "parity_failure_optimizer_action": "discard_complete_atomic_update",
            "parity_failure_scheduler_action": "do_not_step",
            "parity_failure_sampling_action": "advance_without_replacement",
            "maximum_parity_quarantines_per_ten_update_stage": 1,
            "second_parity_failure_in_stage": "abort_before_optimizer_step",
            "resume_restores_append_only_quarantine_counts": True,
        },
        "launch_contract": {
            "base_repo": "Qwen/Qwen3-4B-Instruct-2507",
            "base_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "uv_argv": ["run", "--frozen", "--extra", "flash-attn"],
            "public_base_and_adapter_repo_ids_required": True,
            "orchestrator_lora_schema": "student.model.lora",
            "cpu_rehearsal_before_rental": True,
            "four_gpu_default": {
                "trainer_gpu_ids": [0],
                "inference_gpu_ids": [1, 2, 3],
            },
            "eight_gpu_initial_profile": {
                "trainer_gpu_ids": [0, 1],
                "inference_gpu_ids": [2, 3, 4, 5, 6, 7],
                "rollout_ports": [8001, 8002, 8003, 8004, 8005, 8006],
            },
            "runtime_topology_must_assign_every_gpu_exactly_once": True,
            "runtime_certificate_regenerated_for_exact_topology": True,
            "topology_profile_inputs": "operational_timings_only",
            "topology_profile_minimum_updates": 3,
        },
        "gpu_budget": {
            **parent["gpu_budget"],
            "assumed_hourly_usd": 6.0,
            "maximum_usd": 60.0,
            "target_hardware": "8xL40S",
        },
        "required_preflight": [
            "complete all V14.5 CPU validation before rental",
            "publish and anonymously verify the exact source and bundle",
            "initialize every pinned source submodule before frozen uv validation",
            "render and syntax-check the exact staged launcher",
            "resolve the trainer config and bind four distinct parent adapters",
            "pass pooled and policy-local runtime parity calibration",
            "pass compact public HF and W&B authentication preflights",
            "complete unchanged update-zero evaluation before optimizer work",
            "assign every GPU exactly once and isolate every inference server",
            "arm watcher, recovery supervisor, budget, TTL, and exact teardown",
        ],
        "preventive_fixes": [
            "current student.model.lora preflight schema",
            "one frozen flash-attn uv runtime for every staged subprocess",
            "exact public base and adapter repository binding",
            "bounded no-resampling parity quarantine",
            "truthful logical-versus-optimizer update accounting",
            "restart-safe append-only parity allowance",
            "compact off-node parity decision mirroring",
            "topology-bound multi-rank trainer and dynamic rollout-server launch",
            "reward-blind durable per-update runtime profiling",
        ],
        "v14_4_observed_failure": {
            "durable_logical_updates": 2,
            "durable_optimizer_steps": 2,
            "rejected_pending_policy": "blue-2",
            "rejected_mean_mismatch_kl": 0.004621265,
            "threshold": 0.002,
            "curriculum_stage_evaluated": False,
            "final_pod_cost_usd": 4.42,
            "pod_decommissioned": True,
        },
        "plan": {
            "path": _stable_path(plan_path),
            "sha256": file_sha256(plan_path),
        },
        "code_file_sha256": {
            _stable_path(path): file_sha256(path) for path in sorted(code_paths)
        },
        "stop_contract": parent["stop_contract"],
    }
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--prior-trainer-config", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(
        parent=load_hashed(args.parent_bundle),
        prior_trainer_path=args.prior_trainer_config,
        trainer_path=args.trainer_config,
        plan_path=args.plan,
        code_paths=tuple(args.code_file),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
