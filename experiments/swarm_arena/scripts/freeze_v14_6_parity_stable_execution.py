#!/usr/bin/env python3
"""Freeze V14.6's parity-stable serving and certification contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.6-parity-stable-execution-v1"
EXPECTED_PARENT_SHA256 = (
    "8631449b372fafb1e3affbfbedf55b4996853468281b46d362651f6e24b55586"
)
EXPECTED_TRAINER_SHA256 = (
    "d6e257658468c6e49c29ebdda3a77b987047b9f0592b7daa7274ec55fdd41115"
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


def _stable_path(path: Path) -> str:
    parts = path.resolve().parts
    for root in ("experiments", "packages", "src", "skills"):
        if root in parts:
            return Path(*parts[parts.index(root) :]).as_posix()
    raise ValueError(f"bundle path is outside the tracked execution tree: {path}")


def _strict_inference_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    model = config.get("model", {})
    extra = config.get("vllm_extra", {})
    if model.get("enforce_eager") is not True:
        raise ValueError("V14.6 requires eager vLLM execution")
    if extra.get("generation_config") != "vllm":
        raise ValueError("V14.6 requires the neutral vLLM generation config")
    if extra.get("async_scheduling") is not False:
        raise ValueError("V14.6 requires synchronous vLLM scheduling")
    if extra.get("max_num_seqs") != 4:
        raise ValueError("V14.6 requires a four-sequence serving cap")
    ir_priority = extra.get("ir_op_priority", {})
    if ir_priority != {
        "rms_norm": ["native"],
        "fused_add_rms_norm": ["native"],
    }:
        raise ValueError("V14.6 requires native RMSNorm serving kernels")
    return config


def build_bundle(
    *,
    parent: dict[str, Any],
    trainer_path: Path,
    inference_path: Path,
    plan_path: Path,
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if parent.get("sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("V14.6 parent must be the frozen V14.5 CPU bundle")
    if parent.get("frozen_data_opened") is not False:
        raise ValueError("V14.6 cannot inherit a bundle that opened frozen data")
    if file_sha256(trainer_path) != EXPECTED_TRAINER_SHA256:
        raise ValueError("V14.6 changed the frozen V14.5 trainer config")
    _strict_inference_config(inference_path)

    body = {
        "version": VERSION,
        "status": "cpu_validated",
        "parent_cpu_bundle_sha256": parent["sha256"],
        "initializer": parent["initializer"],
        "routing": parent["routing"],
        "artifact_file_sha256": parent["artifact_file_sha256"],
        "frozen_data_opened": False,
        "trainer": parent["trainer"],
        "execution": {
            **parent["execution"],
            "serving_profile": "parity_stable_l40s_v1",
            "runtime_probe_samples": 128,
            "runtime_probe_concurrency_per_server": 4,
            "runtime_probe_samples_per_policy": 32,
            "runtime_probe_uses_production_like_batching": True,
        },
        "inference": {
            "config": _stable_path(inference_path),
            "config_file_sha256": file_sha256(inference_path),
            "enforce_eager": True,
            "generation_config": "vllm",
            "async_scheduling": False,
            "max_num_seqs": 4,
            "native_rms_norm_kernels": True,
        },
        "launch_contract": parent["launch_contract"],
        "gpu_budget": {
            **parent["gpu_budget"],
            "maximum_usd": 60.0,
            "target_hardware": "single 4x or 8x NVIDIA L40/L40S",
        },
        "required_preflight": [
            "publish and anonymously verify the exact V14.6 source and bundle",
            "bind the strict inference config to the exact runtime certificate",
            "capture 128 predetermined parity decisions at concurrency four",
            "pass pooled and policy-local parity under unchanged thresholds",
            "complete the unchanged 192-row update-zero evaluation",
            "arm exact-pod recovery, idle, terminal, budget, and teardown supervisors",
        ],
        "preventive_fixes": [
            "eager vLLM execution",
            "synchronous scheduling",
            "neutral vLLM generation config",
            "native RMSNorm serving kernels",
            "four-sequence serving cap",
            "production-like concurrent runtime parity certification",
        ],
        "v14_5_observed_failure": {
            "update_zero_rows": 192,
            "update_zero_passed": True,
            "logical_updates_attempted": 4,
            "durable_optimizer_steps": 2,
            "quarantined_logical_updates": [1],
            "first_failed_mean_mismatch_kl": 0.011002085,
            "second_failed_logical_update": 4,
            "second_failed_mean_mismatch_kl": 0.003157224,
            "threshold": 0.002,
            "stage_gate_evaluated": False,
            "final_pod_cost_usd": 5.36,
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
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(
        parent=load_hashed(args.parent_bundle),
        trainer_path=args.trainer_config,
        inference_path=args.inference_config,
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
