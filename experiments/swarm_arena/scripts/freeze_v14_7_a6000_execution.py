#!/usr/bin/env python3
"""Freeze V14.7's four-A6000 execution contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.7-a6000-execution-v1"
EXPECTED_PARENT_SHA256 = (
    "5dd516d131ee1459d4d9f3007b96cf18e82ca83106f7195a88c60780f6697cea"
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


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_bundle(
    *,
    parent: dict[str, Any],
    trainer_path: Path,
    parent_inference_path: Path,
    inference_path: Path,
    plan_path: Path,
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if parent.get("sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("V14.7 parent must be the frozen V14.6 CPU bundle")
    if parent.get("frozen_data_opened") is not False:
        raise ValueError("V14.7 cannot inherit a bundle that opened frozen data")
    if file_sha256(trainer_path) != EXPECTED_TRAINER_SHA256:
        raise ValueError("V14.7 changed the frozen trainer config")
    if _load_toml(inference_path) != _load_toml(parent_inference_path):
        raise ValueError("V14.7 changed the V14.6 strict inference settings")
    launch_contract = {
        key: value
        for key, value in parent["launch_contract"].items()
        if key != "eight_gpu_initial_profile"
    }

    body = {
        **{key: value for key, value in parent.items() if key != "sha256"},
        "version": VERSION,
        "parent_cpu_bundle_sha256": parent["sha256"],
        "frozen_data_opened": False,
        "execution": {
            **parent["execution"],
            "serving_profile": "parity_stable_a6000_v1",
        },
        "inference": {
            **parent["inference"],
            "config": _stable_path(inference_path),
            "config_file_sha256": file_sha256(inference_path),
        },
        "launch_contract": {
            **launch_contract,
            "allowed_topology": "four_gpu_default",
            "required_gpu_name": "NVIDIA RTX A6000",
            "required_gpu_count": 4,
            "minimum_vram_gb_per_gpu": 48,
        },
        "gpu_budget": {
            **parent["gpu_budget"],
            "assumed_hourly_usd": 1.68,
            "target_hardware": "single 4x NVIDIA RTX A6000 48 GB",
        },
        "required_preflight": [
            "publish and anonymously verify the exact V14.7 source and bundle",
            "verify exactly four NVIDIA RTX A6000 GPUs with at least 48 GB each",
            "bind the strict inference config to the exact runtime certificate",
            "capture 128 predetermined parity decisions at concurrency four",
            "pass pooled and policy-local parity under unchanged thresholds",
            "complete the unchanged 192-row update-zero evaluation",
            "arm exact-pod recovery, idle, terminal, budget, and teardown supervisors",
        ],
        "hardware_rebind": {
            "source": "V14.6 parity-stable execution",
            "only_declared_change": "target hardware L40/L40S to RTX A6000",
            "scientific_settings_changed": False,
            "strict_inference_settings_changed": False,
        },
        "plan": {
            "path": _stable_path(plan_path),
            "sha256": file_sha256(plan_path),
        },
        "code_file_sha256": {
            _stable_path(path): file_sha256(path) for path in sorted(code_paths)
        },
    }
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--parent-inference-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(
        parent=load_hashed(args.parent_bundle),
        trainer_path=args.trainer_config,
        parent_inference_path=args.parent_inference_config,
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
