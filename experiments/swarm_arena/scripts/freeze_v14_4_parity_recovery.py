#!/usr/bin/env python3
"""Freeze the CPU-complete V14.4 numerical-parity recovery contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

VERSION = "arena-rl-v14.4-parity-recovery-v1"
EXPECTED_PARENT_SHA256 = (
    "2741872a8a4d9f632752c56a7f0c58537155812679427ea5c355d5806401ea32"
)
EXPECTED_MEAN_MISMATCH_KL = 0.002


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
    if "experiments" not in parts:
        raise ValueError(f"bundle path is outside the experiment tree: {path}")
    return Path(*parts[parts.index("experiments") :]).as_posix()


def _scientific_trainer_body(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"experimental", "rollout_parity_gate", "wandb"}
    }


def build_bundle(
    *,
    parent: dict[str, Any],
    prior_trainer_path: Path,
    trainer_path: Path,
    code_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if parent.get("sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("V14.4 parent must be the final frozen V14.3 CPU bundle")
    if parent.get("frozen_data_opened") is not False:
        raise ValueError("V14.4 cannot inherit a bundle that opened frozen data")
    prior_trainer = _trainer_config(prior_trainer_path)
    trainer = _trainer_config(trainer_path)
    if _scientific_trainer_body(trainer) != _scientific_trainer_body(prior_trainer):
        raise ValueError("V14.4 changed trainer science outside the parity recovery")
    parity_gate = trainer.get("rollout_parity_gate")
    if parity_gate != {
        "max_mean_logprob_error": 0.25,
        "probability_tail_threshold": 0.05,
        "max_mean_mismatch_kl": EXPECTED_MEAN_MISMATCH_KL,
    }:
        raise ValueError("V14.4 trainer parity gate differs from the frozen recovery")
    if trainer.get("experimental", {}).get("token_export") != {}:
        raise ValueError("V14.4 must enable local token export for failure diagnosis")
    if trainer.get("optim") != prior_trainer.get("optim"):
        raise ValueError("V14.4 cannot change optimizer settings")
    if trainer.get("loss") != prior_trainer.get("loss"):
        raise ValueError("V14.4 cannot change loss or DPPO masking")

    body = {
        "version": VERSION,
        "status": "cpu_frozen_runtime_preflight_pending",
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
            "token_export": "enabled_local_only_not_mirrored",
            "optimization_dtype_changed": False,
            "reduce_dtype_changed": False,
        },
        "runtime_certificate": {
            "probe_samples": 32,
            "samples_per_policy": 8,
            "required_policy_ids": [f"blue-{index}" for index in range(4)],
            "pooled_and_every_policy_must_pass": True,
            "thresholds_derived_from_resolved_trainer_config": True,
            "conflicting_cli_thresholds_rejected": True,
            "threshold_body_retained_and_rehashed_by_preflight": True,
            "fresh_certificate_required": True,
        },
        "execution": {
            **parent["execution"],
            "first_spend_decision_update": 10,
            "resume_rejected_v14_3_run": False,
            "restart_from_parent_initializer": "V13 update 80",
        },
        "gpu_budget": parent["gpu_budget"],
        "required_preflight": [
            "publish and anonymously verify the exact source and this bundle before renting",
            "resolve the trainer config and bind four distinct parent policy adapters",
            "capture exactly eight parity samples for each policy across all three servers",
            "pass pooled and all four policy-local parity gates at the trainer-declared thresholds",
            "bind and rehash the complete threshold body in the runtime certificate",
            "pass a compact public-mirror write/read/hash preflight before optimizer update 1",
            "complete the unchanged update-0 baseline before optimizer update 1",
            "arm progress watcher, pulse recovery, budget/TTL, and immediate exact-pod teardown",
        ],
        "preventive_fixes": [
            "no hidden certificate threshold default when a trainer config is supplied",
            "no pooled-only certificate for a policy-local live gate",
            "no missing or unbalanced policy slot in the 32-sample runtime probe",
            "no certificate whose threshold body differs from its recorded gate hash",
            "no reuse of rejected V14.3 progress under a changed runtime contract",
            "local token evidence retained for any future parity rejection",
        ],
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
    parser.add_argument("--code-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(
        parent=load_hashed(args.parent_bundle),
        prior_trainer_path=args.prior_trainer_config,
        trainer_path=args.trainer_config,
        code_paths=tuple(args.code_file),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
