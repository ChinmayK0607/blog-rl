"""Bind measurement repairs without rewriting any historical CPU bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_bundle(repo: Path) -> dict:
    arena = repo / "experiments/swarm_arena"
    parent = json.loads((arena / "data/rl_v14_7/cpu_bundle.json").read_text())
    parent_sha = parent.pop("sha256")
    if parent_sha != "b3dafe8339cc68180707f772e39b3ed6e72d9690530d49034ebfac831f819fa7" or digest(parent) != parent_sha:
        raise ValueError("V14.7 parent identity mismatch")
    body = copy.deepcopy(parent)
    for name in ("trainer", "inference"):
        if hashlib.sha256((repo / body[name]["config"]).read_bytes()).hexdigest() != body[name]["config_file_sha256"]:
            raise ValueError(f"frozen {name} configuration changed")
    body["version"] = "arena-rl-v14.8-measurement-repair-v1"
    body["status"] = "cpu_bound_pending_exact_host_certification"
    body["parent_cpu_bundle_sha256"] = parent_sha
    body.pop("hardware_rebind", None)
    body["measurement_repair"] = {
        "evaluation_identity": "actual-initializer-v1",
        "historical_evaluation_resume_allowed": False,
        "sft_reference_from_frozen_snapshot": True,
        "served_adapter_hash_verification": True,
        "paired_initializer_comparison": True,
        "scientific_training_settings_changed": False,
        "semantic_probe_in_staged_schedule": False,
    }
    body["gpu_budget"]["maximum_wall_hours"] = 30
    body["gpu_budget"]["maximum_rental_usd_at_declared_rate"] = 50.40
    body["gpu_budget"]["timing_basis"] = "historical_evaluation_rate_plus_unmeasured_update_reservation"
    body["gpu_budget"].pop("never_rent_before_cpu_and_zero_update_preflight_pass", None)
    body["gpu_budget"]["never_optimize_before_runtime_and_zero_update_preflight_pass"] = True
    body["required_preflight"][0] = "publish and independently verify exact V14.8 source and CPU bundle"
    body["required_preflight"] += [
        "admit explicitly labeled extended-time reservation against actual provider deadline",
        "HF and W&B compact mirror preflights before update-zero evaluation",
        "actual initializer aliases and SFT snapshot identity verification",
    ]
    plan = "experiments/swarm_arena/V14_8_MEASUREMENT_REPAIR_PLAN.md"
    body["plan"] = {"path": plan, "sha256": hashlib.sha256((repo / plan).read_bytes()).hexdigest()}
    paths = set(parent["code_file_sha256"]) | {
        "experiments/swarm_arena/scripts/" + name for name in (
            "freeze_v14_8_measurement_repair.py", "preflight_staged_budget.py",
            "supervise_staged_role.py", "run_progress_eval_v4.py", "run_staged_pulses.py",
            "run_live_artifact_mirror.py", "log_live_rl_wandb.py",
        )
    } | {"experiments/swarm_arena/swarm_ctf_eval/evaluation_contract.py"}
    body["code_file_sha256"] = {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in sorted(paths)}
    body["sha256"] = digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = build_bundle(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"body_sha256": body["sha256"], "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
