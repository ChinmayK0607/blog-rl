from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.swarm_arena.scripts.bind_runtime_certificate import (
    _validated_per_policy_parity,
)
from experiments.swarm_arena.scripts.freeze_v14_6_parity_stable_execution import (
    _strict_inference_config,
    build_bundle,
    load_hashed,
)
from experiments.swarm_arena.scripts.freeze_v14_7_a6000_execution import (
    build_bundle as build_a6000_bundle,
)
from experiments.swarm_arena.scripts.preflight_staged_rl import (
    _validate_strict_parity_probe,
)


def _parity_fixture(samples_per_policy: int = 32) -> tuple[dict, dict]:
    samples = []
    per_policy = {}
    for policy_slot in range(4):
        for sample_index in range(samples_per_policy):
            samples.append(
                {
                    "policy_slot": policy_slot,
                    "completion_ids": [policy_slot, sample_index],
                }
            )
        per_policy[f"blue-{policy_slot}"] = {
            "policy_slot": policy_slot,
            "samples": samples_per_policy,
            "completion_tokens": samples_per_policy * 2,
            "parity_passed": True,
        }
    return {"per_policy_parity": per_policy}, {"samples": samples}


def test_runtime_certificate_accepts_balanced_128_sample_probe() -> None:
    parity, probe = _parity_fixture()
    assert _validated_per_policy_parity(parity, probe) == parity["per_policy_parity"]


def test_runtime_certificate_rejects_policy_imbalanced_probe() -> None:
    parity, probe = _parity_fixture()
    probe["samples"][-1]["policy_slot"] = 2
    with pytest.raises(ValueError, match="equal samples"):
        _validated_per_policy_parity(parity, probe)


def test_strict_inference_profile_fails_closed(tmp_path: Path) -> None:
    arena_root = Path(__file__).parents[1]
    strict_path = arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
    assert _strict_inference_config(strict_path)["vllm_extra"]["max_num_seqs"] == 4
    changed = strict_path.read_text(encoding="utf-8").replace(
        "max_num_seqs = 4", "max_num_seqs = 8"
    )
    changed_path = tmp_path / "changed.toml"
    changed_path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="four-sequence"):
        _strict_inference_config(changed_path)


def test_strict_preflight_requires_production_like_probe() -> None:
    arena_root = Path(__file__).parents[1]
    inference = _strict_inference_config(
        arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
    )
    certificate = {
        "parity_report": {
            "samples": 128,
            "capture_concurrency_per_server": 4,
        }
    }
    _validate_strict_parity_probe(inference, certificate)
    certificate["parity_report"]["capture_concurrency_per_server"] = 1
    with pytest.raises(ValueError, match="per-server concurrency four"):
        _validate_strict_parity_probe(inference, certificate)


def test_v14_6_probe_shape_does_not_rebind_1_7b_profile() -> None:
    arena_root = Path(__file__).parents[1]
    inference = _strict_inference_config(
        arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
    )
    inference["model"]["name"] = "/workspace/models/qwen3-1.7b"
    inference["max_lora_rank"] = 16
    _validate_strict_parity_probe(inference, {"parity_report": {}})


def test_v14_6_cpu_bundle_reproduces_exactly() -> None:
    arena_root = Path(__file__).parents[1]
    repo_root = arena_root.parents[1]
    code_paths = tuple(
        repo_root / path
        for path in (
            "experiments/swarm_arena/scripts/capture_runtime_parity_probe.py",
            "experiments/swarm_arena/scripts/bind_runtime_certificate.py",
            "experiments/swarm_arena/scripts/preflight_staged_rl.py",
            "experiments/swarm_arena/scripts/launch_inference_pool.sh",
            "experiments/swarm_arena/scripts/launch_staged_rl.sh",
            "experiments/swarm_arena/scripts/freeze_v14_6_parity_stable_execution.py",
        )
    )
    actual = build_bundle(
        parent=load_hashed(arena_root / "data" / "rl_v14_5" / "cpu_bundle.json"),
        trainer_path=(
            arena_root / "configs" / "rl_v14_5_4b_policy_routed_40.toml"
        ),
        inference_path=(
            arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
        ),
        plan_path=arena_root / "V14_6_EXECUTION_PLAN.md",
        code_paths=code_paths,
    )
    expected = load_hashed(arena_root / "data" / "rl_v14_6" / "cpu_bundle.json")
    assert actual == expected


def test_v14_6_bundle_rejects_changed_parent() -> None:
    arena_root = Path(__file__).parents[1]
    parent = copy.deepcopy(
        load_hashed(arena_root / "data" / "rl_v14_5" / "cpu_bundle.json")
    )
    parent["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen V14.5"):
        build_bundle(
            parent=parent,
            trainer_path=(
                arena_root / "configs" / "rl_v14_5_4b_policy_routed_40.toml"
            ),
            inference_path=(
                arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
            ),
            plan_path=arena_root / "V14_6_EXECUTION_PLAN.md",
            code_paths=(),
        )


def test_v14_7_a6000_bundle_reproduces_exactly() -> None:
    arena_root = Path(__file__).parents[1]
    repo_root = arena_root.parents[1]
    code_paths = tuple(
        repo_root / path
        for path in (
            "experiments/swarm_arena/scripts/capture_runtime_parity_probe.py",
            "experiments/swarm_arena/scripts/bind_runtime_certificate.py",
            "experiments/swarm_arena/scripts/preflight_staged_rl.py",
            "experiments/swarm_arena/scripts/launch_inference_pool.sh",
            "experiments/swarm_arena/scripts/launch_staged_rl.sh",
            "experiments/swarm_arena/scripts/freeze_v14_7_a6000_execution.py",
        )
    )
    actual = build_a6000_bundle(
        parent=load_hashed(arena_root / "data" / "rl_v14_6" / "cpu_bundle.json"),
        trainer_path=(
            arena_root / "configs" / "rl_v14_5_4b_policy_routed_40.toml"
        ),
        parent_inference_path=(
            arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
        ),
        inference_path=(
            arena_root / "configs" / "inference_4b_a6000_parity_strict.toml"
        ),
        plan_path=arena_root / "V14_7_A6000_EXECUTION_PLAN.md",
        code_paths=code_paths,
    )
    expected = load_hashed(arena_root / "data" / "rl_v14_7" / "cpu_bundle.json")
    assert actual == expected


def test_v14_7_rejects_inference_setting_change(tmp_path: Path) -> None:
    arena_root = Path(__file__).parents[1]
    changed = (
        arena_root / "configs" / "inference_4b_a6000_parity_strict.toml"
    ).read_text(encoding="utf-8").replace("max_num_seqs = 4", "max_num_seqs = 8")
    changed_path = tmp_path / "changed.toml"
    changed_path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="strict inference settings"):
        build_a6000_bundle(
            parent=load_hashed(
                arena_root / "data" / "rl_v14_6" / "cpu_bundle.json"
            ),
            trainer_path=(
                arena_root / "configs" / "rl_v14_5_4b_policy_routed_40.toml"
            ),
            parent_inference_path=(
                arena_root / "configs" / "inference_4b_l40s_parity_strict.toml"
            ),
            inference_path=changed_path,
            plan_path=arena_root / "V14_7_A6000_EXECUTION_PLAN.md",
            code_paths=(),
        )
