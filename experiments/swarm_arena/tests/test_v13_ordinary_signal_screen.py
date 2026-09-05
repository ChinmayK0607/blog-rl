from __future__ import annotations

from experiments.swarm_arena.scripts.assess_v13_ordinary_signal_screen import assess
from experiments.swarm_arena.scripts.build_v13_ordinary_signal_screen import (
    build_screen_manifest,
)


def _curriculum() -> dict:
    handoff_cases = [[0, "left_exposed"]] * 100
    return {
        "groups_per_update": 4,
        "schedule": {"pair_offset": 0, "ordinary_seed_base": 1000, "shuffle_seed": 7},
        "stages": [
            {
                "name": "screen",
                "updates": 40,
                "update_pattern": [{"ordinary": 2, "critical": 1, "decoy": 1}],
                "ordinary_sizes": [12, 14],
                "ordinary_horizons": [4, 6],
                "handoff_remaining_turns": 1,
                "handoff_cases": handoff_cases,
            }
        ],
    }


def _initializer() -> dict:
    return {
        "ready": {
            "step": 160,
            "policy_revision": "r" * 64,
            "policy_adapter_sha256": {f"blue-{index}": str(index) * 64 for index in range(4)},
        }
    }


def _diagnostic(case: dict, *, variable: bool = True) -> list[dict]:
    focused = case["focused_agent"]
    returns = [0.0, 1.0, 0.0, 1.0] if variable else [0.0] * 4
    advantages = [-1.0, 1.0, -1.0, 1.0] if variable else [0.0] * 4
    return [
        {
            "groups": [
                {
                    "scenario": {
                        "source": "ordinary",
                        "seed": case["seed"],
                        "size": case["size"],
                        "scheduled_horizon": case["horizon"],
                        "focused_agent": focused,
                    },
                    "replicas": [
                        {
                            "return": value,
                            "advantages": {focused: advantage},
                            "focused_action": {"type": "PROBE", "target": f"V{index % 2}"},
                        }
                        for index, (value, advantage) in enumerate(
                            zip(returns, advantages, strict=True)
                        )
                    ],
                }
            ]
        }
    ]


def test_screen_manifest_is_balanced_and_hash_bound() -> None:
    manifest = build_screen_manifest(
        _curriculum(),
        {"status": "cpu_schedule_passed_gpu_gates_pending"},
        _initializer(),
        curriculum_file_sha256="a" * 64,
        curriculum_audit_file_sha256="b" * 64,
        initializer_manifest_file_sha256="c" * 64,
    )
    assert manifest["case_count"] == 64
    assert manifest["games"] == 256
    assert set(manifest["policy_case_counts"].values()) == {16}
    assert set(manifest["opponent_case_counts"].values()) == {16}


def test_assessment_passes_balanced_nonzero_signal() -> None:
    manifest = build_screen_manifest(
        _curriculum(),
        {"status": "cpu_schedule_passed_gpu_gates_pending"},
        _initializer(),
        curriculum_file_sha256="a" * 64,
        curriculum_audit_file_sha256="b" * 64,
        initializer_manifest_file_sha256="c" * 64,
    )
    diagnostics = {case["case_id"]: _diagnostic(case) for case in manifest["cases"]}
    result = assess(manifest, diagnostics)
    assert result["status"] == "passed"
    assert result["protocol_admission_rate"] == 1.0
    assert not result["failed_gates"]


def test_assessment_fails_zero_credit_slot() -> None:
    manifest = build_screen_manifest(
        _curriculum(),
        {"status": "cpu_schedule_passed_gpu_gates_pending"},
        _initializer(),
        curriculum_file_sha256="a" * 64,
        curriculum_audit_file_sha256="b" * 64,
        initializer_manifest_file_sha256="c" * 64,
    )
    diagnostics = {
        case["case_id"]: _diagnostic(
            case,
            variable=case["focused_agent"] != "blue-2",
        )
        for case in manifest["cases"]
    }
    result = assess(manifest, diagnostics)
    assert result["status"] == "failed"
    assert any(gate.startswith("blue-2/") for gate in result["failed_gates"])
