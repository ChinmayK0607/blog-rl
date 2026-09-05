from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from swarm_ctf_eval.rl_production import load_production_plan

from experiments.swarm_arena.scripts.build_staged_rl_plan import _production_stage
from experiments.swarm_arena.scripts.build_v14_grounded_curriculum import build


def test_v14_curriculum_targets_multiturn_followthrough_and_fails_closed(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    v13 = json.loads((root / "data" / "rl_v13" / "curriculum.json").read_text())
    train = json.loads((root / "data" / "rl_v12" / "handoff_train.json").read_text())
    summary = {
        "version": "arena-rl-progress-eval-v5-rl-specific-communication",
        "tier": "pulse",
        "rows": 192,
        "communication_effects": {
            "normal_minus_dropped": {"mean_difference": 0.01, "mean_difference_95": [-0.1, 0.1]}
        },
        "critical_minus_decoy_specificity": {
            "mean_difference": 0.01,
            "mean_difference_95": [-0.1, 0.1],
        },
        "rl_specific_communication_lift": {
            "mean_difference": 0.01,
            "mean_difference_95": [-0.1, 0.1],
        },
        "communication_mechanism": {
            "candidate_capture_normal_minus_dropped": {"mean_difference": 0.125}
        },
        "capability_rl_minus_sft": {
            "ordinary_hard": {"mean_difference": 0.02},
            "ordinary_legacy": {"mean_difference": 0.02},
        },
        "overall_gameplay_rl_minus_sft": {"mean_difference": 0.03},
    }
    hashes = {policy: policy[-1] * 64 for policy in ("blue-0", "blue-1", "blue-2", "blue-3")}
    manifest = {
        "ready": {"step": 80, "policy_revision": "r", "policy_adapter_sha256": hashes},
        "files_sha256": {
            f"checkpoints/step-80/policy-{policy}/adapter_model.safetensors": value
            for policy, value in hashes.items()
        },
    }
    artifacts = build(
        v13_curriculum=v13,
        train_manifest=train,
        update60=summary,
        update80=summary,
        initializer_manifest=manifest,
        v13_progress=[],
        source_hashes={"fixture": "f" * 64},
    )
    curriculum = artifacts["curriculum.json"]
    audit = artifacts["curriculum_audit.json"]
    gates = artifacts["stage_gates.json"]

    assert curriculum["total_updates"] == 40
    assert curriculum["runtime"]["monitor_logical_update_signal"] is True
    assert curriculum["adaptive_curriculum"]["stage_updates"] == 10
    assert curriculum["adaptive_scope"]["evidence"].startswith("immediately preceding")
    assert [stage["handoff_trainable_turn_offsets"] for stage in curriculum["stages"]] == [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ]
    assert audit["group_counts"] == {"critical": 70, "decoy": 40, "ordinary": 50}
    assert audit["receiver_counts"]["decoy"] == {
        "blue-0": 10,
        "blue-1": 10,
        "blue-2": 10,
        "blue-3": 10,
    }
    assert Counter(audit["receiver_counts"]["critical"].values()) == Counter({18: 2, 17: 2})
    assert audit["decoys_are_update_local_matched_critical_subset"] is True
    assert list(gates["checkpoints"]) == ["10", "20", "30", "40"]
    assert all(
        checkpoint["on_fail"] == "stop_before_next_optimizer_update"
        for checkpoint in gates["checkpoints"].values()
    )
    assert artifacts["cpu_bundle.json"]["gpu_budget"]["maximum_usd"] == 15.0
    assert artifacts["cpu_bundle.json"]["artifact_body_sha256"]["stage_gates"] == gates["sha256"]
    assert artifacts["cpu_bundle.json"]["frozen_data_opened"] is False
    assert artifacts["v13_frontier_analysis.json"]["scope"].endswith(
        "no_development_or_frozen_data"
    )
    rendered_stage = _production_stage(curriculum["stages"][0])
    assert rendered_stage["handoff_cases"][0] == {
        "pair_index": curriculum["stages"][0]["handoff_cases"][0][0],
        "world": curriculum["stages"][0]["handoff_cases"][0][1],
    }
    base = json.loads((root / "configs" / "rl_v11_4b_base_plan.json").read_text())
    plan_path = tmp_path / "v14-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                **base,
                "version": "arena-rl-v4-staged-production-plan-v1",
                "curriculum_stages": [
                    _production_stage(stage) for stage in curriculum["stages"]
                ],
                "rollout_runtime": curriculum["runtime"],
                "adaptive_curriculum": curriculum["adaptive_curriculum"],
            }
        )
    )
    plan, _ = load_production_plan(plan_path)
    assert plan.expected_updates == 40
    assert len(plan.curriculum_schedule(steps=40)) == 160
    assert plan.adaptive_curriculum is not None
    assert len(plan.adaptive_curriculum.candidate_cases) == 33
