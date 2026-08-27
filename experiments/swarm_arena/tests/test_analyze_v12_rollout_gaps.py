from __future__ import annotations

from experiments.swarm_arena.scripts.analyze_v12_rollout_gaps import training_report


def _group(kind: str, *, action_target: str, effect: float) -> dict:
    replica = {
        "advantages": {"blue-0": effect, "blue-1": 0.0, "blue-2": 0.0, "blue-3": 0.0},
        "challenge_effect": effect if kind == "decoy" else None,
        "focused_action": {"target": action_target, "type": "CAPTURE"},
        "return": 0.1,
        "semantic_effect": effect if kind == "critical" else None,
    }
    return {
        "replicas": [replica, replica.copy()],
        "scenario": {
            "active_target": "V1",
            "candidate_targets": ["V1", "V2"],
            "curriculum_stage": "stage",
            "focused_agent": "blue-0",
            "kind": kind,
            "opponent": {"family": "sft"},
            "scheduled_horizon": 2,
            "world": "left_exposed",
        },
    }


def test_training_report_separates_factual_and_challenge_action_classes() -> None:
    updates = [
        {
            "groups": [
                _group("critical", action_target="V1", effect=0.2),
                _group("decoy", action_target="V2", effect=-0.2),
            ],
            "policy_revision": "revision",
            "step": 0,
        }
    ]
    report = training_report(updates)
    critical = report["by_kind"]["critical"]
    challenge = report["by_kind"]["decoy"]
    assert critical["focused_action_class_rates"]["active_target"] == 1.0
    assert critical["effect"]["mean"] == 0.2
    assert challenge["focused_action_class_rates"]["alternate_target"] == 1.0
    assert challenge["effect"]["negative_rate"] == 1.0


def test_training_report_rejects_noncontiguous_progress() -> None:
    updates = [{"groups": [], "policy_revision": "revision", "step": 1}]
    try:
        training_report(updates)
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("noncontiguous progress must fail closed")
