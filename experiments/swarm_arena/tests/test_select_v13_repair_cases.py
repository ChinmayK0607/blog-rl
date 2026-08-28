from __future__ import annotations

from experiments.swarm_arena.scripts.select_v13_repair_cases import select_repair_cases


def _handoff_group(kind: str, receiver: str, pair_index: int, alternate: bool) -> dict:
    effect = -0.2 if kind == "decoy" and alternate else 0.2
    return {
        "scenario": {
            "active_target": "V1",
            "candidate_targets": ["V1", "V2"],
            "focused_agent": receiver,
            "kind": kind,
            "pair_index": pair_index,
            "receiver": receiver,
            "world": "left_exposed",
        },
        "replicas": [
            {
                "advantages": {receiver: effect},
                "challenge_effect": effect if kind == "decoy" else None,
                "focused_action": {
                    "target": "V2" if alternate else "V1",
                    "type": "CAPTURE",
                },
                "semantic_effect": effect if kind == "critical" else None,
            }
        ],
    }


def _ordinary_group(receiver: str, seed: int) -> dict:
    return {
        "scenario": {
            "curriculum_stage": "repair",
            "focused_agent": receiver,
            "opponent": {"family": "base"},
            "seed": seed,
            "source": "ordinary",
        },
        "replicas": [
            {
                "advantages": {receiver: 0.1},
                "focused_action": {"target": f"V{seed}", "type": "PROBE"},
            }
        ],
    }


def _updates() -> list[dict]:
    groups = []
    for policy_index, policy in enumerate(("blue-0", "blue-1", "blue-2", "blue-3")):
        groups.extend(
            (
                _handoff_group("decoy", policy, policy_index, alternate=policy == "blue-0"),
                _handoff_group("critical", policy, policy_index, alternate=False),
                _ordinary_group(policy, 100 + policy_index),
            )
        )
    return [{"groups": groups, "step": 0}]


def test_selector_prioritizes_message_obedient_decoy() -> None:
    result = select_repair_cases(
        _updates(),
        window_updates=1,
        challenge_per_receiver=1,
        critical_per_receiver=1,
        ordinary_per_receiver=1,
    )
    selected = result["selected"]["challenge_repair"]
    blue_zero = next(row for row in selected if row["receiver"] == "blue-0")
    assert blue_zero["alternate_target_rate"] == 1.0
    assert blue_zero["negative_effect_rate"] == 1.0
    assert result["admission"]["status"] == "interim_only"


def test_selector_can_bind_completed_training_only_progress() -> None:
    result = select_repair_cases(
        _updates(),
        window_updates=1,
        challenge_per_receiver=1,
        critical_per_receiver=1,
        ordinary_per_receiver=1,
        completed_run=True,
    )
    assert result["admission"]["status"] == "training_only_complete"


def test_selector_rejects_noncontiguous_progress() -> None:
    updates = _updates()
    updates[0]["step"] = 1
    try:
        select_repair_cases(
            updates,
            window_updates=1,
            challenge_per_receiver=1,
            critical_per_receiver=1,
            ordinary_per_receiver=1,
        )
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("noncontiguous progress must fail closed")
