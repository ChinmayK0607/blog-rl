from __future__ import annotations

from collections import Counter

from experiments.swarm_arena.scripts.build_v13_role_adaptive_curriculum import (
    build,
    challenge_role_quotas,
    handoff_case_order,
    role_order,
)


def _selection() -> dict:
    challenge = []
    rehearsal = []
    for index, policy in enumerate(("blue-0", "blue-1", "blue-2", "blue-3")):
        for offset in range(2):
            row = {
                "pair_index": index,
                "world": "left_exposed" if offset == 0 else "right_exposed",
                "receiver": policy,
                "priority": 1.0,
            }
            challenge.append(row)
            rehearsal.append(row)
    return {
        "admission": {"status": "interim_only"},
        "selected": {
            "challenge_repair": challenge,
            "critical_rehearsal": rehearsal,
            "ordinary_replay_anchors": [],
        },
    }


def _train() -> dict:
    return {
        "pairs": [
            {
                "critical": {"receiver": f"blue-{index}"},
                "decoy": {"receiver": f"blue-{index}"},
            }
            for index in range(4)
        ]
    }


def test_role_order_preserves_exact_quotas_without_adjacent_blue_zero() -> None:
    quotas = {"blue-0": 32, "blue-1": 14, "blue-2": 14, "blue-3": 20}
    order = role_order(quotas)
    assert Counter(order) == Counter(quotas)
    assert len(order) == 80
    assert all(left != right for left, right in zip(order, order[1:]))


def test_v13_schedule_is_matched_and_has_expected_counts() -> None:
    selection = _selection()
    quotas = challenge_role_quotas(selection)
    assert quotas == {policy: 20 for policy in ("blue-0", "blue-1", "blue-2", "blue-3")}
    assert len(handoff_case_order(selection)) == 100
    plan, audit = build(selection, _train())
    assert plan["total_updates"] == 80
    assert "nonadmitted_warmstart" in plan["initializer"]
    assert audit["group_counts"] == {"critical": 100, "decoy": 80, "ordinary": 140}
    assert audit["decoys_are_matched_critical_subset"] is True
    assert audit["ordinary_seeds_unique"] is True
    assert audit["frozen_data_used"] is False
