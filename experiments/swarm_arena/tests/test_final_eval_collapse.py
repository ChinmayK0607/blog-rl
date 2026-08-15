from __future__ import annotations

import pytest
from scripts.audit_final_eval_collapse import _behavior_summary, _evaluation_metrics


def test_behavior_summary_detects_action_and_speaking_collapse() -> None:
    broadcasts = [
        {
            "agent_id": f"blue-{index}",
            "team": "BLUE",
            "parsed_message": {"facts": [], "intent": None, "request_resource": 0},
        }
        for index in range(4)
    ]
    actions = [
        {
            "agent_id": f"blue-{index}",
            "team": "BLUE",
            "selected_action": {"type": "WAIT"},
        }
        for index in range(4)
    ]
    raw = {
        "blue_agent_models": {f"blue-{index}": f"blue-{index}" for index in range(4)},
        "turns": [{"broadcasts": broadcasts, "actions": actions}] * 20,
    }
    kl = {
        "per_policy": {
            f"blue-{index}": {
                "candidate_to_baseline_kl": {"mean": 0.01, "p99": 0.02}
            }
            for index in range(4)
        }
    }
    report = _behavior_summary([({"side": "BLUE"}, raw)], kl)
    assert all(item["never_speaking"] for item in report.values())
    assert all(item["action_collapse"] for item in report.values())
    assert not any(item["excessive_kl"] for item in report.values())


def test_behavior_summary_maps_explicit_served_policy_aliases() -> None:
    raw = {
        "blue_agent_models": {
            f"blue-{index}": f"served-step3-blue-{index}" for index in range(4)
        },
        "turns": [
            {
                "broadcasts": [
                    {
                        "agent_id": f"blue-{index}",
                        "team": "BLUE",
                        "parsed_message": {
                            "facts": [],
                            "intent": None,
                            "request_resource": 0,
                        },
                    }
                    for index in range(4)
                ],
                "actions": [
                    {
                        "agent_id": f"blue-{index}",
                        "team": "BLUE",
                        "selected_action": {"type": "WAIT"},
                    }
                    for index in range(4)
                ],
            }
        ],
    }
    kl = {
        "per_policy": {
            f"blue-{index}": {
                "candidate_to_baseline_kl": {"mean": 0.01, "p99": 0.02}
            }
            for index in range(4)
        }
    }
    aliases = {
        f"served-step3-blue-{index}": f"blue-{index}" for index in range(4)
    }
    report = _behavior_summary(
        [({"side": "BLUE"}, raw)],
        kl,
        policy_aliases=aliases,
    )
    assert set(report) == set(kl["per_policy"])


def test_evaluation_metrics_are_derived_from_paired_v4_rows() -> None:
    rows = []
    for opponent, candidate, baseline in (("base", 0.3, 0.1), ("sft", 0.2, 0.25)):
        common = {
            "case_id": f"ordinary-{opponent}",
            "suite": "ordinary_hard",
            "opponent_id": opponent,
            "side": "BLUE",
            "condition": "normal",
            "option_order": "canonical",
        }
        rows.extend(
            (
                {**common, "policy_variant": "candidate_rl", "terminal_return": candidate},
                {**common, "policy_variant": "sft_init", "terminal_return": baseline},
            )
        )
    for case_id, normal, dropped in (("critical-a", 0.4, 0.1), ("critical-b", 0.2, 0.1)):
        common = {
            "case_id": case_id,
            "suite": "handoff_critical",
            "opponent_id": "sft",
            "side": "BLUE",
            "policy_variant": "candidate_rl",
        }
        rows.extend(
            (
                {**common, "condition": "normal", "terminal_return": normal},
                {**common, "condition": "dropped", "terminal_return": dropped},
            )
        )

    opponent_returns, return_gain, message_gain = _evaluation_metrics(rows)
    assert opponent_returns == {"base": 0.3, "sft": 0.2}
    assert return_gain == pytest.approx(0.075)
    assert message_gain == pytest.approx(0.2)


def test_evaluation_metrics_reject_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match="paired ordinary"):
        _evaluation_metrics(
            [
                {
                    "case_id": "ordinary-a",
                    "suite": "ordinary_hard",
                    "opponent_id": "sft",
                    "side": "BLUE",
                    "condition": "normal",
                    "policy_variant": "candidate_rl",
                    "terminal_return": 0.1,
                }
            ]
        )
