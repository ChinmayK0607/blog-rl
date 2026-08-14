from __future__ import annotations

from scripts.audit_final_eval_collapse import _behavior_summary


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
