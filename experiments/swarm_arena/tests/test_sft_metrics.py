from __future__ import annotations

import json

from swarm_ctf_eval.sft_metrics import validate_dataset_response


def _broadcast_row() -> dict:
    facts = [
        {
            "node": "V1",
            "owner": "BLUE",
            "status": "SECURE",
            "value": 2,
            "critical": False,
            "observed_turn": 0,
        },
        {
            "node": "V2",
            "owner": "NEUTRAL",
            "status": "SECURE",
            "value": 1,
            "critical": True,
            "observed_turn": 0,
        },
    ]
    target = {
        "facts": facts,
        "intent": {"type": "PROBE", "target": "V2"},
        "request_resource": 0,
    }
    user = {
        "observation": {"known_nodes": facts},
        "legal_actions": [{"id": "A0", "type": "PROBE", "target": "V2"}],
    }
    return {
        "messages": [
            {"role": "user", "content": json.dumps(user)},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "metadata": {"phase": "BROADCAST"},
    }


def test_broadcast_fact_order_is_semantically_irrelevant() -> None:
    row = _broadcast_row()
    predicted = json.loads(row["messages"][-1]["content"])
    predicted["facts"].reverse()

    result = validate_dataset_response(row, json.dumps(predicted))

    assert result["schema_valid"]
    assert result["supported"]
    assert result["exact"]
    assert not result["ordered_exact"]


def test_broadcast_duplicates_and_boolean_resource_are_rejected() -> None:
    row = _broadcast_row()
    predicted = json.loads(row["messages"][-1]["content"])
    predicted["facts"] = [predicted["facts"][0], predicted["facts"][0]]
    predicted["request_resource"] = True

    result = validate_dataset_response(row, json.dumps(predicted))

    assert result["schema_valid"]
    assert not result["supported"]
    assert not result["legal"]
    assert not result["exact"]
