from __future__ import annotations

import json
from typing import Any

STRUCTURED_PROTOCOL_VERSION = "arena-structured-protocol-v1-dynamic-json-schema"


def _const(value: Any) -> dict[str, Any]:
    return {"const": value}


def _object_branch(
    facts: dict[str, Any],
    intent: Any,
    request_resource: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "facts": facts,
            "intent": _const(intent),
            "request_resource": _const(request_resource),
        },
        "required": ["facts", "intent", "request_resource"],
        "additionalProperties": False,
    }


def broadcast_json_schema(body: dict[str, Any]) -> dict[str, Any]:
    """Enumerate only grounded, legal, budget-feasible Broadcast shapes.

    The schema constrains mechanics, not strategy: the model still chooses
    whether to speak, which observed facts to send, which currently legal intent
    to declare, and whether to request a resource.
    """
    observation = body["observation"]
    known_facts = list(observation["known_nodes"])
    legal_intents = list(body["legal_intents"])
    budget = int(observation["message_budget_remaining"])
    max_facts = min(int(body["max_facts"]), len(known_facts))
    branches = []
    for fact_count in range(max_facts + 1):
        facts_schema: dict[str, Any]
        if fact_count == 0:
            facts_schema = _const([])
        else:
            facts_schema = {
                "type": "array",
                "minItems": fact_count,
                "maxItems": fact_count,
                "items": {"anyOf": [_const(fact) for fact in known_facts]},
                "uniqueItems": True,
            }
        for intent in [None, *legal_intents]:
            for request in (0, 1):
                empty = fact_count == 0 and intent is None and request == 0
                units = 0 if empty else 1 + fact_count + int(intent is not None) + request
                if units <= budget:
                    branches.append(_object_branch(facts_schema, intent, request))
    if not branches:
        raise ValueError("broadcast schema must always permit the empty message")
    return {"anyOf": branches}


def action_json_schema(body: dict[str, Any]) -> dict[str, Any]:
    identifiers = [item["id"] for item in body["legal_actions"]]
    if not identifiers:
        raise ValueError("action prompt must contain at least WAIT")
    return {
        "type": "object",
        "properties": {"action_id": {"type": "string", "enum": identifiers}},
        "required": ["action_id"],
        "additionalProperties": False,
    }


def protocol_response_format(messages: list[dict[str, str]]) -> dict[str, Any]:
    body = json.loads(messages[-1]["content"])
    phase = body.get("phase")
    if phase == "BROADCAST":
        schema = broadcast_json_schema(body)
        name = "arena_broadcast"
    elif phase == "ACT":
        schema = action_json_schema(body)
        name = "arena_action"
    else:
        raise ValueError(f"structured protocol received unknown phase: {phase}")
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
