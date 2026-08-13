from __future__ import annotations

import json
from itertools import permutations
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
                "enum": [list(choice) for choice in permutations(known_facts, fact_count)]
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


def protocol_choices(messages: list[dict[str, str]]) -> tuple[str, ...]:
    """Enumerate canonical legal outputs for exact choice-trie decoding."""
    body = json.loads(messages[-1]["content"])
    if body.get("phase") == "ACT":
        values = ({"action_id": item["id"]} for item in body["legal_actions"])
    elif body.get("phase") == "BROADCAST":
        observation = body["observation"]
        facts = list(observation["known_nodes"])
        budget = int(observation["message_budget_remaining"])
        max_facts = min(int(body["max_facts"]), len(facts))
        values = (
            {"facts": list(selected), "intent": intent, "request_resource": request}
            for count in range(max_facts + 1)
            for selected in permutations(facts, count)
            for intent in [None, *body["legal_intents"]]
            for request in (0, 1)
            if (
                0
                if count == 0 and intent is None and request == 0
                else 1 + count + int(intent is not None) + request
            )
            <= budget
        )
    else:
        raise ValueError("protocol choices received an unknown phase")
    choices = tuple(
        sorted(
            {
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in values
            }
        )
    )
    if not choices:
        raise ValueError("protocol choice set cannot be empty")
    return choices


def completion_allowed_token_ids(
    completion_ids: list[int],
    choice_token_ids: list[list[int]],
) -> list[list[int]]:
    """Return the legal next-token set along one selected path through a choice trie."""
    prefix: list[int] = []
    rows = []
    for token_id in completion_ids:
        candidates = [choice for choice in choice_token_ids if choice[: len(prefix)] == prefix]
        allowed = sorted(
            {
                choice[len(prefix)]
                for choice in candidates
                if len(choice) > len(prefix)
            }
        )
        if token_id not in allowed:
            raise ValueError("completion is not a member of the canonical protocol choice trie")
        rows.append(allowed)
        prefix.append(token_id)
    if prefix not in choice_token_ids:
        raise ValueError("completion ended before reaching a canonical protocol choice")
    return rows
