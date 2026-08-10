from __future__ import annotations

import json
from typing import Any


def parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def validate_dataset_response(row: dict[str, Any], raw: str) -> dict[str, bool]:
    """Strict, data-only validation used during checkpoint selection.

    The frozen simulator evaluation remains the authoritative outcome metric.
    This check catches formatting, unsupported broadcast facts, illegal action IDs,
    and exact imitation of the unique filtered target without needing an arena seed.
    """

    predicted = parse_json_object(raw)
    target = json.loads(row["messages"][-1]["content"])
    phase = row["metadata"]["phase"]
    user = json.loads(row["messages"][-2]["content"])
    schema = False
    supported = False
    legal = False

    if phase == "ACT" and predicted is not None:
        schema = set(predicted) == {"action_id"} and isinstance(
            predicted.get("action_id"), str
        )
        legal_ids = {item["id"] for item in user["legal_actions"]}
        legal = schema and predicted["action_id"] in legal_ids
        supported = legal
    elif phase == "BROADCAST" and predicted is not None:
        schema = set(predicted) == {"facts", "intent", "request_resource"}
        facts = predicted.get("facts") if schema else None
        if schema and isinstance(facts, list) and len(facts) <= 3:
            known = user["observation"]["known_nodes"]
            supported_facts = all(
                isinstance(fact, dict) and fact in known for fact in facts
            )
            legal_actions = [
                {key: value for key, value in action.items() if key != "id"}
                for action in user["legal_actions"]
            ]
            intent = predicted.get("intent")
            legal_intent = intent is None or intent in legal_actions
            request_ok = predicted.get("request_resource") in (0, 1)
            supported = supported_facts and legal_intent and request_ok
            legal = supported

    return {
        "schema_valid": schema,
        "supported": supported,
        "legal": legal,
        "exact": predicted == target,
    }
