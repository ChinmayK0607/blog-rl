from __future__ import annotations

import json
from collections import Counter
from typing import Any


def parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _canonical_facts(facts: Any) -> Counter[str] | None:
    if not isinstance(facts, list) or not all(isinstance(fact, dict) for fact in facts):
        return None
    return Counter(json.dumps(fact, sort_keys=True, separators=(",", ":")) for fact in facts)


def _target_equivalent(phase: str, predicted: dict[str, Any] | None, target: dict[str, Any]) -> bool:
    if predicted is None:
        return False
    if phase == "ACT":
        return predicted == target
    predicted_facts = _canonical_facts(predicted.get("facts"))
    target_facts = _canonical_facts(target.get("facts"))
    return (
        predicted_facts is not None
        and predicted_facts == target_facts
        and predicted.get("intent") == target.get("intent")
        and type(predicted.get("request_resource")) is int
        and predicted.get("request_resource") == target.get("request_resource")
    )


def validate_dataset_response(row: dict[str, Any], raw: str) -> dict[str, bool]:
    """Strict, data-only validation used during checkpoint selection.

    The frozen simulator evaluation remains the authoritative outcome metric.
    This check catches formatting, unsupported broadcast facts, illegal action IDs,
    and semantically exact imitation of the unique filtered target without needing
    an arena seed. Broadcast fact order is intentionally ignored.
    """

    predicted = parse_json_object(raw)
    target = json.loads(row["messages"][-1]["content"])
    phase = row["metadata"]["phase"]
    user = json.loads(row["messages"][-2]["content"])
    schema = False
    supported = False
    legal = False

    if phase == "ACT" and predicted is not None:
        schema = set(predicted) == {"action_id"} and isinstance(predicted.get("action_id"), str)
        legal_ids = {item["id"] for item in user["legal_actions"]}
        legal = schema and predicted["action_id"] in legal_ids
        supported = legal
    elif phase == "BROADCAST" and predicted is not None:
        schema = set(predicted) == {"facts", "intent", "request_resource"}
        facts = predicted.get("facts") if schema else None
        if schema and isinstance(facts, list) and len(facts) <= 3:
            known = user["observation"]["known_nodes"]
            supported_facts = all(isinstance(fact, dict) and fact in known for fact in facts)
            canonical_facts = _canonical_facts(facts)
            facts_are_unique = canonical_facts is not None and all(count == 1 for count in canonical_facts.values())
            legal_actions = [
                {key: value for key, value in action.items() if key != "id"} for action in user["legal_actions"]
            ]
            intent = predicted.get("intent")
            legal_intent = intent is None or intent in legal_actions
            request_resource = predicted.get("request_resource")
            request_ok = type(request_resource) is int and request_resource in (0, 1)
            supported = supported_facts and facts_are_unique and legal_intent and request_ok
            legal = supported

    return {
        "schema_valid": schema,
        "supported": supported,
        "legal": legal,
        "ordered_exact": predicted == target,
        "exact": _target_equivalent(phase, predicted, target),
    }
