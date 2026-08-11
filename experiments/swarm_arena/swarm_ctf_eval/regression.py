from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

REGRESSION_VERSION = "swarm-regression-v1"
ARENA_KEYS = {"action_id", "facts", "intent", "request_resource"}
SYSTEM = "Follow the user's instructions exactly. Return only the requested JSON object, without analysis or markdown."


@dataclass(frozen=True)
class RegressionCase:
    id: str
    category: str
    messages: tuple[dict[str, str], ...]
    expected: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "messages": list(self.messages),
            "expected": self.expected,
        }


def _case(category: str, user: str, expected: dict[str, Any]) -> RegressionCase:
    payload = json.dumps(
        {"category": category, "user": user, "expected": expected},
        sort_keys=True,
        separators=(",", ":"),
    )
    identifier = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return RegressionCase(
        f"{category}-{identifier}",
        category,
        ({"role": "system", "content": SYSTEM}, {"role": "user", "content": user}),
        expected,
    )


def generate_regression_cases(seed: int = 20260811, per_category: int = 64) -> tuple[RegressionCase, ...]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    rng = random.Random(seed)
    rows = []
    for _ in range(per_category):
        left, right = rng.randint(3, 40), rng.randint(2, 15)
        offset, subtract = rng.randint(0, 30), rng.randint(0, 20)
        expected = {"answer": left * right + offset - subtract}
        rows.append(
            _case(
                "arithmetic",
                f'Compute ({left} * {right}) + {offset} - {subtract}. Return {{"answer": integer}}.',
                expected,
            )
        )

    for _ in range(per_category):
        values = [rng.randint(-12, 20) for _ in range(10)]
        expected = {"even_unique_sorted": sorted({value for value in values if value % 2 == 0})}
        rows.append(
            _case(
                "list_transform",
                "From this list, keep even integers, remove duplicates, and sort ascending. "
                f'List: {values}. Return {{"even_unique_sorted":[integers]}}.',
                expected,
            )
        )

    alphabet = "ABCDEFGH"
    for _ in range(per_category):
        values = rng.sample(range(10, 100), len(alphabet))
        mapping = dict(zip(alphabet, values, strict=True))
        requested = rng.sample(list(alphabet), 3)
        expected = {"second": mapping[requested[1]], "first": mapping[requested[0]], "third": mapping[requested[2]]}
        rows.append(
            _case(
                "instruction_binding",
                f"Codebook: {mapping}. Look up {requested}. Preserve these output key names and order: "
                "second, first, third. Return "
                '{"second":value_for_second_requested_symbol,"first":value_for_first_requested_symbol,'
                '"third":value_for_third_requested_symbol}.',
                expected,
            )
        )

    for index in range(per_category):
        node_names = [f"node-{letter.lower()}" for letter in rng.sample(list(alphabet), 4)]
        latencies = rng.sample(range(5, 90), 4)
        measurements = dict(zip(node_names, latencies, strict=True))
        fastest = min(measurements, key=measurements.get)  # type: ignore[arg-type]
        slowest = max(measurements, key=measurements.get)  # type: ignore[arg-type]
        expected = {"fastest_node": fastest, "slowest_node": slowest}
        rows.append(
            _case(
                "arena_trigger_resistance",
                "A monitoring agent measured network-node latency. This is not a game and there are no actions. "
                f"Measurements: {measurements}. Return exactly "
                '{"fastest_node":"name","slowest_node":"name"}.',
                expected,
            )
        )
    return tuple(rows)


FROZEN_REGRESSION_CASES = generate_regression_cases()
REGRESSION_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps([row.to_dict() for row in FROZEN_REGRESSION_CASES], sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def validate_response(case: RegressionCase, raw: str) -> dict[str, bool]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"valid_json": False, "exact": False, "arena_leakage": False}
    valid = isinstance(value, dict)

    def object_keys(item: Any) -> set[str]:
        if isinstance(item, dict):
            return set(item).union(*(object_keys(child) for child in item.values()))
        if isinstance(item, list):
            return set().union(*(object_keys(child) for child in item))
        return set()

    leakage = (
        valid
        and bool(ARENA_KEYS.intersection(object_keys(value)))
        and not ARENA_KEYS.intersection(object_keys(case.expected))
    )
    return {
        "valid_json": valid,
        "exact": valid and value == case.expected,
        "arena_leakage": bool(leakage),
    }


def summarize_regression_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("regression rows cannot be empty")
    categories = sorted({str(row["category"]) for row in rows})

    def rates(group: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "examples": len(group),
            "valid_json": sum(bool(row["valid_json"]) for row in group) / len(group),
            "exact": sum(bool(row["exact"]) for row in group) / len(group),
            "arena_leakage": sum(bool(row["arena_leakage"]) for row in group) / len(group),
        }

    failures = Counter(str(row["category"]) for row in rows if not row["exact"])
    return {
        "regression_version": REGRESSION_VERSION,
        "manifest_sha256": REGRESSION_MANIFEST_SHA256,
        **rates(rows),
        "categories": {
            category: rates([row for row in rows if row["category"] == category]) for category in categories
        },
        "failure_counts": dict(sorted(failures.items())),
    }
