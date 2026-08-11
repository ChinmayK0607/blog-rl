from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from .regression import ARENA_KEYS, SYSTEM, RegressionCase

REGRESSION_VERSION = "swarm-regression-v2"


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


def generate_regression_v2_cases(seed: int = 20260831, per_category: int = 64) -> tuple[RegressionCase, ...]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    rng = random.Random(seed)
    rows = []
    labels = tuple("ABCDEFGH")

    for _ in range(per_category):
        records = [
            {"code": label, "score": score}
            for label, score in zip(rng.sample(labels, 5), rng.sample(range(20, 100), 5), strict=True)
        ]
        ranked = sorted(records, key=lambda item: (-item["score"], item["code"]))
        expected = {"runner_up": ranked[1]["code"], "leader": ranked[0]["code"]}
        rows.append(
            _case(
                "record_projection",
                f"Records: {records}. Rank by score descending. Return only "
                '{"runner_up":"code in second place","leader":"code in first place"}.',
                expected,
            )
        )

    for _ in range(per_category):
        values = rng.sample(range(2, 30), 7)
        shift = rng.randint(1, 6)
        rotated = values[shift:] + values[:shift]
        expected = {"tail_sum": sum(rotated[-3:]), "rotated": rotated}
        rows.append(
            _case(
                "conditional_transform",
                f"Rotate {values} left by {shift} positions, then sum its final three values. "
                'Return {"tail_sum":integer,"rotated":[integers]}.',
                expected,
            )
        )

    for _ in range(per_category):
        services = [
            {"service": f"svc-{label.lower()}", "uptime": uptime, "agents": rng.randint(1, 8)}
            for label, uptime in zip(rng.sample(labels, 4), rng.sample(range(910, 1000), 4), strict=True)
        ]
        best = max(services, key=lambda item: (item["uptime"], -item["agents"], item["service"]))
        expected = {"selected_service": best["service"], "uptime_basis_points": best["uptime"]}
        rows.append(
            _case(
                "domain_trigger_resistance",
                "These services use agents, network nodes, probes, and capture logs, but none are game actions. "
                f"Choose the highest-uptime service from {services}; break ties by fewer agents. Return exactly "
                '{"selected_service":"name","uptime_basis_points":integer}.',
                expected,
            )
        )

    for _ in range(per_category):
        start = rng.randint(20, 90)
        multiplier = rng.randint(2, 7)
        fee = rng.randint(3, 25)
        divisor = rng.randint(2, 6)
        total = start * multiplier - fee
        expected = {"quotient": total // divisor, "remainder": total % divisor}
        rows.append(
            _case(
                "arithmetic_division",
                f"Compute ({start} * {multiplier} - {fee}), divide by {divisor}, and return "
                '{"quotient":integer,"remainder":integer}.',
                expected,
            )
        )
    return tuple(rows)


FROZEN_REGRESSION_V2_CASES = generate_regression_v2_cases()
REGRESSION_V2_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(
        [row.to_dict() for row in FROZEN_REGRESSION_V2_CASES],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def validate_v2_response(case: RegressionCase, raw: str) -> dict[str, bool]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"valid_json": False, "exact": False, "arena_leakage": False}
    valid = isinstance(value, dict)

    def keys(item: Any) -> set[str]:
        if isinstance(item, dict):
            return set(item).union(*(keys(child) for child in item.values()))
        if isinstance(item, list):
            return set().union(*(keys(child) for child in item))
        return set()

    leakage = valid and bool(ARENA_KEYS.intersection(keys(value))) and not ARENA_KEYS.intersection(keys(case.expected))
    return {"valid_json": valid, "exact": valid and value == case.expected, "arena_leakage": bool(leakage)}


def summarize_v2_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "manifest_sha256": REGRESSION_V2_MANIFEST_SHA256,
        **rates(rows),
        "categories": {
            category: rates([row for row in rows if row["category"] == category]) for category in categories
        },
        "failure_counts": dict(sorted(failures.items())),
    }
