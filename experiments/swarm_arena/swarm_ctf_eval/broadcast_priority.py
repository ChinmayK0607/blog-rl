from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any

PROMPT_VARIANTS = {
    "current": "",
    "recency_scoped": (
        "\n\nFACT PRIORITY: Recency resolves conflicting reports about the same node. "
        "Do not prefer an unrelated newer fact over an older fact merely because "
        "its observed_turn is larger."
    ),
    "actionable_priority": (
        "\n\nFACT PRIORITY: Use the limited fact slots for information most likely to "
        "change a teammate's legal action. Prefer actionable EXPOSED or COMPROMISED "
        "nodes, then critical or high-value nodes, then other facts. A slightly "
        "older actionable fact can be more useful than an unrelated fresh fact. "
        "Use recency only to resolve conflicting reports about the same node."
    ),
}


def apply_prompt_variant(
    messages: list[dict[str, str]],
    variant: str,
) -> list[dict[str, str]]:
    try:
        suffix = PROMPT_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown broadcast-priority variant: {variant}") from error
    updated = [dict(message) for message in messages]
    system_rows = [row for row in updated if row.get("role") == "system"]
    if len(system_rows) != 1:
        raise ValueError("broadcast prompt must contain exactly one system message")
    system_rows[0]["content"] += suffix
    return updated


def summarize_priority_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    summary = {}
    for variant, values in sorted(grouped.items()):
        pair_ids = sorted({int(row["pair_index"]) for row in values})
        repetitions = sorted({int(row["repetition"]) for row in values})
        target_by_pair = {
            str(pair_index): fmean(
                row["target_fact_present"]
                for row in values
                if int(row["pair_index"]) == pair_index
            )
            for pair_index in pair_ids
        }
        summary[variant] = {
            "samples": len(values),
            "pairs": len(pair_ids),
            "repetitions": len(repetitions),
            "protocol_valid_rate": fmean(row["protocol_valid"] for row in values),
            "target_fact_rate": fmean(row["target_fact_present"] for row in values),
            "mean_fact_count": fmean(row["fact_count"] for row in values),
            "pairs_target_fact_majority": sum(
                rate > 0.5 for rate in target_by_pair.values()
            ),
            "target_fact_rate_by_pair": target_by_pair,
        }
    return summary
