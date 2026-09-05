from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAX_OVERALL_DROP = 0.02
MAX_CATEGORY_DROP = 0.05
MAX_LEAKAGE = 0.02


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    *,
    comparison_protocol: str = "paired-swarm-regression-v1",
) -> dict[str, Any]:
    base = {row["id"]: row for row in base_rows}
    adapter = {row["id"]: row for row in adapter_rows}
    if len(base) != len(base_rows) or len(adapter) != len(adapter_rows):
        raise ValueError("duplicate regression case id")
    if set(base) != set(adapter):
        raise ValueError("base and adapter regression cases differ")
    categories = sorted({str(row["category"]) for row in base_rows})

    def group(category: str | None) -> list[str]:
        return [
            identifier for identifier in sorted(base) if category is None or base[identifier]["category"] == category
        ]

    def result(identifiers: list[str]) -> dict[str, Any]:
        base_accuracy = sum(bool(base[item]["exact"]) for item in identifiers) / len(identifiers)
        adapter_accuracy = sum(bool(adapter[item]["exact"]) for item in identifiers) / len(identifiers)
        regressions = sum(bool(base[item]["exact"]) and not bool(adapter[item]["exact"]) for item in identifiers)
        improvements = sum(not bool(base[item]["exact"]) and bool(adapter[item]["exact"]) for item in identifiers)
        return {
            "examples": len(identifiers),
            "base_exact": base_accuracy,
            "adapter_exact": adapter_accuracy,
            "adapter_minus_base": adapter_accuracy - base_accuracy,
            "paired_regressions": regressions,
            "paired_improvements": improvements,
        }

    overall = result(group(None))
    by_category = {category: result(group(category)) for category in categories}
    adapter_leakage = sum(bool(row["arena_leakage"]) for row in adapter_rows) / len(adapter_rows)
    base_leakage = sum(bool(row["arena_leakage"]) for row in base_rows) / len(base_rows)
    gates = {
        "overall_drop_within_0_02": overall["adapter_minus_base"] >= -MAX_OVERALL_DROP,
        "every_category_drop_within_0_05": all(
            value["adapter_minus_base"] >= -MAX_CATEGORY_DROP for value in by_category.values()
        ),
        "adapter_arena_leakage_at_most_0_02": adapter_leakage <= MAX_LEAKAGE,
        "arena_leakage_not_increased": adapter_leakage <= base_leakage,
    }
    return {
        "comparison_protocol": comparison_protocol,
        "overall": overall,
        "categories": by_category,
        "base_arena_leakage": base_leakage,
        "adapter_arena_leakage": adapter_leakage,
        "gates": {**gates, "passed": all(gates.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and adapter non-arena regressions.")
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--adapter-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comparison-protocol",
        choices=("paired-swarm-regression-v1", "paired-swarm-regression-v2"),
        default="paired-swarm-regression-v1",
    )
    args = parser.parse_args()
    result = compare(
        load_rows(args.base_rows),
        load_rows(args.adapter_rows),
        comparison_protocol=args.comparison_protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
