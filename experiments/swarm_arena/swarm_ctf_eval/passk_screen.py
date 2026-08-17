from __future__ import annotations

import math
import statistics
from collections import defaultdict
from itertools import combinations
from typing import Any

PASSK_SCREEN_VERSION = "arena-training-passk-screen-v1"


def pass_at_k(successes: int, samples: int, k: int) -> float | None:
    """Unbiased pass@k estimator used by code-generation evaluations."""
    if not 0 <= successes <= samples:
        raise ValueError("successes must be between zero and samples")
    if not 1 <= k <= samples:
        return None
    if samples - successes < k:
        return 1.0
    return 1.0 - math.comb(samples - successes, k) / math.comb(samples, k)


def expected_best_at_k(values: list[float], k: int) -> float | None:
    if not 1 <= k <= len(values):
        return None
    return statistics.fmean(max(group) for group in combinations(values, k))


def contrast_at_k(values: list[float], k: int) -> float | None:
    """Probability that k samples contain more than one terminal return."""
    if not 1 <= k <= len(values):
        return None
    groups = list(combinations(values, k))
    return statistics.fmean(len(set(group)) > 1 for group in groups)


def summarize_passk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["pair_index"],
                row["kind"],
                row["world"],
                row["condition"],
            )
        ].append(row)

    cells = []
    for (pair_index, kind, world, condition), values in sorted(grouped.items()):
        returns = [float(row["terminal_return"]) for row in values]
        captures = sum(bool(row["target_captured"]) for row in values)
        turn_zero_captures = sum(bool(row["target_captured_turn_zero"]) for row in values)
        cell = {
            "pair_index": pair_index,
            "kind": kind,
            "world": world,
            "condition": condition,
            "samples": len(values),
            "capture_pass_at_1": captures / len(values),
            "turn_zero_capture_pass_at_1": turn_zero_captures / len(values),
            "sender_target_fact_rate": statistics.fmean(
                bool(row["sender_target_fact"]) for row in values
            ),
            "receiver_target_action_rate": statistics.fmean(
                bool(row["receiver_target_action"]) for row in values
            ),
            "mean_return": statistics.fmean(returns),
            "return_min": min(returns),
            "return_max": max(returns),
            "protocol_valid_rate": statistics.fmean(
                bool(row["protocol_valid"]) for row in values
            ),
        }
        for k in (2, 4, 8):
            cell[f"capture_pass_at_{k}"] = pass_at_k(captures, len(values), k)
            cell[f"turn_zero_capture_pass_at_{k}"] = pass_at_k(
                turn_zero_captures, len(values), k
            )
            cell[f"best_return_at_{k}"] = expected_best_at_k(returns, k)
            cell[f"return_contrast_at_{k}"] = contrast_at_k(returns, k)
        cells.append(cell)

    by_scenario: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        by_scenario[(cell["pair_index"], cell["kind"], cell["world"])][
            cell["condition"]
        ] = cell

    comparisons = []
    for (pair_index, kind, world), conditions in sorted(by_scenario.items()):
        required = {"generated", "dropped", "reference"}
        if set(conditions) != required:
            raise ValueError(
                f"scenario {pair_index}/{kind}/{world} lacks conditions: "
                f"{sorted(required - set(conditions))}"
            )
        generated = conditions["generated"]
        dropped = conditions["dropped"]
        reference = conditions["reference"]
        comparisons.append(
            {
                "pair_index": pair_index,
                "kind": kind,
                "world": world,
                "generated_capture_rate": generated["capture_pass_at_1"],
                "dropped_capture_rate": dropped["capture_pass_at_1"],
                "reference_capture_rate": reference["capture_pass_at_1"],
                "generated_minus_dropped_capture": (
                    generated["capture_pass_at_1"] - dropped["capture_pass_at_1"]
                ),
                "reference_minus_generated_capture": (
                    reference["capture_pass_at_1"] - generated["capture_pass_at_1"]
                ),
                "reference_minus_dropped_capture": (
                    reference["capture_pass_at_1"] - dropped["capture_pass_at_1"]
                ),
                "generated_sender_target_fact_rate": generated["sender_target_fact_rate"],
                "generated_capture_pass_at_4": generated["capture_pass_at_4"],
                "generated_capture_pass_at_8": generated["capture_pass_at_8"],
                "generated_return_contrast_at_4": generated["return_contrast_at_4"],
            }
        )

    critical = [row for row in comparisons if row["kind"] == "critical"]
    decoy = [row for row in comparisons if row["kind"] == "decoy"]

    def mean(rows_: list[dict[str, Any]], field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows_)

    return {
        "version": PASSK_SCREEN_VERSION,
        "games": len(rows),
        "cells": cells,
        "scenario_comparisons": comparisons,
        "aggregate": {
            "critical_scenarios": len(critical),
            "decoy_scenarios": len(decoy),
            "critical_generated_capture_rate": mean(critical, "generated_capture_rate"),
            "critical_generated_pass_at_4": mean(critical, "generated_capture_pass_at_4"),
            "critical_generated_pass_at_8": mean(critical, "generated_capture_pass_at_8"),
            "critical_reference_minus_generated_capture": mean(
                critical, "reference_minus_generated_capture"
            ),
            "critical_generated_minus_dropped_capture": mean(
                critical, "generated_minus_dropped_capture"
            ),
            "critical_reference_minus_dropped_capture": mean(
                critical, "reference_minus_dropped_capture"
            ),
            "critical_sender_target_fact_rate": mean(
                critical, "generated_sender_target_fact_rate"
            ),
            "critical_return_contrast_at_4": mean(
                critical, "generated_return_contrast_at_4"
            ),
            "decoy_generated_minus_dropped_capture": mean(
                decoy, "generated_minus_dropped_capture"
            ),
            "decoy_reference_minus_dropped_capture": mean(
                decoy, "reference_minus_dropped_capture"
            ),
        },
    }
