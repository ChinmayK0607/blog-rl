from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from itertools import combinations
from typing import Any

PASSK_SCREEN_VERSION = "arena-training-passk-screen-v1"
TERMINAL_PROXIMAL_SCREEN_VERSION = "arena-terminal-proximal-screen-v1"


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
        receiver_actions = sum(bool(row["receiver_target_action"]) for row in values)
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
            cell[f"receiver_target_action_pass_at_{k}"] = pass_at_k(
                receiver_actions, len(values), k
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
                "generated_turn_zero_capture_rate": generated[
                    "turn_zero_capture_pass_at_1"
                ],
                "dropped_turn_zero_capture_rate": dropped[
                    "turn_zero_capture_pass_at_1"
                ],
                "reference_turn_zero_capture_rate": reference[
                    "turn_zero_capture_pass_at_1"
                ],
                "generated_receiver_target_action_rate": generated[
                    "receiver_target_action_rate"
                ],
                "dropped_receiver_target_action_rate": dropped[
                    "receiver_target_action_rate"
                ],
                "reference_receiver_target_action_rate": reference[
                    "receiver_target_action_rate"
                ],
                "generated_minus_dropped_capture": (
                    generated["capture_pass_at_1"] - dropped["capture_pass_at_1"]
                ),
                "reference_minus_generated_capture": (
                    reference["capture_pass_at_1"] - generated["capture_pass_at_1"]
                ),
                "reference_minus_dropped_capture": (
                    reference["capture_pass_at_1"] - dropped["capture_pass_at_1"]
                ),
                "generated_minus_dropped_turn_zero_capture": (
                    generated["turn_zero_capture_pass_at_1"]
                    - dropped["turn_zero_capture_pass_at_1"]
                ),
                "reference_minus_generated_turn_zero_capture": (
                    reference["turn_zero_capture_pass_at_1"]
                    - generated["turn_zero_capture_pass_at_1"]
                ),
                "reference_minus_dropped_turn_zero_capture": (
                    reference["turn_zero_capture_pass_at_1"]
                    - dropped["turn_zero_capture_pass_at_1"]
                ),
                "generated_minus_dropped_receiver_action": (
                    generated["receiver_target_action_rate"]
                    - dropped["receiver_target_action_rate"]
                ),
                "reference_minus_generated_receiver_action": (
                    reference["receiver_target_action_rate"]
                    - generated["receiver_target_action_rate"]
                ),
                "reference_minus_dropped_receiver_action": (
                    reference["receiver_target_action_rate"]
                    - dropped["receiver_target_action_rate"]
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
            "critical_generated_minus_dropped_turn_zero_capture": mean(
                critical, "generated_minus_dropped_turn_zero_capture"
            ),
            "critical_reference_minus_generated_turn_zero_capture": mean(
                critical, "reference_minus_generated_turn_zero_capture"
            ),
            "critical_reference_minus_dropped_turn_zero_capture": mean(
                critical, "reference_minus_dropped_turn_zero_capture"
            ),
            "critical_generated_minus_dropped_receiver_action": mean(
                critical, "generated_minus_dropped_receiver_action"
            ),
            "critical_reference_minus_generated_receiver_action": mean(
                critical, "reference_minus_generated_receiver_action"
            ),
            "critical_reference_minus_dropped_receiver_action": mean(
                critical, "reference_minus_dropped_receiver_action"
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
            "decoy_reference_minus_dropped_turn_zero_capture": mean(
                decoy, "reference_minus_dropped_turn_zero_capture"
            ),
            "decoy_reference_minus_dropped_receiver_action": mean(
                decoy, "reference_minus_dropped_receiver_action"
            ),
        },
    }


def _bootstrap_mean_interval(values: list[float], *, seed: int = 20260817) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires at least one independent value")
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(10_000)
    )
    return [draws[249], draws[9749]]


def _condition_effects(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    grouped: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair_index"], row["world"], row["kind"], row["condition"])].append(row)

    metrics = (
        "terminal_return",
        "target_captured",
        "target_captured_turn_zero",
        "receiver_target_action",
    )
    by_world: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for (pair_index, world, kind, condition), cell_rows in grouped.items():
        if condition not in {"generated", "dropped"}:
            continue
        if len(cell_rows) != 4:
            raise ValueError(
                f"terminal-proximal cell {pair_index}/{world}/{kind}/{condition} "
                f"requires exactly four repetitions; got {len(cell_rows)}"
            )
        if not all(bool(row["protocol_valid"]) for row in cell_rows):
            raise ValueError(f"invalid protocol in terminal-proximal cell {pair_index}/{world}")
        for metric in metrics:
            by_world[(pair_index, world)][f"{kind}_{condition}_{metric}"] = statistics.fmean(
                float(row[metric]) for row in cell_rows
            )

    by_pair: dict[int, list[dict[str, float]]] = defaultdict(list)
    for (pair_index, _), values in by_world.items():
        expected = {
            f"{kind}_{condition}_{metric}"
            for kind in ("critical", "decoy")
            for condition in ("generated", "dropped")
            for metric in metrics
        }
        if set(values) != expected:
            raise ValueError(f"incomplete terminal-proximal comparison for pair {pair_index}")
        effects = {}
        for metric in metrics:
            critical = values[f"critical_generated_{metric}"] - values[f"critical_dropped_{metric}"]
            decoy = values[f"decoy_generated_{metric}"] - values[f"decoy_dropped_{metric}"]
            effects[f"critical_{metric}"] = critical
            effects[f"decoy_{metric}"] = decoy
            effects[f"specificity_{metric}"] = critical - decoy
        by_pair[pair_index].append(effects)

    return {
        pair_index: {
            metric: statistics.fmean(row[metric] for row in worlds)
            for metric in worlds[0]
        }
        for pair_index, worlds in by_pair.items()
    }


def summarize_terminal_proximal(
    fresh_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare a short-horizon screen with the same four old-horizon samples."""
    selected = {(row["pair_index"], row["world"]) for row in fresh_rows}
    baseline = [
        row
        for row in baseline_rows
        if (row["pair_index"], row["world"]) in selected
        and row["condition"] in {"generated", "dropped"}
        and int(row["repeat"]) < 4
    ]
    fresh = _condition_effects(fresh_rows)
    old = _condition_effects(baseline)
    if set(fresh) != set(old):
        raise ValueError("fresh and old-horizon screens do not contain the same pair clusters")

    metric_names = tuple(next(iter(fresh.values())))

    def aggregate(source: dict[int, dict[str, float]]) -> dict[str, Any]:
        result = {}
        for metric in metric_names:
            values = [source[pair_index][metric] for pair_index in sorted(source)]
            result[metric] = {
                "mean": statistics.fmean(values),
                "bootstrap_95": _bootstrap_mean_interval(values),
            }
        return result

    change = {}
    for metric in metric_names:
        values = [fresh[pair_index][metric] - old[pair_index][metric] for pair_index in sorted(fresh)]
        change[metric] = {
            "mean": statistics.fmean(values),
            "bootstrap_95": _bootstrap_mean_interval(values, seed=20260818),
        }

    return {
        "version": TERMINAL_PROXIMAL_SCREEN_VERSION,
        "games": len(fresh_rows),
        "selected_worlds": len(selected),
        "independent_pair_clusters": len(fresh),
        "old_horizon": aggregate(old),
        "terminal_proximal": aggregate(fresh),
        "terminal_proximal_minus_old": change,
        "decision": {
            "protocol_valid": all(bool(row["protocol_valid"]) for row in fresh_rows),
            "terminal_specificity_improved": (
                change["specificity_terminal_return"]["mean"] > 0
            ),
            "terminal_capture_specificity_improved": (
                change["specificity_target_captured"]["mean"] > 0
            ),
        },
    }
