from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from typing import Any

PROGRESS_EVAL_VERSION = "arena-rl-progress-eval-v4"
PROGRESS_SUITES = (
    "ordinary_legacy",
    "ordinary_hard",
    "handoff_critical",
    "handoff_decoy",
)
COMMUNICATION_CONDITIONS = (
    "normal",
    "dropped",
    "sender_shuffled",
    "delayed",
    "zero_budget",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_ordinary_manifest(
    *,
    count: int,
    seed_start: int,
    sizes: tuple[int, ...],
    horizons: tuple[int, ...],
) -> dict:
    if count < 1 or not sizes or not horizons:
        raise ValueError("ordinary manifest requires positive cases, sizes, and horizons")
    cases = [
        {
            "case_id": f"ordinary-hard-{index:03d}",
            "seed": seed_start + 211 * index,
            "size": sizes[index % len(sizes)],
            "horizon": horizons[index % len(horizons)],
        }
        for index in range(count)
    ]
    body = {
        "version": PROGRESS_EVAL_VERSION,
        "seed_start": seed_start,
        "sizes": list(sizes),
        "horizons": list(horizons),
        "case_count": count,
        "cases": cases,
    }
    body["sha256"] = _digest(body)
    return body


def _bootstrap(values: list[float], *, trials: int = 20_000, seed: int = 0) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty endpoint")
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    samples = sorted(
        statistics.mean(values[generator.randrange(len(values))] for _ in values)
        for _ in range(trials)
    )
    return [samples[int(0.025 * (trials - 1))], samples[int(0.975 * (trials - 1))]]


def _paired_effect(
    rows: list[dict[str, Any]],
    *,
    field: str,
    left: str,
    right: str,
    match_fields: tuple[str, ...],
    definition: str,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, tuple[str, float]]] = defaultdict(dict)
    for row in rows:
        level = str(row[field])
        if level not in {left, right}:
            continue
        key = tuple(row[name] for name in match_fields)
        if level in grouped[key]:
            raise ValueError(f"duplicate paired row: {key}/{level}")
        grouped[key][level] = (
            str(row["independent_id"]),
            float(row["terminal_return"]),
        )
    incomplete = [key for key, values in grouped.items() if set(values) != {left, right}]
    if incomplete:
        raise ValueError(f"incomplete paired endpoint: {incomplete[:3]}")
    if not grouped:
        raise ValueError(f"no rows for endpoint: {definition}")
    per_unit: dict[str, list[float]] = defaultdict(list)
    for values in grouped.values():
        left_unit, left_value = values[left]
        right_unit, right_value = values[right]
        if left_unit != right_unit:
            raise ValueError("paired rows disagree on their independent unit")
        per_unit[left_unit].append(left_value - right_value)
    unit_effects = [statistics.mean(per_unit[key]) for key in sorted(per_unit)]
    return {
        "definition": definition,
        "paired_cells": len(grouped),
        "independent_units": len(unit_effects),
        "mean_difference": statistics.mean(unit_effects),
        "mean_difference_95": _bootstrap(unit_effects),
        "positive_unit_rate": statistics.mean(value > 0 for value in unit_effects),
        "unit_effects_sha256": _digest(unit_effects),
    }


def _select(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str | None = None,
    condition: str | None = None,
    opponent: str | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["suite"] == suite
        and (variant is None or row["policy_variant"] == variant)
        and (condition is None or row["condition"] == condition)
        and (opponent is None or row["opponent_id"] == opponent)
    ]
    if not selected:
        raise ValueError(
            f"no rows for suite={suite}, variant={variant}, condition={condition}, opponent={opponent}"
        )
    return selected


def summarize_progress_eval(
    rows: list[dict[str, Any]],
    *,
    intervention_conditions: tuple[str, ...] = COMMUNICATION_CONDITIONS[1:],
) -> dict[str, Any]:
    required = {
        "independent_id",
        "case_id",
        "suite",
        "opponent_id",
        "opponent_revision",
        "side",
        "policy_variant",
        "policy_revision",
        "condition",
        "sampling_key",
        "terminal_return",
        "messages_nonempty",
        "broadcast_protocol_rate",
        "broadcast_grounded_rate",
        "action_protocol_rate",
    }
    if not rows:
        raise ValueError("progress evaluation requires rows")
    if (
        not intervention_conditions
        or len(intervention_conditions) != len(set(intervention_conditions))
        or any(
            condition not in COMMUNICATION_CONDITIONS[1:]
            for condition in intervention_conditions
        )
    ):
        raise ValueError("progress evaluation requires unique, known message interventions")
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} is missing fields: {sorted(missing)}")
        if row["suite"] not in PROGRESS_SUITES:
            raise ValueError(f"unknown progress suite: {row['suite']}")
        if row["condition"] not in COMMUNICATION_CONDITIONS:
            raise ValueError(f"unknown communication condition: {row['condition']}")
        if row["side"] not in {"BLUE", "RED"}:
            raise ValueError(f"unknown focal side: {row['side']}")

    ordinary_match = (
        "case_id",
        "suite",
        "opponent_id",
        "opponent_revision",
        "side",
        "condition",
        "sampling_key",
    )
    capability = {}
    for suite in ("ordinary_legacy", "ordinary_hard"):
        ordinary = _select(rows, suite=suite, condition="normal")
        capability[suite] = _paired_effect(
            ordinary,
            field="policy_variant",
            left="candidate_rl",
            right="sft_init",
            match_fields=ordinary_match,
            definition="candidate RL minus SFT, paired by map/opponent/side",
        )

    communication_match = (
        "case_id",
        "suite",
        "opponent_id",
        "opponent_revision",
        "side",
        "policy_variant",
        "policy_revision",
        "sampling_key",
    )
    critical = _select(rows, suite="handoff_critical", variant="candidate_rl")
    communication = {
        f"normal_minus_{condition}": _paired_effect(
            critical,
            field="condition",
            left="normal",
            right=condition,
            match_fields=communication_match,
            definition=f"critical handoff normal minus {condition}",
        )
        for condition in intervention_conditions
    }
    decoy = _select(rows, suite="handoff_decoy", variant="candidate_rl")
    decoy_effect = _paired_effect(
        decoy,
        field="condition",
        left="normal",
        right="dropped",
        match_fields=communication_match,
        definition="matched-decoy normal minus dropped",
    )
    opponents = sorted({str(row["opponent_id"]) for row in critical})
    communication_by_opponent = {
        opponent_id: {
            f"normal_minus_{condition}": _paired_effect(
                _select(
                    rows,
                    suite="handoff_critical",
                    variant="candidate_rl",
                    opponent=opponent_id,
                ),
                field="condition",
                left="normal",
                right=condition,
                match_fields=communication_match,
                definition=f"critical handoff normal minus {condition} vs {opponent_id}",
            )
            for condition in intervention_conditions
        }
        for opponent_id in opponents
    }
    candidate_rows = [row for row in rows if row["policy_variant"] == "candidate_rl"]
    protocol = {}
    protocol_denominators = {}
    for field in (
        "broadcast_protocol_rate",
        "broadcast_grounded_rate",
        "action_protocol_rate",
    ):
        defined = [float(row[field]) for row in candidate_rows if row[field] is not None]
        protocol[field] = statistics.mean(defined) if defined else None
        protocol_denominators[field] = {
            "defined_rows": len(defined),
            "undefined_rows": len(candidate_rows) - len(defined),
        }
    tested_communication_positive = all(
        endpoint["mean_difference_95"][0] > 0 for endpoint in communication.values()
    )
    opponent_communication = all(
        endpoint["mean_difference"] > 0
        for effects in communication_by_opponent.values()
        for endpoint in effects.values()
    )
    decoy_includes_zero = (
        decoy_effect["mean_difference_95"][0]
        <= 0
        <= decoy_effect["mean_difference_95"][1]
    )
    full_intervention_matrix = set(intervention_conditions) == set(
        COMMUNICATION_CONDITIONS[1:]
    )
    return {
        "version": PROGRESS_EVAL_VERSION,
        "rows": len(rows),
        "independent_unit": "game seed for ordinary cases; two-world latent bundle for handoffs",
        "capability_rl_minus_sft": capability,
        "communication_effects": communication,
        "communication_effects_by_opponent": communication_by_opponent,
        "matched_decoy_normal_minus_dropped": decoy_effect,
        "candidate_protocol": protocol,
        "candidate_protocol_denominators": protocol_denominators,
        "claim_checks": {
            "legacy_capability_interval_positive": capability["ordinary_legacy"][
                "mean_difference_95"
            ][0]
            > 0,
            "hard_capability_interval_positive": capability["ordinary_hard"][
                "mean_difference_95"
            ][0]
            > 0,
            "full_intervention_matrix": full_intervention_matrix,
            "all_tested_critical_intervention_intervals_positive": (
                tested_communication_positive
            ),
            "critical_effect_positive_against_every_opponent": opponent_communication,
            "matched_decoy_interval_includes_zero": decoy_includes_zero,
            "communication_claim_passed": full_intervention_matrix
            and tested_communication_positive
            and opponent_communication
            and decoy_includes_zero,
        },
        "claim_boundary": (
            "Ordinary return measures game capability. Communication requires positive causal "
            "message interventions on two-world handoffs and a null matched-decoy effect."
        ),
    }
