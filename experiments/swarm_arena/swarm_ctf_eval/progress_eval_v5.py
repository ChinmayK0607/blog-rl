from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .progress_eval_v4 import _bootstrap, _digest, summarize_progress_eval

PROGRESS_EVAL_V5_VERSION = "arena-rl-progress-eval-v5-rl-specific-communication"


def _unit_effects(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    left: str = "normal",
    right: str = "dropped",
    value_field: str = "terminal_return",
) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["suite"] != suite or row["policy_variant"] != variant:
            continue
        condition = str(row["condition"])
        if condition not in {left, right}:
            continue
        key = (
            str(row["independent_id"]),
            str(row["opponent_id"]),
            str(row["opponent_revision"]),
            str(row["side"]),
        )
        grouped[key][condition].append(float(row[value_field]))
    if not grouped:
        raise ValueError(f"no {suite}/{variant} rows for {left} minus {right}")
    incomplete = [key for key, values in grouped.items() if set(values) != {left, right}]
    if incomplete:
        raise ValueError(f"incomplete unit-level communication endpoint: {incomplete[:3]}")
    return {key: statistics.mean(values[left]) - statistics.mean(values[right]) for key, values in grouped.items()}


def _endpoint(values: list[float], definition: str) -> dict[str, Any]:
    if not values:
        raise ValueError(f"empty endpoint: {definition}")
    return {
        "definition": definition,
        "independent_units": len(values),
        "mean_difference": statistics.mean(values),
        "mean_difference_95": _bootstrap(values),
        "positive_unit_rate": statistics.mean(value > 0 for value in values),
        "unit_effects_sha256": _digest(values),
    }


def _aggregate_independent_units(
    effects: dict[tuple[str, str, str, str], float],
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for (independent_id, _opponent_id, _opponent_revision, _side), value in effects.items():
        grouped[independent_id].append(value)
    return [statistics.mean(grouped[key]) for key in sorted(grouped)]


def _variant_unit_effects(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    condition: str = "normal",
) -> list[float]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["suite"] != suite or row["condition"] != condition:
            continue
        key = (
            str(row["independent_id"]),
            str(row["opponent_id"]),
            str(row["opponent_revision"]),
            str(row["side"]),
        )
        grouped[key][str(row["policy_variant"])].append(float(row["terminal_return"]))
    expected = {"candidate_rl", "sft_init"}
    incomplete = [key for key, values in grouped.items() if set(values) != expected]
    if incomplete or not grouped:
        raise ValueError(f"incomplete {suite} candidate/SFT endpoint: {incomplete[:3]}")
    return _aggregate_independent_units(
        {
            key: statistics.mean(values["candidate_rl"]) - statistics.mean(values["sft_init"])
            for key, values in grouped.items()
        }
    )


def _field_rate_endpoint(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    field: str,
    condition: str = "normal",
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["suite"] == suite
            and row["policy_variant"] == variant
            and row["condition"] == condition
        ):
            grouped[str(row["independent_id"])].append(float(row[field]))
    if not grouped:
        raise ValueError(f"no {suite}/{variant}/{condition} rows for {field}")
    return _endpoint(
        [statistics.mean(grouped[key]) for key in sorted(grouped)],
        f"{variant} {suite} {condition} {field} rate",
    )


def summarize_rl_specific_progress_eval(
    rows: list[dict[str, Any]],
    *,
    intervention_conditions: tuple[str, ...] = ("dropped",),
) -> dict[str, Any]:
    """Add RL-specific and critical-vs-decoy causal contrasts to the v4 summary."""
    summary = summarize_progress_eval(
        rows,
        intervention_conditions=intervention_conditions,
    )
    candidate_critical = _unit_effects(rows, suite="handoff_critical", variant="candidate_rl")
    baseline_critical = _unit_effects(rows, suite="handoff_critical", variant="sft_init")
    candidate_decoy = _unit_effects(rows, suite="handoff_decoy", variant="candidate_rl")
    candidate_capture = _unit_effects(
        rows,
        suite="handoff_critical",
        variant="candidate_rl",
        value_field="critical_capture",
    )
    baseline_capture = _unit_effects(
        rows,
        suite="handoff_critical",
        variant="sft_init",
        value_field="critical_capture",
    )
    if set(candidate_critical) != set(baseline_critical):
        raise ValueError("candidate and SFT critical communication units do not match")
    if set(candidate_critical) != set(candidate_decoy):
        raise ValueError("critical and matched-decoy communication units do not match")

    baseline_values = _aggregate_independent_units(baseline_critical)
    rl_specific = _aggregate_independent_units(
        {key: candidate_critical[key] - baseline_critical[key] for key in candidate_critical}
    )
    specificity = _aggregate_independent_units(
        {key: candidate_critical[key] - candidate_decoy[key] for key in candidate_critical}
    )
    legacy_capability = _variant_unit_effects(rows, suite="ordinary_legacy")
    hard_capability = _variant_unit_effects(rows, suite="ordinary_hard")
    handoff_capability = _variant_unit_effects(rows, suite="handoff_critical")
    summary["version"] = PROGRESS_EVAL_V5_VERSION
    summary["sft_critical_normal_minus_dropped"] = _endpoint(
        baseline_values,
        "SFT critical normal minus dropped",
    )
    summary["rl_specific_communication_lift"] = _endpoint(
        rl_specific,
        "(RL critical normal-minus-dropped) minus (SFT critical normal-minus-dropped)",
    )
    summary["critical_minus_decoy_specificity"] = _endpoint(
        specificity,
        "RL critical normal-minus-dropped minus matched-decoy normal-minus-dropped",
    )
    summary["communication_mechanism"] = {
        "candidate_sender_target_fact_rate": _field_rate_endpoint(
            rows,
            suite="handoff_critical",
            variant="candidate_rl",
            field="sender_target_fact",
        ),
        "sft_sender_target_fact_rate": _field_rate_endpoint(
            rows,
            suite="handoff_critical",
            variant="sft_init",
            field="sender_target_fact",
        ),
        "candidate_capture_normal_minus_dropped": _endpoint(
            _aggregate_independent_units(candidate_capture),
            "candidate critical capture rate under normal minus dropped messaging",
        ),
        "sft_capture_normal_minus_dropped": _endpoint(
            _aggregate_independent_units(baseline_capture),
            "SFT critical capture rate under normal minus dropped messaging",
        ),
        "rl_specific_capture_lift": _endpoint(
            _aggregate_independent_units(
                {key: candidate_capture[key] - baseline_capture[key] for key in candidate_capture}
            ),
            "candidate-minus-SFT lift in critical capture dependence on messages",
        ),
    }
    summary["handoff_capability_rl_minus_sft"] = _endpoint(
        handoff_capability,
        "candidate RL minus SFT return on critical handoffs under normal messaging",
    )
    summary["overall_gameplay_rl_minus_sft"] = _endpoint(
        legacy_capability + hard_capability + handoff_capability,
        "candidate RL minus SFT return over equally represented legacy, hard, and critical-handoff units",
    )
    summary["claim_checks"]["rl_specific_lift_interval_positive"] = (
        summary["rl_specific_communication_lift"]["mean_difference_95"][0] > 0
    )
    summary["claim_checks"]["critical_minus_decoy_interval_positive"] = (
        summary["critical_minus_decoy_specificity"]["mean_difference_95"][0] > 0
    )
    summary["claim_boundary"] = (
        "A communication-learning claim requires more message sensitivity than the SFT "
        "initializer and more sensitivity on critical than matched-decoy cases. Absolute "
        "candidate message sensitivity alone is insufficient."
    )
    return summary
