from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .progress_eval_v4 import _bootstrap, _digest
from .progress_eval_v5 import _aggregate_independent_units, _endpoint, _unit_effects

SEMANTIC_HOLDOUT_VERSION = "arena-rl-v10-clean-semantic-holdout-v1"


def _rate_endpoint(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    condition: str,
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["suite"] == suite
            and row["policy_variant"] == variant
            and row["condition"] == condition
            and row.get(field) is not None
        ):
            grouped[str(row["independent_id"])].append(float(row[field]))
    if not grouped:
        raise ValueError(f"no {suite}/{variant}/{condition} values for {field}")
    values = [statistics.mean(grouped[key]) for key in sorted(grouped)]
    return {
        "definition": f"{variant} {suite} {condition} {field}",
        "independent_units": len(values),
        "mean": statistics.mean(values),
        "mean_95": _bootstrap(values),
        "values_sha256": _digest(values),
    }


def _semantic_effects(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    value_field: str = "terminal_return",
) -> dict[tuple[str, str, str, str], float]:
    return _unit_effects(
        rows,
        suite=suite,
        variant=variant,
        left="normal",
        right="target_swapped",
        value_field=value_field,
    )


def summarize_semantic_holdout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_critical = _semantic_effects(
        rows,
        suite="handoff_critical",
        variant="candidate_rl",
    )
    baseline_critical = _semantic_effects(
        rows,
        suite="handoff_critical",
        variant="sft_init",
    )
    candidate_decoy = _semantic_effects(
        rows,
        suite="handoff_decoy",
        variant="candidate_rl",
    )
    candidate_receiver_action = _semantic_effects(
        rows,
        suite="handoff_critical",
        variant="candidate_rl",
        value_field="receiver_target_action",
    )
    if set(candidate_critical) != set(baseline_critical):
        raise ValueError("candidate and SFT semantic units do not match")
    if set(candidate_critical) != set(candidate_decoy):
        raise ValueError("critical and decoy semantic units do not match")
    if set(candidate_critical) != set(candidate_receiver_action):
        raise ValueError("return and receiver-action semantic units do not match")

    candidate_values = _aggregate_independent_units(candidate_critical)
    baseline_values = _aggregate_independent_units(baseline_critical)
    rl_specific_values = _aggregate_independent_units(
        {key: candidate_critical[key] - baseline_critical[key] for key in candidate_critical}
    )
    specificity_values = _aggregate_independent_units(
        {key: candidate_critical[key] - candidate_decoy[key] for key in candidate_critical}
    )
    receiver_action_values = _aggregate_independent_units(candidate_receiver_action)
    opponents = sorted({key[1] for key in candidate_critical})
    by_opponent = {
        opponent: _endpoint(
            _aggregate_independent_units(
                {key: value for key, value in candidate_critical.items() if key[1] == opponent}
            ),
            f"candidate critical normal minus receiver-only target-swapped return vs {opponent}",
        )
        for opponent in opponents
    }

    candidate_endpoint = _endpoint(
        candidate_values,
        "candidate critical normal minus receiver-only target-swapped terminal return (ITT)",
    )
    baseline_endpoint = _endpoint(
        baseline_values,
        "SFT critical normal minus receiver-only target-swapped terminal return (ITT)",
    )
    rl_specific_endpoint = _endpoint(
        rl_specific_values,
        "candidate-minus-SFT receiver-only target-swap return sensitivity",
    )
    specificity_endpoint = _endpoint(
        specificity_values,
        "candidate critical-minus-decoy receiver-only target-swap sensitivity",
    )
    receiver_action_endpoint = _endpoint(
        receiver_action_values,
        "candidate critical normal-minus-target-swapped receiver target-action rate",
    )
    candidate_eligibility = _rate_endpoint(
        rows,
        suite="handoff_critical",
        variant="candidate_rl",
        condition="target_swapped",
        field="target_swap_eligible",
    )
    baseline_eligibility = _rate_endpoint(
        rows,
        suite="handoff_critical",
        variant="sft_init",
        condition="target_swapped",
        field="target_swap_eligible",
    )

    return {
        "version": SEMANTIC_HOLDOUT_VERSION,
        "independent_unit": "two-world latent handoff bundle",
        "estimand": (
            "intention-to-treat: ineligible sender messages remain zero-strength "
            "interventions rather than being retried or discarded"
        ),
        "candidate_critical_normal_minus_target_swapped": candidate_endpoint,
        "sft_critical_normal_minus_target_swapped": baseline_endpoint,
        "rl_specific_semantic_lift": rl_specific_endpoint,
        "critical_minus_decoy_semantic_specificity": specificity_endpoint,
        "receiver_target_action_gap": receiver_action_endpoint,
        "candidate_target_swap_eligibility": candidate_eligibility,
        "sft_target_swap_eligibility": baseline_eligibility,
        "candidate_normal_receiver_target_action_rate": _rate_endpoint(
            rows,
            suite="handoff_critical",
            variant="candidate_rl",
            condition="normal",
            field="receiver_target_action",
        ),
        "candidate_swapped_receiver_target_action_rate": _rate_endpoint(
            rows,
            suite="handoff_critical",
            variant="candidate_rl",
            condition="target_swapped",
            field="receiver_target_action",
        ),
        "candidate_normal_sender_target_fact_rate": _rate_endpoint(
            rows,
            suite="handoff_critical",
            variant="candidate_rl",
            condition="normal",
            field="sender_target_fact",
        ),
        "by_opponent": by_opponent,
        "claim_checks": {
            "candidate_semantic_interval_positive": candidate_endpoint["mean_difference_95"][0] > 0,
            "rl_specific_semantic_interval_positive": rl_specific_endpoint["mean_difference_95"][0] > 0,
            "critical_specificity_interval_positive": specificity_endpoint["mean_difference_95"][0] > 0,
            "receiver_action_gap_interval_positive": receiver_action_endpoint["mean_difference_95"][0] > 0,
            "effect_positive_against_every_opponent": all(
                endpoint["mean_difference"] > 0 for endpoint in by_opponent.values()
            ),
            "candidate_swap_eligibility_at_least_80pct": (candidate_eligibility["mean"] >= 0.8),
        },
    }
