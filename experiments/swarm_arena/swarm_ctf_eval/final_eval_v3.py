from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from typing import Any

FINAL_EVAL_VERSION = "arena-final-eval-v3"
COMMUNICATION_CONDITIONS = ("normal", "dropped", "sender_shuffled", "delayed", "zero_budget")
SUITES = ("ordinary_ood", "critical", "decoy")
BASELINE_VARIANT = "sft_init"
CANDIDATE_VARIANT = "candidate_rl"
ACTION_ONLY_VARIANT = "action_only_rl"


def _bootstrap_mean(values: list[float], trials: int = 20_000, seed: int = 0) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty effect")
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    samples = sorted(
        statistics.mean(values[generator.randrange(len(values))] for _ in values)
        for _ in range(trials)
    )
    return [samples[int(0.025 * (trials - 1))], samples[int(0.975 * (trials - 1))]]


def _summarize_differences(
    cell_differences: dict[tuple[Any, ...], float],
    *,
    definition: str,
) -> dict[str, Any]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for key, difference in cell_differences.items():
        by_case[str(key[0])].append(difference)
    case_differences = [statistics.mean(by_case[case]) for case in sorted(by_case)]
    return {
        "definition": definition,
        "paired_cells": len(cell_differences),
        "independent_seed_units": len(case_differences),
        "mean_difference": statistics.mean(case_differences),
        "mean_difference_95": _bootstrap_mean(case_differences),
        "positive_seed_rate": statistics.mean(value > 0 for value in case_differences),
        "seed_differences_sha256": hashlib.sha256(
            json.dumps(case_differences, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _paired_field_effect(
    rows: list[dict[str, Any]],
    contrast_field: str,
    left: str,
    right: str,
    *,
    match_fields: tuple[str, ...],
    metric: str = "terminal_return",
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        level = str(row[contrast_field])
        if level not in {left, right}:
            continue
        key = tuple(row[field] for field in match_fields)
        if level in grouped[key]:
            raise ValueError(f"duplicate paired evaluation row: {key}/{level}")
        grouped[key][level] = float(row[metric])
    incomplete = [key for key, values in grouped.items() if set(values) != {left, right}]
    if incomplete:
        raise ValueError(f"incomplete paired evaluation rows: {incomplete[:3]}")
    if not grouped:
        raise ValueError(f"no paired rows for {contrast_field}: {left} versus {right}")
    differences = {
        key: values[left] - values[right]
        for key, values in grouped.items()
    }
    return _summarize_differences(
        differences,
        definition=f"{left} minus {right}, paired and bootstrapped by game seed",
    )


def _identity_vs_permutations(
    rows: list[dict[str, Any]],
    field: str,
    *,
    match_fields: tuple[str, ...],
    metric: str = "terminal_return",
    identity_value: str = "identity",
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: {"identity": [], "permuted": []}
    )
    for row in rows:
        key = tuple(row[name] for name in match_fields)
        bucket = "identity" if row[field] == identity_value else "permuted"
        grouped[key][bucket].append(float(row[metric]))
    incomplete = [
        key
        for key, values in grouped.items()
        if len(values["identity"]) != 1 or not values["permuted"]
    ]
    if incomplete:
        raise ValueError(f"incomplete identity/permutation rows for {field}: {incomplete[:3]}")
    if not grouped:
        raise ValueError(f"no identity/permutation rows for {field}")
    differences = {
        key: values["identity"][0] - statistics.mean(values["permuted"])
        for key, values in grouped.items()
    }
    return _summarize_differences(
        differences,
        definition=f"identity minus mean non-identity {field}, bootstrapped by game seed",
    )


def _select(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    description: str,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        raise ValueError(f"final evaluation has no rows for {description}")
    return selected


def summarize_final_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("final evaluation requires rows")
    required = {
        "case_id",
        "suite",
        "opponent_id",
        "opponent_revision",
        "side",
        "policy_variant",
        "policy_revision",
        "policy_assignment",
        "role_assignment",
        "option_order",
        "condition",
        "sampling_key",
        "terminal_return",
        "messages_nonempty",
        "critical_capture",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} is missing fields: {sorted(missing)}")
        if row["suite"] not in SUITES:
            raise ValueError(f"unknown suite: {row['suite']}")
        if row["condition"] not in COMMUNICATION_CONDITIONS:
            raise ValueError(f"unknown intervention: {row['condition']}")
        if row["side"] not in {"BLUE", "RED"}:
            raise ValueError(f"unknown side: {row['side']}")
        for field in (
            "case_id",
            "opponent_id",
            "opponent_revision",
            "policy_variant",
            "policy_revision",
            "sampling_key",
        ):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"row {index} requires a non-empty string {field}")

    canonical_candidate = lambda row: (
        row["policy_variant"] == CANDIDATE_VARIANT
        and row["policy_assignment"] == "identity"
        and row["role_assignment"] == "identity"
        and row["option_order"] == "canonical"
    )
    critical = _select(
        rows,
        lambda row: row["suite"] == "critical" and canonical_candidate(row),
        "canonical candidate critical suite",
    )
    decoy = _select(
        rows,
        lambda row: row["suite"] == "decoy" and canonical_candidate(row),
        "canonical candidate decoy suite",
    )
    communication_match = (
        "case_id",
        "suite",
        "opponent_id",
        "opponent_revision",
        "side",
        "policy_variant",
        "policy_revision",
        "policy_assignment",
        "role_assignment",
        "option_order",
        "sampling_key",
    )
    communication = {
        f"normal_minus_{condition}": _paired_field_effect(
            critical,
            "condition",
            "normal",
            condition,
            match_fields=communication_match,
        )
        for condition in COMMUNICATION_CONDITIONS[1:]
    }
    decoy_effect = _paired_field_effect(
        decoy,
        "condition",
        "normal",
        "dropped",
        match_fields=communication_match,
    )
    opponent_ids = sorted({str(row["opponent_id"]) for row in critical})
    communication_by_opponent = {
        opponent_id: {
            f"normal_minus_{condition}": _paired_field_effect(
                [row for row in critical if row["opponent_id"] == opponent_id],
                "condition",
                "normal",
                condition,
                match_fields=communication_match,
            )
            for condition in COMMUNICATION_CONDITIONS[1:]
        }
        for opponent_id in opponent_ids
    }

    ordinary_capability = _select(
        rows,
        lambda row: row["suite"] == "ordinary_ood"
        and row["condition"] == "normal"
        and row["policy_assignment"] == "identity"
        and row["role_assignment"] == "identity"
        and row["option_order"] == "canonical"
        and row["policy_variant"] in {CANDIDATE_VARIANT, BASELINE_VARIANT},
        "ordinary OOD candidate/baseline comparison",
    )
    capability = _paired_field_effect(
        ordinary_capability,
        "policy_variant",
        CANDIDATE_VARIANT,
        BASELINE_VARIANT,
        match_fields=(
            "case_id",
            "suite",
            "opponent_id",
            "opponent_revision",
            "side",
            "condition",
            "policy_assignment",
            "role_assignment",
            "option_order",
            "sampling_key",
        ),
    )
    capability_opponents = sorted(
        {str(row["opponent_id"]) for row in ordinary_capability}
    )
    capability_by_opponent = {
        opponent_id: _paired_field_effect(
            [row for row in ordinary_capability if row["opponent_id"] == opponent_id],
            "policy_variant",
            CANDIDATE_VARIANT,
            BASELINE_VARIANT,
            match_fields=(
                "case_id",
                "suite",
                "opponent_id",
                "opponent_revision",
                "side",
                "condition",
                "policy_assignment",
                "role_assignment",
                "option_order",
                "sampling_key",
            ),
        )
        for opponent_id in capability_opponents
    }
    action_only_comparison = _select(
        rows,
        lambda row: row["suite"] == "ordinary_ood"
        and row["condition"] == "normal"
        and row["policy_assignment"] == "identity"
        and row["role_assignment"] == "identity"
        and row["option_order"] == "canonical"
        and row["policy_variant"] in {CANDIDATE_VARIANT, ACTION_ONLY_VARIANT},
        "ordinary OOD candidate/action-only comparison",
    )
    communication_policy_effect = _paired_field_effect(
        action_only_comparison,
        "policy_variant",
        CANDIDATE_VARIANT,
        ACTION_ONLY_VARIANT,
        match_fields=(
            "case_id",
            "suite",
            "opponent_id",
            "opponent_revision",
            "side",
            "condition",
            "policy_assignment",
            "role_assignment",
            "option_order",
            "sampling_key",
        ),
    )

    candidate_ordinary = [
        row for row in ordinary_capability if row["policy_variant"] == CANDIDATE_VARIANT
    ]
    opponent_returns = {
        opponent_id: statistics.mean(
            float(row["terminal_return"])
            for row in candidate_ordinary
            if row["opponent_id"] == opponent_id
        )
        for opponent_id in sorted({str(row["opponent_id"]) for row in candidate_ordinary})
    }
    specialization_rows = _select(
        rows,
        lambda row: row["suite"] in {"critical", "ordinary_ood"}
        and row["condition"] == "normal"
        and row["policy_variant"] == CANDIDATE_VARIANT
        and row["role_assignment"] == "identity"
        and row["option_order"] == "canonical",
        "adapter specialization permutations",
    )
    adapter_shuffle = _identity_vs_permutations(
        specialization_rows,
        "policy_assignment",
        match_fields=(
            "case_id",
            "suite",
            "opponent_id",
            "opponent_revision",
            "side",
            "policy_variant",
            "policy_revision",
            "condition",
            "role_assignment",
            "option_order",
            "sampling_key",
        ),
    )
    role_rows = _select(
        rows,
        lambda row: row["suite"] in {"critical", "ordinary_ood"}
        and row["condition"] == "normal"
        and row["policy_variant"] == CANDIDATE_VARIANT
        and row["policy_assignment"] == "identity"
        and row["option_order"] == "canonical",
        "role-label permutations",
    )
    role_label_effect = _identity_vs_permutations(
        role_rows,
        "role_assignment",
        match_fields=(
            "case_id",
            "suite",
            "opponent_id",
            "opponent_revision",
            "side",
            "policy_variant",
            "policy_revision",
            "condition",
            "policy_assignment",
            "option_order",
            "sampling_key",
        ),
    )
    option_rows = _select(
        rows,
        lambda row: row["suite"] in {"critical", "ordinary_ood"}
        and row["condition"] == "normal"
        and row["policy_variant"] == CANDIDATE_VARIANT
        and row["policy_assignment"] == "identity"
        and row["role_assignment"] == "identity",
        "legal-action option permutations",
    )
    option_order_effect = _identity_vs_permutations(
        option_rows,
        "option_order",
        match_fields=(
            "case_id",
            "suite",
            "opponent_id",
            "opponent_revision",
            "side",
            "policy_variant",
            "policy_revision",
            "condition",
            "policy_assignment",
            "role_assignment",
            "sampling_key",
        ),
        identity_value="canonical",
    )

    normal_critical = [row for row in critical if row["condition"] == "normal"]
    normal_decoy = [row for row in decoy if row["condition"] == "normal"]
    pooled_positive = all(
        effect["mean_difference_95"][0] > 0 for effect in communication.values()
    )
    each_opponent_positive = all(
        effect["mean_difference_95"][0] > 0
        for effects in communication_by_opponent.values()
        for effect in effects.values()
    )
    side_swap_complete = all(
        {str(row["side"]) for row in suite_rows} == {"BLUE", "RED"}
        for suite_rows in (critical, decoy, ordinary_capability)
    )
    return {
        "version": FINAL_EVAL_VERSION,
        "rows": len(rows),
        "independent_unit": "case_id/game seed",
        "capability_rl_minus_sft": capability,
        "capability_rl_minus_sft_by_opponent": capability_by_opponent,
        "candidate_minus_action_only": communication_policy_effect,
        "communication_effects": communication,
        "matched_decoy_normal_minus_dropped": decoy_effect,
        "communication_effects_by_opponent": communication_by_opponent,
        "adapter_identity_minus_shuffle": adapter_shuffle,
        "role_identity_minus_label_permutation": role_label_effect,
        "canonical_minus_option_permutation": option_order_effect,
        "critical_capture_rate": statistics.mean(
            bool(row["critical_capture"]) for row in normal_critical
        ),
        "critical_nonempty_message_rate": statistics.mean(
            int(row["messages_nonempty"]) > 0 for row in normal_critical
        ),
        "decoy_nonempty_message_rate": statistics.mean(
            int(row["messages_nonempty"]) > 0 for row in normal_decoy
        ),
        "ordinary_ood_mean_return_by_opponent": opponent_returns,
        "claim_checks": {
            "capability_interval_positive": capability["mean_difference_95"][0] > 0,
            "capability_interval_positive_against_every_opponent": all(
                effect["mean_difference_95"][0] > 0
                for effect in capability_by_opponent.values()
            ),
            "communication_all_pooled_intervals_positive": pooled_positive,
            "matched_decoy_effect_includes_zero": (
                decoy_effect["mean_difference_95"][0] <= 0
                <= decoy_effect["mean_difference_95"][1]
            ),
            "communication_positive_against_every_opponent": each_opponent_positive,
            "opponent_pool_complete": len(opponent_returns) >= 3,
            "side_swap_complete": side_swap_complete,
            "specialization_interval_positive": adapter_shuffle["mean_difference_95"][0] > 0,
            "role_label_equivalence_within_0_02": (
                role_label_effect["mean_difference_95"][0] >= -0.02
                and role_label_effect["mean_difference_95"][1] <= 0.02
            ),
            "option_order_equivalence_within_0_02": (
                option_order_effect["mean_difference_95"][0] >= -0.02
                and option_order_effect["mean_difference_95"][1] <= 0.02
            ),
            "communication_claim_passed": pooled_positive
            and each_opponent_positive
            and len(opponent_returns) >= 3
            and side_swap_complete
            and decoy_effect["mean_difference_95"][0] <= 0 <= decoy_effect["mean_difference_95"][1],
        },
        "claim_boundary": (
            "Return improvement alone is capability learning. Communication requires positive "
            "paired normal-minus-dropped, shuffled, delayed, and zero-budget effects across "
            "unseen opponents and both sides. Adapter shuffling is a separate specialization test."
        ),
    }
