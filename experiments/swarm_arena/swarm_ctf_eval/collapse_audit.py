from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

COLLAPSE_AUDIT_VERSION = "arena-rl-collapse-audit-v1"


def _paired_checkpoint_effect(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
) -> float:
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["condition"] != "normal" or row["checkpoint_id"] not in {candidate, baseline}:
            continue
        key = (str(row["game_id"]), str(row["opponent_id"]), str(row["side"]))
        grouped[key][str(row["checkpoint_id"])] = float(row["terminal_return"])
    complete = [values for values in grouped.values() if set(values) == {candidate, baseline}]
    if not complete:
        raise ValueError("collapse audit has no paired candidate/baseline games")
    return statistics.mean(values[candidate] - values[baseline] for values in complete)


def _paired_condition_effect(rows: list[dict[str, Any]], checkpoint: str) -> float:
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["checkpoint_id"] != checkpoint or row["condition"] not in {"normal", "dropped"}:
            continue
        key = (str(row["game_id"]), str(row["opponent_id"]), str(row["side"]))
        grouped[key][str(row["condition"])] = float(row["terminal_return"])
    complete = [values for values in grouped.values() if set(values) == {"normal", "dropped"}]
    if not complete:
        raise ValueError("collapse audit has no paired normal/dropped games")
    return statistics.mean(values["normal"] - values["dropped"] for values in complete)


def _concentration(values: list[str]) -> float | None:
    if not values:
        return None
    return max(Counter(values).values()) / len(values)


def audit_training_collapse(
    rows: list[dict[str, Any]],
    *,
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    speaking_extreme: float = 0.98,
    concentration_limit: float = 0.95,
    kl_mean_limit: float = 0.08,
    kl_p99_limit: float = 0.30,
) -> dict[str, Any]:
    required = {
        "checkpoint_id",
        "game_id",
        "agent_id",
        "policy_id",
        "opponent_id",
        "side",
        "condition",
        "terminal_return",
        "message_nonempty",
        "message_target",
        "action_signature",
        "kl",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"collapse row {index} is missing fields: {sorted(missing)}")
        if not math.isfinite(float(row["kl"])) or float(row["kl"]) < 0:
            raise ValueError(f"collapse row {index} has invalid KL")

    candidate_rows = [
        row
        for row in rows
        if row["checkpoint_id"] == candidate_checkpoint and row["condition"] == "normal"
    ]
    if not candidate_rows:
        raise ValueError("collapse audit has no candidate normal rows")
    policy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        policy_rows[str(row["policy_id"])].append(row)

    per_policy = {}
    for policy_id, owned in sorted(policy_rows.items()):
        speaking_rate = statistics.mean(bool(row["message_nonempty"]) for row in owned)
        targets = [
            str(row["message_target"])
            for row in owned
            if row["message_nonempty"] and row["message_target"] is not None
        ]
        actions = [str(row["action_signature"]) for row in owned]
        kls = sorted(float(row["kl"]) for row in owned)
        p99 = kls[min(len(kls) - 1, math.ceil(0.99 * len(kls)) - 1)]
        target_concentration = _concentration(targets)
        action_concentration = _concentration(actions)
        per_policy[policy_id] = {
            "decisions": len(owned),
            "speaking_rate": speaking_rate,
            "always_speaking": speaking_rate >= speaking_extreme,
            "never_speaking": speaking_rate <= 1 - speaking_extreme,
            "message_target_concentration": target_concentration,
            "repeated_target_collapse": len(targets) >= 20
            and target_concentration is not None
            and target_concentration >= concentration_limit,
            "action_concentration": action_concentration,
            "action_collapse": len(actions) >= 20
            and action_concentration is not None
            and action_concentration >= concentration_limit,
            "kl_mean": statistics.mean(kls),
            "kl_p99": p99,
            "excessive_kl": statistics.mean(kls) > kl_mean_limit or p99 > kl_p99_limit,
        }

    opponent_returns = {
        opponent: statistics.mean(
            float(row["terminal_return"])
            for row in candidate_rows
            if str(row["opponent_id"]) == opponent
        )
        for opponent in sorted({str(row["opponent_id"]) for row in candidate_rows})
    }
    return_gain = _paired_checkpoint_effect(rows, candidate_checkpoint, baseline_checkpoint)
    message_effect = _paired_condition_effect(rows, candidate_checkpoint)
    flags = {
        "always_or_never_speaking": any(
            report["always_speaking"] or report["never_speaking"]
            for report in per_policy.values()
        ),
        "repeated_target_collapse": any(
            report["repeated_target_collapse"] for report in per_policy.values()
        ),
        "action_collapse": any(report["action_collapse"] for report in per_policy.values()),
        "excessive_kl": any(report["excessive_kl"] for report in per_policy.values()),
        "performance_against_only_one_opponent": len(opponent_returns) >= 3
        and sum(value > 0 for value in opponent_returns.values()) <= 1,
        "return_gain_without_message_gain": return_gain > 0.01 and message_effect <= 0.005,
    }
    return {
        "version": COLLAPSE_AUDIT_VERSION,
        "candidate_checkpoint": candidate_checkpoint,
        "baseline_checkpoint": baseline_checkpoint,
        "per_policy": per_policy,
        "mean_return_gain": return_gain,
        "mean_normal_minus_dropped": message_effect,
        "mean_return_by_opponent": opponent_returns,
        "flags": flags,
        "passed": not any(flags.values()),
        "scope": "diagnostic stop/inspect gates; never reward terms",
    }
