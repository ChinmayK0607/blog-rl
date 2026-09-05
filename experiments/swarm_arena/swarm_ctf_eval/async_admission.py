from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .prime_rl_bridge import RolloutDecision, validate_rollout_decision


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _exp_or_infinity(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class PolicySnapshot:
    policy_id: str
    revision: str
    adapter_sha256: str
    update_index: int
    trainable: bool

    def validate(self) -> None:
        if not self.policy_id or not self.revision:
            raise ValueError("policy snapshots require immutable IDs and revisions")
        if not _is_sha256(self.adapter_sha256):
            raise ValueError(f"invalid adapter SHA-256 for policy {self.policy_id}")
        if self.update_index < 0:
            raise ValueError(f"negative update index for policy {self.policy_id}")


@dataclass(frozen=True)
class AsyncRolloutHeader:
    rollout_id: str
    backend_name: str
    backend_version: str
    kernel_config_sha256: str
    calibration_sha256: str
    policy_snapshots: tuple[PolicySnapshot, ...]

    def validate(self) -> None:
        if not self.rollout_id or not self.backend_name or not self.backend_version:
            raise ValueError("async rollout headers require rollout and backend identities")
        if not _is_sha256(self.kernel_config_sha256):
            raise ValueError("invalid rollout kernel-configuration SHA-256")
        if not _is_sha256(self.calibration_sha256):
            raise ValueError("invalid rollout calibration SHA-256")
        if not self.policy_snapshots:
            raise ValueError("async rollout headers require policy snapshots")
        policy_ids = [snapshot.policy_id for snapshot in self.policy_snapshots]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("async rollout header contains duplicate policy snapshots")
        for snapshot in self.policy_snapshots:
            snapshot.validate()


@dataclass(frozen=True)
class AsyncAdmissionLimits:
    max_policy_lag: int
    max_mean_abs_log_ratio: float | None
    max_mean_mismatch_kl: float | None
    max_p99_abs_log_ratio: float | None
    max_symmetric_importance_ratio: float | None
    max_p99_probability_error: float | None
    probability_tail_threshold: float
    max_probability_tail_fraction: float | None

    def validate(self) -> None:
        if self.max_policy_lag < 0:
            raise ValueError("maximum policy lag cannot be negative")
        optional_positive = {
            "max_mean_abs_log_ratio": self.max_mean_abs_log_ratio,
            "max_mean_mismatch_kl": self.max_mean_mismatch_kl,
            "max_p99_abs_log_ratio": self.max_p99_abs_log_ratio,
            "max_symmetric_importance_ratio": self.max_symmetric_importance_ratio,
            "max_p99_probability_error": self.max_p99_probability_error,
        }
        if (
            any(value is not None and (not math.isfinite(value) or value <= 0) for value in optional_positive.values())
            or not math.isfinite(self.probability_tail_threshold)
            or self.probability_tail_threshold <= 0
        ):
            raise ValueError("async numerical limits must be finite and positive")
        if self.max_symmetric_importance_ratio is not None and self.max_symmetric_importance_ratio < 1:
            raise ValueError("symmetric importance-ratio limit must be at least one")
        if self.max_probability_tail_fraction is not None and (
            not math.isfinite(self.max_probability_tail_fraction) or not 0 <= self.max_probability_tail_fraction <= 1
        ):
            raise ValueError("probability-tail fraction limit must be in [0, 1]")

    @property
    def sha256(self) -> str:
        self.validate()
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class AsyncAdmissionResult:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


def admit_async_rollout(
    header: AsyncRolloutHeader,
    decisions: tuple[RolloutDecision, ...],
    current_snapshots: tuple[PolicySnapshot, ...],
    trainer_logprobs: dict[str, tuple[float, ...]],
    *,
    trainable_decision_ids: frozenset[str] | None = None,
    trainable_branch: str = "actual",
    allowed_backend_calibrations: frozenset[tuple[str, str, str, str]],
    allowed_constraint_sha256s: frozenset[str],
    limits: AsyncAdmissionLimits,
) -> AsyncAdmissionResult:
    """Admit a bounded-staleness rollout using current-policy token log-probabilities.

    Malformed or incomplete evidence raises ``ValueError``. Well-formed evidence
    outside a precommitted staleness or numerical envelope returns a rejection.
    The function never clips or silently repairs a rollout.
    """
    header.validate()
    limits.validate()
    if trainable_branch not in {"actual", "message_swap"}:
        raise ValueError("async admission trainable branch must be actual or message_swap")
    if not decisions:
        raise ValueError("async admission requires rollout decisions")
    if not allowed_backend_calibrations or not allowed_constraint_sha256s:
        raise ValueError("async admission requires non-empty calibration and constraint allowlists")
    for record in allowed_backend_calibrations:
        if len(record) != 4 or not record[0] or not record[1]:
            raise ValueError("backend calibration allowlist contains a malformed identity")
        if not _is_sha256(record[2]) or not _is_sha256(record[3]):
            raise ValueError("backend calibration allowlist contains an invalid SHA-256")
    if any(not _is_sha256(value) for value in allowed_constraint_sha256s):
        raise ValueError("constraint allowlist contains an invalid SHA-256")

    behavior_by_policy = {snapshot.policy_id: snapshot for snapshot in header.policy_snapshots}
    current_by_policy: dict[str, PolicySnapshot] = {}
    for snapshot in current_snapshots:
        snapshot.validate()
        if snapshot.policy_id in current_by_policy:
            raise ValueError("current state contains duplicate policy snapshots")
        current_by_policy[snapshot.policy_id] = snapshot
    if set(current_by_policy) != set(behavior_by_policy):
        raise ValueError("current and behavior policy rosters do not match")

    decision_ids: set[str] = set()
    for decision in decisions:
        validate_rollout_decision(decision)
        if decision.decision_id in decision_ids:
            raise ValueError(f"duplicate async decision ID: {decision.decision_id}")
        decision_ids.add(decision.decision_id)
        snapshot = behavior_by_policy.get(decision.policy_id)
        if snapshot is None:
            raise ValueError(f"decision references an unbound policy: {decision.policy_id}")
        if decision.policy_revision != snapshot.revision:
            raise ValueError(f"decision policy revision disagrees with header: {decision.decision_id}")
        if decision.constraint_sha256 not in allowed_constraint_sha256s:
            raise ValueError(f"decision uses an unapproved dynamic constraint: {decision.decision_id}")
        if len(decision.allowed_token_ids) != len(decision.completion_ids):
            raise ValueError(f"async decision lacks exact constraint rows: {decision.decision_id}")

    observed_policy_ids = {decision.policy_id for decision in decisions}
    if observed_policy_ids != set(behavior_by_policy):
        raise ValueError("rollout decisions do not cover the complete snapshotted policy roster")

    eligible = tuple(
        decision
        for decision in decisions
        if decision.branch == trainable_branch and behavior_by_policy[decision.policy_id].trainable
    )
    eligible_ids = {decision.decision_id for decision in eligible}
    if trainable_decision_ids is None:
        selected = eligible
    else:
        if not trainable_decision_ids or not trainable_decision_ids <= eligible_ids:
            raise ValueError("trainable decision selection is empty or contains ineligible spans")
        selected = tuple(decision for decision in eligible if decision.decision_id in trainable_decision_ids)
    if not selected:
        raise ValueError(f"async admission requires {trainable_branch} decisions from trainable policies")
    selected_policy_ids = {decision.policy_id for decision in selected}
    trainable_policy_ids = {policy_id for policy_id, snapshot in behavior_by_policy.items() if snapshot.trainable}
    if selected_policy_ids != trainable_policy_ids:
        raise ValueError(f"{trainable_branch} decisions do not cover every trainable policy")
    expected_logprob_ids = {decision.decision_id for decision in selected}
    if set(trainer_logprobs) != expected_logprob_ids:
        raise ValueError("current-policy log-prob rows do not exactly match trainable decisions")

    reasons: list[str] = []
    calibration_identity = (
        header.backend_name,
        header.backend_version,
        header.kernel_config_sha256,
        header.calibration_sha256,
    )
    if calibration_identity not in allowed_backend_calibrations:
        reasons.append("uncertified rollout backend/kernel calibration")

    policy_lags: dict[str, int] = {}
    for policy_id, behavior in behavior_by_policy.items():
        current = current_by_policy[policy_id]
        if current.trainable != behavior.trainable:
            raise ValueError(f"trainability changed for policy {policy_id}")
        lag = current.update_index - behavior.update_index
        if lag < 0:
            reasons.append(f"policy {policy_id} comes from a future update")
        if behavior.trainable:
            policy_lags[policy_id] = lag
            if lag == 0 and (
                current.revision != behavior.revision or current.adapter_sha256 != behavior.adapter_sha256
            ):
                reasons.append(f"policy {policy_id} changed without an update-index change")
            if lag > limits.max_policy_lag:
                reasons.append(f"policy {policy_id} lag {lag} exceeds {limits.max_policy_lag}")
        elif (
            current.revision != behavior.revision
            or current.adapter_sha256 != behavior.adapter_sha256
            or current.update_index != behavior.update_index
        ):
            reasons.append(f"frozen policy {policy_id} changed after rollout")

    per_policy_ratios: dict[str, list[float]] = {}
    per_policy_probability_errors: dict[str, list[float]] = {}
    for decision in selected:
        compared = trainer_logprobs[decision.decision_id]
        if len(compared) != len(decision.rollout_logprobs):
            raise ValueError(f"current-policy log-prob length mismatch: {decision.decision_id}")
        if not all(math.isfinite(value) for value in compared):
            raise ValueError(f"non-finite current-policy log probability: {decision.decision_id}")
        policy_ratios = per_policy_ratios.setdefault(decision.policy_id, [])
        policy_probability_errors = per_policy_probability_errors.setdefault(decision.policy_id, [])
        for behavior_logprob, current_logprob in zip(decision.rollout_logprobs, compared, strict=True):
            policy_ratios.append(current_logprob - behavior_logprob)
            policy_probability_errors.append(
                abs(_exp_or_infinity(current_logprob) - _exp_or_infinity(behavior_logprob))
            )

    all_ratios = [value for values in per_policy_ratios.values() for value in values]
    all_probability_errors = [value for values in per_policy_probability_errors.values() for value in values]
    abs_ratios = [abs(value) for value in all_ratios]
    symmetric_importance_ratios = [_exp_or_infinity(value) for value in abs_ratios]
    mean_abs_log_ratio = sum(abs_ratios) / len(abs_ratios)
    mismatch_kls = [_exp_or_infinity(value) - value - 1.0 for value in all_ratios]
    mean_mismatch_kl = sum(mismatch_kls) / len(mismatch_kls)
    p99_abs_log_ratio = _quantile(abs_ratios, 0.99)
    max_symmetric_importance_ratio = max(symmetric_importance_ratios)
    p99_probability_error = _quantile(all_probability_errors, 0.99)
    probability_tail_fraction = sum(
        value > limits.probability_tail_threshold for value in all_probability_errors
    ) / len(all_probability_errors)

    numerical_checks = {
        "mean absolute log ratio": (
            mean_abs_log_ratio,
            limits.max_mean_abs_log_ratio,
        ),
        "mean mismatch KL": (
            mean_mismatch_kl,
            limits.max_mean_mismatch_kl,
        ),
        "p99 absolute log ratio": (
            p99_abs_log_ratio,
            limits.max_p99_abs_log_ratio,
        ),
        "maximum symmetric importance ratio": (
            max_symmetric_importance_ratio,
            limits.max_symmetric_importance_ratio,
        ),
        "p99 probability error": (
            p99_probability_error,
            limits.max_p99_probability_error,
        ),
        "probability tail fraction": (
            probability_tail_fraction,
            limits.max_probability_tail_fraction,
        ),
    }
    reasons.extend(
        f"{name} {value:.8g} exceeds {bound:.8g}"
        for name, (value, bound) in numerical_checks.items()
        if bound is not None and (not math.isfinite(value) or value > bound)
    )
    if not math.isfinite(mean_mismatch_kl):
        reasons.append("mean mismatch KL is non-finite")

    per_policy = {}
    for policy_id, ratios in sorted(per_policy_ratios.items()):
        absolute = [abs(value) for value in ratios]
        probability_errors = per_policy_probability_errors[policy_id]
        per_policy[policy_id] = {
            "tokens": len(ratios),
            "policy_lag": policy_lags[policy_id],
            "mean_abs_log_ratio": sum(absolute) / len(absolute),
            "p99_abs_log_ratio": _quantile(absolute, 0.99),
            "max_symmetric_importance_ratio": max(_exp_or_infinity(value) for value in absolute),
            "p99_probability_error": _quantile(probability_errors, 0.99),
        }

    return AsyncAdmissionResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        metrics={
            "rollout_id": header.rollout_id,
            "backend": f"{header.backend_name}@{header.backend_version}",
            "limits_sha256": limits.sha256,
            "decisions": len(selected),
            "tokens": len(all_ratios),
            "max_policy_lag": max(policy_lags.values()),
            "mean_abs_log_ratio": mean_abs_log_ratio,
            "mean_mismatch_kl": mean_mismatch_kl,
            "p99_abs_log_ratio": p99_abs_log_ratio,
            "max_symmetric_importance_ratio": max_symmetric_importance_ratio,
            "p99_probability_error": p99_probability_error,
            "probability_tail_fraction": probability_tail_fraction,
            "per_policy": per_policy,
        },
    )
