from __future__ import annotations

from dataclasses import replace

import pytest
from swarm_ctf_eval.async_admission import (
    AsyncAdmissionLimits,
    AsyncRolloutHeader,
    PolicySnapshot,
    admit_async_rollout,
)
from swarm_ctf_eval.prime_rl_bridge import RolloutDecision

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _snapshot(
    policy_id: str,
    revision: str,
    update: int,
    *,
    trainable: bool,
    adapter_sha256: str,
) -> PolicySnapshot:
    return PolicySnapshot(policy_id, revision, adapter_sha256, update, trainable)


def _decision(
    policy_id: str,
    revision: str,
    *,
    agent_id: str = "blue-0",
    team: str = "BLUE",
) -> RolloutDecision:
    return RolloutDecision(
        game_id="game-1",
        branch="actual",
        replaced_agent=None,
        agent_id=agent_id,
        policy_id=policy_id,
        policy_revision=revision,
        team=team,
        turn=0,
        phase="ACT",
        trajectory_index=0,
        prompt_ids=(1, 2),
        completion_ids=(3, 4),
        rollout_logprobs=(-0.2, -0.3),
        constraint_sha256=SHA_C,
        sampling_key=f"game-1:{agent_id}:0:ACT",
        context_sha256=SHA_A,
        request_sha256=SHA_B,
        output_sha256=SHA_D,
        allowed_token_ids=((3, 8), (4, 9)),
    )


def _limits() -> AsyncAdmissionLimits:
    return AsyncAdmissionLimits(
        max_policy_lag=2,
        max_mean_abs_log_ratio=0.05,
        max_p99_abs_log_ratio=0.1,
        max_symmetric_importance_ratio=1.2,
        max_p99_probability_error=0.05,
        probability_tail_threshold=0.02,
        max_probability_tail_fraction=0.5,
    )


def _admit(
    *,
    current_update: int = 6,
    trainer=(-0.21, -0.29),
    calibration: str = SHA_D,
):
    behavior = _snapshot("blue-0-policy", "step-4", 4, trainable=True, adapter_sha256=SHA_A)
    opponent = _snapshot("red-policy", "frozen", 0, trainable=False, adapter_sha256=SHA_B)
    header = AsyncRolloutHeader(
        "rollout-1",
        "vllm",
        "0.10.2",
        SHA_C,
        calibration,
        (behavior, opponent),
    )
    decisions = (
        _decision("blue-0-policy", "step-4"),
        _decision("red-policy", "frozen", agent_id="red-0", team="RED"),
    )
    current = (
        _snapshot(
            "blue-0-policy",
            f"step-{current_update}",
            current_update,
            trainable=True,
            adapter_sha256=SHA_D,
        ),
        opponent,
    )
    return admit_async_rollout(
        header,
        decisions,
        current,
        {decisions[0].decision_id: trainer},
        allowed_backend_calibrations=frozenset({("vllm", "0.10.2", SHA_C, SHA_D)}),
        allowed_constraint_sha256s=frozenset({SHA_C}),
        limits=_limits(),
    )


def test_admits_calibrated_bounded_off_policy_rollout() -> None:
    result = _admit()
    assert result.accepted
    assert result.reasons == ()
    assert result.metrics["max_policy_lag"] == 2
    assert result.metrics["per_policy"]["blue-0-policy"]["tokens"] == 2


def test_rejects_stale_or_divergent_rollout_without_repairing_it() -> None:
    stale = _admit(current_update=7)
    assert not stale.accepted
    assert any("lag 3 exceeds 2" in reason for reason in stale.reasons)

    divergent = _admit(trainer=(-0.9, -1.0))
    assert not divergent.accepted
    assert any("log ratio" in reason for reason in divergent.reasons)
    assert any("importance ratio" in reason for reason in divergent.reasons)


def test_rejects_unknown_calibration_and_malformed_constraint_evidence() -> None:
    uncertified = _admit(calibration=SHA_A)
    assert not uncertified.accepted
    assert "uncertified rollout backend/kernel calibration" in uncertified.reasons

    behavior = _snapshot("blue-0-policy", "step-4", 4, trainable=True, adapter_sha256=SHA_A)
    opponent = _snapshot("red-policy", "frozen", 0, trainable=False, adapter_sha256=SHA_B)
    header = AsyncRolloutHeader("rollout-1", "vllm", "0.10.2", SHA_C, SHA_D, (behavior, opponent))
    decision = replace(_decision("blue-0-policy", "step-4"), allowed_token_ids=())
    opponent_decision = _decision("red-policy", "frozen", agent_id="red-0", team="RED")
    with pytest.raises(ValueError, match="lacks exact constraint rows"):
        admit_async_rollout(
            header,
            (decision, opponent_decision),
            (
                _snapshot("blue-0-policy", "step-4", 4, trainable=True, adapter_sha256=SHA_A),
                opponent,
            ),
            {decision.decision_id: decision.rollout_logprobs},
            allowed_backend_calibrations=frozenset({("vllm", "0.10.2", SHA_C, SHA_D)}),
            allowed_constraint_sha256s=frozenset({SHA_C}),
            limits=_limits(),
        )
