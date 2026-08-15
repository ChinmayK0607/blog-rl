from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from swarm_ctf_eval.async_admission import (
    AsyncAdmissionLimits,
    AsyncRolloutHeader,
    PolicySnapshot,
    admit_async_rollout,
)
from swarm_ctf_eval.async_training_queue import AtomicAsyncTrainingQueue
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


def test_admission_checks_only_the_signed_trainable_span_selection() -> None:
    behavior = _snapshot("blue-0-policy", "step-4", 4, trainable=True, adapter_sha256=SHA_A)
    opponent = _snapshot("red-policy", "frozen", 0, trainable=False, adapter_sha256=SHA_B)
    header = AsyncRolloutHeader(
        "rollout-selected-spans",
        "vllm",
        "0.10.2",
        SHA_C,
        SHA_D,
        (behavior, opponent),
    )
    action = _decision("blue-0-policy", "step-4")
    broadcast = replace(
        action,
        phase="BROADCAST",
        sampling_key="game-1:blue-0:0:BROADCAST",
    )
    red = _decision("red-policy", "frozen", agent_id="red-0", team="RED")
    result = admit_async_rollout(
        header,
        (action, broadcast, red),
        (
            _snapshot("blue-0-policy", "step-4", 4, trainable=True, adapter_sha256=SHA_A),
            opponent,
        ),
        {action.decision_id: action.rollout_logprobs},
        trainable_decision_ids=frozenset({action.decision_id}),
        allowed_backend_calibrations=frozenset({("vllm", "0.10.2", SHA_C, SHA_D)}),
        allowed_constraint_sha256s=frozenset({SHA_C}),
        limits=_limits(),
    )

    assert result.accepted
    assert result.metrics["decisions"] == 1


def test_atomic_queue_never_routes_a_partial_four_policy_group(tmp_path) -> None:
    behavior = tuple(
        _snapshot(
            f"blue-{index}-policy",
            "step-0",
            0,
            trainable=True,
            adapter_sha256=SHA_A,
        )
        for index in range(4)
    )
    opponent = _snapshot("red-policy", "frozen", 0, trainable=False, adapter_sha256=SHA_B)
    header = AsyncRolloutHeader(
        "rollout-four-policy",
        "vllm",
        "0.10.2",
        SHA_C,
        SHA_D,
        (*behavior, opponent),
    )
    blue_decisions = tuple(
        _decision(
            snapshot.policy_id,
            snapshot.revision,
            agent_id=f"blue-{index}",
        )
        for index, snapshot in enumerate(behavior)
    )
    red = _decision("red-policy", "frozen", agent_id="red-0", team="RED")
    queue = AtomicAsyncTrainingQueue(
        capacity=1,
        audit_path=tmp_path / "async.jsonl",
        allowed_backend_calibrations=frozenset({("vllm", "0.10.2", SHA_C, SHA_D)}),
        allowed_constraint_sha256s=frozenset({SHA_C}),
        limits=_limits(),
    )
    batches = {
        f"run_blue_{index}": SimpleNamespace(step=0, examples=[index])
        for index in range(4)
    }
    result = queue.admit(
        header=header,
        decisions=(*blue_decisions, red),
        trainable_decision_ids=frozenset(row.decision_id for row in blue_decisions),
        current_snapshots=(*behavior, opponent),
        current_policy_logprobs={
            row.decision_id: row.rollout_logprobs for row in blue_decisions
        },
        routed_batches=batches,
        trainer_step=0,
    )

    assert result.accepted
    assert queue.size == 1
    merged = queue.pop_logical_update(groups=1, trainer_step=0)
    assert set(merged) == set(batches)
    assert queue.size == 0

    with pytest.raises(ValueError, match="all four isolated policy batches"):
        queue.admit(
            header=replace(header, rollout_id="rollout-partial"),
            decisions=(*blue_decisions, red),
            trainable_decision_ids=frozenset(row.decision_id for row in blue_decisions),
            current_snapshots=(*behavior, opponent),
            current_policy_logprobs={
                row.decision_id: row.rollout_logprobs for row in blue_decisions
            },
            routed_batches={key: value for key, value in batches.items() if key != "run_blue_3"},
            trainer_step=0,
        )
