from __future__ import annotations

import asyncio
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm_ctf_eval.episode import EpisodeConfig
from swarm_ctf_eval.live_rl_rollout import (
    ChoiceCompletion,
    PolicyEndpoint,
    build_live_shared_return_group,
    protocol_constraint_sha256,
)
from swarm_ctf_eval.multi_policy_contract import AgentPolicy
from swarm_ctf_eval.prime_multi_run_router import (
    PolicyRunRoute,
    merge_routed_batch_groups,
    route_approved_samples,
    validate_single_trajectory_packing,
)
from swarm_ctf_eval.safety_supervisor import (
    RunLock,
    SharedReturnSpec,
    approve_shared_return_group,
    canonical_sha256,
    leave_one_out_advantages,
)
from swarm_ctf_eval.shared_return_parity import build_shared_return_parity_probe
from swarm_ctf_eval.structured_protocol import protocol_choices


class FirstChoiceGenerator:
    async def generate(
        self,
        endpoint: PolicyEndpoint,
        messages: list[dict[str, str]],
        *,
        sampling_key: str,
    ) -> ChoiceCompletion:
        text = protocol_choices(messages)[0]
        request_sha256 = canonical_sha256(
            {
                "policy_id": endpoint.policy_id,
                "revision": endpoint.revision,
                "messages": messages,
                "sampling_key": sampling_key,
            }
        )
        return ChoiceCompletion(
            (10,),
            (11,),
            (0.0,),
            ((11,),),
            text,
            request_sha256,
            serving_allowed_logprobs=(((11, 0.0),),),
        )


def _bindings() -> tuple[AgentPolicy, ...]:
    return tuple(
        AgentPolicy(
            f"{team.lower()}-{index}",
            team,
            f"blue-policy-{index}" if team == "BLUE" else "red-opponent",
            team == "BLUE",
        )
        for team in ("BLUE", "RED")
        for index in range(4)
    )


def _endpoints() -> tuple[PolicyEndpoint, ...]:
    return tuple(
        PolicyEndpoint(
            f"blue-policy-{index}",
            "trainable-r0",
            f"blue-{index}",
            ("http://unused",),
        )
        for index in range(4)
    ) + (
        PolicyEndpoint("red-opponent", "opponent-r0", "red", ("http://unused",)),
    )


def _lock(spec: SharedReturnSpec) -> RunLock:
    return RunLock(
        run_id="shared-return-smoke",
        source_commit="commit-1234567",
        task_version="arena-rl-v3-terminal-control",
        train_manifest_sha256="1" * 64,
        development_manifest_sha256="2" * 64,
        final_eval_manifest_sha256="3" * 64,
        base_model_revision="base-r0",
        trainable_policy_revisions=tuple(
            (f"blue-policy-{index}", "trainable-r0") for index in range(4)
        ),
        frozen_policy_revisions=(("red-opponent", "opponent-r0"),),
        replacement_policy_id=None,
        opponent_id="sft-opponent",
        opponent_revision="opponent-r0",
        allowed_constraint_hashes=(
            protocol_constraint_sha256("BROADCAST"),
            protocol_constraint_sha256("ACT"),
        ),
        trainer_parity_gate_sha256="c" * 64,
        credit_estimator="shared_return",
        credit_estimator_config_sha256=spec.sha256,
        trainer_config_sha256="d" * 64,
        serving_config_sha256="e" * 64,
    )


def test_leave_one_out_advantages_are_zero_sum_and_unshaped() -> None:
    values = leave_one_out_advantages((0.0, 1.0, 2.0, 3.0))
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            values,
            (-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0),
            strict=True,
        )
    )
    assert math.isclose(sum(values), 0.0, abs_tol=1e-12)


def test_shared_return_group_is_replayed_signed_and_routed_fail_closed() -> None:
    spec = SharedReturnSpec(replicas=4)
    lock = _lock(spec)
    bindings = _bindings()
    group = asyncio.run(
        build_live_shared_return_group(
            FirstChoiceGenerator(),  # type: ignore[arg-type]
            group_id="shared-game-1",
            seed=101,
            size=12,
            config=EpisodeConfig(
                horizon=2,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=bindings,
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            sampling_namespace="shared-state-101",
        )
    )
    key = b"shared-return-test-signing-key-32-bytes"
    approvals = approve_shared_return_group(lock, group.evidence, bindings, "BLUE", key)

    assert len(approvals) == 4
    assert {row.evidence_sha256 for row in approvals} == {approvals[0].evidence_sha256}
    assert math.isclose(
        sum(row.envelopes[0].advantage for row in approvals),
        0.0,
        abs_tol=1e-10,
    )
    for approval in approvals:
        assert len(approval.envelopes) == 4
        assert len({row.advantage for row in approval.envelopes}) == 1
        assert all(
            decision_id.endswith(":0:BROADCAST")
            for envelope in approval.envelopes
            for decision_id in envelope.decision_ids
        )

    routes = tuple(
        PolicyRunRoute(f"blue-policy-{index}", f"run_blue_{index}")
        for index in range(4)
    )
    routed = tuple(
        route_approved_samples(
            approval,
            owned,
            routes,
            step=0,
            signing_key=key,
            trainer_parity_gate_sha256=lock.trainer_parity_gate_sha256,
        )
        for approval, owned in zip(
            approvals,
            group.owned_samples_by_replica,
            strict=True,
        )
    )
    merged = merge_routed_batch_groups(routed, step=0)
    assert set(merged) == {f"run_blue_{index}" for index in range(4)}
    assert all(len(batch.examples) == spec.replicas for batch in merged.values())

    first_owned = group.owned_samples_by_replica[0]
    swapped = deepcopy(first_owned[0].samples[0])
    swapped.prompt_ids = [999 for _ in swapped.prompt_ids]
    tampered_owned = (
        replace(first_owned[0], samples=(swapped, *first_owned[0].samples[1:])),
        *first_owned[1:],
    )
    try:
        route_approved_samples(
            approvals[0],
            tampered_owned,
            routes,
            step=0,
            signing_key=key,
            trainer_parity_gate_sha256=lock.trainer_parity_gate_sha256,
        )
    except ValueError as error:
        assert "committed training-sample payload mismatch" in str(error)
    else:
        raise AssertionError("same-length sample-content substitution must fail closed")

    first = group.evidence.replicas[0]
    bad_replay = replace(first.replay, terminal_return=first.replay.terminal_return + 1.0)
    tampered = replace(
        group.evidence,
        replicas=(replace(first, replay=bad_replay), *group.evidence.replicas[1:]),
    )
    try:
        approve_shared_return_group(lock, tampered, bindings, "BLUE", key)
    except ValueError as error:
        assert "independent replay" in str(error)
    else:
        raise AssertionError("tampered terminal return must fail closed")

    duplicate_namespace = replace(
        group.evidence.replicas[1],
        sampling_namespace=group.evidence.replicas[0].sampling_namespace,
    )
    tampered = replace(
        group.evidence,
        replicas=(group.evidence.replicas[0], duplicate_namespace, *group.evidence.replicas[2:]),
    )
    try:
        approve_shared_return_group(lock, tampered, bindings, "BLUE", key)
    except ValueError as error:
        assert "unique games and sampling namespaces" in str(error)
    else:
        raise AssertionError("reused sampling namespace must fail closed")

    stale_lock = replace(lock, credit_estimator_config_sha256="d" * 64)
    try:
        approve_shared_return_group(stale_lock, group.evidence, bindings, "BLUE", key)
    except ValueError as error:
        assert "spec does not match" in str(error)
    else:
        raise AssertionError("stale shared-return spec must fail closed")

    try:
        replace(lock, serving_config_sha256=None).validate()
    except ValueError as error:
        assert "immutable serving config" in str(error)
    else:
        raise AssertionError("unbound shared-return serving config must fail closed")


def test_published_v4_evidence_builds_policy_bound_parity_probe() -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "results/pre_rl_1_7b/shared_return_smoke_ab981247/shared_return_evidence.jsonl"
    )
    probe = build_shared_return_parity_probe(evidence)
    assert probe["version"] == "arena-shared-return-parity-probe-v1"
    assert len(probe["samples"]) == 16
    assert {row["policy_slot"] for row in probe["samples"]} == set(range(4))
    assert all(
        row["completion_ids"]
        and len(row["completion_ids"])
        == len(row["completion_logprobs"])
        == len(row["allowed_token_ids"])
        for row in probe["samples"]
    )


def test_live_batches_must_prove_single_trajectory_packing() -> None:
    def sample(length: int) -> SimpleNamespace:
        return SimpleNamespace(prompt_ids=[1] * (length - 1), completion_ids=[2])

    validate_single_trajectory_packing(
        {"run_blue_0": SimpleNamespace(examples=[sample(799), sample(975)])},
        seq_len=1024,
    )
    with pytest.raises(ValueError, match="could co-pack multiple trajectories"):
        validate_single_trajectory_packing(
            {"run_blue_0": SimpleNamespace(examples=[sample(500), sample(524)])},
            seq_len=1024,
        )
    with pytest.raises(ValueError, match="empty or over-length"):
        validate_single_trajectory_packing(
            {"run_blue_0": SimpleNamespace(examples=[sample(1025)])},
            seq_len=1024,
        )
