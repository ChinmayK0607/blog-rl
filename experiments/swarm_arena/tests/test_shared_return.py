from __future__ import annotations

import asyncio
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from swarm_ctf_eval.episode import EpisodeConfig
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.live_rl_rollout import (
    ChoiceCompletion,
    PolicyEndpoint,
    build_live_shared_return_group,
    protocol_constraint_sha256,
)
from swarm_ctf_eval.message_interventions import target_swapped_broadcast
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
    paired_message_drop_advantages,
    paired_terminal_contrast_advantages,
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


class ExposedFactGenerator(FirstChoiceGenerator):
    async def generate(
        self,
        endpoint: PolicyEndpoint,
        messages: list[dict[str, str]],
        *,
        sampling_key: str,
    ) -> ChoiceCompletion:
        choices = protocol_choices(messages)
        selected = choices[0]
        for choice in choices:
            payload = json.loads(choice)
            if any(fact.get("status") == "EXPOSED" for fact in payload.get("facts", [])):
                selected = choice
                break
        original = await super().generate(endpoint, messages, sampling_key=sampling_key)
        return replace(
            original,
            text=selected,
            request_sha256=canonical_sha256(
                {
                    "policy_id": endpoint.policy_id,
                    "revision": endpoint.revision,
                    "messages": messages,
                    "sampling_key": sampling_key,
                }
            ),
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
    ) + (PolicyEndpoint("red-opponent", "opponent-r0", "red", ("http://unused",)),)


def _lock(spec: SharedReturnSpec) -> RunLock:
    return RunLock(
        run_id="shared-return-smoke",
        source_commit="commit-1234567",
        task_version="arena-rl-v3-terminal-control",
        train_manifest_sha256="1" * 64,
        development_manifest_sha256="2" * 64,
        final_eval_manifest_sha256="3" * 64,
        base_model_revision="base-r0",
        trainable_policy_revisions=tuple((f"blue-policy-{index}", "trainable-r0") for index in range(4)),
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


def test_paired_message_drop_advantages_center_only_terminal_return_effects() -> None:
    values = paired_message_drop_advantages(
        (0.5, 0.4, 0.9, 0.2),
        (0.5, 0.2, 0.3, 0.4),
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(values, (-0.2, 0.06666666666666667, 0.6, -0.4666666666666667), strict=True)
    )
    assert math.isclose(sum(values), 0.0, abs_tol=1e-12)


def test_paired_message_drop_group_replays_both_conditions_and_routes_normal_receiver_only() -> None:
    spec = SharedReturnSpec(
        replicas=2,
        trainable_phases=("ACT",),
        trainable_turn_offsets=(0,),
        baseline="paired_message_drop",
        credit_assignment="focused_agent",
        action_prompt_profile="focused_handoff_compact",
    )
    lock = _lock(spec)
    bindings = _bindings()
    group = asyncio.run(
        build_live_shared_return_group(
            FirstChoiceGenerator(),  # type: ignore[arg-type]
            group_id="paired-drop-receiver",
            seed=305,
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
            focused_agent="blue-1",
            message_drop_agent="blue-0",
            message_drop_turn=0,
        )
    )
    approvals = approve_shared_return_group(
        lock,
        group.evidence,
        bindings,
        "BLUE",
        b"shared-return-test-signing-key-32-bytes",
    )
    assert len(approvals) == 2
    assert all(replica.dropped_replay is not None for replica in group.evidence.replicas)
    assert all(
        decision.branch == "message_drop"
        for replica in group.evidence.replicas
        for decision in replica.dropped_decisions
    )
    assert {
        envelope.agent_id
        for approval in approvals
        for envelope in approval.envelopes
        if envelope.advantage != 0.0
    } <= {"blue-1"}
    assert all(
        decision_id.endswith(":0:ACT")
        for approval in approvals
        for envelope in approval.envelopes
        for decision_id in envelope.decision_ids
    )

    first = group.evidence.replicas[0]
    assert first.dropped_replay is not None
    tampered = replace(
        group.evidence,
        replicas=(
            replace(first, dropped_replay=replace(first.dropped_replay, replaced_agent="blue-2")),
            *group.evidence.replicas[1:],
        ),
    )
    with pytest.raises(ValueError, match="wrong sender"):
        approve_shared_return_group(
            lock,
            tampered,
            bindings,
            "BLUE",
            b"shared-return-test-signing-key-32-bytes",
        )


def test_paired_target_swap_is_replayed_and_routes_only_the_receiver() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data" / "rl_v4" / "handoff_train.json").read_text())
    critical = reconstruct_manifest_scenario(manifest["pairs"][12]["critical"])
    world = critical.worlds[0]
    spec = SharedReturnSpec(
        replicas=2,
        trainable_phases=("ACT",),
        trainable_turn_offsets=(0,),
        baseline="paired_target_swap",
        credit_assignment="focused_agent",
        action_prompt_profile="focused_handoff_compact",
    )
    lock = _lock(spec)
    group = asyncio.run(
        build_live_shared_return_group(
            ExposedFactGenerator(),  # type: ignore[arg-type]
            group_id="paired-swap-receiver",
            seed=critical.seed,
            size=critical.size,
            config=EpisodeConfig(
                horizon=world.state.turn + 2,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=_bindings(),
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            initial_state=world.state,
            focused_agent=critical.receiver,
            message_swap_agent=critical.sender,
            message_swap_turn=world.state.turn,
            message_swap_targets=critical.candidate_targets,
            message_swap_active_target=world.active_target,
        )
    )
    approvals = approve_shared_return_group(
        lock,
        group.evidence,
        _bindings(),
        "BLUE",
        b"shared-return-test-signing-key-32-bytes",
    )
    assert len(approvals) == 2
    assert all(replica.swapped_replay is not None for replica in group.evidence.replicas)
    assert all(
        decision.branch == "message_swap"
        for replica in group.evidence.replicas
        for decision in replica.swapped_decisions
    )
    assert {
        envelope.agent_id
        for approval in approvals
        for envelope in approval.envelopes
        if envelope.advantage != 0.0
    } <= {critical.receiver}
    first = group.evidence.replicas[0]
    assert first.swapped_replay is not None
    actual_message = dict(first.replay.turns[0].delivered_broadcasts)[critical.sender]
    swapped_message = dict(first.swapped_replay.turns[0].delivered_broadcasts)[critical.sender]
    assert swapped_message == target_swapped_broadcast(
        actual_message,
        candidate_targets=critical.candidate_targets,
        active_target=world.active_target,
    )
    tampered_turn = replace(
        first.swapped_replay.turns[0],
        delivered_broadcasts=tuple(
            (agent_id, actual_message if agent_id == critical.sender else broadcast)
            for agent_id, broadcast in first.swapped_replay.turns[0].delivered_broadcasts
        ),
    )
    tampered_replica = replace(
        first,
        swapped_replay=replace(
            first.swapped_replay,
            turns=(tampered_turn, *first.swapped_replay.turns[1:]),
        ),
    )
    tampered_evidence = replace(
        group.evidence,
        replicas=(tampered_replica, *group.evidence.replicas[1:]),
    )
    with pytest.raises(ValueError, match="replay post-state mismatch"):
        approve_shared_return_group(
            lock,
            tampered_evidence,
            _bindings(),
            "BLUE",
            b"shared-return-test-signing-key-32-bytes",
        )


def test_receiver_only_target_swap_isolates_the_counterfactual_context() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data" / "rl_v4" / "handoff_train.json").read_text())
    critical = reconstruct_manifest_scenario(manifest["pairs"][12]["critical"])
    world = critical.worlds[0]
    spec = SharedReturnSpec(
        replicas=2,
        trainable_phases=("ACT",),
        trainable_turn_offsets=(0,),
        baseline="paired_receiver_target_swap",
        credit_assignment="focused_agent",
        action_prompt_profile="focused_handoff_compact",
    )
    lock = _lock(spec)
    group = asyncio.run(
        build_live_shared_return_group(
            ExposedFactGenerator(),  # type: ignore[arg-type]
            group_id="paired-receiver-only-swap",
            seed=critical.seed,
            size=critical.size,
            config=EpisodeConfig(
                horizon=world.state.turn + 1,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=_bindings(),
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            initial_state=world.state,
            focused_agent=critical.receiver,
            message_swap_agent=critical.sender,
            message_swap_turn=world.state.turn,
            message_swap_targets=critical.candidate_targets,
            message_swap_active_target=world.active_target,
            message_swap_sender_sampling_namespace="sender-eligibility-retry-1",
        )
    )
    control = asyncio.run(
        build_live_shared_return_group(
            ExposedFactGenerator(),  # type: ignore[arg-type]
            group_id="paired-receiver-only-swap",
            seed=critical.seed,
            size=critical.size,
            config=EpisodeConfig(
                horizon=world.state.turn + 1,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=_bindings(),
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            initial_state=world.state,
            focused_agent=critical.receiver,
            message_swap_agent=critical.sender,
            message_swap_turn=world.state.turn,
            message_swap_targets=critical.candidate_targets,
            message_swap_active_target=world.active_target,
        )
    )
    approve_shared_return_group(
        lock,
        group.evidence,
        _bindings(),
        "BLUE",
        b"shared-return-test-signing-key-32-bytes",
    )

    first = group.evidence.replicas[0]
    assert first.swapped_replay is not None
    actual_turn = first.replay.turns[0]
    swapped_turn = first.swapped_replay.turns[0]
    assert swapped_turn.delivered_broadcasts == actual_turn.delivered_broadcasts
    overrides = {
        receiver: dict(values)
        for receiver, values in swapped_turn.receiver_delivery_overrides
    }
    actual_sender_message = dict(actual_turn.delivered_broadcasts)[critical.sender]
    assert overrides == {
        critical.receiver: {
            critical.sender: target_swapped_broadcast(
                actual_sender_message,
                candidate_targets=critical.candidate_targets,
                active_target=world.active_target,
            )
        }
    }

    actual = {
        (row.agent_id, row.phase): row
        for row in first.decisions
        if row.turn == world.state.turn
    }
    swapped = {
        (row.agent_id, row.phase): row
        for row in first.swapped_decisions
        if row.turn == world.state.turn
    }
    control_actual = {
        (row.agent_id, row.phase): row
        for row in control.evidence.replicas[0].decisions
        if row.turn == world.state.turn
    }
    assert actual[(critical.sender, "BROADCAST")].sampling_key != control_actual[
        (critical.sender, "BROADCAST")
    ].sampling_key
    assert all(
        actual[key].sampling_key == control_actual[key].sampling_key
        for key in actual
        if key != (critical.sender, "BROADCAST")
    )
    assert actual[(critical.sender, "BROADCAST")].sampling_key == swapped[
        (critical.sender, "BROADCAST")
    ].sampling_key
    assert actual[(critical.receiver, "ACT")].context_sha256 != swapped[
        (critical.receiver, "ACT")
    ].context_sha256
    assert actual[(critical.receiver, "ACT")].sampling_key == swapped[
        (critical.receiver, "ACT")
    ].sampling_key
    assert all(
        actual[key].context_sha256 == swapped[key].context_sha256
        and actual[key].output_sha256 == swapped[key].output_sha256
        for key in actual
        if key != (critical.receiver, "ACT")
    )
    tampered_replica = replace(
        first,
        swapped_replay=replace(
            first.swapped_replay,
            turns=(
                replace(swapped_turn, receiver_delivery_overrides=()),
                *first.swapped_replay.turns[1:],
            ),
        ),
    )
    with pytest.raises(ValueError):
        approve_shared_return_group(
            lock,
            replace(
                group.evidence,
                replicas=(tampered_replica, *group.evidence.replicas[1:]),
            ),
            _bindings(),
            "BLUE",
            b"shared-return-test-signing-key-32-bytes",
        )


def test_receiver_target_swap_challenge_trains_only_the_counterfactual_receiver() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data" / "rl_v4" / "handoff_train.json").read_text())
    decoy = reconstruct_manifest_scenario(manifest["pairs"][12]["decoy"])
    world = decoy.worlds[0]
    spec = SharedReturnSpec(
        replicas=3,
        trainable_phases=("ACT",),
        trainable_turn_offsets=(0,),
        baseline="paired_receiver_target_swap_challenge",
        credit_assignment="focused_agent",
        action_prompt_profile="focused_handoff_compact",
        paired_contrast_centering="none",
    )
    lock = _lock(spec)
    group = asyncio.run(
        build_live_shared_return_group(
            ExposedFactGenerator(),  # type: ignore[arg-type]
            group_id="paired-receiver-swap-challenge",
            seed=decoy.seed,
            size=decoy.size,
            config=EpisodeConfig(
                horizon=world.state.turn + 1,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=_bindings(),
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            initial_state=world.state,
            focused_agent=decoy.receiver,
            message_swap_agent=decoy.sender,
            message_swap_turn=world.state.turn,
            message_swap_targets=decoy.candidate_targets,
            message_swap_active_target=world.active_target,
        )
    )
    approvals = approve_shared_return_group(
        lock,
        group.evidence,
        _bindings(),
        "BLUE",
        b"shared-return-test-signing-key-32-bytes",
    )

    factual_returns = tuple(
        replica.replay.terminal_return for replica in group.evidence.replicas
    )
    swapped_returns = tuple(
        replica.swapped_replay.terminal_return
        for replica in group.evidence.replicas
        if replica.swapped_replay is not None
    )
    expected = paired_terminal_contrast_advantages(
        swapped_returns,
        factual_returns,
        centering="none",
    )
    assert len(approvals) == spec.replicas
    assert tuple(
        next(
            envelope.advantage
            for envelope in approval.envelopes
            if envelope.agent_id == decoy.receiver
        )
        for approval in approvals
    ) == expected
    assert all(approval.game_id.endswith(":swap") for approval in approvals)
    assert all(
        decision.branch == "message_swap"
        for replica in group.evidence.replicas
        for decision in replica.swapped_decisions
    )
    assert all(
        envelope.advantage == 0.0
        for approval in approvals
        for envelope in approval.envelopes
        if envelope.agent_id != decoy.receiver
    )
    assert all(
        ":swap:" in decision_id
        for approval in approvals
        for envelope in approval.envelopes
        if envelope.agent_id == decoy.receiver
        for decision_id in envelope.decision_ids
    )


def test_compact_prompt_profile_is_bound_without_changing_legacy_spec_identity() -> None:
    legacy = SharedReturnSpec(replicas=4)
    assert legacy.sha256 == canonical_sha256(
        {
            "replicas": 4,
            "trainable_phases": ("BROADCAST",),
            "trainable_turn_offsets": (0,),
            "baseline": "leave_one_out_mean",
            "reward": "verified_terminal_team_return",
            "credit_assignment": "shared_team",
        }
    )
    compact = SharedReturnSpec(
        replicas=8,
        trainable_phases=("ACT",),
        trainable_turn_offsets=(0,),
        credit_assignment="focused_agent",
        action_prompt_profile="focused_handoff_compact",
    )
    compact.validate()
    assert compact.sha256 != replace(compact, action_prompt_profile="full").sha256
    with pytest.raises(ValueError, match="focused-agent ACT"):
        replace(compact, credit_assignment="shared_team").validate()


def test_uncentered_paired_contrast_preserves_uniform_success_and_failure_signal() -> None:
    factual = (0.2, 0.2, 0.2, 0.2)
    misleading = (-0.2, -0.2, -0.2, -0.2)

    assert paired_terminal_contrast_advantages(
        factual,
        misleading,
        centering="none",
    ) == (0.4, 0.4, 0.4, 0.4)
    assert paired_terminal_contrast_advantages(
        misleading,
        factual,
        centering="none",
    ) == (-0.4, -0.4, -0.4, -0.4)
    assert all(
        math.isclose(value, 0.0, abs_tol=1e-12)
        for value in paired_terminal_contrast_advantages(factual, misleading)
    )


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

    routes = tuple(PolicyRunRoute(f"blue-policy-{index}", f"run_blue_{index}") for index in range(4))
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


def test_focused_shared_return_couples_other_agents_and_credits_only_focus() -> None:
    spec = SharedReturnSpec(
        replicas=2,
        trainable_phases=("ACT",),
        trainable_turn_offsets=None,
        credit_assignment="focused_agent",
    )
    lock = _lock(spec)
    group = asyncio.run(
        build_live_shared_return_group(
            FirstChoiceGenerator(),  # type: ignore[arg-type]
            group_id="shared-all-turns",
            seed=303,
            size=12,
            config=EpisodeConfig(
                horizon=2,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=_bindings(),
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            focused_agent="blue-2",
        )
    )
    approvals = approve_shared_return_group(
        lock,
        group.evidence,
        _bindings(),
        "BLUE",
        b"shared-return-test-signing-key-32-bytes",
    )

    assert len(approvals) == 2
    for approval, owned in zip(
        approvals,
        group.owned_samples_by_replica,
        strict=True,
    ):
        assert all(len(envelope.decision_ids) == 2 for envelope in approval.envelopes)
        assert all(len(agent.samples) == 2 for agent in owned)
        assert all(
            decision_id.endswith(":ACT") for envelope in approval.envelopes for decision_id in envelope.decision_ids
        )
        assert {envelope.agent_id for envelope in approval.envelopes if envelope.advantage != 0.0} <= {"blue-2"}

    decisions = [decision for replica in group.evidence.replicas for decision in replica.decisions]
    focus_keys = {
        decision.sampling_key for decision in decisions if decision.agent_id == "blue-2" and decision.phase == "ACT"
    }
    assert len(focus_keys) == 4
    blue_zero_action_keys = [
        decision.sampling_key for decision in decisions if decision.agent_id == "blue-0" and decision.phase == "ACT"
    ]
    assert len(set(blue_zero_action_keys)) == 2


def test_focused_shared_return_can_isolate_sender_broadcast_credit() -> None:
    spec = SharedReturnSpec(
        replicas=2,
        trainable_phases=("BROADCAST",),
        trainable_turn_offsets=(0,),
        credit_assignment="focused_agent",
    )
    lock = _lock(spec)
    group = asyncio.run(
        build_live_shared_return_group(
            FirstChoiceGenerator(),  # type: ignore[arg-type]
            group_id="focused-sender-broadcast",
            seed=304,
            size=12,
            config=EpisodeConfig(
                horizon=2,
                communication_cost=0.0,
                invalid_broadcast_cost=0.0,
                invalid_action_cost=0.0,
            ),
            spec=spec,
            bindings=_bindings(),
            policies=_endpoints(),
            run_lock_sha256=lock.sha256,
            focused_agent="blue-1",
        )
    )
    approvals = approve_shared_return_group(
        lock,
        group.evidence,
        _bindings(),
        "BLUE",
        b"shared-return-test-signing-key-32-bytes",
    )

    for approval in approvals:
        assert all(
            decision_id.endswith(":0:BROADCAST")
            for envelope in approval.envelopes
            for decision_id in envelope.decision_ids
        )
        assert {
            envelope.agent_id for envelope in approval.envelopes if envelope.advantage != 0.0
        } <= {"blue-1"}


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
        and len(row["completion_ids"]) == len(row["completion_logprobs"]) == len(row["allowed_token_ids"])
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
