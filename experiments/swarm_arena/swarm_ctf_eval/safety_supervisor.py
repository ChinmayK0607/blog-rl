from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .arena import Action, GameState, Team, state_to_dict
from .arena_protocol import Broadcast
from .episode import EMPTY_BROADCAST, EpisodeConfig
from .message_interventions import target_swapped_broadcast
from .multi_policy_contract import AgentPolicy, MessageCredit, ReplacementCredit
from .prime_rl_bridge import (
    PolicyTrainingEnvelope,
    RolloutDecision,
    build_message_training_envelopes,
    build_shared_return_training_envelopes,
    build_training_envelopes,
    verify_trainer_logprob_parity,
)
from .rl_v3 import ArenaRLEnv

SUPERVISOR_VERSION = "arena-fail-closed-supervisor-v5-config-sample-bound"
ZERO_HASH = "0" * 64


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RunLock:
    run_id: str
    source_commit: str
    task_version: str
    train_manifest_sha256: str
    development_manifest_sha256: str
    final_eval_manifest_sha256: str
    base_model_revision: str
    trainable_policy_revisions: tuple[tuple[str, str], ...]
    frozen_policy_revisions: tuple[tuple[str, str], ...]
    replacement_policy_id: str | None
    opponent_id: str
    opponent_revision: str
    allowed_constraint_hashes: tuple[str, ...]
    trainer_parity_gate_sha256: str | None = None
    credit_estimator: Literal["policy_replacement", "message_drop", "shared_return"] = "policy_replacement"
    credit_estimator_config_sha256: str | None = None
    trainer_config_sha256: str | None = None
    serving_config_sha256: str | None = None
    production_plan_sha256: str | None = None

    def validate(self) -> None:
        nonempty = (
            self.run_id,
            self.source_commit,
            self.task_version,
            self.base_model_revision,
            self.opponent_id,
            self.opponent_revision,
        )
        if not all(nonempty):
            raise ValueError("run lock contains an empty immutable identifier")
        digests = (
            self.train_manifest_sha256,
            self.development_manifest_sha256,
            self.final_eval_manifest_sha256,
            *self.allowed_constraint_hashes,
        )
        if not digests or any(not _is_sha256(value) for value in digests):
            raise ValueError("run lock contains an invalid SHA-256 digest")
        if len(self.trainable_policy_revisions) != 4:
            raise ValueError("run lock requires exactly four trainable policy revisions")
        if len({policy for policy, _ in self.trainable_policy_revisions}) != 4:
            raise ValueError("run lock policy IDs must be distinct")
        if any(not revision for _, revision in self.trainable_policy_revisions):
            raise ValueError("run lock policy revisions must be immutable")
        if self.trainer_parity_gate_sha256 is not None and not _is_sha256(self.trainer_parity_gate_sha256):
            raise ValueError("run lock trainer parity gate has an invalid SHA-256 digest")
        if self.production_plan_sha256 is not None and not _is_sha256(self.production_plan_sha256):
            raise ValueError("run lock production plan has an invalid SHA-256 digest")
        frozen = dict(self.frozen_policy_revisions)
        if len(frozen) != len(self.frozen_policy_revisions):
            raise ValueError("run lock frozen policy IDs must be distinct")
        if self.credit_estimator == "policy_replacement":
            if not self.replacement_policy_id or self.replacement_policy_id not in frozen:
                raise ValueError("replacement policy must have a frozen revision")
            if self.credit_estimator_config_sha256 is not None:
                raise ValueError("policy-replacement credit cannot bind a shared-return config")
        elif self.credit_estimator == "message_drop":
            if self.replacement_policy_id is not None:
                raise ValueError("message-drop credit cannot bind a replacement policy")
            if self.credit_estimator_config_sha256 is not None:
                raise ValueError("message-drop credit cannot bind a shared-return config")
        elif self.credit_estimator == "shared_return":
            if self.replacement_policy_id is not None:
                raise ValueError("shared-return credit cannot bind a replacement policy")
            if self.credit_estimator_config_sha256 is None or not _is_sha256(self.credit_estimator_config_sha256):
                raise ValueError("shared-return credit requires an immutable estimator config")
            if self.trainer_config_sha256 is None or not _is_sha256(self.trainer_config_sha256):
                raise ValueError("shared-return credit requires an immutable trainer config")
            if self.serving_config_sha256 is None or not _is_sha256(self.serving_config_sha256):
                raise ValueError("shared-return credit requires an immutable serving config")
        else:
            raise ValueError(f"unknown credit estimator: {self.credit_estimator}")
        if set(frozen) & {policy for policy, _ in self.trainable_policy_revisions}:
            raise ValueError("a policy cannot be both trainable and frozen")
        if any(not revision for revision in frozen.values()):
            raise ValueError("frozen policy revisions must be immutable")

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ReplayTurn:
    turn: int
    broadcasts: tuple[tuple[str, Broadcast], ...]
    delivered_broadcasts: tuple[tuple[str, Broadcast], ...]
    actions: tuple[tuple[str, Action], ...]
    pre_state_sha256: str
    post_state_sha256: str
    receiver_delivery_overrides: tuple[
        tuple[str, tuple[tuple[str, Broadcast], ...]], ...
    ] = ()


@dataclass(frozen=True)
class BranchReplay:
    replaced_agent: str | None
    turns: tuple[ReplayTurn, ...]
    terminal_return: float


@dataclass(frozen=True)
class CreditGroupEvidence:
    run_lock_sha256: str
    game_id: str
    initial_state: GameState
    initial_state_sha256: str
    episode_config: EpisodeConfig
    actual: BranchReplay
    replacements: tuple[BranchReplay, ...]
    decisions: tuple[RolloutDecision, ...]
    trainer_logprobs: dict[str, tuple[float, ...]] | None


@dataclass(frozen=True)
class MessageCreditGroupEvidence:
    run_lock_sha256: str
    game_id: str
    initial_state: GameState
    initial_state_sha256: str
    episode_config: EpisodeConfig
    intervention_turn: int
    actual: BranchReplay
    drops: tuple[BranchReplay, ...]
    decisions: tuple[RolloutDecision, ...]
    trainer_logprobs: dict[str, tuple[float, ...]] | None


@dataclass(frozen=True)
class SharedReturnSpec:
    replicas: int
    trainable_phases: tuple[Literal["BROADCAST", "ACT"], ...] = ("BROADCAST",)
    trainable_turn_offsets: tuple[int, ...] | None = (0,)
    baseline: Literal[
        "leave_one_out_mean",
        "paired_message_drop",
        "paired_target_swap",
        "paired_receiver_target_swap",
        "paired_receiver_target_swap_challenge",
    ] = "leave_one_out_mean"
    reward: Literal["verified_terminal_team_return"] = "verified_terminal_team_return"
    credit_assignment: Literal["shared_team", "focused_agent"] = "shared_team"
    action_prompt_profile: Literal["full", "focused_handoff_compact"] = "full"
    paired_contrast_centering: Literal["replica_mean", "none"] = "replica_mean"

    def validate(self) -> None:
        if self.replicas < 2 or self.replicas > 32:
            raise ValueError("shared-return replicas must be between 2 and 32")
        if not self.trainable_phases or len(set(self.trainable_phases)) != len(self.trainable_phases):
            raise ValueError("shared-return phases must be non-empty and unique")
        if any(phase not in {"BROADCAST", "ACT"} for phase in self.trainable_phases):
            raise ValueError("shared-return spec contains an unknown phase")
        if self.trainable_turn_offsets is not None:
            if not self.trainable_turn_offsets or len(set(self.trainable_turn_offsets)) != len(
                self.trainable_turn_offsets
            ):
                raise ValueError("shared-return turn offsets must be non-empty and unique")
            if any(offset < 0 for offset in self.trainable_turn_offsets):
                raise ValueError("shared-return turn offsets cannot be negative")
        if self.baseline not in {
            "leave_one_out_mean",
            "paired_message_drop",
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }:
            raise ValueError("shared-return spec contains an unsupported baseline")
        if self.reward != "verified_terminal_team_return":
            raise ValueError("shared-return spec contains an unsupported reward")
        if self.credit_assignment not in {"shared_team", "focused_agent"}:
            raise ValueError("shared-return spec contains an unsupported credit assignment")
        if self.action_prompt_profile not in {"full", "focused_handoff_compact"}:
            raise ValueError("shared-return spec contains an unsupported action prompt profile")
        if self.paired_contrast_centering not in {"replica_mean", "none"}:
            raise ValueError("shared-return spec contains an unsupported paired-contrast centering")
        if self.action_prompt_profile == "focused_handoff_compact" and (
            self.credit_assignment != "focused_agent" or self.trainable_phases != ("ACT",)
        ):
            raise ValueError("compact handoff prompts require focused-agent ACT credit")
        if self.credit_assignment == "focused_agent" and (
            len(self.trainable_phases) != 1
            or self.trainable_phases[0] not in {"BROADCAST", "ACT"}
        ):
            raise ValueError("focused-agent credit requires exactly one causal decision phase")
        if self.baseline in {
            "paired_message_drop",
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        } and (
            self.credit_assignment != "focused_agent" or self.trainable_phases != ("ACT",)
        ):
            raise ValueError("paired message intervention credit requires focused-agent ACT credit")
        if self.paired_contrast_centering == "none" and self.baseline not in {
            "paired_message_drop",
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }:
            raise ValueError("uncentered paired contrast requires a paired intervention baseline")

    @property
    def sha256(self) -> str:
        self.validate()
        payload = asdict(self)
        if self.action_prompt_profile == "full":
            # Preserve the immutable identity of completed runs created before
            # prompt profiles were introduced.
            payload.pop("action_prompt_profile")
        if self.paired_contrast_centering == "replica_mean":
            payload.pop("paired_contrast_centering")
        return canonical_sha256(payload)


@dataclass(frozen=True)
class SharedReturnReplicaEvidence:
    replica_index: int
    game_id: str
    sampling_namespace: str
    replay: BranchReplay
    decisions: tuple[RolloutDecision, ...]
    dropped_replay: BranchReplay | None = None
    dropped_decisions: tuple[RolloutDecision, ...] = ()
    swapped_replay: BranchReplay | None = None
    swapped_decisions: tuple[RolloutDecision, ...] = ()


@dataclass(frozen=True)
class SharedReturnGroupEvidence:
    run_lock_sha256: str
    group_id: str
    initial_state: GameState
    initial_state_sha256: str
    episode_config: EpisodeConfig
    spec: SharedReturnSpec
    focused_agent: str | None
    common_sampling_namespace: str | None
    replicas: tuple[SharedReturnReplicaEvidence, ...]
    trainer_logprobs: dict[str, tuple[float, ...]] | None
    message_drop_agent: str | None = None
    message_drop_turn: int | None = None
    message_swap_agent: str | None = None
    message_swap_turn: int | None = None
    message_swap_targets: tuple[str, str] | None = None
    message_swap_active_target: str | None = None
    message_swap_sender_sampling_namespace: str | None = None


@dataclass(frozen=True)
class Approval:
    supervisor_version: str
    run_lock_sha256: str
    game_id: str
    evidence_sha256: str
    replay_return: float
    logprob_max_abs_error: float | None
    logprob_mean_abs_error: float | None
    logprob_p99_abs_error: float | None
    probability_max_abs_error: float | None
    probability_p99_abs_error: float | None
    probability_tail_fraction: float | None
    mismatch_kl_mean: float | None
    mismatch_kl_max: float | None
    envelopes: tuple[PolicyTrainingEnvelope, ...]
    parity_mode: Literal["pre_admission", "trainer_pre_step"]
    trainer_parity_gate_sha256: str | None
    signature: str


@dataclass(frozen=True)
class ReplayVerification:
    terminal_return: float
    context_sha256: dict[tuple[str, int, str], str]
    output_sha256: dict[tuple[str, int, str], str]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _state_sha256(state: GameState) -> str:
    return canonical_sha256(state_to_dict(state))


def _evidence_payload(evidence: CreditGroupEvidence) -> dict[str, Any]:
    return {
        "run_lock_sha256": evidence.run_lock_sha256,
        "game_id": evidence.game_id,
        "initial_state_sha256": evidence.initial_state_sha256,
        "episode_config": asdict(evidence.episode_config),
        "actual": _branch_payload(evidence.actual),
        "replacements": [_branch_payload(row) for row in evidence.replacements],
        "decisions": [asdict(row) for row in evidence.decisions],
        "trainer_logprobs": evidence.trainer_logprobs,
    }


def _message_evidence_payload(evidence: MessageCreditGroupEvidence) -> dict[str, Any]:
    return {
        "run_lock_sha256": evidence.run_lock_sha256,
        "game_id": evidence.game_id,
        "initial_state_sha256": evidence.initial_state_sha256,
        "episode_config": asdict(evidence.episode_config),
        "intervention_turn": evidence.intervention_turn,
        "actual": _branch_payload(evidence.actual),
        "drops": [_branch_payload(row) for row in evidence.drops],
        "decisions": [asdict(row) for row in evidence.decisions],
        "trainer_logprobs": evidence.trainer_logprobs,
    }


def shared_return_evidence_payload(evidence: SharedReturnGroupEvidence) -> dict[str, Any]:
    payload = {
        "run_lock_sha256": evidence.run_lock_sha256,
        "group_id": evidence.group_id,
        "initial_state_sha256": evidence.initial_state_sha256,
        "initial_state": state_to_dict(evidence.initial_state),
        "episode_config": asdict(evidence.episode_config),
        "spec": asdict(evidence.spec),
        "focused_agent": evidence.focused_agent,
        "common_sampling_namespace": evidence.common_sampling_namespace,
        "replicas": [
            {
                "replica_index": row.replica_index,
                "game_id": row.game_id,
                "sampling_namespace": row.sampling_namespace,
                "replay": _branch_payload(row.replay),
                "decisions": [asdict(decision) for decision in row.decisions],
            }
            for row in evidence.replicas
        ],
        "trainer_logprobs": evidence.trainer_logprobs,
    }
    if evidence.spec.baseline == "paired_message_drop":
        payload["message_drop_agent"] = evidence.message_drop_agent
        payload["message_drop_turn"] = evidence.message_drop_turn
        for replica_payload, replica in zip(payload["replicas"], evidence.replicas, strict=True):
            replica_payload["dropped_replay"] = (
                _branch_payload(replica.dropped_replay) if replica.dropped_replay else None
            )
            replica_payload["dropped_decisions"] = [asdict(row) for row in replica.dropped_decisions]
    elif evidence.spec.baseline in {
        "paired_target_swap",
        "paired_receiver_target_swap",
        "paired_receiver_target_swap_challenge",
    }:
        payload["message_swap_agent"] = evidence.message_swap_agent
        payload["message_swap_turn"] = evidence.message_swap_turn
        payload["message_swap_targets"] = evidence.message_swap_targets
        payload["message_swap_active_target"] = evidence.message_swap_active_target
        if evidence.message_swap_sender_sampling_namespace is not None:
            payload["message_swap_sender_sampling_namespace"] = (
                evidence.message_swap_sender_sampling_namespace
            )
        for replica_payload, replica in zip(payload["replicas"], evidence.replicas, strict=True):
            replica_payload["swapped_replay"] = (
                _branch_payload(replica.swapped_replay) if replica.swapped_replay else None
            )
            replica_payload["swapped_decisions"] = [asdict(row) for row in replica.swapped_decisions]
    return payload


def _branch_payload(branch: BranchReplay) -> dict[str, Any]:
    return {
        "replaced_agent": branch.replaced_agent,
        "terminal_return": branch.terminal_return,
        "turns": [
            {
                "turn": row.turn,
                "broadcasts": [(agent, value.to_dict()) for agent, value in row.broadcasts],
                "delivered_broadcasts": [(agent, value.to_dict()) for agent, value in row.delivered_broadcasts],
                "actions": [(agent, value.to_dict()) for agent, value in row.actions],
                "pre_state_sha256": row.pre_state_sha256,
                "post_state_sha256": row.post_state_sha256,
                "receiver_delivery_overrides": [
                    (
                        receiver,
                        [(sender, message.to_dict()) for sender, message in overrides],
                    )
                    for receiver, overrides in row.receiver_delivery_overrides
                ],
            }
            for row in branch.turns
        ],
    }


def verify_replay(
    initial_state: GameState,
    config: EpisodeConfig,
    branch: BranchReplay,
    trainable_team: Team,
) -> ReplayVerification:
    env = ArenaRLEnv(
        size=len(initial_state.nodes),
        config=config,
    )
    env.reset_from_state(initial_state)
    final = None
    contexts: dict[tuple[str, int, str], str] = {}
    outputs: dict[tuple[str, int, str], str] = {}
    for expected_turn, row in enumerate(branch.turns, start=initial_state.turn):
        if row.turn != expected_turn:
            raise ValueError("replay turns are not contiguous")
        if _state_sha256(env._require_state()) != row.pre_state_sha256:
            raise ValueError(f"replay pre-state mismatch at turn {row.turn}")
        for agent_id, observation in env.observations().items():
            contexts[(agent_id, row.turn, "BROADCAST")] = canonical_sha256(observation)
        env.broadcast_phase(
            dict(row.broadcasts),
            delivered_broadcasts=dict(row.delivered_broadcasts),
            receiver_delivery_overrides={
                receiver: dict(overrides)
                for receiver, overrides in row.receiver_delivery_overrides
            },
        )
        accepted = env._phase.accepted if env._phase is not None else {}
        for agent_id, broadcast in accepted.items():
            outputs[(agent_id, row.turn, "BROADCAST")] = canonical_sha256(broadcast.to_dict())
        for agent_id, observation in env.action_observations().items():
            contexts[(agent_id, row.turn, "ACT")] = canonical_sha256(observation)
        final = env.advance(dict(row.actions))
        for agent_id, action in row.actions:
            outputs[(agent_id, row.turn, "ACT")] = canonical_sha256(action.to_dict())
        if _state_sha256(env._require_state()) != row.post_state_sha256:
            raise ValueError(f"replay post-state mismatch at turn {row.turn}")
    if final is None or not (final.terminated or final.truncated):
        raise ValueError("credit-group replay does not reach a terminal state")
    replay_return = float(final.rewards[trainable_team])
    if not math.isclose(replay_return, branch.terminal_return, abs_tol=1e-12):
        raise ValueError("recorded branch return disagrees with independent replay")
    return ReplayVerification(replay_return, contexts, outputs)


def _verify_private_contexts(
    decisions: tuple[RolloutDecision, ...],
    verification: ReplayVerification,
    *,
    branch: str,
) -> None:
    actual = {(decision.agent_id, decision.turn, decision.phase): decision.context_sha256 for decision in decisions}
    if len(actual) != len(decisions):
        raise ValueError(f"duplicate private contexts in {branch} branch")
    if set(actual) != set(verification.context_sha256):
        raise ValueError(f"{branch} decisions do not cover every referee-generated private context")
    mismatched = [key for key, digest in actual.items() if digest != verification.context_sha256[key]]
    if mismatched:
        raise ValueError(f"private observation/inbox leakage or mismatch in {branch}: {mismatched[:3]}")
    output_mismatches = [
        (decision.agent_id, decision.turn, decision.phase)
        for decision in decisions
        if decision.output_sha256 != verification.output_sha256[(decision.agent_id, decision.turn, decision.phase)]
    ]
    if output_mismatches:
        raise ValueError(f"decoded model output does not match replayed action in {branch}: {output_mismatches[:3]}")


def _verify_delivery_contract(
    initial_state: GameState,
    config: EpisodeConfig,
    branch: BranchReplay,
    *,
    dropped_sender: str | None,
    dropped_turn: int | None = None,
    swapped_sender: str | None = None,
    swapped_turn: int | None = None,
    swapped_targets: tuple[str, str] | None = None,
    swapped_active_target: str | None = None,
    swapped_receiver: str | None = None,
) -> None:
    if (dropped_sender is None) != (dropped_turn is None):
        raise ValueError("message delivery intervention requires both sender and turn")
    swap_fields = (swapped_sender, swapped_turn, swapped_targets, swapped_active_target)
    if any(value is not None for value in swap_fields) != all(value is not None for value in swap_fields):
        raise ValueError("target swap requires sender, turn, candidates, and active target")
    if dropped_sender is not None and swapped_sender is not None:
        raise ValueError("one replay cannot drop and swap a message")
    env = ArenaRLEnv(size=len(initial_state.nodes), config=config)
    env.reset_from_state(initial_state)
    for row in branch.turns:
        overrides = {
            receiver: dict(values)
            for receiver, values in row.receiver_delivery_overrides
        }
        phase = env.broadcast_phase(
            dict(row.broadcasts),
            delivered_broadcasts=dict(row.delivered_broadcasts),
            receiver_delivery_overrides=overrides,
        )
        expected = dict(phase.accepted)
        if dropped_sender is not None and row.turn == dropped_turn:
            if dropped_sender not in expected:
                raise ValueError(f"message-drop branch names an unknown sender: {dropped_sender}")
            expected[dropped_sender] = EMPTY_BROADCAST
        if swapped_sender is not None and row.turn == swapped_turn:
            if swapped_sender not in expected:
                raise ValueError(f"message-swap branch names an unknown sender: {swapped_sender}")
            assert swapped_targets is not None and swapped_active_target is not None
            swapped = target_swapped_broadcast(
                expected[swapped_sender],
                candidate_targets=swapped_targets,
                active_target=swapped_active_target,
            )
            if swapped_receiver is None:
                expected[swapped_sender] = swapped
            else:
                expected_receiver_overrides = {
                    swapped_receiver: {swapped_sender: swapped}
                }
                if overrides != expected_receiver_overrides:
                    raise ValueError("receiver-only target-swap branch violates its delivery intervention")
        elif overrides:
            raise ValueError("non-swap replay cannot contain receiver delivery overrides")
        if phase.delivered != expected:
            label = (
                f"drop-message-{dropped_sender}"
                if dropped_sender is not None
                else f"swap-message-{swapped_sender}"
                if swapped_sender is not None
                else "actual"
            )
            raise ValueError(f"{label} branch violates its delivery intervention")
        env.advance(dict(row.actions))


def _verify_common_random_outputs(decisions: tuple[RolloutDecision, ...]) -> None:
    requests: dict[tuple[str, str, str, str, str], str] = {}
    responses: dict[str, str] = {}
    for decision in decisions:
        key = (
            decision.policy_id,
            decision.policy_revision,
            decision.sampling_key,
            decision.context_sha256,
            decision.constraint_sha256,
        )
        previous_request = requests.setdefault(key, decision.request_sha256)
        if previous_request != decision.request_sha256:
            raise ValueError(
                "identical policy, private context, constraint, and sampling key produced "
                f"different inference requests: {decision.decision_id}"
            )
        response_sha256 = canonical_sha256(
            {
                "completion_ids": decision.completion_ids,
                "rollout_logprobs": decision.rollout_logprobs,
                "allowed_token_ids": decision.allowed_token_ids,
                "serving_allowed_logprobs": decision.serving_allowed_logprobs,
                "output_sha256": decision.output_sha256,
            }
        )
        previous = responses.setdefault(decision.request_sha256, response_sha256)
        if previous != response_sha256:
            raise ValueError(
                f"identical inference request produced different outputs or distributions: {decision.decision_id}"
            )


def approve_credit_group(
    lock: RunLock,
    evidence: CreditGroupEvidence,
    bindings: tuple[AgentPolicy, ...],
    trainable_team: Team,
    signing_key: bytes,
) -> Approval:
    if len(signing_key) < 32:
        raise ValueError("supervisor signing key must contain at least 32 bytes")
    lock.validate()
    if lock.credit_estimator != "policy_replacement":
        raise ValueError("policy-replacement evidence cannot use a message-drop run lock")
    if evidence.run_lock_sha256 != lock.sha256:
        raise ValueError("credit group was produced under a different run lock")
    if evidence.game_id == "" or {row.game_id for row in evidence.decisions} != {evidence.game_id}:
        raise ValueError("credit group mixes or omits game IDs")
    if _state_sha256(evidence.initial_state) != evidence.initial_state_sha256:
        raise ValueError("initial state does not match its committed hash")
    policy_revisions = {
        **dict(lock.trainable_policy_revisions),
        **dict(lock.frozen_policy_revisions),
    }
    binding_by_agent = {binding.agent_id: binding for binding in bindings}
    allowed_constraints = set(lock.allowed_constraint_hashes)
    for decision in evidence.decisions:
        if decision.constraint_sha256 not in allowed_constraints:
            raise ValueError(f"unapproved dynamic constraint: {decision.decision_id}")
        binding = binding_by_agent.get(decision.agent_id)
        if binding is None:
            raise ValueError(f"decision has an unknown agent: {decision.decision_id}")
        expected_policy = (
            lock.replacement_policy_id
            if decision.branch == "replacement" and decision.agent_id == decision.replaced_agent
            else binding.policy_id
        )
        if decision.policy_id != expected_policy:
            raise ValueError(f"wrong actual/replacement policy routing: {decision.decision_id}")
        expected_revision = policy_revisions.get(expected_policy)
        if expected_revision is None or decision.policy_revision != expected_revision:
            raise ValueError(f"stale or unexpected trainable policy revision: {decision.decision_id}")

    if evidence.actual.replaced_agent is not None:
        raise ValueError("actual replay cannot name a replaced agent")
    actual_verification = verify_replay(
        evidence.initial_state,
        evidence.episode_config,
        evidence.actual,
        trainable_team,
    )
    _verify_private_contexts(
        tuple(row for row in evidence.decisions if row.branch == "actual"),
        actual_verification,
        branch="actual",
    )
    replay_return = actual_verification.terminal_return
    replacement_agents = [row.replaced_agent for row in evidence.replacements]
    expected_replacements = sorted(
        binding.agent_id for binding in bindings if binding.trainable and binding.team == trainable_team
    )
    if (
        len(replacement_agents) != 4
        or sorted(agent for agent in replacement_agents if agent is not None) != expected_replacements
    ):
        raise ValueError("replacement replays do not cover each trainable agent exactly once")
    replacement_returns = {}
    for branch in evidence.replacements:
        verification = verify_replay(
            evidence.initial_state,
            evidence.episode_config,
            branch,
            trainable_team,
        )
        _verify_private_contexts(
            tuple(
                row
                for row in evidence.decisions
                if row.branch == "replacement" and row.replaced_agent == branch.replaced_agent
            ),
            verification,
            branch=f"replace-{branch.replaced_agent}",
        )
        replacement_returns[branch.replaced_agent] = verification.terminal_return
    credits = tuple(
        ReplacementCredit(
            binding.agent_id,
            binding.policy_id,
            evidence.actual.terminal_return,
            replacement_returns[binding.agent_id],
            evidence.actual.terminal_return - replacement_returns[binding.agent_id],
        )
        for binding in bindings
        if binding.trainable and binding.team == trainable_team
    )
    envelopes = build_training_envelopes(
        evidence.decisions,
        bindings,
        credits,
        trainable_team,
    )
    if evidence.trainer_logprobs is None:
        if lock.trainer_parity_gate_sha256 is None:
            raise ValueError("deferred parity requires an immutable trainer pre-step gate")
        parity_mode = "trainer_pre_step"
        parity: dict[str, float | int | str | None] = {
            "max_abs_error": None,
            "mean_abs_error": None,
            "p99_abs_error": None,
            "max_probability_error": None,
            "p99_probability_error": None,
            "probability_tail_fraction": None,
            "mean_mismatch_kl": None,
            "max_mismatch_kl": None,
        }
    else:
        parity_mode = "pre_admission"
        parity = verify_trainer_logprob_parity(
            evidence.decisions,
            evidence.trainer_logprobs,
            frozenset(dict(lock.trainable_policy_revisions)),
        )

    def optional_float(value: float | int | str | None) -> float | None:
        return None if value is None else float(value)

    unsigned = {
        "supervisor_version": SUPERVISOR_VERSION,
        "run_lock_sha256": lock.sha256,
        "game_id": evidence.game_id,
        "evidence_sha256": canonical_sha256(_evidence_payload(evidence)),
        "replay_return": replay_return,
        "logprob_max_abs_error": optional_float(parity["max_abs_error"]),
        "logprob_mean_abs_error": optional_float(parity["mean_abs_error"]),
        "logprob_p99_abs_error": optional_float(parity["p99_abs_error"]),
        "probability_max_abs_error": optional_float(parity["max_probability_error"]),
        "probability_p99_abs_error": optional_float(parity["p99_probability_error"]),
        "probability_tail_fraction": optional_float(parity["probability_tail_fraction"]),
        "mismatch_kl_mean": optional_float(parity["mean_mismatch_kl"]),
        "mismatch_kl_max": optional_float(parity["max_mismatch_kl"]),
        "envelopes": [asdict(row) for row in envelopes],
        "parity_mode": parity_mode,
        "trainer_parity_gate_sha256": lock.trainer_parity_gate_sha256,
    }
    signature = hmac.new(
        signing_key,
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    return Approval(
        SUPERVISOR_VERSION,
        lock.sha256,
        evidence.game_id,
        unsigned["evidence_sha256"],
        replay_return,
        optional_float(parity["max_abs_error"]),
        optional_float(parity["mean_abs_error"]),
        optional_float(parity["p99_abs_error"]),
        optional_float(parity["max_probability_error"]),
        optional_float(parity["p99_probability_error"]),
        optional_float(parity["probability_tail_fraction"]),
        optional_float(parity["mean_mismatch_kl"]),
        optional_float(parity["max_mismatch_kl"]),
        envelopes,
        parity_mode,
        lock.trainer_parity_gate_sha256,
        signature,
    )


def approve_message_credit_group(
    lock: RunLock,
    evidence: MessageCreditGroupEvidence,
    bindings: tuple[AgentPolicy, ...],
    trainable_team: Team,
    signing_key: bytes,
) -> Approval:
    """Approve terminal-return credit for sender broadcasts only."""
    if len(signing_key) < 32:
        raise ValueError("supervisor signing key must contain at least 32 bytes")
    lock.validate()
    if lock.credit_estimator != "message_drop":
        raise ValueError("message-drop evidence requires a message-drop run lock")
    if evidence.run_lock_sha256 != lock.sha256:
        raise ValueError("message-credit group was produced under a different run lock")
    if evidence.game_id == "" or {row.game_id for row in evidence.decisions} != {evidence.game_id}:
        raise ValueError("message-credit group mixes or omits game IDs")
    if _state_sha256(evidence.initial_state) != evidence.initial_state_sha256:
        raise ValueError("initial state does not match its committed hash")

    policy_revisions = {
        **dict(lock.trainable_policy_revisions),
        **dict(lock.frozen_policy_revisions),
    }
    binding_by_agent = {binding.agent_id: binding for binding in bindings}
    allowed_constraints = set(lock.allowed_constraint_hashes)
    for decision in evidence.decisions:
        if decision.branch not in {"actual", "message_drop"}:
            raise ValueError(f"unexpected branch in message-credit evidence: {decision.decision_id}")
        if decision.constraint_sha256 not in allowed_constraints:
            raise ValueError(f"unapproved dynamic constraint: {decision.decision_id}")
        binding = binding_by_agent.get(decision.agent_id)
        if binding is None:
            raise ValueError(f"decision has an unknown agent: {decision.decision_id}")
        if decision.policy_id != binding.policy_id:
            raise ValueError(f"message intervention changed policy routing: {decision.decision_id}")
        expected_revision = policy_revisions.get(binding.policy_id)
        if expected_revision is None or decision.policy_revision != expected_revision:
            raise ValueError(f"stale or unexpected policy revision: {decision.decision_id}")
    _verify_common_random_outputs(evidence.decisions)

    if evidence.actual.replaced_agent is not None:
        raise ValueError("actual replay cannot name an intervened sender")
    actual_verification = verify_replay(
        evidence.initial_state,
        evidence.episode_config,
        evidence.actual,
        trainable_team,
    )
    _verify_delivery_contract(
        evidence.initial_state,
        evidence.episode_config,
        evidence.actual,
        dropped_sender=None,
        dropped_turn=None,
    )
    _verify_private_contexts(
        tuple(row for row in evidence.decisions if row.branch == "actual"),
        actual_verification,
        branch="actual",
    )

    expected_senders = sorted(
        binding.agent_id for binding in bindings if binding.trainable and binding.team == trainable_team
    )
    dropped_senders = [row.replaced_agent for row in evidence.drops]
    if (
        len(dropped_senders) != 4
        or sorted(sender for sender in dropped_senders if sender is not None) != expected_senders
    ):
        raise ValueError("message-drop replays do not cover each trainable sender exactly once")
    if evidence.intervention_turn != evidence.initial_state.turn:
        raise ValueError("bootstrap message credit must intervene on the first episode turn")

    dropped_returns: dict[str, float] = {}
    actual_turns_by_number = {row.turn: row for row in evidence.actual.turns}
    for branch in evidence.drops:
        sender = branch.replaced_agent
        if sender is None:
            raise ValueError("message-drop replay is missing its sender")
        verification = verify_replay(
            evidence.initial_state,
            evidence.episode_config,
            branch,
            trainable_team,
        )
        _verify_delivery_contract(
            evidence.initial_state,
            evidence.episode_config,
            branch,
            dropped_sender=sender,
            dropped_turn=evidence.intervention_turn,
        )
        branch_decisions = tuple(
            row for row in evidence.decisions if row.branch == "message_drop" and row.replaced_agent == sender
        )
        _verify_private_contexts(
            branch_decisions,
            verification,
            branch=f"drop-message-{sender}",
        )
        intervention_row = next(
            (row for row in branch.turns if row.turn == evidence.intervention_turn),
            None,
        )
        actual_intervention_row = actual_turns_by_number.get(evidence.intervention_turn)
        if intervention_row is None or actual_intervention_row is None:
            raise ValueError("message-drop branch omits the intervention turn")
        if dict(intervention_row.broadcasts) != dict(actual_intervention_row.broadcasts):
            raise ValueError("emitted broadcasts changed before delivery intervention")
        actual_sender_message = dict(actual_intervention_row.broadcasts).get(sender, EMPTY_BROADCAST)
        if actual_sender_message == EMPTY_BROADCAST:
            if branch.turns != evidence.actual.turns or not math.isclose(
                branch.terminal_return,
                evidence.actual.terminal_return,
                abs_tol=1e-12,
            ):
                raise ValueError("dropping an already-empty sender message changed the trajectory")
        dropped_returns[sender] = verification.terminal_return

    credits = tuple(
        MessageCredit(
            binding.agent_id,
            binding.policy_id,
            evidence.actual.terminal_return,
            dropped_returns[binding.agent_id],
            evidence.actual.terminal_return - dropped_returns[binding.agent_id],
        )
        for binding in bindings
        if binding.trainable and binding.team == trainable_team
    )
    envelopes = build_message_training_envelopes(
        evidence.decisions,
        bindings,
        credits,
        trainable_team,
        credit_turn=evidence.intervention_turn,
    )

    if evidence.trainer_logprobs is None:
        if lock.trainer_parity_gate_sha256 is None:
            raise ValueError("deferred parity requires an immutable trainer pre-step gate")
        parity_mode: Literal["pre_admission", "trainer_pre_step"] = "trainer_pre_step"
        parity: dict[str, float | int | str | None] = {
            "max_abs_error": None,
            "mean_abs_error": None,
            "p99_abs_error": None,
            "max_probability_error": None,
            "p99_probability_error": None,
            "probability_tail_fraction": None,
            "mean_mismatch_kl": None,
            "max_mismatch_kl": None,
        }
    else:
        parity_mode = "pre_admission"
        parity = verify_trainer_logprob_parity(
            evidence.decisions,
            evidence.trainer_logprobs,
            frozenset(dict(lock.trainable_policy_revisions)),
            trainable_phases=frozenset({"BROADCAST"}),
            trainable_turns=frozenset({evidence.intervention_turn}),
        )

    def optional_float(value: float | int | str | None) -> float | None:
        return None if value is None else float(value)

    unsigned = {
        "supervisor_version": SUPERVISOR_VERSION,
        "run_lock_sha256": lock.sha256,
        "game_id": evidence.game_id,
        "evidence_sha256": canonical_sha256(_message_evidence_payload(evidence)),
        "replay_return": actual_verification.terminal_return,
        "logprob_max_abs_error": optional_float(parity["max_abs_error"]),
        "logprob_mean_abs_error": optional_float(parity["mean_abs_error"]),
        "logprob_p99_abs_error": optional_float(parity["p99_abs_error"]),
        "probability_max_abs_error": optional_float(parity["max_probability_error"]),
        "probability_p99_abs_error": optional_float(parity["p99_probability_error"]),
        "probability_tail_fraction": optional_float(parity["probability_tail_fraction"]),
        "mismatch_kl_mean": optional_float(parity["mean_mismatch_kl"]),
        "mismatch_kl_max": optional_float(parity["max_mismatch_kl"]),
        "envelopes": [asdict(row) for row in envelopes],
        "parity_mode": parity_mode,
        "trainer_parity_gate_sha256": lock.trainer_parity_gate_sha256,
    }
    signature = hmac.new(
        signing_key,
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    return Approval(
        SUPERVISOR_VERSION,
        lock.sha256,
        evidence.game_id,
        unsigned["evidence_sha256"],
        actual_verification.terminal_return,
        optional_float(parity["max_abs_error"]),
        optional_float(parity["mean_abs_error"]),
        optional_float(parity["p99_abs_error"]),
        optional_float(parity["max_probability_error"]),
        optional_float(parity["p99_probability_error"]),
        optional_float(parity["probability_tail_fraction"]),
        optional_float(parity["mean_mismatch_kl"]),
        optional_float(parity["max_mismatch_kl"]),
        envelopes,
        parity_mode,
        lock.trainer_parity_gate_sha256,
        signature,
    )


def leave_one_out_advantages(returns: tuple[float, ...]) -> tuple[float, ...]:
    """Compute a zero-sum leave-one-out baseline without reward shaping."""
    if len(returns) < 2:
        raise ValueError("leave-one-out credit requires at least two returns")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("leave-one-out returns must be finite")
    total = sum(returns)
    denominator = len(returns) - 1
    advantages = tuple(value - (total - value) / denominator for value in returns)
    if not math.isclose(sum(advantages), 0.0, abs_tol=1e-10):
        raise ValueError("leave-one-out advantages failed their zero-sum invariant")
    return advantages


def paired_terminal_contrast_advantages(
    actual_returns: tuple[float, ...],
    counterfactual_returns: tuple[float, ...],
    *,
    centering: Literal["replica_mean", "none"] = "replica_mean",
) -> tuple[float, ...]:
    """Compute verified actual-minus-counterfactual terminal effects."""
    if len(actual_returns) != len(counterfactual_returns):
        raise ValueError("paired terminal-return contrasts must have equal lengths")
    if len(actual_returns) < 2:
        raise ValueError("paired terminal-return contrasts require at least two replicas")
    if centering not in {"replica_mean", "none"}:
        raise ValueError("unsupported paired terminal-return contrast centering")
    effects = tuple(
        actual - counterfactual
        for actual, counterfactual in zip(actual_returns, counterfactual_returns, strict=True)
    )
    if not all(math.isfinite(value) for value in effects):
        raise ValueError("paired terminal-return contrasts must be finite")
    if centering == "none":
        return effects
    return leave_one_out_advantages(effects)


def paired_message_drop_advantages(
    normal_returns: tuple[float, ...],
    dropped_returns: tuple[float, ...],
    *,
    centering: Literal["replica_mean", "none"] = "replica_mean",
) -> tuple[float, ...]:
    """Compute verified normal-minus-dropped terminal-return effects."""
    return paired_terminal_contrast_advantages(
        normal_returns,
        dropped_returns,
        centering=centering,
    )


def approve_shared_return_group(
    lock: RunLock,
    evidence: SharedReturnGroupEvidence,
    bindings: tuple[AgentPolicy, ...],
    trainable_team: Team,
    signing_key: bytes,
) -> tuple[Approval, ...]:
    """Replay and atomically approve a group of joint trajectories for four policies."""
    if len(signing_key) < 32:
        raise ValueError("supervisor signing key must contain at least 32 bytes")
    lock.validate()
    evidence.spec.validate()
    if lock.credit_estimator != "shared_return":
        raise ValueError("shared-return evidence requires a shared-return run lock")
    if lock.credit_estimator_config_sha256 != evidence.spec.sha256:
        raise ValueError("shared-return spec does not match the immutable run lock")
    if evidence.run_lock_sha256 != lock.sha256:
        raise ValueError("shared-return group was produced under a different run lock")
    if not evidence.group_id:
        raise ValueError("shared-return group ID cannot be empty")
    if _state_sha256(evidence.initial_state) != evidence.initial_state_sha256:
        raise ValueError("initial state does not match its committed hash")
    if (
        evidence.spec.trainable_turn_offsets is not None
        and evidence.initial_state.turn + max(evidence.spec.trainable_turn_offsets) >= evidence.episode_config.horizon
    ):
        raise ValueError("shared-return trainable turn falls outside the episode horizon")
    if len(evidence.replicas) != evidence.spec.replicas:
        raise ValueError("shared-return evidence has the wrong replica count")
    if {row.replica_index for row in evidence.replicas} != set(range(evidence.spec.replicas)):
        raise ValueError("shared-return replica indices must be contiguous and unique")
    game_ids = [row.game_id for row in evidence.replicas]
    namespaces = [row.sampling_namespace for row in evidence.replicas]
    if any(not value for value in game_ids + namespaces):
        raise ValueError("shared-return replica identifiers cannot be empty")
    if len(set(game_ids)) != len(game_ids) or len(set(namespaces)) != len(namespaces):
        raise ValueError("shared-return replicas require unique games and sampling namespaces")

    if evidence.spec.credit_assignment == "focused_agent":
        trainable_agents = {
            binding.agent_id for binding in bindings if binding.trainable and binding.team == trainable_team
        }
        if evidence.focused_agent not in trainable_agents:
            raise ValueError("focused-agent evidence names a non-trainable agent")
        if not evidence.common_sampling_namespace:
            raise ValueError("focused-agent evidence requires a common sampling namespace")
    elif evidence.focused_agent is not None or evidence.common_sampling_namespace is not None:
        raise ValueError("shared-team evidence cannot name focused sampling fields")
    if evidence.spec.baseline == "paired_message_drop":
        same_team_agents = {binding.agent_id for binding in bindings if binding.team == trainable_team}
        if evidence.message_drop_agent not in same_team_agents:
            raise ValueError("paired message-drop evidence names an invalid same-team sender")
        if evidence.message_drop_agent == evidence.focused_agent:
            raise ValueError("paired message-drop sender and receiver must differ")
        if evidence.message_drop_turn is None:
            raise ValueError("paired message-drop evidence requires an intervention turn")
        if any(
            value is not None
            for value in (
                evidence.message_swap_agent,
                evidence.message_swap_turn,
                evidence.message_swap_targets,
                evidence.message_swap_active_target,
                evidence.message_swap_sender_sampling_namespace,
            )
        ):
            raise ValueError("message-drop evidence cannot also name a target swap")
    elif evidence.spec.baseline in {
        "paired_target_swap",
        "paired_receiver_target_swap",
        "paired_receiver_target_swap_challenge",
    }:
        same_team_agents = {binding.agent_id for binding in bindings if binding.team == trainable_team}
        if evidence.message_swap_agent not in same_team_agents:
            raise ValueError("paired target-swap evidence names an invalid same-team sender")
        if evidence.message_swap_agent == evidence.focused_agent:
            raise ValueError("paired target-swap sender and receiver must differ")
        if (
            evidence.message_swap_turn is None
            or evidence.message_swap_targets is None
            or evidence.message_swap_active_target is None
        ):
            raise ValueError("paired target-swap evidence requires complete intervention metadata")
        if evidence.message_swap_sender_sampling_namespace == "":
            raise ValueError("sender sampling override namespace cannot be empty")
        if evidence.message_drop_agent is not None or evidence.message_drop_turn is not None:
            raise ValueError("target-swap evidence cannot also name a message drop")
    elif any(
        value is not None
        for value in (
            evidence.message_drop_agent,
            evidence.message_drop_turn,
            evidence.message_swap_agent,
            evidence.message_swap_turn,
            evidence.message_swap_targets,
            evidence.message_swap_active_target,
            evidence.message_swap_sender_sampling_namespace,
        )
    ):
        raise ValueError("leave-one-out evidence cannot name a message intervention")

    policy_revisions = {
        **dict(lock.trainable_policy_revisions),
        **dict(lock.frozen_policy_revisions),
    }
    binding_by_agent = {binding.agent_id: binding for binding in bindings}
    allowed_constraints = set(lock.allowed_constraint_hashes)
    seen_sampling_keys: set[str] = set()
    verifications: list[ReplayVerification] = []
    dropped_verifications: list[ReplayVerification] = []
    swapped_verifications: list[ReplayVerification] = []
    absolute_turns = (
        None
        if evidence.spec.trainable_turn_offsets is None
        else frozenset(evidence.initial_state.turn + offset for offset in evidence.spec.trainable_turn_offsets)
    )
    phases = frozenset(evidence.spec.trainable_phases)

    for replica in sorted(evidence.replicas, key=lambda row: row.replica_index):
        if replica.replay.replaced_agent is not None:
            raise ValueError("shared-return replay cannot name an intervention")
        if not replica.decisions or {row.game_id for row in replica.decisions} != {replica.game_id}:
            raise ValueError("shared-return replica mixes or omits game IDs")
        replica_keys = {row.sampling_key for row in replica.decisions}
        if len(replica_keys) != len(replica.decisions):
            raise ValueError("shared-return replica contains duplicate sampling keys")
        if evidence.spec.credit_assignment == "shared_team":
            if any(not key.startswith(f"{replica.sampling_namespace}:") for key in replica_keys):
                raise ValueError("shared-return decision escaped its sampling namespace")
            if seen_sampling_keys & replica_keys:
                raise ValueError("shared-return replicas do not have independent sampling keys")
            seen_sampling_keys.update(replica_keys)
        else:
            assert evidence.focused_agent is not None
            assert evidence.common_sampling_namespace is not None
            for decision in replica.decisions:
                independently_sampled = (
                    decision.agent_id == evidence.focused_agent
                    and decision.phase in phases
                    and (absolute_turns is None or decision.turn in absolute_turns)
                )
                expected_namespace = (
                    evidence.message_swap_sender_sampling_namespace
                    if evidence.message_swap_sender_sampling_namespace is not None
                    and decision.agent_id == evidence.message_swap_agent
                    and decision.phase == "BROADCAST"
                    and decision.turn == evidence.message_swap_turn
                    else replica.sampling_namespace
                    if independently_sampled
                    else evidence.common_sampling_namespace
                )
                if not decision.sampling_key.startswith(f"{expected_namespace}:"):
                    raise ValueError("focused-agent decision escaped its declared sampling namespace")
                if independently_sampled:
                    if decision.sampling_key in seen_sampling_keys:
                        raise ValueError("focused-agent replicas reused an independent sampling key")
                    seen_sampling_keys.add(decision.sampling_key)

        for decision in replica.decisions:
            if decision.branch != "actual" or decision.replaced_agent is not None:
                raise ValueError("shared-return evidence can contain only actual decisions")
            if decision.constraint_sha256 not in allowed_constraints:
                raise ValueError(f"unapproved dynamic constraint: {decision.decision_id}")
            binding = binding_by_agent.get(decision.agent_id)
            if binding is None or decision.policy_id != binding.policy_id:
                raise ValueError(f"wrong shared-return policy routing: {decision.decision_id}")
            expected_revision = policy_revisions.get(binding.policy_id)
            if expected_revision is None or decision.policy_revision != expected_revision:
                raise ValueError(f"stale or unexpected policy revision: {decision.decision_id}")

        verification = verify_replay(
            evidence.initial_state,
            evidence.episode_config,
            replica.replay,
            trainable_team,
        )
        _verify_delivery_contract(
            evidence.initial_state,
            evidence.episode_config,
            replica.replay,
            dropped_sender=None,
            dropped_turn=None,
        )
        _verify_private_contexts(
            replica.decisions,
            verification,
            branch=f"replica-{replica.replica_index}",
        )
        verifications.append(verification)
        if evidence.spec.baseline == "paired_message_drop":
            if replica.dropped_replay is None or not replica.dropped_decisions:
                raise ValueError("paired message-drop replica omits its dropped branch")
            if replica.dropped_replay.replaced_agent != evidence.message_drop_agent:
                raise ValueError("paired message-drop replay names the wrong sender")
            if {row.game_id for row in replica.dropped_decisions} != {f"{replica.game_id}:drop"}:
                raise ValueError("paired message-drop decisions mix or omit their drop game ID")
            for decision in replica.dropped_decisions:
                if decision.branch != "message_drop" or decision.replaced_agent != evidence.message_drop_agent:
                    raise ValueError("paired message-drop evidence contains a non-drop decision")
                if decision.constraint_sha256 not in allowed_constraints:
                    raise ValueError(f"unapproved dynamic constraint: {decision.decision_id}")
                binding = binding_by_agent.get(decision.agent_id)
                if binding is None or decision.policy_id != binding.policy_id:
                    raise ValueError(f"wrong paired-drop policy routing: {decision.decision_id}")
                expected_revision = policy_revisions.get(binding.policy_id)
                if expected_revision is None or decision.policy_revision != expected_revision:
                    raise ValueError(f"stale or unexpected policy revision: {decision.decision_id}")
            dropped_verification = verify_replay(
                evidence.initial_state,
                evidence.episode_config,
                replica.dropped_replay,
                trainable_team,
            )
            _verify_delivery_contract(
                evidence.initial_state,
                evidence.episode_config,
                replica.dropped_replay,
                dropped_sender=evidence.message_drop_agent,
                dropped_turn=evidence.message_drop_turn,
            )
            _verify_private_contexts(
                replica.dropped_decisions,
                dropped_verification,
                branch=f"replica-{replica.replica_index}-drop",
            )
            dropped_verifications.append(dropped_verification)
        elif replica.dropped_replay is not None or replica.dropped_decisions:
            raise ValueError("non-drop replica cannot contain a dropped branch")

        if evidence.spec.baseline in {
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }:
            if replica.swapped_replay is None or not replica.swapped_decisions:
                raise ValueError("paired target-swap replica omits its swapped branch")
            if replica.swapped_replay.replaced_agent != evidence.message_swap_agent:
                raise ValueError("paired target-swap replay names the wrong sender")
            if {row.game_id for row in replica.swapped_decisions} != {f"{replica.game_id}:swap"}:
                raise ValueError("paired target-swap decisions mix or omit their swap game ID")
            for decision in replica.swapped_decisions:
                if decision.branch != "message_swap" or decision.replaced_agent != evidence.message_swap_agent:
                    raise ValueError("paired target-swap evidence contains a non-swap decision")
                if decision.constraint_sha256 not in allowed_constraints:
                    raise ValueError(f"unapproved dynamic constraint: {decision.decision_id}")
                binding = binding_by_agent.get(decision.agent_id)
                if binding is None or decision.policy_id != binding.policy_id:
                    raise ValueError(f"wrong paired-swap policy routing: {decision.decision_id}")
                expected_revision = policy_revisions.get(binding.policy_id)
                if expected_revision is None or decision.policy_revision != expected_revision:
                    raise ValueError(f"stale or unexpected policy revision: {decision.decision_id}")
            swapped_verification = verify_replay(
                evidence.initial_state,
                evidence.episode_config,
                replica.swapped_replay,
                trainable_team,
            )
            _verify_delivery_contract(
                evidence.initial_state,
                evidence.episode_config,
                replica.swapped_replay,
                dropped_sender=None,
                swapped_sender=evidence.message_swap_agent,
                swapped_turn=evidence.message_swap_turn,
                swapped_targets=evidence.message_swap_targets,
                swapped_active_target=evidence.message_swap_active_target,
                swapped_receiver=(
                    evidence.focused_agent
                    if evidence.spec.baseline in {
                        "paired_receiver_target_swap",
                        "paired_receiver_target_swap_challenge",
                    }
                    else None
                ),
            )
            _verify_private_contexts(
                replica.swapped_decisions,
                swapped_verification,
                branch=f"replica-{replica.replica_index}-swap",
            )
            swapped_verifications.append(swapped_verification)
            if evidence.spec.baseline in {
                "paired_receiver_target_swap",
                "paired_receiver_target_swap_challenge",
            }:
                assert evidence.focused_agent is not None
                assert evidence.message_swap_turn is not None
                actual_at_intervention = {
                    (row.agent_id, row.phase): row
                    for row in replica.decisions
                    if row.turn == evidence.message_swap_turn
                }
                swapped_at_intervention = {
                    (row.agent_id, row.phase): row
                    for row in replica.swapped_decisions
                    if row.turn == evidence.message_swap_turn
                }
                if set(actual_at_intervention) != set(swapped_at_intervention):
                    raise ValueError("receiver-only target swap changed decision coverage")
                for key, actual_decision in actual_at_intervention.items():
                    swapped_decision = swapped_at_intervention[key]
                    agent_id, phase = key
                    if agent_id == evidence.focused_agent and phase == "ACT":
                        if actual_decision.context_sha256 == swapped_decision.context_sha256:
                            raise ValueError("receiver-only target swap did not change the receiver inbox")
                        if actual_decision.sampling_key != swapped_decision.sampling_key:
                            raise ValueError(
                                "receiver-only target swap did not preserve common randomness"
                            )
                        continue
                    if (
                        actual_decision.context_sha256 != swapped_decision.context_sha256
                        or actual_decision.output_sha256 != swapped_decision.output_sha256
                    ):
                        raise ValueError(
                            "receiver-only target swap changed a non-receiver decision"
                        )
        elif replica.swapped_replay is not None or replica.swapped_decisions:
            raise ValueError("non-swap replica cannot contain a swapped branch")
    _verify_common_random_outputs(
        tuple(
            decision
            for row in evidence.replicas
            for decision in row.decisions + row.dropped_decisions + row.swapped_decisions
        )
    )

    challenge_baseline = (
        evidence.spec.baseline == "paired_receiver_target_swap_challenge"
    )
    advantages = (
        paired_message_drop_advantages(
            tuple(row.terminal_return for row in verifications),
            tuple(row.terminal_return for row in dropped_verifications),
            centering=evidence.spec.paired_contrast_centering,
        )
        if evidence.spec.baseline == "paired_message_drop"
        else paired_terminal_contrast_advantages(
            tuple(
                row.terminal_return
                for row in (
                    swapped_verifications if challenge_baseline else verifications
                )
            ),
            tuple(
                row.terminal_return
                for row in (
                    verifications if challenge_baseline else swapped_verifications
                )
            ),
            centering=evidence.spec.paired_contrast_centering,
        )
        if evidence.spec.baseline in {
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }
        else leave_one_out_advantages(tuple(row.terminal_return for row in verifications))
    )
    envelopes = tuple(
        build_shared_return_training_envelopes(
            replica.swapped_decisions if challenge_baseline else replica.decisions,
            bindings,
            trainable_team,
            advantage,
            trainable_phases=phases,
            trainable_turns=absolute_turns,
            focused_agent=evidence.focused_agent,
            expected_branch=("message_swap" if challenge_baseline else "actual"),
            expected_replaced_agent=(
                evidence.message_swap_agent if challenge_baseline else None
            ),
        )
        for replica, advantage in zip(
            sorted(evidence.replicas, key=lambda row: row.replica_index),
            advantages,
            strict=True,
        )
    )

    if evidence.trainer_logprobs is None:
        if lock.trainer_parity_gate_sha256 is None:
            raise ValueError("deferred parity requires an immutable trainer pre-step gate")
        parity_mode: Literal["pre_admission", "trainer_pre_step"] = "trainer_pre_step"
        parities: tuple[dict[str, float | int | str | None], ...] = tuple(
            {
                "max_abs_error": None,
                "mean_abs_error": None,
                "p99_abs_error": None,
                "max_probability_error": None,
                "p99_probability_error": None,
                "probability_tail_fraction": None,
                "mean_mismatch_kl": None,
                "max_mismatch_kl": None,
            }
            for _ in evidence.replicas
        )
    else:
        parity_mode = "pre_admission"
        expected_ids = {
            decision.decision_id
            for replica in evidence.replicas
            for decision in (
                replica.swapped_decisions if challenge_baseline else replica.decisions
            )
            if decision.policy_id in dict(lock.trainable_policy_revisions)
            and decision.phase in phases
            and (absolute_turns is None or decision.turn in absolute_turns)
        }
        if set(evidence.trainer_logprobs) != expected_ids:
            raise ValueError("trainer log-prob rows do not exactly match shared-return spans")
        parities = tuple(
            verify_trainer_logprob_parity(
                replica.swapped_decisions if challenge_baseline else replica.decisions,
                {
                    decision_id: values
                    for decision_id, values in evidence.trainer_logprobs.items()
                    if decision_id.startswith(f"{replica.game_id}:")
                },
                frozenset(dict(lock.trainable_policy_revisions)),
                trainable_phases=phases,
                trainable_turns=absolute_turns,
            )
            for replica in sorted(evidence.replicas, key=lambda row: row.replica_index)
        )

    evidence_sha256 = canonical_sha256(shared_return_evidence_payload(evidence))

    def optional_float(value: float | int | str | None) -> float | None:
        return None if value is None else float(value)

    approvals = []
    training_verifications = (
        swapped_verifications if challenge_baseline else verifications
    )
    for replica, verification, replica_envelopes, parity in zip(
        sorted(evidence.replicas, key=lambda row: row.replica_index),
        training_verifications,
        envelopes,
        parities,
        strict=True,
    ):
        approval_game_id = replica_envelopes[0].game_id
        unsigned = {
            "supervisor_version": SUPERVISOR_VERSION,
            "run_lock_sha256": lock.sha256,
            "game_id": approval_game_id,
            "evidence_sha256": evidence_sha256,
            "replay_return": verification.terminal_return,
            "logprob_max_abs_error": optional_float(parity["max_abs_error"]),
            "logprob_mean_abs_error": optional_float(parity["mean_abs_error"]),
            "logprob_p99_abs_error": optional_float(parity["p99_abs_error"]),
            "probability_max_abs_error": optional_float(parity["max_probability_error"]),
            "probability_p99_abs_error": optional_float(parity["p99_probability_error"]),
            "probability_tail_fraction": optional_float(parity["probability_tail_fraction"]),
            "mismatch_kl_mean": optional_float(parity["mean_mismatch_kl"]),
            "mismatch_kl_max": optional_float(parity["max_mismatch_kl"]),
            "envelopes": [asdict(row) for row in replica_envelopes],
            "parity_mode": parity_mode,
            "trainer_parity_gate_sha256": lock.trainer_parity_gate_sha256,
        }
        signature = hmac.new(
            signing_key,
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()
        approvals.append(
            Approval(
                SUPERVISOR_VERSION,
                lock.sha256,
                approval_game_id,
                evidence_sha256,
                verification.terminal_return,
                optional_float(parity["max_abs_error"]),
                optional_float(parity["mean_abs_error"]),
                optional_float(parity["p99_abs_error"]),
                optional_float(parity["max_probability_error"]),
                optional_float(parity["p99_probability_error"]),
                optional_float(parity["probability_tail_fraction"]),
                optional_float(parity["mean_mismatch_kl"]),
                optional_float(parity["max_mismatch_kl"]),
                replica_envelopes,
                parity_mode,
                lock.trainer_parity_gate_sha256,
                signature,
            )
        )
    return tuple(approvals)


def verify_approval_signature(approval: Approval, signing_key: bytes) -> None:
    unsigned = {
        "supervisor_version": approval.supervisor_version,
        "run_lock_sha256": approval.run_lock_sha256,
        "game_id": approval.game_id,
        "evidence_sha256": approval.evidence_sha256,
        "replay_return": approval.replay_return,
        "logprob_max_abs_error": approval.logprob_max_abs_error,
        "logprob_mean_abs_error": approval.logprob_mean_abs_error,
        "logprob_p99_abs_error": approval.logprob_p99_abs_error,
        "probability_max_abs_error": approval.probability_max_abs_error,
        "probability_p99_abs_error": approval.probability_p99_abs_error,
        "probability_tail_fraction": approval.probability_tail_fraction,
        "mismatch_kl_mean": approval.mismatch_kl_mean,
        "mismatch_kl_max": approval.mismatch_kl_max,
        "envelopes": [asdict(row) for row in approval.envelopes],
        "parity_mode": approval.parity_mode,
        "trainer_parity_gate_sha256": approval.trainer_parity_gate_sha256,
    }
    expected = hmac.new(
        signing_key,
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(approval.signature, expected):
        raise ValueError("invalid supervisor approval signature")


def append_hash_chained_record(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = ZERO_HASH
    sequence = 0
    if path.exists():
        records = verify_hash_chain(path)
        if records:
            previous = records[-1]["record_sha256"]
            sequence = int(records[-1]["sequence"]) + 1
    body = {"sequence": sequence, "previous_sha256": previous, "payload": payload}
    record = {**body, "record_sha256": canonical_sha256(body)}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return str(record["record_sha256"])


def verify_hash_chain(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    previous = ZERO_HASH
    for sequence, record in enumerate(records):
        if record.get("sequence") != sequence or record.get("previous_sha256") != previous:
            raise ValueError(f"trace chain ordering mismatch at record {sequence}")
        body = {
            "sequence": record["sequence"],
            "previous_sha256": record["previous_sha256"],
            "payload": record["payload"],
        }
        if canonical_sha256(body) != record.get("record_sha256"):
            raise ValueError(f"trace record hash mismatch at record {sequence}")
        previous = record["record_sha256"]
    return records
