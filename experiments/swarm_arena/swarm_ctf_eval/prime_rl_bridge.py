from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .arena import Team
from .multi_policy_contract import (
    AgentPolicy,
    AgentTokenSpan,
    ReplacementCredit,
    validate_policy_roster,
    validate_token_spans,
)

BranchKind = Literal["actual", "replacement"]


@dataclass(frozen=True)
class RolloutDecision:
    game_id: str
    branch: BranchKind
    replaced_agent: str | None
    agent_id: str
    policy_id: str
    policy_revision: str
    team: Team
    turn: int
    phase: Literal["BROADCAST", "ACT"]
    trajectory_index: int
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    constraint_sha256: str
    sampling_key: str
    context_sha256: str
    output_sha256: str

    @property
    def decision_id(self) -> str:
        branch_id = self.branch if self.replaced_agent is None else f"replace-{self.replaced_agent}"
        return f"{self.game_id}:{branch_id}:{self.agent_id}:{self.turn}:{self.phase}"


@dataclass(frozen=True)
class PolicyTrainingEnvelope:
    game_id: str
    agent_id: str
    policy_id: str
    policy_revision: str
    advantage: float
    decision_ids: tuple[str, ...]
    completion_tokens: int


def _validate_decision(decision: RolloutDecision) -> None:
    if not decision.prompt_ids:
        raise ValueError(f"decision has an empty prompt: {decision.decision_id}")
    if not decision.completion_ids:
        raise ValueError(f"decision has an empty completion: {decision.decision_id}")
    if len(decision.completion_ids) != len(decision.rollout_logprobs):
        raise ValueError(f"completion/log-prob length mismatch: {decision.decision_id}")
    if not all(math.isfinite(value) for value in decision.rollout_logprobs):
        raise ValueError(f"non-finite rollout log probability: {decision.decision_id}")
    if len(decision.constraint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in decision.constraint_sha256
    ):
        raise ValueError(f"invalid dynamic-constraint hash: {decision.decision_id}")
    if not decision.sampling_key or not decision.policy_revision:
        raise ValueError(f"decision is missing immutable sampling/policy metadata: {decision.decision_id}")
    if len(decision.context_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in decision.context_sha256
    ):
        raise ValueError(f"invalid private-context hash: {decision.decision_id}")
    if len(decision.output_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in decision.output_sha256
    ):
        raise ValueError(f"invalid decoded-output hash: {decision.decision_id}")
    if decision.branch == "actual" and decision.replaced_agent is not None:
        raise ValueError("actual branches cannot name a replaced agent")
    if decision.branch == "replacement" and decision.replaced_agent is None:
        raise ValueError("replacement branches must name the replaced agent")


def build_training_envelopes(
    decisions: tuple[RolloutDecision, ...],
    bindings: tuple[AgentPolicy, ...],
    credits: tuple[ReplacementCredit, ...],
    trainable_team: Team,
) -> tuple[PolicyTrainingEnvelope, ...]:
    """Route only actual-branch owned tokens into four independent policy batches."""
    validate_policy_roster(bindings, trainable_team)
    if not decisions:
        raise ValueError("a credit group requires rollout decisions")
    for decision in decisions:
        _validate_decision(decision)
    game_ids = {decision.game_id for decision in decisions}
    if len(game_ids) != 1:
        raise ValueError("a training envelope cannot mix games")

    binding_by_agent = {binding.agent_id: binding for binding in bindings}
    credit_by_agent = {credit.agent_id: credit for credit in credits}
    expected_agents = {
        binding.agent_id
        for binding in bindings
        if binding.trainable and binding.team == trainable_team
    }
    if set(credit_by_agent) != expected_agents:
        raise ValueError("replacement credits do not cover the four trainable agents")
    for agent_id, credit in credit_by_agent.items():
        if credit.policy_id != binding_by_agent[agent_id].policy_id:
            raise ValueError(f"replacement credit is routed to the wrong policy: {agent_id}")
    replacement_agents = {
        decision.replaced_agent
        for decision in decisions
        if decision.branch == "replacement"
    }
    if replacement_agents != expected_agents:
        raise ValueError(
            "an atomic credit group requires one replacement branch for each trainable agent"
        )

    actual = tuple(decision for decision in decisions if decision.branch == "actual")
    actual_schedule = {
        (decision.agent_id, decision.turn, decision.phase): decision.sampling_key
        for decision in actual
    }
    if len(actual_schedule) != len(actual):
        raise ValueError("actual branch contains duplicate agent/turn/phase decisions")
    for replaced_agent in sorted(expected_agents):
        branch = [
            decision
            for decision in decisions
            if decision.branch == "replacement"
            and decision.replaced_agent == replaced_agent
        ]
        branch_schedule = {
            (decision.agent_id, decision.turn, decision.phase): decision.sampling_key
            for decision in branch
        }
        if len(branch_schedule) != len(branch):
            raise ValueError(f"replacement branch has duplicate decisions: {replaced_agent}")
        if branch_schedule != actual_schedule:
            raise ValueError(
                f"replacement branch does not share the complete random-key schedule: {replaced_agent}"
            )
    spans = tuple(
        AgentTokenSpan(
            decision.game_id,
            decision.agent_id,
            decision.policy_id,
            decision.team,
            decision.turn,
            decision.phase,
            decision.trajectory_index,
            len(decision.prompt_ids),
            len(decision.completion_ids),
        )
        for decision in actual
    )
    validate_token_spans(spans, bindings)
    owned: dict[str, list[RolloutDecision]] = {agent_id: [] for agent_id in expected_agents}
    for decision in actual:
        binding = binding_by_agent[decision.agent_id]
        if binding.trainable:
            if decision.policy_id != binding.policy_id:
                raise ValueError(f"actual decision is routed to the wrong policy: {decision.decision_id}")
            owned[decision.agent_id].append(decision)
    missing = [agent_id for agent_id, rows in owned.items() if not rows]
    if missing:
        raise ValueError(f"actual branch has no trainable decisions for: {missing}")

    envelopes = []
    for agent_id in sorted(owned):
        rows = sorted(owned[agent_id], key=lambda row: row.trajectory_index)
        revisions = {row.policy_revision for row in rows}
        if len(revisions) != 1:
            raise ValueError(f"policy revision changed inside one game: {agent_id}")
        credit = credit_by_agent[agent_id]
        envelopes.append(
            PolicyTrainingEnvelope(
                next(iter(game_ids)),
                agent_id,
                credit.policy_id,
                next(iter(revisions)),
                credit.advantage,
                tuple(row.decision_id for row in rows),
                sum(len(row.completion_ids) for row in rows),
            )
        )
    return tuple(envelopes)


def verify_trainer_logprob_parity(
    decisions: tuple[RolloutDecision, ...],
    trainer_logprobs: dict[str, tuple[float, ...]],
    trainable_policy_ids: frozenset[str],
    *,
    absolute_tolerance: float = 2e-4,
) -> dict[str, float | int | str]:
    actual_trainable = [
        decision
        for decision in decisions
        if decision.branch == "actual" and decision.policy_id in trainable_policy_ids
    ]
    expected = {decision.decision_id for decision in actual_trainable}
    if not expected:
        raise ValueError("log-prob parity requires trainable actual decisions")
    for decision in actual_trainable:
        _validate_decision(decision)
    if set(trainer_logprobs) != expected:
        raise ValueError("trainer log-prob rows do not exactly match actual decision IDs")
    errors = []
    token_count = 0
    for decision in actual_trainable:
        compared = trainer_logprobs[decision.decision_id]
        if len(compared) != len(decision.rollout_logprobs):
            raise ValueError(f"trainer log-prob length mismatch: {decision.decision_id}")
        if not all(math.isfinite(value) for value in compared):
            raise ValueError(f"non-finite trainer log probability: {decision.decision_id}")
        token_count += len(compared)
        errors.extend(
            abs(rollout_value - trainer_value)
            for rollout_value, trainer_value in zip(
                decision.rollout_logprobs, compared, strict=True
            )
        )
    maximum = max(errors, default=0.0)
    if maximum > absolute_tolerance:
        raise ValueError(
            f"rollout/trainer log-prob mismatch: max_abs_error={maximum:.8g} "
            f"> tolerance={absolute_tolerance:.8g}"
        )
    return {
        "status": "passed",
        "decisions": len(actual_trainable),
        "tokens": token_count,
        "max_abs_error": maximum,
        "absolute_tolerance": absolute_tolerance,
    }
