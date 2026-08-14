from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .arena import Team

MULTI_POLICY_CONTRACT_VERSION = "arena-four-policy-v2-message-credit"
Phase = Literal["BROADCAST", "ACT"]


@dataclass(frozen=True)
class AgentPolicy:
    agent_id: str
    team: Team
    policy_id: str
    trainable: bool


@dataclass(frozen=True)
class AgentTokenSpan:
    game_id: str
    agent_id: str
    policy_id: str
    team: Team
    turn: int
    phase: Phase
    trajectory_index: int
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ReplacementCredit:
    agent_id: str
    policy_id: str
    actual_return: float
    replacement_return: float
    advantage: float


@dataclass(frozen=True)
class MessageCredit:
    agent_id: str
    policy_id: str
    actual_return: float
    dropped_return: float
    advantage: float


def validate_policy_roster(bindings: tuple[AgentPolicy, ...], trainable_team: Team) -> None:
    if len(bindings) != 8:
        raise ValueError("a 4v4 rollout requires eight policy bindings")
    agent_ids = {binding.agent_id for binding in bindings}
    if len(agent_ids) != 8:
        raise ValueError("every in-game agent must have exactly one policy binding")
    trainable = [binding for binding in bindings if binding.trainable]
    if len(trainable) != 4 or {binding.team for binding in trainable} != {trainable_team}:
        raise ValueError("exactly the four members of one team must be trainable")
    policy_ids = [binding.policy_id for binding in trainable]
    if len(set(policy_ids)) != 4:
        raise ValueError("the primary swarm experiment requires four distinct trainable policies")
    if any(binding.trainable for binding in bindings if binding.team != trainable_team):
        raise ValueError("opponent policies must be frozen for an update epoch")


def validate_token_spans(
    spans: tuple[AgentTokenSpan, ...],
    bindings: tuple[AgentPolicy, ...],
) -> None:
    lookup = {binding.agent_id: binding for binding in bindings}
    previous_index = -1
    seen = set()
    for span in spans:
        binding = lookup.get(span.agent_id)
        if binding is None:
            raise ValueError(f"token span has no policy binding: {span.agent_id}")
        if (span.policy_id, span.team) != (binding.policy_id, binding.team):
            raise ValueError(f"token span is routed to the wrong policy: {span.agent_id}")
        if span.trajectory_index <= previous_index:
            raise ValueError("trajectory indices must preserve the environment emission order")
        if min(span.prompt_tokens, span.completion_tokens) < 0 or span.completion_tokens == 0:
            raise ValueError("every trainable decision needs a non-empty completion span")
        key = (span.agent_id, span.turn, span.phase)
        if key in seen:
            raise ValueError(f"duplicate decision span: {key}")
        seen.add(key)
        previous_index = span.trajectory_index


def replacement_credits(
    actual_return: float,
    replacement_returns: dict[str, float],
    bindings: tuple[AgentPolicy, ...],
    trainable_team: Team,
) -> tuple[ReplacementCredit, ...]:
    trainable = sorted(
        (binding for binding in bindings if binding.trainable and binding.team == trainable_team),
        key=lambda binding: binding.agent_id,
    )
    expected = {binding.agent_id for binding in trainable}
    if set(replacement_returns) != expected:
        raise ValueError(
            f"replacement branches must cover exactly the trainable agents: expected={sorted(expected)}"
        )
    return tuple(
        ReplacementCredit(
            binding.agent_id,
            binding.policy_id,
            actual_return,
            replacement_returns[binding.agent_id],
            actual_return - replacement_returns[binding.agent_id],
        )
        for binding in trainable
    )


def attach_credits_to_spans(
    spans: tuple[AgentTokenSpan, ...],
    credits: tuple[ReplacementCredit, ...],
) -> dict[str, dict[str, float | int | str]]:
    credit_by_agent = {credit.agent_id: credit for credit in credits}
    result: dict[str, dict[str, float | int | str]] = {}
    for span in spans:
        credit = credit_by_agent.get(span.agent_id)
        if credit is None:
            continue
        key = f"{span.game_id}:{span.agent_id}:{span.turn}:{span.phase}"
        result[key] = {
            "policy_id": span.policy_id,
            "agent_id": span.agent_id,
            "advantage": credit.advantage,
            "trajectory_index": span.trajectory_index,
            "completion_tokens": span.completion_tokens,
        }
    return result
