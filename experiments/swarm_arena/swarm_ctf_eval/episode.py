from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .arena import (
    TEAMS,
    Action,
    Event,
    GameState,
    Team,
    legal_actions,
    observation_for,
    step,
    team_value,
)
from .arena_generation import generate_state
from .arena_protocol import Broadcast

EPISODE_VERSION = "arena-episode-v2"
EMPTY_BROADCAST = Broadcast((), None, 0)


@dataclass(frozen=True)
class EpisodeConfig:
    horizon: int = 6
    message_budget_per_agent: int = 8
    max_facts_per_message: int = 2
    communication_cost: float = 0.05
    invalid_broadcast_cost: float = 0.5
    invalid_action_cost: float = 1.0
    resource_regen_period: int = 2

    def validate(self) -> None:
        if self.horizon < 2:
            raise ValueError("RL episodes require a horizon of at least two")
        if self.message_budget_per_agent < 0:
            raise ValueError("message budget cannot be negative")
        if not 0 <= self.max_facts_per_message <= 3:
            raise ValueError("max facts must be between zero and three")
        if min(self.communication_cost, self.invalid_broadcast_cost, self.invalid_action_cost) < 0:
            raise ValueError("costs cannot be negative")
        if self.resource_regen_period < 1:
            raise ValueError("resource regeneration period must be positive")


@dataclass(frozen=True)
class BroadcastPhase:
    inboxes: dict[str, tuple[dict[str, Any], ...]]
    accepted: dict[str, Broadcast]
    delivered: dict[str, Broadcast]
    errors: dict[str, tuple[str, ...]]
    message_units: dict[str, int]
    remaining_budget: dict[str, int]
    shared_fact_updates: dict[str, int]


@dataclass(frozen=True)
class EpisodeTransition:
    observations: dict[str, dict[str, Any]]
    rewards: dict[Team, float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def message_units(message: Broadcast) -> int:
    if message == EMPTY_BROADCAST:
        return 0
    return 1 + len(message.facts) + int(message.intent is not None) + int(message.request_resource > 0)


def validate_message(
    state: GameState,
    agent_id: str,
    message: Broadcast,
    *,
    max_facts: int,
) -> tuple[str, ...]:
    errors = []
    if len(message.facts) > max_facts:
        errors.append("fact_budget")
    if len({fact.node for fact in message.facts}) != len(message.facts):
        errors.append("duplicate_fact")
    known = state.knowledge[agent_id]
    for fact in message.facts:
        if known.get(fact.node) != fact:
            errors.append("unsupported_fact")
            break
    if message.intent is not None and message.intent not in legal_actions(state, agent_id):
        errors.append("illegal_intent")
    if message.request_resource not in (0, 1):
        errors.append("invalid_request")
    return tuple(errors)


class ArenaEpisodeEnv:
    """Two-phase, multi-turn MARL wrapper with terminal-only team reward.

    Broadcasts are actions rather than labels. They are delivered only to
    teammates for the current action phase, consume a private episodic budget,
    and are charged at the terminal reward. This keeps communication useful but
    prevents an always-broadcast policy from being free.
    """

    def __init__(self, seed: int = 0, size: int = 12, config: EpisodeConfig | None = None) -> None:
        self.seed = seed
        self.size = size
        self.config = config or EpisodeConfig()
        self.config.validate()
        self.state: GameState | None = None
        self.initial_values: dict[Team, float] = {}
        self.remaining_budget: dict[str, int] = {}
        self.communication_spend: dict[Team, int] = {team: 0 for team in TEAMS}
        self.invalid_broadcasts: dict[Team, int] = {team: 0 for team in TEAMS}
        self.invalid_actions: dict[Team, int] = {team: 0 for team in TEAMS}
        self.last_events: tuple[Event, ...] = ()
        self._phase: BroadcastPhase | None = None

    def reset(self, seed: int | None = None) -> dict[str, dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        return self.reset_from_state(generate_state(self.seed, self.size))

    def reset_from_state(self, state: GameState) -> dict[str, dict[str, Any]]:
        state.validate()
        if state.turn >= self.config.horizon:
            raise ValueError("initial state must precede the episode horizon")
        self.state = state.clone()
        self.size = len(self.state.nodes)
        self.initial_values = {team: team_value(self.state, team) for team in TEAMS}
        self.remaining_budget = {agent_id: self.config.message_budget_per_agent for agent_id in self.state.agents}
        self.communication_spend = {team: 0 for team in TEAMS}
        self.invalid_broadcasts = {team: 0 for team in TEAMS}
        self.invalid_actions = {team: 0 for team in TEAMS}
        self.last_events = ()
        self._phase = None
        return self.observations()

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("call reset before using the episode")
        return self.state

    def _local_events(self, agent_id: str) -> list[dict[str, Any]]:
        state = self._require_state()
        position = state.agents[agent_id].position
        visible_nodes = {position, *state.nodes[position].neighbors}
        return [
            event.to_dict()
            for event in self.last_events
            if event.actor == agent_id or (event.target is not None and event.target in visible_nodes)
        ]

    def observations(self) -> dict[str, dict[str, Any]]:
        state = self._require_state()
        result = {}
        for agent_id in sorted(state.agents):
            result[agent_id] = {
                **observation_for(state, agent_id),
                "episode_version": EPISODE_VERSION,
                "remaining_turns": self.config.horizon - state.turn,
                "message_budget_remaining": self.remaining_budget[agent_id],
                "last_local_events": self._local_events(agent_id),
            }
        return result

    def broadcast_phase(
        self,
        broadcasts: dict[str, Broadcast],
        *,
        delivered_broadcasts: dict[str, Broadcast] | None = None,
    ) -> BroadcastPhase:
        state = self._require_state()
        if self._phase is not None:
            raise RuntimeError("broadcast phase already submitted for this turn")
        unknown = set(broadcasts) - set(state.agents)
        if unknown:
            raise ValueError(f"broadcasts contain unknown agents: {sorted(unknown)}")

        accepted: dict[str, Broadcast] = {}
        errors: dict[str, tuple[str, ...]] = {}
        units: dict[str, int] = {}
        for agent_id in sorted(state.agents):
            message = broadcasts.get(agent_id, EMPTY_BROADCAST)
            if not isinstance(message, Broadcast):
                current_errors = ("message_type",)
                message = EMPTY_BROADCAST
            else:
                current_errors = validate_message(
                    state,
                    agent_id,
                    message,
                    max_facts=self.config.max_facts_per_message,
                )
            cost = message_units(message)
            if not current_errors and cost > self.remaining_budget[agent_id]:
                current_errors = ("message_budget_exceeded",)
            if current_errors:
                team = state.agents[agent_id].team
                self.invalid_broadcasts[team] += 1
                accepted[agent_id] = EMPTY_BROADCAST
                units[agent_id] = 0
            else:
                accepted[agent_id] = message
                units[agent_id] = cost
                self.remaining_budget[agent_id] -= cost
                self.communication_spend[state.agents[agent_id].team] += cost
            errors[agent_id] = current_errors

        if delivered_broadcasts is None:
            delivered = accepted
        else:
            unknown_deliveries = set(delivered_broadcasts) - set(state.agents)
            if unknown_deliveries:
                raise ValueError(f"delivered broadcasts contain unknown agents: {sorted(unknown_deliveries)}")
            if any(not isinstance(message, Broadcast) for message in delivered_broadcasts.values()):
                raise TypeError("delivered broadcasts must already be validated Broadcast values")
            delivered = {
                agent_id: delivered_broadcasts.get(agent_id, EMPTY_BROADCAST) for agent_id in sorted(state.agents)
            }

        shared_fact_updates = {agent_id: 0 for agent_id in state.agents}
        for receiver, receiver_state in sorted(state.agents.items()):
            memory = state.knowledge[receiver]
            for sender, sender_state in sorted(state.agents.items()):
                if sender == receiver or sender_state.team != receiver_state.team:
                    continue
                for fact in delivered[sender].facts:
                    previous = memory.get(fact.node)
                    if previous is None or fact.observed_turn > previous.observed_turn:
                        memory[fact.node] = fact
                        shared_fact_updates[receiver] += 1

        inboxes = {}
        for receiver, receiver_state in sorted(state.agents.items()):
            inboxes[receiver] = tuple(
                {"sender": sender, "broadcast": delivered[sender].to_dict()}
                for sender, sender_state in sorted(state.agents.items())
                if sender_state.team == receiver_state.team
                and sender != receiver
                and delivered[sender] != EMPTY_BROADCAST
            )
        self._phase = BroadcastPhase(
            inboxes,
            accepted,
            delivered,
            errors,
            units,
            dict(self.remaining_budget),
            shared_fact_updates,
        )
        return self._phase

    def action_observations(self) -> dict[str, dict[str, Any]]:
        if self._phase is None:
            raise RuntimeError("submit broadcasts before requesting action observations")
        observations = self.observations()
        return {
            agent_id: {**observation, "inbox": list(self._phase.inboxes[agent_id])}
            for agent_id, observation in observations.items()
        }

    def _terminal_rewards(self) -> dict[Team, float]:
        state = self._require_state()
        # team_value is an antisymmetric relative-advantage coordinate:
        # team_value(BLUE) == -team_value(RED). A RED capture therefore lowers
        # this coordinate and gives RED the corresponding positive zero-sum reward.
        blue_value = team_value(state, "BLUE")
        red_value = team_value(state, "RED")
        if abs(blue_value + red_value) > 1e-9:
            raise RuntimeError("team-value coordinates must remain antisymmetric")
        blue_delta = blue_value - self.initial_values["BLUE"]
        regularization = (
            -self.config.communication_cost * self.communication_spend["BLUE"]
            + self.config.communication_cost * self.communication_spend["RED"]
            - self.config.invalid_broadcast_cost * self.invalid_broadcasts["BLUE"]
            + self.config.invalid_broadcast_cost * self.invalid_broadcasts["RED"]
            - self.config.invalid_action_cost * self.invalid_actions["BLUE"]
            + self.config.invalid_action_cost * self.invalid_actions["RED"]
        )
        blue = blue_delta + regularization
        return {"BLUE": blue, "RED": -blue}

    def advance(self, actions: dict[str, Action]) -> EpisodeTransition:
        state = self._require_state()
        if self._phase is None:
            raise RuntimeError("submit broadcasts before actions")
        unknown = set(actions) - set(state.agents)
        if unknown:
            raise ValueError(f"actions contain unknown agents: {sorted(unknown)}")
        result = step(state, actions)
        for agent_id in result.invalid_agents:
            self.invalid_actions[state.agents[agent_id].team] += 1
        self.state = result.state
        if self.state.turn % self.config.resource_regen_period == 0:
            for agent in self.state.agents.values():
                agent.resource = min(4, agent.resource + 1)
            self.state.validate()
        self.last_events = result.events
        phase = self._phase
        self._phase = None

        counts = {team: sum(node.owner == team for node in self.state.nodes.values()) for team in TEAMS}
        terminated = any(counts[team] == 0 for team in TEAMS)
        truncated = self.state.turn >= self.config.horizon and not terminated
        done = terminated or truncated
        rewards = self._terminal_rewards() if done else {team: 0.0 for team in TEAMS}
        info = {
            "episode_version": EPISODE_VERSION,
            "turn": self.state.turn,
            "terminal_only_reward": True,
            "events": [event.to_dict() for event in result.events],
            "broadcast_errors": {agent: list(value) for agent, value in phase.errors.items()},
            "message_units": phase.message_units,
            "shared_fact_updates": phase.shared_fact_updates,
            "remaining_budget": dict(self.remaining_budget),
            "communication_spend": dict(self.communication_spend),
            "invalid_broadcasts": dict(self.invalid_broadcasts),
            "invalid_actions": dict(self.invalid_actions),
            "duplicate_targets": {team: list(result.duplicate_targets[team]) for team in TEAMS},
            "team_value": {team: team_value(self.state, team) for team in TEAMS},
        }
        return EpisodeTransition(self.observations(), rewards, terminated, truncated, info)
