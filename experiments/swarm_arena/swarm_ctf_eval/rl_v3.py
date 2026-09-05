from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

from .arena import Action, GameState, Team, legal_actions, opponent, step
from .episode import ArenaEpisodeEnv, BroadcastPhase, EpisodeConfig, EpisodeTransition

RL_TASK_VERSION = "arena-rl-v3-terminal-control"


class ProtocolContractError(RuntimeError):
    """A rollout violated mechanics that structured decoding must guarantee."""


def node_control_weight(state: GameState, node_id: str) -> int:
    node = state.nodes[node_id]
    return node.value + int(node.critical)


def terminal_control_margin(state: GameState, team: Team) -> float:
    """Normalized zero-sum territory score in [-1, 1]."""
    rival = opponent(team)
    total = sum(node_control_weight(state, node_id) for node_id in state.nodes)
    if total <= 0:
        raise ValueError("control score requires positive total node value")
    owned = sum(
        node_control_weight(state, node_id)
        for node_id, node in state.nodes.items()
        if node.owner == team
    )
    opposed = sum(
        node_control_weight(state, node_id)
        for node_id, node in state.nodes.items()
        if node.owner == rival
    )
    return (owned - opposed) / total


def terminal_control_delta(initial: GameState, final: GameState, team: Team) -> float:
    """Terminal objective with the seed-specific initial margin as a baseline."""
    return terminal_control_margin(final, team) - terminal_control_margin(initial, team)


@dataclass(frozen=True)
class JointValue:
    value: float
    assignment: tuple[tuple[str, Action], ...]
    explored: int


def maximize_terminal_control(
    state: GameState,
    team: Team,
    opponent_actions: dict[str, Action],
    *,
    initial_state: GameState | None = None,
    score: Callable[[GameState, GameState, Team], float] = terminal_control_delta,
) -> JointValue:
    """Exactly maximize one team's one-turn terminal objective against fixed actions."""
    initial = initial_state or state
    members = sorted(agent.id for agent in state.agents.values() if agent.team == team)
    choices = [legal_actions(state, agent_id) for agent_id in members]
    best_value = float("-inf")
    best_assignment: tuple[tuple[str, Action], ...] | None = None
    explored = 0
    for actions in itertools.product(*choices):
        explored += 1
        assignment = tuple(zip(members, actions, strict=True))
        outcome = step(state, {**opponent_actions, **dict(assignment)})
        value = score(initial, outcome.state, team)
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12
            and (best_assignment is None or assignment < best_assignment)
        ):
            best_value = value
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("joint action enumeration produced no assignment")
    return JointValue(best_value, best_assignment, explored)


class ArenaRLEnv(ArenaEpisodeEnv):
    """RL v3: hard protocol contracts and one pure terminal territory objective."""

    def __init__(self, seed: int = 0, size: int = 12, config: EpisodeConfig | None = None) -> None:
        resolved = config or EpisodeConfig(
            communication_cost=0.0,
            invalid_broadcast_cost=0.0,
            invalid_action_cost=0.0,
        )
        if any(
            value != 0.0
            for value in (
                resolved.communication_cost,
                resolved.invalid_broadcast_cost,
                resolved.invalid_action_cost,
            )
        ):
            raise ValueError("RL v3 uses hard constraints, not additive communication/protocol costs")
        super().__init__(seed, size, resolved)
        self._initial_state: GameState | None = None

    def reset(self, seed: int | None = None) -> dict[str, dict]:
        observations = super().reset(seed)
        self._initial_state = self._require_state().clone()
        return observations

    def reset_from_state(self, state: GameState) -> dict[str, dict]:
        observations = super().reset_from_state(state)
        self._initial_state = self._require_state().clone()
        return observations

    def broadcast_phase(self, *args, **kwargs) -> BroadcastPhase:
        phase = super().broadcast_phase(*args, **kwargs)
        failures = {agent_id: errors for agent_id, errors in phase.errors.items() if errors}
        if failures:
            self._phase = None
            raise ProtocolContractError(f"invalid broadcast under structured policy: {failures}")
        return phase

    def advance(self, actions: dict[str, Action]) -> EpisodeTransition:
        state = self._require_state()
        failures = {
            agent_id: action
            for agent_id, action in actions.items()
            if action not in legal_actions(state, agent_id)
        }
        if failures:
            raise ProtocolContractError(f"illegal action under structured policy: {failures}")
        transition = super().advance(actions)
        if not (transition.terminated or transition.truncated):
            return EpisodeTransition(
                transition.observations,
                {"BLUE": 0.0, "RED": 0.0},
                transition.terminated,
                transition.truncated,
                {**transition.info, "rl_task_version": RL_TASK_VERSION},
            )
        if self._initial_state is None:
            raise RuntimeError("missing initial state for terminal control reward")
        blue = terminal_control_delta(self._initial_state, self._require_state(), "BLUE")
        return EpisodeTransition(
            transition.observations,
            {"BLUE": blue, "RED": -blue},
            transition.terminated,
            transition.truncated,
            {
                **transition.info,
                "rl_task_version": RL_TASK_VERSION,
                "reward_definition": "normalized terminal control delta",
            },
        )
