from __future__ import annotations

import itertools
from dataclasses import dataclass

from .arena import Action, GameState, Team, legal_actions, step


@dataclass(frozen=True)
class JointSolution:
    team: Team
    reward: float
    assignments: tuple[tuple[tuple[str, Action], ...], ...]
    canonical_assignment: tuple[tuple[str, Action], ...]
    optimal_count: int
    action_options: tuple[tuple[str, frozenset[Action]], ...]
    explored: int

    @property
    def unique(self) -> bool:
        return self.optimal_count == 1

    def acceptable_for(self, agent_id: str) -> frozenset[Action]:
        return dict(self.action_options)[agent_id]


def local_priority(
    state: GameState, agent_id: str, action: Action, style: str = "balanced"
) -> tuple[float, str, str]:
    agent = state.agents[agent_id]
    score = 0.0
    seen = state.knowledge[agent_id].get(action.target or "")
    if action.kind == "RECOVER" and action.target:
        score = 12.0 + (seen.value if seen else 0) + 3.0 * bool(seen and seen.critical)
    elif action.kind == "CAPTURE" and action.target:
        score = (
            9.0 + (seen.value if seen else 0) + 2.0 * bool(seen and seen.critical)
            if seen and seen.status == "EXPOSED"
            else 1.0
        )
    elif action.kind == "FORTIFY" and action.target:
        score = 7.0 + (seen.value if seen else 0) + 3.0 * bool(seen and seen.critical)
    elif action.kind == "PROBE" and action.target:
        score = (
            1.0
            if seen and seen.status == "EXPOSED"
            else 8.0 + (seen.value if seen else 0) + 2.0 * bool(seen and seen.critical)
        )
    elif action.kind == "TRANSFER" and action.target:
        receiver = state.agents[action.target]
        score = 3.0 + 8.0 * (receiver.resource == 0) + 2.0 * (agent.resource > 1)
    elif action.kind == "SCAN":
        score = 2.0
    if style == "aggressive":
        score += {"CAPTURE": 6.0, "PROBE": 4.0, "FORTIFY": -2.0, "RECOVER": -1.0}.get(action.kind, 0.0)
    elif style == "defensive":
        score += {"RECOVER": 6.0, "FORTIFY": 5.0, "CAPTURE": -2.0, "PROBE": -1.0}.get(action.kind, 0.0)
    elif style != "balanced":
        raise ValueError(f"unknown policy style: {style}")
    return score, action.kind, action.target or agent.position


def deterministic_policy(state: GameState, team: Team, style: str = "balanced") -> dict[str, Action]:
    """A transparent non-learning opponent/base policy used for one-step labels."""

    selected: dict[str, Action] = {}
    claimed: set[tuple[str, str | None]] = set()
    members = sorted(agent.id for agent in state.agents.values() if agent.team == team)
    for agent_id in members:
        ranked = sorted(
            legal_actions(state, agent_id),
            key=lambda action: local_priority(state, agent_id, action, style),
            reverse=True,
        )
        choice = ranked[0]
        for action in ranked:
            key = (action.kind, action.target)
            if action.kind == "WAIT" or key not in claimed:
                choice = action
                break
        selected[agent_id] = choice
        if choice.kind != "WAIT":
            claimed.add((choice.kind, choice.target))
    return selected


def local_policy_action(
    state: GameState, agent_id: str, style: str = "balanced"
) -> Action:
    """Choose from one agent's prompt-visible state without teammate leakage."""

    return max(
        legal_actions(state, agent_id),
        key=lambda action: local_priority(state, agent_id, action, style),
    )


def solve_joint_action(
    state: GameState,
    team: Team,
    opponent_actions: dict[str, Action] | None = None,
    *,
    max_optima: int = 256,
) -> JointSolution:
    """Exactly enumerate one team's legal joint action against a fixed opponent.

    Exact enumeration is deliberately used for SFT labels. `max_optima` prevents
    pathological all-tie states from consuming unbounded memory; such a state is
    marked ambiguous and excluded from SFT either way.
    """

    if opponent_actions is None:
        opponent_actions = deterministic_policy(state, "RED" if team == "BLUE" else "BLUE")
    members = sorted(agent.id for agent in state.agents.values() if agent.team == team)
    choices = [legal_actions(state, agent_id) for agent_id in members]
    best_reward = float("-inf")
    best: list[tuple[tuple[str, Action], ...]] = []
    canonical: tuple[tuple[str, Action], ...] | None = None
    optimal_count = 0
    options: dict[str, set[Action]] = {agent_id: set() for agent_id in members}
    explored = 0
    for combination in itertools.product(*choices):
        explored += 1
        assignment = dict(zip(members, combination, strict=True))
        result = step(state, {**opponent_actions, **assignment})
        reward = result.rewards[team]
        serialized = tuple(zip(members, combination, strict=True))
        if reward > best_reward + 1e-12:
            best_reward = reward
            best = [serialized]
            canonical = serialized
            optimal_count = 1
            options = {agent_id: {action} for agent_id, action in serialized}
        elif abs(reward - best_reward) <= 1e-12:
            optimal_count += 1
            for agent_id, action in serialized:
                options[agent_id].add(action)
            if len(best) < max_optima:
                best.append(serialized)
            assert canonical is not None
            candidate_key = (sum(action.kind != "WAIT" for _, action in serialized), combination)
            canonical_actions = tuple(action for _, action in canonical)
            canonical_key = (sum(action.kind != "WAIT" for action in canonical_actions), canonical_actions)
            if candidate_key < canonical_key:
                canonical = serialized
    assert canonical is not None
    return JointSolution(
        team,
        best_reward,
        tuple(best),
        canonical,
        optimal_count,
        tuple((agent_id, frozenset(options[agent_id])) for agent_id in members),
        explored,
    )
