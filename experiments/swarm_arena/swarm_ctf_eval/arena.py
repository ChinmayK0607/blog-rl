from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Team = Literal["BLUE", "RED"]
Owner = Literal["BLUE", "RED", "NEUTRAL"]
ActionKind = Literal["WAIT", "SCAN", "PROBE", "CAPTURE", "FORTIFY", "RECOVER", "TRANSFER"]

TEAMS: tuple[Team, Team] = ("BLUE", "RED")
ARENA_VERSION = "arena-core-v1"
ACTION_COST = {
    "WAIT": 0,
    "SCAN": 0,
    "PROBE": 1,
    "CAPTURE": 1,
    "FORTIFY": 1,
    "RECOVER": 1,
    "TRANSFER": 1,
}


def opponent(team: Team) -> Team:
    return "RED" if team == "BLUE" else "BLUE"


@dataclass(frozen=True, order=True)
class Action:
    kind: ActionKind
    target: str | None = None
    amount: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"type": self.kind}
        if self.target is not None:
            value["target"] = self.target
        if self.amount is not None:
            value["amount"] = self.amount
        return value


WAIT = Action("WAIT")


@dataclass
class Node:
    id: str
    neighbors: tuple[str, ...]
    owner: Owner
    value: int = 1
    critical: bool = False
    fortification: int = 0
    exposed: bool = False
    compromised: bool = False

    def validate(self) -> None:
        if self.id in self.neighbors:
            raise ValueError(f"node {self.id} cannot neighbor itself")
        if self.value not in (1, 2, 3):
            raise ValueError(f"node {self.id} has invalid value")
        if self.fortification not in (0, 1, 2):
            raise ValueError(f"node {self.id} has invalid fortification")
        if self.owner == "NEUTRAL" and self.compromised:
            raise ValueError(f"neutral node {self.id} cannot be compromised")
        if self.exposed and self.fortification:
            raise ValueError(f"node {self.id} cannot be exposed and fortified")

    @property
    def status(self) -> str:
        if self.compromised:
            return "COMPROMISED"
        if self.exposed:
            return "EXPOSED"
        if self.fortification:
            return "FORTIFIED"
        return "SECURE"


@dataclass
class AgentState:
    id: str
    team: Team
    position: str
    resource: int = 2

    def validate(self, nodes: dict[str, Node]) -> None:
        if self.position not in nodes:
            raise ValueError(f"agent {self.id} has unknown position")
        if not 0 <= self.resource <= 4:
            raise ValueError(f"agent {self.id} has invalid resource")


@dataclass(frozen=True)
class NodeObservation:
    node: str
    owner: Owner
    status: str
    value: int
    critical: bool
    observed_turn: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GameState:
    turn: int
    nodes: dict[str, Node]
    agents: dict[str, AgentState]
    knowledge: dict[str, dict[str, NodeObservation]] = field(default_factory=dict)

    def clone(self) -> GameState:
        return copy.deepcopy(self)

    def validate(self) -> None:
        if self.turn < 0:
            raise ValueError("turn must be non-negative")
        if len(self.agents) != 8:
            raise ValueError("arena requires exactly eight agents")
        for team in TEAMS:
            members = [agent for agent in self.agents.values() if agent.team == team]
            if len(members) != 4:
                raise ValueError(f"arena requires four {team} agents")
        for node in self.nodes.values():
            node.validate()
            for neighbor in node.neighbors:
                if neighbor not in self.nodes:
                    raise ValueError(f"node {node.id} has unknown neighbor {neighbor}")
                if node.id not in self.nodes[neighbor].neighbors:
                    raise ValueError(f"edge {node.id}-{neighbor} is not symmetric")
        for agent in self.agents.values():
            agent.validate(self.nodes)
            if agent.position not in self.knowledge.get(agent.id, {}):
                raise ValueError(f"agent {agent.id} must observe its position")


@dataclass(frozen=True)
class Event:
    kind: str
    actor: str | None = None
    target: str | None = None
    success: bool = True
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    state: GameState
    rewards: dict[Team, float]
    events: tuple[Event, ...]
    invalid_agents: tuple[str, ...]
    duplicate_targets: dict[Team, tuple[str, ...]]


def observe_node(node: Node, turn: int) -> NodeObservation:
    return NodeObservation(node.id, node.owner, node.status, node.value, node.critical, turn)


def refresh_local_knowledge(state: GameState, agent_id: str) -> None:
    agent = state.agents[agent_id]
    memory = state.knowledge.setdefault(agent_id, {})
    # Agents know adjacent node identifiers but must SCAN to reveal an unseen
    # neighbor's state. Previously scanned adjacent nodes are refreshed locally;
    # remote memory remains stale and retains its original observed_turn.
    visible = {
        agent.position,
        *(neighbor for neighbor in state.nodes[agent.position].neighbors if neighbor in memory),
    }
    for node_id in visible:
        memory[node_id] = observe_node(state.nodes[node_id], state.turn)


def observation_for(state: GameState, agent_id: str) -> dict[str, Any]:
    agent = state.agents[agent_id]
    memory = state.knowledge[agent_id]
    return {
        "turn": state.turn,
        "self": {
            "id": agent.id,
            "team": agent.team,
            "position": agent.position,
            "resource": agent.resource,
        },
        "known_nodes": [memory[node_id].to_dict() for node_id in sorted(memory)],
        "adjacent_teammates": [
            {
                "id": teammate.id,
                "position": teammate.position,
                "resource": teammate.resource,
            }
            for teammate in sorted(state.agents.values(), key=lambda item: item.id)
            if teammate.team == agent.team
            and teammate.id != agent.id
            and teammate.position in state.nodes[agent.position].neighbors
        ],
        "unknown_neighbors": sorted(
            neighbor
            for neighbor in state.nodes[agent.position].neighbors
            if neighbor not in memory
        ),
    }


def legal_actions(state: GameState, agent_id: str) -> tuple[Action, ...]:
    agent = state.agents[agent_id]
    node = state.nodes[agent.position]
    memory = state.knowledge[agent_id]
    actions: set[Action] = {WAIT}

    for neighbor_id in node.neighbors:
        if neighbor_id not in memory:
            actions.add(Action("SCAN", neighbor_id))
            continue
        seen = memory[neighbor_id]
        if agent.resource >= 1 and seen.owner != agent.team:
            actions.add(Action("PROBE", neighbor_id))
            # CAPTURE is a valid attempt against a known enemy node. It only
            # succeeds if the node is exposed after simultaneous defenses and
            # probes resolve, allowing genuine probe/capture coordination.
            actions.add(Action("CAPTURE", neighbor_id))
        if agent.resource >= 1 and seen.owner == agent.team:
            if seen.status == "COMPROMISED":
                actions.add(Action("RECOVER", neighbor_id))
            elif seen.status != "FORTIFIED":
                actions.add(Action("FORTIFY", neighbor_id))

    current = memory[agent.position]
    if agent.resource >= 1 and current.owner == agent.team:
        if current.status == "COMPROMISED":
            actions.add(Action("RECOVER", agent.position))
        elif current.status != "FORTIFIED":
            actions.add(Action("FORTIFY", agent.position))

    if agent.resource >= 1:
        for teammate in state.agents.values():
            if (
                teammate.team == agent.team
                and teammate.id != agent.id
                and teammate.position in node.neighbors
                and teammate.resource < 4
            ):
                actions.add(Action("TRANSFER", teammate.id, 1))

    return tuple(sorted(actions))


def team_value(state: GameState, team: Team) -> float:
    value = 0.0
    for node in state.nodes.values():
        weight = float(node.value + int(node.critical))
        if node.owner == team:
            value += weight
            value += 0.15 * node.fortification
            if node.exposed:
                value -= 0.25 * weight
            if node.compromised:
                value -= 0.75 * weight
        elif node.owner == opponent(team):
            value -= weight
            value -= 0.15 * node.fortification
            if node.exposed:
                value += 0.25 * weight
            if node.compromised:
                value += 0.75 * weight
    value += _resource_potential(state, team) - _resource_potential(state, opponent(team))
    return value


def _resource_potential(state: GameState, team: Team) -> float:
    potential = 0.0
    for agent in state.agents.values():
        if agent.team != team or agent.resource == 0:
            continue
        opportunity = 0.0
        for seen in state.knowledge[agent.id].values():
            if seen.owner == team and seen.status == "COMPROMISED":
                opportunity = max(opportunity, 3.0 + seen.value)
            elif seen.owner != team and seen.status == "EXPOSED":
                opportunity = max(opportunity, 3.0 + seen.value)
            elif seen.owner != team:
                opportunity = max(opportunity, 2.0 + seen.value)
            elif seen.owner == team and seen.critical:
                opportunity = max(opportunity, 1.0 + seen.value)
        potential += 0.05 * opportunity
    return potential


def _duplicate_targets(
    state: GameState, joint_actions: dict[str, Action]
) -> dict[Team, tuple[str, ...]]:
    result: dict[Team, tuple[str, ...]] = {}
    for team in TEAMS:
        counts = Counter(
            (action.kind, action.target)
            for agent_id, action in joint_actions.items()
            if state.agents[agent_id].team == team and action.kind != "WAIT" and action.target is not None
        )
        result[team] = tuple(
            sorted(f"{kind}:{target}" for (kind, target), count in counts.items() if count > 1)
        )
    return result


def step(state: GameState, joint_actions: dict[str, Action]) -> StepResult:
    """Resolve one simultaneous turn using a fixed, documented phase order.

    The action set is checked against the pre-turn state. Transfers resolve first,
    then recovery/fortification, information actions, and capture. A successful
    same-turn PROBE may therefore enable a teammate's CAPTURE, while same-turn
    FORTIFY can block it. This is the environment's main coordination primitive.
    """

    state.validate()
    next_state = state.clone()
    events: list[Event] = []
    invalid: list[str] = []
    resolved: dict[str, Action] = {}
    for agent_id in sorted(state.agents):
        action = joint_actions.get(agent_id, WAIT)
        if action not in legal_actions(state, agent_id):
            invalid.append(agent_id)
            resolved[agent_id] = WAIT
            events.append(Event("INVALID", agent_id, action.target, False, action.kind))
        else:
            resolved[agent_id] = action

    before = {team: team_value(state, team) for team in TEAMS}
    duplicates = _duplicate_targets(state, resolved)

    # Pay non-transfer action costs up front. Transfers move, rather than destroy,
    # one unit and are handled separately.
    for agent_id, action in resolved.items():
        if action.kind not in ("WAIT", "SCAN", "TRANSFER"):
            next_state.agents[agent_id].resource -= ACTION_COST[action.kind]

    for agent_id, action in resolved.items():
        if action.kind != "TRANSFER" or action.target is None:
            continue
        sender = next_state.agents[agent_id]
        receiver = next_state.agents[action.target]
        if receiver.resource >= 4:
            events.append(Event("TRANSFER", agent_id, action.target, False, "receiver_full"))
            continue
        sender.resource -= 1
        receiver.resource += 1
        events.append(Event("TRANSFER", agent_id, action.target))

    for agent_id, action in resolved.items():
        if action.target not in next_state.nodes:
            continue
        node = next_state.nodes[action.target]
        if action.kind == "RECOVER":
            node.compromised = False
            node.exposed = False
            events.append(Event("RECOVER", agent_id, node.id))
        elif action.kind == "FORTIFY":
            node.fortification = min(2, node.fortification + 1)
            node.exposed = False
            events.append(Event("FORTIFY", agent_id, node.id))

    scan_bonus: dict[Team, float] = defaultdict(float)
    probe_attempts: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in resolved.items():
        agent = next_state.agents[agent_id]
        if action.kind == "SCAN" and action.target in next_state.nodes:
            was_unknown = action.target not in next_state.knowledge[agent_id]
            next_state.knowledge[agent_id][action.target] = observe_node(next_state.nodes[action.target], state.turn)
            scan_bonus[agent.team] += 0.10 * float(was_unknown)
            events.append(Event("SCAN", agent_id, action.target, was_unknown))
        elif action.kind == "PROBE" and action.target in next_state.nodes:
            probe_attempts[action.target].append(agent_id)

    for target_id, agent_ids in sorted(probe_attempts.items()):
        target = next_state.nodes[target_id]
        valid = [agent_id for agent_id in agent_ids if next_state.agents[agent_id].team != target.owner]
        for agent_id in set(agent_ids) - set(valid):
            events.append(Event("PROBE", agent_id, target.id, False, "owner_changed"))
        if not valid:
            continue
        shields = target.fortification
        target.fortification = max(0, shields - len(valid))
        target.exposed = len(valid) > shields
        for index, agent_id in enumerate(valid):
            detail = "exposed" if index >= shields else "fortification_reduced"
            events.append(Event("PROBE", agent_id, target.id, True, detail))

    # Capture is based on the post-defense, post-probe state. Opposing attempts
    # against the same neutral target cancel rather than depending on iteration
    # order. Duplicate same-team attempts can succeed but incur the team collision
    # penalty computed above.
    capture_attempts: dict[str, list[str]] = defaultdict(list)
    for agent_id, action in resolved.items():
        if action.kind == "CAPTURE" and action.target in next_state.nodes:
            capture_attempts[action.target].append(agent_id)
    for target_id, agent_ids in sorted(capture_attempts.items()):
        target = next_state.nodes[target_id]
        teams = {next_state.agents[agent_id].team for agent_id in agent_ids}
        viable = target.exposed and target.fortification == 0
        if len(teams) != 1:
            for agent_id in agent_ids:
                events.append(Event("CAPTURE", agent_id, target_id, False, "contested"))
            continue
        team = next(iter(teams))
        if viable and target.owner != team:
            target.owner = team
            target.exposed = False
            target.compromised = False
            for agent_id in agent_ids:
                next_state.agents[agent_id].position = target.id
                events.append(Event("CAPTURE", agent_id, target.id))
        else:
            for agent_id in agent_ids:
                events.append(Event("CAPTURE", agent_id, target.id, False, "not_exposed"))

    next_state.turn += 1
    for agent_id in next_state.agents:
        refresh_local_knowledge(next_state, agent_id)
    next_state.validate()

    rewards: dict[Team, float] = {}
    for team in TEAMS:
        delta = team_value(next_state, team) - before[team]
        other = opponent(team)
        rewards[team] = (
            delta
            + scan_bonus[team] - scan_bonus[other]
            - 1.0 * sum(state.agents[agent_id].team == team for agent_id in invalid)
            + 1.0 * sum(state.agents[agent_id].team == other for agent_id in invalid)
        )
    return StepResult(next_state, rewards, tuple(events), tuple(invalid), duplicates)


def redundant_agents(
    state: GameState, joint_actions: dict[str, Action], team: Team
) -> tuple[str, ...]:
    """Return actions with non-positive leave-one-out marginal team reward.

    This counterfactual definition avoids falsely calling complementary repeated
    actions—such as two probes removing two shield levels—a collision.
    """

    baseline = step(state, joint_actions).rewards[team]
    redundant = []
    for agent_id, action in sorted(joint_actions.items()):
        if state.agents[agent_id].team != team or action.kind == "WAIT":
            continue
        counterfactual = dict(joint_actions)
        counterfactual[agent_id] = WAIT
        if step(state, counterfactual).rewards[team] >= baseline - 1e-12:
            redundant.append(agent_id)
    return tuple(redundant)


def state_to_dict(state: GameState) -> dict[str, Any]:
    return {
        "turn": state.turn,
        "nodes": {node_id: asdict(node) for node_id, node in sorted(state.nodes.items())},
        "agents": {agent_id: asdict(agent) for agent_id, agent in sorted(state.agents.items())},
        "knowledge": {
            agent_id: {node_id: observation.to_dict() for node_id, observation in sorted(memory.items())}
            for agent_id, memory in sorted(state.knowledge.items())
        },
    }


class ArenaEnv:
    """Small dependency-free parallel multi-agent environment wrapper."""

    def __init__(self, seed: int = 0, size: int = 12, horizon: int = 8) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.seed = seed
        self.size = size
        self.horizon = horizon
        self.state: GameState | None = None

    def reset(self, seed: int | None = None) -> dict[str, dict[str, Any]]:
        from .arena_generation import generate_state

        if seed is not None:
            self.seed = seed
        self.state = generate_state(self.seed, self.size)
        return self.observations()

    def observations(self) -> dict[str, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("call reset before observations")
        return {
            agent_id: observation_for(self.state, agent_id)
            for agent_id in sorted(self.state.agents)
        }

    def legal_action_map(self) -> dict[str, tuple[Action, ...]]:
        if self.state is None:
            raise RuntimeError("call reset before legal_action_map")
        return {
            agent_id: legal_actions(self.state, agent_id)
            for agent_id in sorted(self.state.agents)
        }

    def advance(
        self, joint_actions: dict[str, Action]
    ) -> tuple[dict[str, dict[str, Any]], dict[Team, float], bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("call reset before advance")
        result = step(self.state, joint_actions)
        self.state = result.state
        blue_nodes = sum(node.owner == "BLUE" for node in self.state.nodes.values())
        red_nodes = sum(node.owner == "RED" for node in self.state.nodes.values())
        terminated = blue_nodes == 0 or red_nodes == 0
        truncated = self.state.turn >= self.horizon and not terminated
        info = {
            "events": [event.to_dict() for event in result.events],
            "invalid_agents": list(result.invalid_agents),
            "same_action_targets": {
                team: list(targets) for team, targets in result.duplicate_targets.items()
            },
            "team_value": {team: team_value(self.state, team) for team in TEAMS},
        }
        return self.observations(), result.rewards, terminated, truncated, info
