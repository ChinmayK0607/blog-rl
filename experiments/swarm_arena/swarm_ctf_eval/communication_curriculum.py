from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from .arena import (
    Action,
    GameState,
    NodeObservation,
    Team,
    legal_actions,
    observation_for,
    observe_node,
    state_to_dict,
)
from .arena_generation import GENERATOR_VERSION, generate_state
from .arena_oracle import deterministic_policy
from .rl_v3 import JointValue, maximize_terminal_control

COMMUNICATION_CURRICULUM_VERSION = "arena-communication-curriculum-v1-certified"
ScenarioKind = Literal["critical", "decoy"]
OPPONENT_STYLES = ("balanced", "aggressive", "defensive")


@dataclass(frozen=True)
class StyleCertificate:
    style: str
    informed_value: float
    dropped_value: float
    advantage: float
    informed_assignment: tuple[tuple[str, dict], ...]
    dropped_assignment: tuple[tuple[str, dict], ...]
    informed_explored: int
    dropped_explored: int


@dataclass(frozen=True)
class CommunicationScenario:
    version: str
    generator_version: str
    seed: int
    size: int
    kind: ScenarioKind
    team: Team
    sender: str
    receiver: str
    target: str
    minimum_advantage: float
    certificates: tuple[StyleCertificate, ...]
    state: GameState

    def manifest_row(self) -> dict:
        serialized_state = state_to_dict(self.state)
        return {
            "version": self.version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "size": self.size,
            "kind": self.kind,
            "team": self.team,
            "sender": self.sender,
            "receiver": self.receiver,
            "target": self.target,
            "minimum_advantage": self.minimum_advantage,
            "certificates": [asdict(certificate) for certificate in self.certificates],
            "state_sha256": _digest(serialized_state),
        }


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def matched_pair_audit(
    critical: CommunicationScenario,
    decoy: CommunicationScenario,
) -> dict[str, object]:
    critical_state = state_to_dict(critical.state)
    decoy_state = state_to_dict(decoy.state)
    critical_knowledge = critical_state.pop("knowledge")
    decoy_knowledge = decoy_state.pop("knowledge")
    changed_agents = sorted(
        agent_id
        for agent_id in critical_knowledge
        if critical_knowledge[agent_id] != decoy_knowledge[agent_id]
    )
    receiver_delta = {
        "critical": critical_knowledge[critical.receiver].get(critical.target),
        "decoy": decoy_knowledge[decoy.receiver].get(decoy.target),
    }
    return {
        "structural_state_identical": critical_state == decoy_state,
        "structural_state_sha256": _digest(critical_state),
        "sender_observation_identical": observation_for(
            critical.state, critical.sender
        )
        == observation_for(decoy.state, decoy.sender),
        "sender_observation_sha256": _digest(
            observation_for(critical.state, critical.sender)
        ),
        "changed_knowledge_agents": changed_agents,
        "only_receiver_target_knowledge_changes": changed_agents == [critical.receiver]
        and receiver_delta["critical"] is None
        and receiver_delta["decoy"] is not None,
    }


def swap_team_labels(state: GameState) -> GameState:
    swapped = state.clone()

    def swap_team(team: str) -> str:
        return {"BLUE": "RED", "RED": "BLUE", "NEUTRAL": "NEUTRAL"}[team]

    id_map = {
        agent_id: ("red-" if agent.team == "BLUE" else "blue-") + agent_id.split("-", 1)[1]
        for agent_id, agent in state.agents.items()
    }
    for node in swapped.nodes.values():
        node.owner = swap_team(node.owner)
    agents = {}
    knowledge = {}
    for old_id, agent in swapped.agents.items():
        new_id = id_map[old_id]
        agent.id = new_id
        agent.team = swap_team(agent.team)
        agents[new_id] = agent
        knowledge[new_id] = {
            node_id: NodeObservation(
                observation.node,
                swap_team(observation.owner),
                observation.status,
                observation.value,
                observation.critical,
                observation.observed_turn,
            )
            for node_id, observation in swapped.knowledge[old_id].items()
        }
    swapped.agents = agents
    swapped.knowledge = knowledge
    swapped.validate()
    return swapped


def permute_agent_labels(
    state: GameState,
    team: Team,
    permutation: tuple[int, int, int, int],
) -> GameState:
    if sorted(permutation) != [0, 1, 2, 3]:
        raise ValueError(f"agent-label assignment must be a permutation: {permutation}")
    permuted = state.clone()
    members = sorted(agent.id for agent in permuted.agents.values() if agent.team == team)
    prefix = team.lower()
    expected = [f"{prefix}-{index}" for index in range(4)]
    if members != expected:
        raise ValueError(f"agent-label permutation requires canonical IDs: {members}")
    mapping = {
        old_id: f"{prefix}-{permutation[index]}"
        for index, old_id in enumerate(expected)
    }
    agents = {}
    knowledge = {}
    for old_id, agent in permuted.agents.items():
        new_id = mapping.get(old_id, old_id)
        agent.id = new_id
        agents[new_id] = agent
        knowledge[new_id] = permuted.knowledge[old_id]
    permuted.agents = agents
    permuted.knowledge = knowledge
    permuted.validate()
    return permuted


def _candidate(state: GameState, team: Team) -> tuple[str, str, str] | None:
    targets = sorted(
        state.nodes.values(),
        key=lambda node: (-node.value, node.id),
    )
    members = sorted(
        (agent for agent in state.agents.values() if agent.team == team),
        key=lambda agent: agent.id,
    )
    for target in targets:
        if target.owner != "NEUTRAL":
            continue
        receivers = [
            agent
            for agent in members
            if target.id in state.nodes[agent.position].neighbors
        ]
        if len(receivers) != 1:
            continue
        receiver = receivers[0]
        senders = [
            agent
            for agent in members
            if agent.id != receiver.id
            and target.id not in state.nodes[agent.position].neighbors
        ]
        if senders:
            return senders[0].id, receiver.id, target.id
    return None


def _relabel_team(
    state: GameState,
    team: Team,
    sender: str,
    receiver: str,
    desired: tuple[str, str],
) -> tuple[str, str]:
    current = sorted(agent.id for agent in state.agents.values() if agent.team == team)
    desired_sender, desired_receiver = desired
    if desired_sender == desired_receiver or {desired_sender, desired_receiver} - set(current):
        raise ValueError(f"invalid requested role identities: {desired}")
    old_remaining = [agent_id for agent_id in current if agent_id not in {sender, receiver}]
    new_remaining = [agent_id for agent_id in current if agent_id not in set(desired)]
    mapping = {
        sender: desired_sender,
        receiver: desired_receiver,
        **dict(zip(old_remaining, new_remaining, strict=True)),
    }
    remapped_agents = {}
    remapped_knowledge = {}
    for old_id, agent in state.agents.items():
        new_id = mapping.get(old_id, old_id)
        agent.id = new_id
        remapped_agents[new_id] = agent
        remapped_knowledge[new_id] = state.knowledge[old_id]
    state.agents = remapped_agents
    state.knowledge = remapped_knowledge
    state.validate()
    return desired


def _base_state(
    seed: int,
    size: int,
    team: Team,
    role_pair: tuple[str, str] | None,
) -> tuple[GameState, str, str, str] | None:
    state = generate_state(seed, size)
    selected = _candidate(state, team)
    if selected is None:
        return None
    sender, receiver, target_id = selected
    if role_pair is not None:
        sender, receiver = _relabel_team(state, team, sender, receiver, role_pair)
    target = state.nodes[target_id]
    target.value = 3
    target.critical = True
    target.fortification = 0
    target.exposed = True
    target.compromised = False
    state.turn = 1
    for agent in state.agents.values():
        state.knowledge[agent.id].pop(target_id, None)
        if agent.team != team:
            continue
        if agent.id == receiver:
            agent.resource = 1
            home = state.nodes[agent.position]
            home.compromised = False
            home.exposed = False
            home.fortification = 2
            state.knowledge[agent.id] = {
                agent.position: observe_node(home, state.turn),
            }
        else:
            agent.resource = 0
            local = (agent.position, *state.nodes[agent.position].neighbors)
            state.knowledge[agent.id] = {
                node_id: observe_node(state.nodes[node_id], state.turn)
                for node_id in local
                if node_id != target_id
            }
    state.knowledge[sender][target_id] = observe_node(target, 0)
    state.validate()
    return state, sender, receiver, target_id


def informed_state(scenario_state: GameState, sender: str, team: Team, target: str) -> GameState:
    state = copy.deepcopy(scenario_state)
    fact = state.knowledge[sender][target]
    for agent in state.agents.values():
        if agent.team == team and agent.id != sender:
            state.knowledge[agent.id][target] = fact
    state.validate()
    return state


def _serialized_assignment(solution: JointValue) -> tuple[tuple[str, dict], ...]:
    return tuple((agent_id, action.to_dict()) for agent_id, action in solution.assignment)


def certify(
    state: GameState,
    sender: str,
    receiver: str,
    target: str,
    team: Team,
) -> tuple[StyleCertificate, ...]:
    informed = informed_state(state, sender, team, target)
    if Action("CAPTURE", target) in legal_actions(state, receiver):
        raise ValueError("dropped condition already permits the receiver to capture")
    if Action("CAPTURE", target) not in legal_actions(informed, receiver):
        raise ValueError("broadcast does not enable the receiver's capture")
    certificates = []
    for style in OPPONENT_STYLES:
        opponent_team: Team = "RED" if team == "BLUE" else "BLUE"
        rival_actions = deterministic_policy(state, opponent_team, style)
        allowed = maximize_terminal_control(
            informed,
            team,
            rival_actions,
            initial_state=state,
        )
        dropped = maximize_terminal_control(
            state,
            team,
            rival_actions,
            initial_state=state,
        )
        certificates.append(
            StyleCertificate(
                style,
                allowed.value,
                dropped.value,
                allowed.value - dropped.value,
                _serialized_assignment(allowed),
                _serialized_assignment(dropped),
                allowed.explored,
                dropped.explored,
            )
        )
    return tuple(certificates)


def generate_pair(
    seed: int,
    size: int = 12,
    team: Team = "BLUE",
    role_pair: tuple[str, str] | None = None,
) -> tuple[CommunicationScenario, CommunicationScenario] | None:
    built = _base_state(seed, size, team, role_pair)
    if built is None:
        return None
    state, sender, receiver, target = built
    certificates = certify(state, sender, receiver, target, team)
    minimum = min(item.advantage for item in certificates)
    if minimum <= 1e-12:
        return None
    critical = CommunicationScenario(
        COMMUNICATION_CURRICULUM_VERSION,
        GENERATOR_VERSION,
        seed,
        size,
        "critical",
        team,
        sender,
        receiver,
        target,
        minimum,
        certificates,
        state,
    )

    decoy_state = copy.deepcopy(state)
    decoy_state.knowledge[receiver][target] = decoy_state.knowledge[sender][target]
    decoy_certificates = certify_decoy(decoy_state, sender, target, team)
    decoy = CommunicationScenario(
        COMMUNICATION_CURRICULUM_VERSION,
        GENERATOR_VERSION,
        seed,
        size,
        "decoy",
        team,
        sender,
        receiver,
        target,
        min(item.advantage for item in decoy_certificates),
        decoy_certificates,
        decoy_state,
    )
    return critical, decoy


def certify_decoy(state: GameState, sender: str, target: str, team: Team) -> tuple[StyleCertificate, ...]:
    informed = informed_state(state, sender, team, target)
    certificates = []
    for style in OPPONENT_STYLES:
        opponent_team: Team = "RED" if team == "BLUE" else "BLUE"
        rival_actions = deterministic_policy(state, opponent_team, style)
        allowed = maximize_terminal_control(informed, team, rival_actions, initial_state=state)
        dropped = maximize_terminal_control(state, team, rival_actions, initial_state=state)
        certificates.append(
            StyleCertificate(
                style,
                allowed.value,
                dropped.value,
                allowed.value - dropped.value,
                _serialized_assignment(allowed),
                _serialized_assignment(dropped),
                allowed.explored,
                dropped.explored,
            )
        )
    if any(abs(item.advantage) > 1e-12 for item in certificates):
        raise ValueError("matched decoy unexpectedly benefits from the sender's fact")
    return tuple(certificates)


def generate_manifest(
    *,
    count: int,
    seed_start: int,
    sizes: tuple[int, ...] = (12, 13),
) -> dict:
    pairs = []
    candidate = seed_start
    prefix = "blue"
    role_pairs = tuple(
        (f"{prefix}-{sender}", f"{prefix}-{receiver}")
        for sender in range(4)
        for receiver in range(4)
        if sender != receiver
    )
    while len(pairs) < count:
        size = sizes[len(pairs) % len(sizes)]
        pair = generate_pair(candidate, size, role_pair=role_pairs[len(pairs) % len(role_pairs)])
        candidate += 1
        if pair is None:
            continue
        critical, decoy = pair
        pairs.append(
            {
                "critical": critical.manifest_row(),
                "decoy": decoy.manifest_row(),
                "matched_pair_audit": matched_pair_audit(critical, decoy),
            }
        )
    body = {
        "version": COMMUNICATION_CURRICULUM_VERSION,
        "seed_start": seed_start,
        "sizes": list(sizes),
        "pair_count": count,
        "pairs": pairs,
    }
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body
