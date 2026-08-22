from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Literal, cast

from .arena import (
    WAIT,
    Action,
    GameState,
    NodeObservation,
    Team,
    legal_actions,
    observation_for,
    observe_node,
    opponent,
    state_to_dict,
    step,
)
from .arena_generation import GENERATOR_VERSION, generate_state
from .arena_oracle import deterministic_policy
from .rl_v3 import terminal_control_delta

HANDOFF_CURRICULUM_VERSION = "arena-information-handoff-v2"
HandoffKind = Literal["critical", "decoy"]
OPPONENT_STYLES = ("balanced", "aggressive", "defensive")
WORLD_LABELS = ("left_exposed", "right_exposed")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _action_rows(actions: tuple[tuple[str, Action], ...]) -> tuple[tuple[str, dict], ...]:
    return tuple((world, action.to_dict()) for world, action in actions)


@dataclass(frozen=True)
class HandoffCertificate:
    style: str
    informed_value: float
    dropped_value: float
    advantage: float
    informed_actions: tuple[tuple[str, dict], ...]
    dropped_actions: tuple[tuple[str, dict], ...]
    informed_information_sets: int
    dropped_information_sets: int
    action_evaluations: int


@dataclass(frozen=True)
class HandoffWorld:
    label: str
    active_target: str
    state: GameState

    def manifest_row(self, receiver: str) -> dict:
        return {
            "label": self.label,
            "active_target": self.active_target,
            "state_sha256": _digest(state_to_dict(self.state)),
            "receiver_observation_sha256": _digest(
                observation_for(self.state, receiver)
            ),
            "receiver_legal_actions_sha256": _digest(
                [action.to_dict() for action in legal_actions(self.state, receiver)]
            ),
        }


@dataclass(frozen=True)
class HandoffScenario:
    version: str
    generator_version: str
    seed: int
    size: int
    horizon: int
    kind: HandoffKind
    team: Team
    sender: str
    receiver: str
    candidate_targets: tuple[str, str]
    minimum_advantage: float
    certificates: tuple[HandoffCertificate, ...]
    worlds: tuple[HandoffWorld, HandoffWorld]

    def manifest_row(self) -> dict:
        return {
            "version": self.version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "size": self.size,
            "horizon": self.horizon,
            "kind": self.kind,
            "team": self.team,
            "sender": self.sender,
            "receiver": self.receiver,
            "candidate_targets": list(self.candidate_targets),
            "minimum_advantage": self.minimum_advantage,
            "certificates": [asdict(certificate) for certificate in self.certificates],
            "worlds": [world.manifest_row(self.receiver) for world in self.worlds],
        }


@dataclass(frozen=True)
class _PolicySolution:
    value: float
    actions: tuple[tuple[str, Action], ...]
    information_sets: int
    action_evaluations: int


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
    agents = {}
    knowledge = {}
    for old_id, agent in state.agents.items():
        new_id = mapping.get(old_id, old_id)
        agent.id = new_id
        agents[new_id] = agent
        knowledge[new_id] = state.knowledge[old_id]
    state.agents = agents
    state.knowledge = knowledge
    state.validate()
    return desired


def _candidate(state: GameState, team: Team) -> tuple[str, str, tuple[str, str]] | None:
    members = sorted(
        (agent for agent in state.agents.values() if agent.team == team),
        key=lambda agent: agent.id,
    )
    for receiver in members:
        candidates = sorted(
            (
                node_id
                for node_id in state.nodes[receiver.position].neighbors
                if state.nodes[node_id].owner == "NEUTRAL"
            ),
            key=lambda node_id: (-state.nodes[node_id].value, node_id),
        )
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                senders = [
                    agent
                    for agent in members
                    if agent.id != receiver.id
                    and left not in state.nodes[agent.position].neighbors
                    and right not in state.nodes[agent.position].neighbors
                ]
                if senders:
                    return senders[0].id, receiver.id, tuple(sorted((left, right)))
    return None


def _stale_target_observation(node_id: str) -> NodeObservation:
    return NodeObservation(node_id, "NEUTRAL", "SECURE", 3, True, 0)


def _configure_world(
    base: GameState,
    *,
    team: Team,
    sender: str,
    receiver: str,
    candidates: tuple[str, str],
    active_target: str,
) -> GameState:
    state = base.clone()
    state.turn = 2
    for node_id in candidates:
        node = state.nodes[node_id]
        node.owner = "NEUTRAL"
        node.value = 3
        node.critical = True
        node.fortification = 0
        node.compromised = False
        node.exposed = node_id == active_target

    for agent in state.agents.values():
        if agent.team != team:
            continue
        home = state.nodes[agent.position]
        home.owner = team
        home.compromised = False
        home.exposed = False
        home.fortification = 2
        agent.resource = 1 if agent.id == receiver else 0
        memory = {
            agent.position: observe_node(home, state.turn),
        }
        if agent.id == receiver:
            memory.update(
                {node_id: _stale_target_observation(node_id) for node_id in candidates}
            )
        elif agent.id == sender:
            memory.update(
                {
                    node_id: observe_node(state.nodes[node_id], state.turn)
                    for node_id in candidates
                }
            )
        else:
            memory.update(
                {
                    node_id: observe_node(state.nodes[node_id], state.turn)
                    for node_id in state.nodes[agent.position].neighbors
                    if node_id not in candidates
                }
            )
        state.knowledge[agent.id] = memory
    state.validate()
    return state


def _delivered_state(
    state: GameState,
    sender: str,
    receiver: str,
    active_target: str,
) -> GameState:
    delivered = state.clone()
    fact = delivered.knowledge[sender][active_target]
    previous = delivered.knowledge[receiver].get(active_target)
    if previous is None or fact.observed_turn > previous.observed_turn:
        delivered.knowledge[receiver][active_target] = fact
    delivered.validate()
    return delivered


def _receiver_policy_solution(
    worlds: tuple[HandoffWorld, HandoffWorld],
    *,
    team: Team,
    sender: str,
    receiver: str,
    style: str,
    informed: bool,
) -> _PolicySolution:
    records = []
    for world in worlds:
        decision_state = (
            _delivered_state(world.state, sender, receiver, world.active_target)
            if informed
            else world.state
        )
        actions = legal_actions(decision_state, receiver)
        rival_actions = deterministic_policy(decision_state, opponent(team), style)
        fixed_team_actions = {
            agent.id: WAIT
            for agent in decision_state.agents.values()
            if agent.team == team
        }
        values = {}
        for action in actions:
            outcome = step(
                decision_state,
                {**rival_actions, **fixed_team_actions, receiver: action},
            )
            values[action] = terminal_control_delta(
                decision_state,
                outcome.state,
                team,
            )
        records.append(
            {
                "world": world,
                "information": _digest(observation_for(decision_state, receiver)),
                "actions": actions,
                "values": values,
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[cast(str, record["information"])].append(record)
    selected: dict[str, Action] = {}
    evaluations = 0
    for information, group in sorted(grouped.items()):
        action_sets = {tuple(record["actions"]) for record in group}
        if len(action_sets) != 1:
            raise ValueError("one information set exposes different legal actions")
        actions = cast(tuple[Action, ...], group[0]["actions"])
        evaluations += len(actions) * len(group)
        selected[information] = min(
            actions,
            key=lambda action: (
                -statistics.mean(record["values"][action] for record in group),
                action,
            ),
        )
    world_actions = tuple(
        (
            cast(HandoffWorld, record["world"]).label,
            selected[cast(str, record["information"])],
        )
        for record in records
    )
    value = statistics.mean(
        record["values"][selected[cast(str, record["information"])]]
        for record in records
    )
    return _PolicySolution(value, world_actions, len(grouped), evaluations)


def _certify(
    worlds: tuple[HandoffWorld, HandoffWorld],
    *,
    team: Team,
    sender: str,
    receiver: str,
) -> tuple[HandoffCertificate, ...]:
    certificates = []
    for style in OPPONENT_STYLES:
        informed = _receiver_policy_solution(
            worlds,
            team=team,
            sender=sender,
            receiver=receiver,
            style=style,
            informed=True,
        )
        dropped = _receiver_policy_solution(
            worlds,
            team=team,
            sender=sender,
            receiver=receiver,
            style=style,
            informed=False,
        )
        certificates.append(
            HandoffCertificate(
                style,
                informed.value,
                dropped.value,
                informed.value - dropped.value,
                _action_rows(informed.actions),
                _action_rows(dropped.actions),
                informed.information_sets,
                dropped.information_sets,
                informed.action_evaluations + dropped.action_evaluations,
            )
        )
    return tuple(certificates)


def _decoy_worlds(
    worlds: tuple[HandoffWorld, HandoffWorld],
    receiver: str,
) -> tuple[HandoffWorld, HandoffWorld]:
    result = []
    for world in worlds:
        state = world.state.clone()
        for node_id in world.state.nodes[state.agents[receiver].position].neighbors:
            if node_id in state.knowledge[receiver]:
                state.knowledge[receiver][node_id] = observe_node(
                    state.nodes[node_id], state.turn
                )
        state.validate()
        result.append(HandoffWorld(world.label, world.active_target, state))
    return cast(tuple[HandoffWorld, HandoffWorld], tuple(result))


def _mechanical_invariants(scenario: HandoffScenario) -> dict[str, object]:
    receiver_actions = [
        legal_actions(world.state, scenario.receiver) for world in scenario.worlds
    ]
    informed_actions = [
        legal_actions(
            _delivered_state(
                world.state,
                scenario.sender,
                scenario.receiver,
                world.active_target,
            ),
            scenario.receiver,
        )
        for world in scenario.worlds
    ]
    return {
        "receiver_action_sets_match_across_worlds": len(set(receiver_actions)) == 1,
        "message_does_not_change_receiver_legal_actions": receiver_actions
        == informed_actions,
        "both_candidate_captures_already_legal": all(
            all(Action("CAPTURE", target) in actions for target in scenario.candidate_targets)
            for actions in receiver_actions
        ),
        "sender_cannot_act_on_candidates": all(
            all(
                action.target not in scenario.candidate_targets
                for action in legal_actions(world.state, scenario.sender)
            )
            for world in scenario.worlds
        ),
    }


def matched_pair_audit(
    critical: HandoffScenario,
    decoy: HandoffScenario,
) -> dict[str, object]:
    changed_agents = []
    structural_matches = []
    sender_matches = []
    for critical_world, decoy_world in zip(
        critical.worlds, decoy.worlds, strict=True
    ):
        critical_state = state_to_dict(critical_world.state)
        decoy_state = state_to_dict(decoy_world.state)
        critical_knowledge = critical_state.pop("knowledge")
        decoy_knowledge = decoy_state.pop("knowledge")
        structural_matches.append(critical_state == decoy_state)
        changed_agents.append(
            sorted(
                agent_id
                for agent_id in critical_knowledge
                if critical_knowledge[agent_id] != decoy_knowledge[agent_id]
            )
        )
        sender_matches.append(
            observation_for(critical_world.state, critical.sender)
            == observation_for(decoy_world.state, decoy.sender)
        )
    critical_receiver_observations = {
        _digest(observation_for(world.state, critical.receiver))
        for world in critical.worlds
    }
    decoy_receiver_observations = {
        _digest(observation_for(world.state, decoy.receiver)) for world in decoy.worlds
    }
    result = {
        "structural_worlds_identical": all(structural_matches),
        "sender_observations_identical": all(sender_matches),
        "only_receiver_knowledge_changes": all(
            agents == [critical.receiver] for agents in changed_agents
        ),
        "critical_receiver_worlds_indistinguishable_without_message": len(
            critical_receiver_observations
        )
        == 1,
        "decoy_receiver_worlds_distinguishable_without_message": len(
            decoy_receiver_observations
        )
        == 2,
        **_mechanical_invariants(critical),
    }
    return result


def exhaustive_receiver_target_separation(scenario: HandoffScenario) -> dict[str, object]:
    """Certify the one-turn value of the informed target under every teammate action.

    The opponent remains one of the three immutable deterministic audit styles. The
    other same-team agents range over every legal joint action. Only the receiver's
    action changes between the factual and target-swapped branches, matching the
    receiver-isolated rollout intervention.
    """
    if scenario.kind != "critical":
        raise ValueError("terminal target separation is defined only for critical scenarios")
    rows = []
    for world in scenario.worlds:
        alternate_target = next(
            target for target in scenario.candidate_targets if target != world.active_target
        )
        teammate_ids = sorted(
            agent.id
            for agent in world.state.agents.values()
            if agent.team == scenario.team and agent.id != scenario.receiver
        )
        teammate_action_sets = tuple(
            legal_actions(world.state, agent_id) for agent_id in teammate_ids
        )
        for teammate_actions in itertools.product(*teammate_action_sets):
            fixed_teammates = dict(zip(teammate_ids, teammate_actions, strict=True))
            for style in OPPONENT_STYLES:
                rival_actions = deterministic_policy(
                    world.state,
                    opponent(scenario.team),
                    style,
                )
                factual = step(
                    world.state,
                    {
                        **rival_actions,
                        **fixed_teammates,
                        scenario.receiver: Action("CAPTURE", world.active_target),
                    },
                )
                swapped = step(
                    world.state,
                    {
                        **rival_actions,
                        **fixed_teammates,
                        scenario.receiver: Action("CAPTURE", alternate_target),
                    },
                )
                factual_value = terminal_control_delta(
                    world.state,
                    factual.state,
                    scenario.team,
                )
                swapped_value = terminal_control_delta(
                    world.state,
                    swapped.state,
                    scenario.team,
                )
                rows.append(
                    {
                        "world": world.label,
                        "opponent_style": style,
                        "advantage": factual_value - swapped_value,
                    }
                )
    advantages = tuple(float(row["advantage"]) for row in rows)
    return {
        "joint_action_evaluations": len(rows),
        "minimum_advantage": min(advantages),
        "maximum_advantage": max(advantages),
        "all_strictly_positive": all(value > 1e-12 for value in advantages),
        "worlds": sorted({str(row["world"]) for row in rows}),
        "opponent_styles": sorted({str(row["opponent_style"]) for row in rows}),
    }


def generate_pair(
    seed: int,
    size: int = 12,
    horizon: int = 4,
    team: Team = "BLUE",
    role_pair: tuple[str, str] | None = None,
) -> tuple[HandoffScenario, HandoffScenario] | None:
    if horizon <= 2:
        raise ValueError("handoff evaluation horizon must extend beyond the initial turn")
    state = generate_state(seed, size)
    selected = _candidate(state, team)
    if selected is None:
        return None
    sender, receiver, candidates = selected
    if role_pair is not None:
        sender, receiver = _relabel_team(state, team, sender, receiver, role_pair)
    worlds = cast(
        tuple[HandoffWorld, HandoffWorld],
        tuple(
            HandoffWorld(
                label,
                target,
                _configure_world(
                    state,
                    team=team,
                    sender=sender,
                    receiver=receiver,
                    candidates=candidates,
                    active_target=target,
                ),
            )
            for label, target in zip(WORLD_LABELS, candidates, strict=True)
        ),
    )
    critical_certificates = _certify(
        worlds,
        team=team,
        sender=sender,
        receiver=receiver,
    )
    if min(certificate.advantage for certificate in critical_certificates) <= 1e-12:
        return None
    if any(
        dict(certificate.informed_actions)[world.label]
        != Action("CAPTURE", world.active_target).to_dict()
        for certificate in critical_certificates
        for world in worlds
    ):
        return None
    critical = HandoffScenario(
        HANDOFF_CURRICULUM_VERSION,
        GENERATOR_VERSION,
        seed,
        size,
        horizon,
        "critical",
        team,
        sender,
        receiver,
        candidates,
        min(certificate.advantage for certificate in critical_certificates),
        critical_certificates,
        worlds,
    )

    decoy_worlds = _decoy_worlds(worlds, receiver)
    decoy_certificates = _certify(
        decoy_worlds,
        team=team,
        sender=sender,
        receiver=receiver,
    )
    if any(abs(certificate.advantage) > 1e-12 for certificate in decoy_certificates):
        raise ValueError("matched handoff decoy unexpectedly benefits from delivery")
    decoy = HandoffScenario(
        HANDOFF_CURRICULUM_VERSION,
        GENERATOR_VERSION,
        seed,
        size,
        horizon,
        "decoy",
        team,
        sender,
        receiver,
        candidates,
        0.0,
        decoy_certificates,
        decoy_worlds,
    )
    if not all(bool(value) for value in matched_pair_audit(critical, decoy).values()):
        raise ValueError("generated handoff pair violates a frozen invariant")
    return critical, decoy


def reconstruct_manifest_scenario(row: dict) -> HandoffScenario:
    role_pair = (str(row["sender"]), str(row["receiver"]))
    pair = generate_pair(
        int(row["seed"]),
        int(row["size"]),
        int(row["horizon"]),
        cast(Team, str(row["team"])),
        role_pair,
    )
    if pair is None:
        raise ValueError("committed handoff scenario no longer reconstructs")
    kind = str(row["kind"])
    if kind not in {"critical", "decoy"}:
        raise ValueError(f"unknown handoff scenario kind: {kind}")
    scenario = pair[0 if kind == "critical" else 1]
    if _digest(scenario.manifest_row()) != _digest(row):
        raise ValueError("reconstructed handoff scenario differs from its manifest")
    return scenario


def generate_manifest(
    *,
    count: int,
    seed_start: int,
    sizes: tuple[int, ...],
    horizons: tuple[int, ...],
) -> dict:
    if not count or count % 12:
        raise ValueError("handoff manifest count must be a positive multiple of 12")
    if not sizes or not horizons:
        raise ValueError("handoff manifests require sizes and horizons")
    role_pairs = tuple(
        (f"blue-{sender}", f"blue-{receiver}")
        for sender in range(4)
        for receiver in range(4)
        if sender != receiver
    )
    pairs = []
    candidate = seed_start
    while len(pairs) < count:
        index = len(pairs)
        pair = generate_pair(
            candidate,
            sizes[index % len(sizes)],
            horizons[index % len(horizons)],
            role_pair=role_pairs[index % len(role_pairs)],
        )
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
        "version": HANDOFF_CURRICULUM_VERSION,
        "seed_start": seed_start,
        "sizes": list(sizes),
        "horizons": list(horizons),
        "pair_count": count,
        "pairs": pairs,
    }
    body["sha256"] = _digest(body)
    return body
