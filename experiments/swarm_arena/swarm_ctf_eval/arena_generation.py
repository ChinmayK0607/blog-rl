from __future__ import annotations

import random

from .arena import AgentState, GameState, Node, observe_node


GENERATOR_VERSION = "arena-gen-v1"


def _connected_edges(rng: random.Random, size: int, extra_edges: int) -> set[tuple[int, int]]:
    edges = {(index, (index + 1) % size) for index in range(size)}
    edges = {tuple(sorted(edge)) for edge in edges}
    candidates = [
        (left, right)
        for left in range(size)
        for right in range(left + 1, size)
        if (left, right) not in edges
    ]
    rng.shuffle(candidates)
    for edge in candidates:
        if extra_edges <= 0:
            break
        left, right = edge
        degree_left = sum(left in item for item in edges)
        degree_right = sum(right in item for item in edges)
        if degree_left < 4 and degree_right < 4:
            edges.add(edge)
            extra_edges -= 1
    return edges


def generate_state(seed: int, size: int = 12) -> GameState:
    """Generate a deterministic connected 4v4 state from an integer seed."""

    if size < 10:
        raise ValueError("size must be at least 10")
    rng = random.Random(seed)
    node_ids = [f"V{value:02d}" for value in rng.sample(range(10, 100), size)]
    edges = _connected_edges(rng, size, max(2, size // 3))
    neighbors: dict[int, set[int]] = {index: set() for index in range(size)}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)

    # Alternating ownership guarantees contested frontiers without encoding team
    # identity in node names. Remaining nodes are neutral exploration targets.
    owner_slots = ["BLUE", "RED"] * 4 + ["NEUTRAL"] * (size - 8)
    rng.shuffle(owner_slots)
    values = [rng.randint(1, 3) for _ in range(size)]
    critical_indices = set(sorted(range(size), key=lambda index: (-values[index], index))[:2])
    nodes: dict[str, Node] = {}
    for index, node_id in enumerate(node_ids):
        owner = owner_slots[index]
        fortification = rng.choice((0, 0, 1)) if owner != "NEUTRAL" else 0
        exposed = owner != "NEUTRAL" and fortification == 0 and rng.random() < 0.28
        compromised = owner != "NEUTRAL" and not exposed and rng.random() < 0.12
        nodes[node_id] = Node(
            id=node_id,
            neighbors=tuple(sorted(node_ids[item] for item in neighbors[index])),
            owner=owner,  # type: ignore[arg-type]
            value=values[index],
            critical=index in critical_indices,
            fortification=fortification,
            exposed=exposed,
            compromised=compromised,
        )

    agents: dict[str, AgentState] = {}
    for team in ("BLUE", "RED"):
        positions = sorted(node.id for node in nodes.values() if node.owner == team)
        rng.shuffle(positions)
        for index in range(4):
            agent_id = f"{team.lower()}-{index}"
            agents[agent_id] = AgentState(agent_id, team, positions[index], rng.randint(1, 3))  # type: ignore[arg-type]

    knowledge = {}
    for agent_id, agent in agents.items():
        memory = {agent.position: observe_node(nodes[agent.position], 0)}
        adjacent = list(nodes[agent.position].neighbors)
        rng.shuffle(adjacent)
        # Each agent begins with one or two locally inspected neighbors and knows
        # the remaining adjacent identifiers as scan candidates.
        for node_id in adjacent[: rng.randint(1, min(2, len(adjacent)))]:
            memory[node_id] = observe_node(nodes[node_id], 0)
        knowledge[agent_id] = memory

    state = GameState(turn=0, nodes=nodes, agents=agents, knowledge=knowledge)
    state.validate()
    return state


def generate_mechanics_state(seed: int, skill: str) -> GameState:
    """Create a procedural state with a controlled prompt-visible mechanic."""

    if skill not in {"WAIT", "SCAN", "TRANSFER", "SILENCE"}:
        raise ValueError(f"unsupported targeted skill: {skill}")
    state = generate_state(seed)
    actor = state.agents["blue-0"]
    home = state.nodes[actor.position]
    home.owner = "BLUE"
    home.compromised = False
    home.exposed = False
    home.fortification = 2
    actor.resource = 0 if skill in {"WAIT", "SCAN", "SILENCE"} else 2

    if skill in {"WAIT", "SILENCE"}:
        if skill == "SILENCE":
            for node_id in (actor.position, *home.neighbors):
                node = state.nodes[node_id]
                node.owner = "BLUE"
                node.critical = False
                node.compromised = False
                node.exposed = False
                node.fortification = 0
        state.knowledge[actor.id] = {
            node_id: observe_node(state.nodes[node_id], state.turn)
            for node_id in (actor.position, *home.neighbors)
        }
    elif skill == "SCAN":
        unknown = sorted(home.neighbors)[0]
        state.knowledge[actor.id] = {
            node_id: observe_node(state.nodes[node_id], state.turn)
            for node_id in (actor.position, *(item for item in home.neighbors if item != unknown))
        }
    else:
        receiver = state.agents["blue-1"]
        receiver.position = sorted(home.neighbors)[0]
        receiver.resource = 0
        need = state.nodes[receiver.position]
        need.owner = "BLUE"
        need.compromised = True
        need.exposed = False
        need.fortification = 0
        state.knowledge[actor.id] = {actor.position: observe_node(home, state.turn)}
        state.knowledge[receiver.id] = {
            receiver.position: observe_node(need, state.turn)
        }
    state.validate()
    return state
