import random
from collections.abc import Callable

from symbolic_tool_calling_v1.models import Action, BenchmarkTask

Policy = Callable[[BenchmarkTask, int], list[Action]]


def optimal_policy(task: BenchmarkTask, seed: int = 0) -> list[Action]:
    del seed
    graph = task.hidden_graph_spec
    actions = []
    for room_index in range(task.dependency_depth):
        if room_index == 1:
            actions.append(Action(tool="pickup", arguments={"item": graph.key_item}))
        if room_index == task.dependency_depth // 2:
            actions.append(Action(tool="use", arguments={"item": graph.switch_id}))
        if room_index == task.dependency_depth - 1:
            actions.append(Action(tool="use", arguments={"item": graph.key_item}))
            actions.append(Action(tool="query"))
        actions.append(Action(tool="move", arguments={"direction": "east"}))
    actions.append(Action(tool="submit", arguments={"system": graph.target_system, "code": graph.access_code}))
    return actions


def exploratory_policy(task: BenchmarkTask, seed: int = 0) -> list[Action]:
    rng = random.Random(seed)
    graph = task.hidden_graph_spec
    actions = [Action(tool="inspect")]
    for room_index in range(task.dependency_depth):
        room = graph.rooms[f"room_{room_index:02d}"]
        side_exits = sorted(direction for direction in room.exits if direction in {"north", "south"})
        if side_exits and rng.random() < 0.8:
            direction = rng.choice(side_exits)
            actions.append(Action(tool="move", arguments={"direction": direction}))
            actions.append(Action(tool="inspect"))
            for _ in range(task.recovery_cost - 1):
                actions.append(Action(tool="move", arguments={"direction": "deeper"}))
                actions.append(Action(tool="inspect"))
            for _ in range(task.recovery_cost):
                actions.append(Action(tool="move", arguments={"direction": "back"}))
        if room_index == 1:
            actions.append(Action(tool="pickup", arguments={"item": graph.key_item}))
        if room_index == task.dependency_depth // 2:
            actions.append(Action(tool="use", arguments={"item": graph.switch_id}))
        if room_index == task.dependency_depth - 1:
            actions.append(Action(tool="use", arguments={"item": graph.key_item}))
            actions.append(Action(tool="query"))
        actions.append(Action(tool="move", arguments={"direction": "east"}))
    actions.append(Action(tool="inspect"))
    actions.append(Action(tool="submit", arguments={"system": graph.target_system, "code": graph.access_code}))
    return actions


def broken_code_policy(task: BenchmarkTask, seed: int = 0) -> list[Action]:
    actions = optimal_policy(task, seed)
    actions[-1] = Action(
        tool="submit",
        arguments={"system": task.hidden_graph_spec.target_system, "code": "000000"},
    )
    return actions


POLICIES: dict[str, Policy] = {
    "scripted-optimal-v1": optimal_policy,
    "scripted-exploratory-v1": exploratory_policy,
    "scripted-broken-code-v1": broken_code_policy,
}


def policy_by_id(policy_id: str) -> Policy:
    try:
        return POLICIES[policy_id]
    except KeyError as error:
        raise ValueError(f"unknown policy: {policy_id}") from error
