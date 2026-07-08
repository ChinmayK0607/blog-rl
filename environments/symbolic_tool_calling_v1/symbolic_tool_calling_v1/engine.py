import json
from collections.abc import Iterable

from symbolic_tool_calling_v1.models import Action, BenchmarkState, BenchmarkTask, Transition


def initial_state(task: BenchmarkTask) -> BenchmarkState:
    return BenchmarkState(current_room=task.hidden_graph_spec.start_room)


def _observe(task: BenchmarkTask, state: BenchmarkState) -> str:
    graph = task.hidden_graph_spec
    room = graph.rooms[state.current_room]
    locked_exits = sorted(
        direction
        for direction, destination in room.exits.items()
        if destination == graph.target_room
        and (graph.target_room not in state.unlocked_rooms or graph.switch_id not in state.activated_switches)
    )
    return json.dumps(
        {
            "room": room.room_id,
            "exits": sorted(room.exits),
            "locked_exits": locked_exits,
            "objects": [item for item in room.objects if item not in state.inventory],
            "activated_mechanisms": [item for item in room.objects if item in state.activated_switches],
            "access_terminal": room.terminal_code is not None,
            "detail": room.verbosity_text.strip(),
        },
        sort_keys=True,
    )


def apply_action(task: BenchmarkTask, state: BenchmarkState, action: Action) -> tuple[BenchmarkState, str]:
    if state.terminal:
        return state.model_copy(deep=True), "The episode has already terminated."

    next_state = state.model_copy(deep=True)
    next_state.turns += 1
    graph = task.hidden_graph_spec
    room = graph.rooms[next_state.current_room]
    args = action.arguments
    observation: str

    if action.tool == "inspect":
        observation = _observe(task, next_state)
    elif action.tool == "move":
        direction = str(args.get("direction", ""))
        destination = room.exits.get(direction)
        locked_target = destination == graph.target_room and (
            graph.target_room not in next_state.unlocked_rooms or graph.switch_id not in next_state.activated_switches
        )
        if destination is None or locked_target:
            next_state.invalid_actions += 1
            observation = (
                f"Movement failed: exit {direction!r} is locked."
                if locked_target
                else "Movement failed: no exit in that direction."
            )
        else:
            next_state.current_room = destination
            observation = _observe(task, next_state)
    elif action.tool == "pickup":
        item = str(args.get("item", ""))
        if item in room.objects and item != graph.switch_id:
            if item not in next_state.inventory:
                next_state.inventory.append(item)
            observation = f"Picked up {item}."
        else:
            next_state.invalid_actions += 1
            observation = "Pickup failed."
    elif action.tool == "use":
        item = str(args.get("item", ""))
        if item == graph.switch_id and item in room.objects:
            if item not in next_state.activated_switches:
                next_state.activated_switches.append(item)
            observation = f"Activated {item}."
        elif (
            item == graph.key_item
            and item in next_state.inventory
            and next_state.current_room
            in {room_id for room_id, candidate in graph.rooms.items() if graph.target_room in candidate.exits.values()}
        ):
            if graph.target_room not in next_state.unlocked_rooms:
                next_state.unlocked_rooms.append(graph.target_room)
            observation = "The target lock is open; the safety switch must also be active."
        else:
            next_state.invalid_actions += 1
            observation = "Use failed."
    elif action.tool == "query":
        if room.terminal_code is None:
            next_state.invalid_actions += 1
            observation = "No useful terminal is present."
        else:
            next_state.discovered_code = room.terminal_code
            observation = json.dumps({"access_code": room.terminal_code})
    else:
        system = str(args.get("system", ""))
        code = str(args.get("code", ""))
        next_state.submitted = True
        next_state.success = (
            next_state.current_room == graph.target_room
            and system == graph.target_system
            and code == graph.access_code
            and next_state.discovered_code == graph.access_code
        )
        next_state.terminal = True
        observation = "Goal verified." if next_state.success else "Submission rejected."

    return next_state, observation


def replay(task: BenchmarkTask, actions: Iterable[Action]) -> list[Transition]:
    state = initial_state(task)
    transitions: list[Transition] = []
    for action in actions:
        state, observation = apply_action(task, state, action)
        transitions.append(Transition(action=action, observation=observation, state=state.model_copy(deep=True)))
    return transitions
