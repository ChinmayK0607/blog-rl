from collections import Counter, deque

from symbolic_tool_calling_v1.models import BenchmarkTask


def validate_task(task: BenchmarkTask) -> None:
    """Raise ValueError when a generated task violates benchmark invariants."""
    graph = task.hidden_graph_spec
    if graph.start_room not in graph.rooms or graph.target_room not in graph.rooms:
        raise ValueError("start and target rooms must exist")
    for room_id, room in graph.rooms.items():
        if room.room_id != room_id:
            raise ValueError(f"room key/id mismatch for {room_id}")
        for destination in room.exits.values():
            if destination not in graph.rooms:
                raise ValueError(f"exit from {room_id} points to missing room {destination}")

    reachable = {graph.start_room}
    queue = deque([graph.start_room])
    while queue:
        room_id = queue.popleft()
        for destination in graph.rooms[room_id].exits.values():
            if destination not in reachable:
                reachable.add(destination)
                queue.append(destination)
    if reachable != set(graph.rooms):
        raise ValueError(f"unreachable rooms: {sorted(set(graph.rooms) - reachable)}")

    objects = Counter(item for room in graph.rooms.values() for item in room.objects)
    for required in (graph.key_item, graph.switch_id, graph.target_system):
        if objects[required] != 1:
            raise ValueError(f"required object {required} must occur exactly once")
    terminals = [room for room in graph.rooms.values() if room.terminal_code == graph.access_code]
    if len(terminals) != 1:
        raise ValueError("access code must occur on exactly one terminal")
    if task.optimal_plan_length != task.dependency_depth + 5:
        raise ValueError("optimal plan length does not match the dependency chain")
