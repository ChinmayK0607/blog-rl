import hashlib
import json
import random

from symbolic_tool_calling_v1.models import BenchmarkTask, HiddenGraphSpec, RoomSpec

_DEPTHS = {"short": 3, "medium": 5, "long": 8}


def _task_id(payload: dict) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"stc-{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"


def generate_task(
    seed: int,
    *,
    horizon_bucket: str = "medium",
    branching_factor: int = 2,
    distractor_ratio: float = 0.5,
    recovery_cost: int = 2,
    verbosity_setting: str = "low",
    imbalance_setting: str = "low",
) -> BenchmarkTask:
    if horizon_bucket not in _DEPTHS:
        raise ValueError(f"unknown horizon bucket: {horizon_bucket}")
    if branching_factor < 1:
        raise ValueError("branching_factor must be at least 1")
    if branching_factor > 2:
        raise ValueError("branching_factor cannot exceed the two available side exits")
    if not 0 <= distractor_ratio <= 1:
        raise ValueError("distractor_ratio must be in [0, 1]")
    if recovery_cost < 1:
        raise ValueError("recovery_cost must be at least 1")

    rng = random.Random(seed)
    depth = _DEPTHS[horizon_bucket]
    room_ids = [f"room_{i:02d}" for i in range(depth + 1)]
    access_code = f"{rng.randrange(100000, 1000000)}"
    key_item = f"key_{rng.randrange(1000):03d}"
    switch_id = f"switch_{rng.randrange(1000):03d}"
    target_system = f"vault_{rng.randrange(1000):03d}"
    rooms: dict[str, RoomSpec] = {}

    for i, room_id in enumerate(room_ids):
        exits: dict[str, str] = {}
        if i:
            exits["west"] = room_ids[i - 1]
        if i < depth:
            exits["east"] = room_ids[i + 1]
        objects: list[str] = []
        if i == 1:
            objects.append(key_item)
        if i == depth // 2:
            objects.append(switch_id)
        if i == depth:
            objects.append(target_system)
        terminal_code = access_code if i == depth - 1 else None
        verbosity_text = ""
        if verbosity_setting == "high":
            verbosity_text = " Diagnostic panels report nominal readings; archived labels are irrelevant."
        rooms[room_id] = RoomSpec(
            room_id=room_id,
            exits=exits,
            objects=tuple(objects),
            terminal_code=terminal_code,
            verbosity_text=verbosity_text,
        )

    side_directions = ("north", "south")[: min(branching_factor, 2)]
    distractor_slots = [(parent_id, direction) for parent_id in room_ids[:-1] for direction in side_directions]
    rng.shuffle(distractor_slots)
    distractor_count = round(len(distractor_slots) * distractor_ratio)
    for i, (parent_id, direction) in enumerate(distractor_slots[:distractor_count]):
        distractor_id = f"diagnostic_{i:02d}_00"
        parent = rooms[parent_id]
        rooms[parent_id] = parent.model_copy(update={"exits": {**parent.exits, direction: distractor_id}})
        previous_id = parent_id
        for recovery_step in range(recovery_cost):
            current_id = f"diagnostic_{i:02d}_{recovery_step:02d}"
            next_id = f"diagnostic_{i:02d}_{recovery_step + 1:02d}" if recovery_step + 1 < recovery_cost else None
            exits = {"back": previous_id}
            if next_id is not None:
                exits["deeper"] = next_id
            rooms[current_id] = RoomSpec(
                room_id=current_id,
                exits=exits,
                objects=((f"scrap_{i:02d}",) if next_id is None else ()),
                verbosity_text=(
                    " A lengthy diagnostic dump contains no access credentials." if imbalance_setting == "high" else ""
                ),
            )
            previous_id = current_id

    graph = HiddenGraphSpec(
        start_room=room_ids[0],
        target_room=room_ids[-1],
        target_system=target_system,
        access_code=access_code,
        key_item=key_item,
        switch_id=switch_id,
        rooms=rooms,
    )
    payload = {
        "seed": seed,
        "horizon_bucket": horizon_bucket,
        "branching_factor": branching_factor,
        "distractor_ratio": distractor_ratio,
        "recovery_cost": recovery_cost,
        "verbosity_setting": verbosity_setting,
        "imbalance_setting": imbalance_setting,
        "hidden_graph_spec": graph.model_dump(mode="json"),
    }
    return BenchmarkTask(
        task_id=_task_id(payload),
        seed=seed,
        horizon_bucket=horizon_bucket,
        dependency_depth=depth,
        branching_factor=branching_factor,
        distractor_ratio=distractor_ratio,
        recovery_cost=recovery_cost,
        verbosity_setting=verbosity_setting,
        imbalance_setting=imbalance_setting,
        hidden_graph_spec=graph,
        optimal_plan_length=depth + 5,
    )
