import json

import pytest
from symbolic_tool_calling_v1 import Action, apply_action, generate_task, initial_state, replay, validate_task


def optimal_actions(task):
    graph = task.hidden_graph_spec
    actions = [Action(tool="move", arguments={"direction": "east"})]
    actions.append(Action(tool="pickup", arguments={"item": graph.key_item}))
    for room_index in range(1, task.dependency_depth):
        if room_index == task.dependency_depth // 2:
            actions.append(Action(tool="use", arguments={"item": graph.switch_id}))
        if room_index == task.dependency_depth - 1:
            actions.append(Action(tool="use", arguments={"item": graph.key_item}))
            actions.append(Action(tool="query"))
        actions.append(Action(tool="move", arguments={"direction": "east"}))
    actions.append(Action(tool="submit", arguments={"system": graph.target_system, "code": graph.access_code}))
    return actions


def test_generation_is_deterministic_and_content_addressed():
    first = generate_task(123, horizon_bucket="long", imbalance_setting="high")
    second = generate_task(123, horizon_bucket="long", imbalance_setting="high")
    assert first.model_dump() == second.model_dump()
    assert first.task_id == second.task_id
    assert generate_task(124).task_id != first.task_id
    validate_task(first)


@pytest.mark.parametrize("bucket,depth", [("short", 3), ("medium", 5), ("long", 8)])
@pytest.mark.parametrize("seed", range(20))
def test_generated_graphs_satisfy_invariants(bucket, depth, seed):
    task = generate_task(
        seed,
        horizon_bucket=bucket,
        branching_factor=2,
        distractor_ratio=0.75,
        recovery_cost=3,
        verbosity_setting="high",
        imbalance_setting="high",
    )
    validate_task(task)
    assert task.dependency_depth == depth
    assert task.optimal_plan_length == depth + 5


def test_optimal_trace_succeeds_and_replays_exactly():
    task = generate_task(7, horizon_bucket="long")
    actions = optimal_actions(task)
    first = replay(task, actions)
    second = replay(task, actions)
    assert [transition.model_dump() for transition in first] == [transition.model_dump() for transition in second]
    assert first[-1].state.success


def test_broken_code_fails_exact_verifier():
    task = generate_task(9)
    state = initial_state(task)
    for action in optimal_actions(task)[:-1]:
        state, _ = apply_action(task, state, action)
    state, _ = apply_action(
        task,
        state,
        Action(tool="submit", arguments={"system": task.hidden_graph_spec.target_system, "code": "wrong"}),
    )
    assert state.terminal
    assert not state.success


def test_state_transitions_do_not_mutate_the_input_state():
    task = generate_task(5)
    original = initial_state(task)
    changed, _ = apply_action(task, original, Action(tool="move", arguments={"direction": "east"}))
    assert original.current_room == task.hidden_graph_spec.start_room
    assert changed.current_room != original.current_room


def test_target_requires_unlock_switch_code_and_location():
    task = generate_task(11, horizon_bucket="short")
    graph = task.hidden_graph_spec
    state = initial_state(task)
    state, _ = apply_action(task, state, Action(tool="move", arguments={"direction": "east"}))
    state, _ = apply_action(task, state, Action(tool="pickup", arguments={"item": graph.key_item}))
    state, observation = apply_action(task, state, Action(tool="move", arguments={"direction": "east"}))
    assert state.current_room == "room_02"
    state, observation = apply_action(task, state, Action(tool="move", arguments={"direction": "east"}))
    assert state.current_room == "room_02"
    assert "failed" in observation.lower()


def test_observations_expose_affordances_without_leaking_the_code():
    task = generate_task(13, horizon_bucket="short", distractor_ratio=0)
    graph = task.hidden_graph_spec
    state = initial_state(task)
    state, _ = apply_action(task, state, Action(tool="move", arguments={"direction": "east"}))
    state, _ = apply_action(task, state, Action(tool="pickup", arguments={"item": graph.key_item}))
    state, _ = apply_action(task, state, Action(tool="use", arguments={"item": graph.switch_id}))
    state, observation = apply_action(task, state, Action(tool="move", arguments={"direction": "east"}))
    visible = json.loads(observation)
    assert visible["access_terminal"] is True
    assert visible["locked_exits"] == ["east"]
    assert graph.access_code not in observation

    state, _ = apply_action(task, state, Action(tool="use", arguments={"item": graph.key_item}))
    state, observation = apply_action(task, state, Action(tool="inspect"))
    assert json.loads(observation)["locked_exits"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_bucket": "unknown"},
        {"branching_factor": 0},
        {"branching_factor": 3},
        {"distractor_ratio": -0.1},
        {"distractor_ratio": 1.1},
        {"recovery_cost": 0},
    ],
)
def test_invalid_generator_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_task(1, **kwargs)
